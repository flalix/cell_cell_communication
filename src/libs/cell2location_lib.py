#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Helpers for notebooks/cell2location01_tutorial.ipynb (human lymph node Visium + lymphoid-organ scRNA reference).

Sections
--------
1. Feature lookup and clipping for spatial plots   : _values, resolve_gene_ids, percentile_cap, clip_for_plot
2. Tissue x age-group summaries of the reference   : age_group, add_tissue_age_group, panel_lognorm,
                                                     create_gene_age_tissue_groups, plot_panel_umap_by_tissue_age
3. cell2location model utilities                   : get_reference_signatures, ensure_posterior_samples

Usage (notebook)
----------------
    ROOT_SRC = Path("/home/flavio/uv/squidpy/src")
    sys.path.append(str(ROOT_SRC))
    from libs.cell2location_lib import *

Notes on the lymphoid-organ reference (lymphoid_organ_sc.h5ad, 73,260 cells x 10,237 genes)
------------------------------------------------------------------------------------------
- X holds raw UMI counts; obs['n_counts'] is the library size computed before gene filtering.
- After the tutorial's var.set_index('GeneID-2'), var_names are Ensembl IDs and symbols live in var['symbol'];
  all gene lookups here go through var[symbol_col] when present.
- Tissue and age are confounded (Tonsil = babies/young only; Spleen/iLN/tLN = adult, one donor; mLN spans
  adult / old adult / elder), so between-group differences are not pure age effects.

@created on 2026/10/16
@author: Flavio Lichtenstein
"""

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import scanpy as sc

AGE_ORDER = ("babies", "young", "adult", "old adult", "elder")
TISSUE_ORDER = ("Tonsil", "Spleen", "iLN", "tLN", "mLN")

__all__ = [
    "AGE_ORDER", "TISSUE_ORDER",
    "resolve_gene_ids", "percentile_cap", "clip_for_plot",
    "age_group", "add_tissue_age_group", "panel_lognorm", "create_gene_age_tissue_groups",
    "plot_panel_umap_by_tissue_age", "plot_genes_per_cell_type",
    "get_reference_signatures", "ensure_posterior_samples",
]

__version__ = '1.0.0'

# ---------------------------------------------------------------------------------------------------------
# 1. Feature lookup and clipping
# ---------------------------------------------------------------------------------------------------------
def _symbol_series(adata, symbol_col: str = "symbol") -> pd.Series:
    """var_names -> gene symbol; falls back to var_names when var[symbol_col] is absent."""
    if symbol_col in adata.var:
        return adata.var[symbol_col].astype(str)
    return pd.Series(adata.var_names, index=adata.var_names)


def _values(adata, gene: str, symbol_col: str = "symbol"):
    """Return a 1-D array for an obs column, a var_name, or a gene symbol."""
    if gene in adata.obs:
        return adata.obs[gene]
    if gene in adata.var_names:
        gid = gene
    else:
        hits = adata.var_names[_symbol_series(adata, symbol_col).values == gene]
        if len(hits) == 0:
            raise KeyError(f"{gene} not found in obs, var_names, or var['{symbol_col}']")
        gid = hits[0]
    x = adata[:, gid].X
    return np.asarray(x.toarray() if sp.issparse(x) else x).ravel()


def resolve_gene_ids(adata, genes: Iterable[str], symbol_col: str = "symbol") -> Tuple[List[str], List[str]]:
    """Map gene symbols to var_names (symbol or Ensembl index). Missing genes are warned about and skipped.

    Returns (found_symbols, var_ids) in the input order.
    """
    sym = _symbol_series(adata, symbol_col)
    found = [g for g in genes if (sym == g).any()]
    missing = [g for g in genes if g not in found]
    if missing:
        warnings.warn(f"not in var (check aliases, e.g. CD20 -> MS4A1): {missing}")
    ids = [sym.index[sym == g][0] for g in found]
    return found, ids


def percentile_cap(v: np.ndarray, q: float = 99.2) -> float:
    """q-th percentile of v; if that is 0 (gene detected in < (100-q)% of spots/cells),
    use the q-th percentile of the non-zero values so the colour scale does not collapse."""
    v = np.asarray(v, dtype=float)
    cap = float(np.percentile(v, q))
    if cap == 0:
        nz = v[v > 0]
        cap = float(np.percentile(nz, q)) if nz.size else 1.0
    return cap


def clip_for_plot(adata, genes: Sequence[str], q: float = 99.2, vmin: float = 0, symbol_col: str = "symbol",
                  sparse_fallback: bool = True) -> Tuple[List[str], List[str]]:
    """Write per-feature percentile-clipped copies to adata.obs['_plot_<feature>'].

    Features may be obs columns or gene symbols / var_names; unknown features are warned about and dropped.
    Categorical obs columns are copied unclipped. With sparse_fallback=True, genes whose q-th percentile
    is 0 are capped at the q-th percentile of their non-zero values (see percentile_cap).

    Returns (features_kept, obs_column_names) for e.g.
        sq.pl.spatial_scatter(adata, color=cols, title=features_kept, ...)
    """
    sym = set(_symbol_series(adata, symbol_col))
    kept = [g for g in genes if g in adata.obs or g in adata.var_names or g in sym]
    dropped = [g for g in genes if g not in kept]
    if dropped:
        warnings.warn(f"not found in obs / var_names / var['{symbol_col}'], dropped: {dropped}")

    cols = []
    for gene in kept:
        v = _values(adata, gene, symbol_col)
        col = f"_plot_{gene}"
        if hasattr(v, "cat"):  # categorical obs: no clipping
            adata.obs[col] = v
        else:
            v = np.asarray(v, dtype=float)
            vmax = percentile_cap(v, q) if sparse_fallback else float(np.percentile(v, q))
            adata.obs[col] = np.clip(v, vmin, vmax)
        cols.append(col)
    return kept, cols


# ---------------------------------------------------------------------------------------------------------
# 2. Tissue x age-group summaries of the scRNA reference
# ---------------------------------------------------------------------------------------------------------
def age_group(s) -> str:
    """'0-4y' / '20-25y' ... -> babies (<5), young (<14), adult (<34), old adult (<55), elder (>=55).
    Uses the lower bound of the Age bin."""
    lo = int(str(s).split("-")[0])
    return ("babies" if lo < 5 else "young" if lo < 14 else "adult" if lo < 34
            else "old adult" if lo < 55 else "elder")


def add_tissue_age_group(adata, age_col: str = "Age", tissue_col: str = "Tissue") -> List[Tuple[str, str]]:
    """Write adata.obs['age_group'] (in place) and return the ordered (tissue, age_group) pairs that contain cells."""
    obs = adata.obs
    obs["age_group"] = obs[age_col].map(age_group)
    return [(t, g) for g in AGE_ORDER for t in TISSUE_ORDER
            if ((obs[tissue_col] == t) & (obs["age_group"] == g)).any()]


def panel_lognorm(adata, genes: Sequence[str], symbol_col: str = "symbol",
                  libsize_col: str = "n_counts") -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Raw counts and log1p(CP10k) for the panel genes only (no full-matrix copy).

    Library size = obs[libsize_col] (pre-gene-filter totals in the lymphoid reference).
    Returns (found_symbols, counts[n_cells, n_genes], log1p_cp10k[n_cells, n_genes]).
    """
    found, ids = resolve_gene_ids(adata, genes, symbol_col)
    X = adata[:, ids].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    E = np.log1p(X / adata.obs[libsize_col].values[:, None] * 1e4)
    return found, X, E


def create_gene_age_tissue_groups(adata, genes_plot: Sequence[str], symbol_col: str = "symbol") -> pd.DataFrame:
    """Per tissue x age group and gene: n_cells, pct_expr (% cells with >0 counts), mean_log1p_cp10k.

    Writes adata.obs['age_group'] as a side effect.
    """
    order = add_tissue_age_group(adata)
    found, X, E = panel_lognorm(adata, genes_plot, symbol_col)
    obs = adata.obs
    rows = []
    for t, g in order:
        m = ((obs.Tissue == t) & (obs.age_group == g)).values
        for i, gene in enumerate(found):
            rows.append(dict(tissue=t, age_group=g, gene=gene, n_cells=int(m.sum()),
                             pct_expr=round(100 * (X[m, i] > 0).mean(), 1),
                             mean_log1p_cp10k=round(float(E[m, i].mean()), 3)))
    return pd.DataFrame(rows)


def plot_panel_umap_by_tissue_age(adata, genes: Sequence[str], out_png: Optional[str] = None,
                                  symbol_col: str = "symbol", q: float = 99.2, basis: str = "X_umap",
                                  labels: Optional[Dict[str, str]] = None, n_bg: int = 20000, seed: int = 0,
                                  cmap_num: int = 1, dpi: int = 200):
    """UMAP grid: rows = genes, columns = tissue x age groups.

    Colour = log1p(CP10k) clipped per gene at the q-th percentile (non-zero fallback for sparse genes),
    one scale per gene row shared across columns; grey = random subsample of all cells.
    labels maps symbol -> display label (e.g. {'MS4A1': 'MS4A1 (CD20)'}). Returns the matplotlib Figure.
    """

    labels = labels or {}
    order = add_tissue_age_group(adata)
    found, _, E = panel_lognorm(adata, genes, symbol_col)
    cap = np.array([percentile_cap(E[:, i], q) for i in range(E.shape[1])])
    C = np.clip(E, 0, cap)
    U = adata.obsm[basis]
    obs = adata.obs
    rng = np.random.default_rng(seed)
    bg = rng.choice(len(U), min(n_bg, len(U)), replace=False)

    if cmap_num == 1:
        cmap = 'RdBu_r'
    elif cmap_num == 2:
        colors = ("#57a4e2", "#ebd246", "#f30707")
        cmap = LinearSegmentedColormap.from_list("blue_yellow_red", list(colors))
    else:
        colors = ("#e6e6e6", "#fdbb84", "#e34a33", "#7f0000")   # light grey -> dark red (sequential)
        cmap = LinearSegmentedColormap.from_list("expr", list(colors))

    nr, nc = len(found), len(order)
    fig, axes = plt.subplots(nr, nc, figsize=(1.35 * nc + 0.9, 1.2 * nr + 0.8), squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=0.06))
    for j, (t, g) in enumerate(order):
        m = np.where(((obs.Tissue == t) & (obs.age_group == g)).values)[0]
        nd = obs["Donor"].iloc[m].nunique() if "Donor" in obs else 0
        for i, gene in enumerate(found):
            ax = axes[i, j]
            ax.scatter(U[bg, 0], U[bg, 1], s=0.3, c="#f2f2f2", lw=0, rasterized=True)
            v = C[m, i]
            o = np.argsort(v)  # highest expression drawn on top
            ax.scatter(U[m[o], 0], U[m[o], 1], s=0.6, c=v[o], cmap=cmap, vmin=0, vmax=cap[i],
                       lw=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"{t} · {g}\n{len(m) / 1000:.1f}k cells\n{nd} donor{'s' if nd != 1 else ''}",
                             fontsize=7)
            if j == 0:
                ax.set_ylabel(labels.get(gene, gene), fontsize=8, style="italic", rotation=0,
                              ha="right", va="center")

    for i in range(nr):
        pos = axes[i, -1].get_position()
        cax = fig.add_axes([pos.x1 + 0.006, pos.y0 + pos.height * 0.15, 0.006, pos.height * 0.7])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, cap[i]), cmap=cmap), cax=cax)
        cb.set_ticks([0, round(float(cap[i]), 1)])
        cb.ax.tick_params(labelsize=6, length=2)

    fig.text(0.5, 0.075, f"colour = log1p(CP10k) clipped at the {q}th percentile per gene (non-zero values for "
             "very sparse genes), one scale per row. Grey = all other cells.", ha="center", fontsize=7)
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    return fig



def plot_genes_per_cell_type(slide, genes, ctypes):
    """Spatial maps of total expression (column 0) and cell-type-specific expected expression
    (one column per cell type) for each gene (rows).

    Requires slide.layers[<cell type>] for every factor in slide.uns['mod']['factor_names'],
    written from mod.module.model.compute_expected_per_cell_type(...) (sparse csr layers),
    and slide.var['symbol']. Returns (fig, axs).
    """

    n_genes = len(genes)
    n_ctypes = len(ctypes)
    fig, axs = plt.subplots(
        nrows=n_genes, ncols=n_ctypes + 1, figsize=(4.5 * (n_ctypes + 1) + 2, 5 * n_genes + 1), squeeze=False
    )
    # axs = axs.reshape((n_genes, n_ctypes+1))

    colors = ("#57a4e2", "#ebd246", "#f30707")
    cmap = LinearSegmentedColormap.from_list("blue_yellow_red", list(colors))

    # plots of every gene
    for j in range(n_genes):
        # limit color scale at 99.2% quantile of gene expression (computed across cell types)
        quantile_across_ct = np.array(
            [
                np.quantile(slide.layers[n][:, (slide.var["symbol"] == genes[j]).values].toarray(), 0.992)
                for n in slide.uns["mod"]["factor_names"]
            ]
        )
        quantile_across_ct = np.partition(quantile_across_ct.flatten(), -2)[-2]
        sc.pl.spatial(
            slide,
            cmap=cmap,
            color=genes[j],
            # layer=ctypes[i],
            gene_symbols="symbol",
            ncols=4,
            size=1.3,
            img_key="hires",
            # limit color scale at 99.2% quantile of gene expression
            vmin=0,
            vmax="p99.2",
            ax=axs[j, 0],
            show=False,
        )

        # plots of every cell type
        for i in range(n_ctypes):
            sc.pl.spatial(
                slide,
                cmap=cmap,
                color=genes[j],
                layer=ctypes[i],
                gene_symbols="symbol",
                ncols=4,
                size=1.3,
                img_key="hires",
                # limit color scale at 99.2% quantile of gene expression
                vmin=0,
                vmax=quantile_across_ct,
                ax=axs[j, i + 1],
                show=False,
            )
            axs[j, i + 1].set_title(f"{genes[j]} {ctypes[i]}")

    return fig, axs


# ---------------------------------------------------------------------------------------------------------
# 3. cell2location model utilities
# ---------------------------------------------------------------------------------------------------------
def get_reference_signatures(adata_ref, prefer: Sequence[str] = ("means", "q50", "q05")) -> Tuple[pd.DataFrame, str]:
    """Per-cell-type expression signatures (genes x cell types) exported by RegressionModel.export_posterior.

    Tries each summary in `prefer` in .varm then .var. export_posterior(use_quantiles=True) writes only
    q05/q50/q95/q0001 (no 'means'), which is why a hard-coded 'means_per_cluster_mu_fg' lookup raises KeyError.
    Returns (inf_aver, summary_used).
    """
    factor_names = adata_ref.uns["mod"]["factor_names"]
    for summary in prefer:
        key = f"{summary}_per_cluster_mu_fg"
        cols = [f"{key}_{i}" for i in factor_names]
        if key in adata_ref.varm:
            inf_aver = adata_ref.varm[key][cols].copy()
        elif cols[0] in adata_ref.var:
            inf_aver = adata_ref.var[cols].copy()
        else:
            continue
        inf_aver.columns = factor_names
        return inf_aver, summary
    raise KeyError(f"No per-cluster signatures for {list(prefer)}; rerun mod.export_posterior(adata_ref, ...)")


def ensure_posterior_samples(mod, adata, quantiles: Sequence[str] = ("q05", "q50", "q95", "q0001"),
                             accelerator: str = "gpu", batch_size: Optional[int] = None, spatial: bool = True):
    """Re-create mod.samples after a model was re-loaded from disk (mod.samples is not saved with the model).

    Needed before mod.plot_QC(...) or mod.samples['post_sample_q05'] on a loaded model; otherwise
    'RuntimeError: self.samples is missing' / "'Cell2location' object has no attribute 'samples'".
    No-op if mod.samples already exists. Returns the AnnData returned by export_posterior (or `adata` unchanged).
    spatial=True -> Cell2location (writes obsm); spatial=False -> RegressionModel (writes varm).
    """
    if getattr(mod, "samples", None):
        return adata
    if batch_size is None:
        batch_size = mod.adata.n_obs if spatial else 2500
    slot = {"add_to_obsm": list(quantiles)} if spatial else {"add_to_varm": list(quantiles)}
    return mod.export_posterior(adata, use_quantiles=True,
                                sample_kwargs={"batch_size": batch_size, "accelerator": accelerator}, **slot)

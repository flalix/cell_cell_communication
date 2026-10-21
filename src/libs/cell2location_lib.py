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
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

import scanpy as sc

import scipy.sparse as sp
from scipy.stats import spearmanr
import scipy.sparse as sps

from libs.Basic import pdwritecsv, pdreadcsv

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

colors = ("#57a4e2", "#ebd246", "#f30707")
cmap_byr = LinearSegmentedColormap.from_list("blue_yellow_red", list(colors))

bcell_cmaps={"GC B":"Reds","naive/mem B":"Blues","plasma":"Purples","T":"Greens","NK/ILC":"GnBu","myeloid":"Oranges","stromal":"Greys"}


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


def wtab(btype, rgg):
    dic = {"gene":rgg["names"][btype],
           "lfc_vs_restB":rgg["logfoldchanges"][btype],
           "padj":rgg["pvals_adj"][btype],
           "score":rgg["scores"][btype]}
    
    df=pd.DataFrame(dic).set_index("gene")
    return df

def donor_consistency(adata_bcell, gene, btype, btypes, donors, grp, Xn, cnt):
    j = adata_bcell.var_names.get_loc(gene)
    col = np.asarray(Xn[:,j].todense()).ravel() if sp.issparse(Xn) else Xn[:,j]
    ok=tot=0
    for dn in cnt.index:
        if cnt.loc[dn,btype]<20: continue
        others=[o for o in btypes if o!=btype and cnt.loc[dn,o]>=20]
        if not others: continue
        mi=col[(donors==dn)&(grp==btype)].mean(); mo=max(col[(donors==dn)&(grp==o)].mean() for o in others)
        tot+=1; ok+=mi>mo
    return ok,tot



# import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def isg_hotspot(adata, ISG_GENES, genes=None, q=0.97, symbol_col="symbol"):
    """ISG score per spot (mean z-score of log1p(CP10k)) and the largest spatially connected
    cluster of spots above the q-th quantile. Returns (score Series, hotspot bool array)."""
    if genes is None:
        genes = ISG_GENES
    sym = adata.var[symbol_col].astype(str).values
    tot = np.asarray(adata.X.sum(1)).ravel()
    cols = {}
    for g in genes:
        j = np.where(sym == g)[0]
        if len(j) == 0:
            continue
        x = adata.X[:, j[0]]
        x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
        cols[g] = np.log1p(x / tot * 1e4)
    Z = pd.DataFrame(cols, index=adata.obs_names)
    score = ((Z - Z.mean()) / Z.std()).mean(1)

    xy = adata.obsm["spatial"].astype(float)
    step = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])        # centre-to-centre spot distance
    idx = np.where(score.values >= np.quantile(score, q))[0]
    pairs = np.array(list(cKDTree(xy[idx]).query_pairs(r=1.6 * step)))
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
    _, lab = connected_components(A, directed=False)
    hs = np.zeros(adata.n_obs, bool)
    hs[idx[lab == np.bincount(lab).argmax()]] = True
    return score, hs


def spot_expr(adata_vis, gene):
    """log1p(CP10k) of one gene per spot, looked up by symbol."""
    j = np.where(adata_vis.var["symbol"].astype(str).values == gene)[0][0]
    x = adata_vis.X[:, j]
    x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return np.log1p(x / np.asarray(adata_vis.X.sum(1)).ravel() * 1e4)


def run_isg_hotspot(adata_vis, ISG_GENES: list, root_results:Path, root_figure: Path):
    '''
    Run ISG hotspot detection and visualization.

    spot_xy[hs] uses two variables that exist only in my kernel:
    spot_xy: the spot centres in hires-image pixels, adata_vis.obsm["spatial"] * tissue_hires_scalef. You already have this as xy (cell 116).
    hs: a boolean mask over the 4,035 spots that is True for the 120 hotspot spots, so P[hs] is the coordinates of the hotspot spots.

    '''

    # ---- inputs (needs adata_vis after cell 97, and isg_hotspot() from the previous cell) ----
    isg_score, hs = isg_hotspot(adata_vis, ISG_GENES)
    Q_ab = pd.DataFrame(np.asarray(adata_vis.obsm["q05_cell_abundance_w_sf"]),
                        index=adata_vis.obs_names, columns=list(adata_vis.uns["mod"]["factor_names"]))

    lib_id  = list(adata_vis.uns["spatial"].keys())[0]
    hires   = adata_vis.uns["spatial"][lib_id]["images"]["hires"]
    sf      = adata_vis.uns["spatial"][lib_id]["scalefactors"]["tissue_hires_scalef"]
    spot_xy = adata_vis.obsm["spatial"] * sf                       # spot centres in hires pixels

    # ---- cell-type enrichment in the hotspot vs rest ----
    rows = []
    for c in Q_ab.columns:
        a, b = Q_ab.loc[hs, c], Q_ab.loc[~hs, c]
        rows.append(dict(cell_type=c, mean_hot=a.mean(), mean_rest=b.mean(),
                        log2fc=np.log2((a.mean() + 0.05) / (b.mean() + 0.05)), p=mannwhitneyu(a, b).pvalue))
    hot_res = pd.DataFrame(rows).sort_values("log2fc", ascending=False)
    hot_res["padj"] = multipletests(hot_res.p, method="fdr_bh")[1]
    hot_sel = pd.concat([hot_res.head(8), hot_res.tail(7)]).set_index("cell_type").iloc[::-1]

    # ---- figure ----
    maps = [("ISG score (mean z of 12 ISGs)", isg_score.values, "a"),
            ("B_IFN abundance (q05)",         Q_ab["B_IFN"].values, "b"),
            ("CXCL10 log1p(CP10k)",           spot_expr(adata_vis, "CXCL10"), "c"),
            ("DC_CCR7+ abundance (q05)",      Q_ab["DC_CCR7+"].values, "d")]

    hull = ConvexHull(spot_xy[hs]); hv = np.append(hull.vertices, hull.vertices[0])
    x0, x1 = spot_xy[:, 0].min() - 30, spot_xy[:, 0].max() + 30
    y0, y1 = spot_xy[:, 1].min() - 30, spot_xy[:, 1].max() + 30

    fig = plt.figure(figsize=(14, 7.4))
    gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], wspace=0.55, hspace=0.2)
    for k, (title, v, let) in enumerate(maps):
        ax = fig.add_subplot(gs[k // 2, k % 2]); ax.imshow(hires, alpha=0.5)
        vmin = np.percentile(v, 1) if v.min() < 0 else 0; vmax = np.percentile(v, 99.2); o = np.argsort(v)
        s = ax.scatter(spot_xy[o, 0], spot_xy[o, 1], c=np.clip(v[o], vmin, vmax), cmap="magma",
                    vmin=vmin, vmax=vmax, s=4.5, lw=0)
        ax.plot(spot_xy[hs][hv, 0], spot_xy[hs][hv, 1], color="#00e5ff", lw=1.0, ls="--")
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
        ax.set_title(title, loc="left", fontsize=8)
        ax.text(-0.06, 1.04, let, transform=ax.transAxes, fontweight="bold", fontsize=10)
        cb = fig.colorbar(s, ax=ax, fraction=0.04, pad=0.01); cb.ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[:, 2])
    ax.barh(range(len(hot_sel)), hot_sel.log2fc, height=0.7,
            color=["#b2182b" if x > 0 else "#2166ac" for x in hot_sel.log2fc])
    ax.axvline(0, color="#888", lw=0.8)
    ax.set_yticks(range(len(hot_sel))); ax.set_yticklabels(hot_sel.index)
    for i, (fc, pa) in enumerate(zip(hot_sel.log2fc, hot_sel.padj)):
        if pa >= 0.05: ax.text(fc + 0.05, i, "n.s.", va="center", ha="left", fontsize=6, color="#555")
    ax.set_xlabel("log2 fold change, hotspot vs rest\n(mapped q05 abundance)")
    ax.set_title("Cell types enriched / depleted in the hotspot", loc="left", fontsize=8)
    ax.text(-0.42, 1.02, "e", transform=ax.transAxes, fontweight="bold", fontsize=10); ax.margins(x=0.08)
    fig.text(0.1, 0.03, f"Dashed cyan = hull of the hotspot ({hs.sum()} of {adata_vis.n_obs:,} spots). "
            "V1_Human_Lymph_Node, one section, one donor.", fontsize=7)

    fig.savefig(root_figure / "ifn_hotspot_summary.png", dpi=200, bbox_inches="tight")
    plt.show()
    hot_res.round(4).to_csv(root_results / "ifn_hotspot_celltype_enrichment.csv", index=False)

    return hot_res

def region_names(adata, key='region_cluster',
                 obsm_key='q05_cell_abundance_w_sf', top_n=2):
    """Name each region cluster by its top enriched cell types (cluster mean / global mean)."""
    ab = pd.DataFrame(adata.obsm[obsm_key], index=adata.obs_names)
    ab.columns = [c.replace('q05cell_abundance_w_sf_', '') for c in ab.columns]
    mean_by_cl = ab.groupby(adata.obs[key], observed=False).mean()
    enrich = mean_by_cl / ab.mean(axis=0)            # >1 = enriched in that region
    return {cl: ' / '.join(enrich.loc[cl].nlargest(top_n).index)
            for cl in enrich.index}


def legend_figure_bottom(fig, adata, key, names=None, ncol=4, fontsize=9):
    """Horizontal legend spanning the whole bottom of the figure: 'number: name'."""
    cats = adata.obs[key].cat.categories
    colors = adata.uns[f'{key}_colors']
    names = names or {}
    handles = [
        Line2D([], [], marker='o', linestyle='', markersize=7,
               markerfacecolor=col, markeredgecolor='none',
               label=f'{cat}: {names[cat]}' if cat in names else str(cat))
        for cat, col in zip(cats, colors)
    ]
    fig.legend(handles=handles, loc='outside lower center',
               ncol=min(ncol, len(handles)), frameon=False, fontsize=fontsize,
               handletextpad=0.3, columnspacing=1.5)


def umap_spatial_3plot(adata_vis, root_figure):
    # compute UMAP using KNN graph based on the cell2location output
    sc.tl.umap(adata_vis, min_dist=0.3, spread=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), layout='constrained')

    sc.pl.umap(adata_vis, color='region_cluster', size=30,
            legend_loc='on data', legend_fontsize=20,
            ax=axes[0], show=False, title='region_cluster')

    sc.pl.umap(adata_vis, color='sample', size=30,
            legend_loc='none',
            ax=axes[1], show=False, title='sample')

    axes[2].set_facecolor('black')
    sc.pl.spatial(adata_vis, color='region_cluster',
                ax=axes[2], size=1.3, img_key='hires', alpha=0.5,
                title='spatial', legend_loc='none', show=False)

    names = region_names(adata_vis, top_n=2)
    legend_figure_bottom(fig, adata_vis, 'region_cluster', names=names, ncol=4)

    fig.savefig(root_figure / 'umap_region_sample.png', dpi=300, bbox_inches='tight')
    plt.show()


def umap_cell_type_plot(adata_vis, adata_ref, genes_plot, lineage, focus, root_figure):
    '''
    Per tissue x age-group table
    - Per tissue x age-group summary of the B-cell gene panel in the lymphoid-organ scRNA reference.
      n_cells, % cells with >0 counts, mean log1p(CP10k). Functions in libs/cell2location_lib.py.


    Figure 1: UMAP by cell type + dot plot

    '''
    from libs.matplotlib_figure_style import apply_figure_style
    apply_figure_style(sizes=(8, 7, 6))

    # cell types: ct
    ct = adata_ref.obs["Subset"].astype(str).values          # cell-type label per cell

    # found = genes present in the reference; X = raw counts; E = log1p(CP10k)   (cells x genes)
    found, X, E = panel_lognorm(adata_ref, genes_plot)

    df_pct  = pd.DataFrame(X > 0, columns=found).groupby(ct).mean() * 100   # % cells with >0 counts
    df_mean = pd.DataFrame(E,     columns=found).groupby(ct).mean()         # mean log1p(CP10k)
    n_ct = pd.Series(ct).value_counts()  

    ct_order = [c for v in lineage.values() for c in v]
    assert set(ct_order) == set(ct), set(ct) ^ set(ct_order)

    P = df_pct.loc[ct_order, found]
    M = df_mean.loc[ct_order, found]
    U = adata_ref.obsm["X_umap"]

    lib = list(adata_vis.uns["spatial"].keys())[0]                       # 'V1_Human_Lymph_Node'

    img = adata_vis.uns["spatial"][lib]["images"]["hires"]
    sf  = adata_vis.uns["spatial"][lib]["scalefactors"]["tissue_hires_scalef"]
    xy  = adata_vis.obsm["spatial"] * sf                                 # spot centres in hires-image pixels


    order=[c for v in lineage.values() for c in v]; assert set(order)==set(ct), set(ct)^set(order)
    col={}
    for k,v in lineage.items():
        cm=plt.get_cmap(bcell_cmaps[k]); 
        for j,c in enumerate(v): col[c]=cm(0.45+0.45*(j/(max(len(v)-1,1))))

    # Per tissue x age-group table
    dfg = create_gene_age_tissue_groups(adata_ref, genes_plot)   # also writes adata_ref.obs['age_group']
    # dfg.to_csv(root0 / 'results' / 'bcell_genes_by_tissue_age.csv', index=False)


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,8))

    ax=axes[0]
    rng = np.random.default_rng(0); o=rng.permutation(len(U))
    ax.scatter(U[o,0],U[o,1],s=0.4,c=[col[c] for c in ct[o]],lw=0,rasterized=True)

    for c in focus:
        m=ct==c; x,y=np.median(U[m],0)
        ax.text(x,y,c,fontsize=7,ha="center",va="center",path_effects=[pe.withStroke(linewidth=2.2,foreground="white")])

    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("Reference UMAP coloured by cell type (Subset); labelled = types the panel genes point to",loc="left",fontsize=8)

    h=[Line2D([],[],marker="o",ls="",color=plt.get_cmap(bcell_cmaps[k])(0.75),label=k) for k in lineage]
    ax.legend(handles=h,loc="lower left",frameon=False,fontsize=7,markerscale=1.2)
    ax.text(0.0,-0.02,"UMAP 1 →",transform=ax.transAxes,fontsize=6,va="top"); ax.text(-0.02,0.0,"UMAP 2 →",transform=ax.transAxes,fontsize=6,rotation=90,ha="right")

    ax2=axes[1]
    P=df_pct.loc[order,found]
    M=df_mean.loc[order,found]
    yy,xx=np.meshgrid(np.arange(len(order)),np.arange(len(found)),indexing="ij")
    sca=ax2.scatter(xx.ravel(),yy.ravel(),s=(P.values.ravel()/100)*60,c=M.values.ravel(),cmap="magma_r",vmin=0,vmax=np.nanmax(M.values),lw=0)
    ax2.set_xticks(range(len(found)))
    ax2.set_xticklabels([("MS4A1 (CD20)" if g=="MS4A1" else g) for g in found],rotation=90,style="italic")
    ax2.set_yticks(range(len(order)))

    n_ct = pd.Series(ct).value_counts()          # cells per cell type (skip if already defined)
    ax2.set_yticklabels([f"{c} (n={n_ct[c]:,})" for c in order], fontsize=6)


    for t,c in zip(ax2.get_yticklabels(),order): t.set_color(col[c] if c in focus else "#555555"); t.set_fontweight("bold" if c in focus else "normal")
    ax2.invert_yaxis(); ax2.set_xlim(-0.6,len(found)-0.4); ax2.set_ylim(len(order)-0.4,-0.6)
    ax2.set_title("Panel genes by reference cell type",loc="left",fontsize=8)
    cb=fig.colorbar(sca,ax=ax2,fraction=0.03,pad=0.02); cb.set_label("mean log1p(CP10k)",fontsize=7)
    for p in (25,50,100): ax2.scatter([],[],s=p/100*60,c="#555555",label=f"{p}%")
    ax2.legend(title="% cells >0",loc="upper left",bbox_to_anchor=(1.18,0.35),frameon=False,fontsize=6,title_fontsize=7)

    plt.show()

    filefig = root_figure / "celltype_umap_dotplot.png"
    fig.savefig(filefig, dpi=300, bbox_inches="tight")

    return dfg



def spatial_plot_bcell_types(adata_vis, focus, lineage, root_figure):
    lib = list(adata_vis.uns["spatial"].keys())[0]
    img=adata_vis.uns["spatial"][lib]["images"]["hires"]; sf=adata_vis.uns["spatial"][lib]["scalefactors"]["tissue_hires_scalef"]
    xy=adata_vis.obsm["spatial"]*sf

    fig,axes=plt.subplots(3,4,figsize=(12,9.6),gridspec_kw=dict(wspace=0.08,hspace=0.18))

    x0,x1=xy[:,0].min()-40,xy[:,0].max()+40; y0,y1=xy[:,1].min()-40,xy[:,1].max()+40

    Q   = pd.DataFrame(np.asarray(adata_vis.obsm["q05_cell_abundance_w_sf"]),   # spots x 34 cell types
                    index=adata_vis.obs_names, columns=list(adata_vis.uns["mod"]["factor_names"]))

    col={}
    for k,v in lineage.items():
        cm=plt.get_cmap(bcell_cmaps[k]); 
        for j,c in enumerate(v): col[c]=cm(0.45+0.45*(j/(max(len(v)-1,1))))

    for ax,c in zip(axes.ravel(),focus):
        v=Q[c].values; vmax=np.percentile(v,99.2)
        ax.imshow(img,alpha=0.55)
        o=np.argsort(v)
        s=ax.scatter(xy[o,0],xy[o,1],c=np.clip(v[o],0,vmax),cmap=cmap_byr,vmin=0,vmax=vmax,s=5.5,lw=0,alpha=0.9)
        ax.set_xlim(x0,x1); ax.set_ylim(y1,y0); ax.set_xticks([]); ax.set_yticks([])
        [sp_.set_visible(False) for sp_ in ax.spines.values()]
        ax.set_title(c,fontsize=8,color=col[c],fontweight="bold")
        cbar=fig.colorbar(s,ax=ax,fraction=0.04,pad=0.01)
        cbar.set_ticks([0,round(vmax,1)])
        cbar.ax.tick_params(labelsize=6,length=2)

    fig.text(0.5,0.075,"cell2location q05 abundance (cells per spot, 'at least this many present'), clipped at the 99.2th percentile per cell type; V1_Human_Lymph_Node, 4,035 spots.",ha="center",fontsize=7)

    plt.show()

    filefig = root_figure / "celltype_spatial_maps.png"
    fig.savefig(filefig, dpi=300, bbox_inches="tight")



def spatial_plot_cell_abundance(adata_vis, adata_ref, genes_plot, focus, lineage, root_results, root_figure) -> pd.DataFrame:

    # found = genes present in the reference; X = raw counts; E = log1p(CP10k)   (cells x genes)
    found, X, E = panel_lognorm(adata_ref, genes_plot)

    symV=adata_vis.var["symbol"].astype(str)
    vf,vids=resolve_gene_ids(adata_vis, found)
    Xv=adata_vis[:,vids].X; Xv=Xv.toarray() if sps.issparse(Xv) else np.asarray(Xv)
    tot=np.asarray(adata_vis.X.sum(1)).ravel()

    Q   = pd.DataFrame(np.asarray(adata_vis.obsm["q05_cell_abundance_w_sf"]),   # spots x 34 cell types
                    index=adata_vis.obs_names, columns=list(adata_vis.uns["mod"]["factor_names"]))

    Ev=np.log1p(Xv/tot[:,None]*1e4)
    df_rho=pd.DataFrame([[spearmanr(Ev[:,i],Q[c].values)[0] for c in focus] for i in range(len(vf))],index=vf,columns=focus)

    fig,ax=plt.subplots(figsize=(12, 8))

    im=ax.imshow(df_rho.values,cmap="RdBu_r",vmin=-0.8,vmax=0.8,aspect="auto")
    for i in range(df_rho.shape[0]):
        for j in range(df_rho.shape[1]):
            r=df_rho.values[i,j]; ax.text(j,i,f"{r:.2f}",ha="center",va="center",fontsize=5.5,color="white" if abs(r)>0.5 else "black")

    ax.set_xticks(range(len(focus))); ax.set_xticklabels(focus,rotation=60,ha="right")

    col={}
    for k,v in lineage.items():
        cm=plt.get_cmap(bcell_cmaps[k]); 
        for j,c in enumerate(v): col[c]=cm(0.45+0.45*(j/(max(len(v)-1,1))))

    for t,c in zip(ax.get_xticklabels(),focus): 
        t.set_color(col[c])

    ax.set_yticks(range(len(vf))); ax.set_yticklabels([("MS4A1 (CD20)" if g=="MS4A1" else g) for g in vf],style="italic")
    cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label("Spearman ρ across spots (0 = no association)",fontsize=7); cb.set_ticks([-0.8,-0.4,0,0.4,0.8])
    ax.set_title("Visium gene expression vs mapped cell-type abundance (n = 4,035 spots)",loc="left",fontsize=8)

    plt.show()
    filefig = root_figure / "gene_celltype_spatial_correlation.png"
    fig.savefig(filefig, dpi=300, bbox_inches="tight")

    fname = "gene_celltype_spatial_correlation.tsv"
    df_rho = df_rho.round(3)
    _ = pdwritecsv(df_rho, fname, root_results)

    return df_rho

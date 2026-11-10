#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single_cell_lymphoid_lib.py - SingleCellLymphoid(SingleCell): human lymph node Visium + lymphoid-organ scRNA reference
(Kleshchevnikov et al., Fig 4; lymphoid_organ_sc.h5ad, 73,260 cells; V1_Human_Lymph_Node).

Methods ported from libs/cell2location_lib.py (that file is kept unchanged for ccc01/ccc02):
    data                  : read_visium, reindex_reference_var, symbol_index, calc_df_sample_stats
    tissue x age groups   : age_group, add_tissue_age_group, panel_lognorm, calc_df_gene_age_tissue_groups,
                            plot_panel_umap_by_tissue_age, calc_df_tissue_age_expr,
                            calc_df_panel_celltype_age, plot_dotplot_celltype_age
    B-cell subsets        : prepare_subset_markers, calc_df_group_pct_mean, calc_df_subset_markers,
                            print_subset_markers, calc_df_best_vs_next, donor_consistency, calc_gene_by_subset_tissue
    B-cell figures        : lineage_colors, lineage_order, umap_cell_type_plot, spatial_plot_cell_types,
                            spatial_plot_cell_abundance, dotplot_markers
    NMF factors           : plot_factor_dotplot (cell types grouped by lineage x NMF factors)
    ISG hotspot           : spot_expr, isg_hotspot, run_isg_hotspot

Parameters added to DEFAULT_PARAMS: LYMPHOID_PARAMS (age bins, tissue order, model settings, marker cutoffs,
ISG cutoffs, ...). LYMPHOID_LINEAGE: the 34 reference cell types (Subset) grouped by lineage.
Every DataFrame variable starts with 'df_'.

@created on 2026/11/05
@updated on 2026/11/10  (mp_lymphoid01 / mp_lymphoid02 lymphoid notebooks)
@author: Flavio Lichtenstein
"""

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import mannwhitneyu, spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import anndata as ad
import scanpy as sc

from libs.Basic import pdwritecsv
from libs.single_cell_lib import DEFAULT_PARAMS, SingleCell, cmap_byr
from libs.matplotlib_figure_style import apply_figure_style

__version__ = "1.1.0"

# the 34 reference cell types (obs['Subset']) grouped by lineage: row order of the lineage plots
LYMPHOID_LINEAGE: Dict[str, List[str]] = {
    "GC B":        ["B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "B_Cycling", "B_preGC"],
    "naive/mem B": ["B_naive", "B_activated", "B_IFN", "B_mem"],
    "plasma":      ["B_plasma"],
    "T":           ["T_CD4+_naive", "T_CD4+", "T_CD4+_TfH", "T_CD4+_TfH_GC", "T_TfR", "T_Treg", "T_TIM3+",
                    "T_CD8+_naive", "T_CD8+_cytotoxic", "T_CD8+_CD161+"],
    "NK/ILC":      ["NK", "NKT", "ILC"],
    "myeloid":     ["Monocytes", "Macrophages_M1", "Macrophages_M2", "DC_cDC1", "DC_cDC2", "DC_pDC", "DC_CCR7+", "Mast"],
    "stromal":     ["FDC", "Endo", "VSMC"],
}

LYMPHOID_PARAMS: Dict[str, Any] = dict(
    # --- reference / Visium layout
    ref_id_col="GeneID-2",             # var column with Ensembl IDs in lymphoid_organ_sc.h5ad
    ref_labels_key="Subset",           # RegressionModel labels
    ref_batch_key="Sample",            # RegressionModel batch
    ref_covariate_keys=("Method",),    # 3' vs 5' chemistry
    # --- cell2location mapping (tutorial: N=30, alpha=20, 25000 epochs) and regions (kNN 15, leidenalg 1.1)
    c2l_N_cells_per_location=30,
    c2l_max_epochs=25000,
    region_n_neighbors=15,
    region_resolution=1.1,
    region_leiden_flavor="leidenalg",
    region_top_n=3,
    # --- tissue x age within one tissue (mLN)
    age_tissue="mLN",
    panel_min_cells=20,                # (cell type, age group) shown when it has >= this many cells
    # --- B-cell subset markers (within-B comparison)
    bcell_types=("B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "B_Cycling", "B_preGC",
                 "B_naive", "B_activated", "B_IFN", "B_mem", "B_plasma"),
    gc_types=("B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "B_Cycling"),
    marker_min_gene_frac=0.01,         # genes detected in >= 1% of the subset's cells are tested
    marker_padj=0.05,
    marker_min_pct_in=0.30,
    unique_max_pct_other=0.25, unique_min_log2fc=1.0,     # 'unique'
    shared_max_pct_other=0.50, shared_min_log2fc=0.58,    # 'enriched (shared)'
    marker_eps=0.05,                   # CP10k pseudocount of log2fc_vs_next
    # --- plotting
    age_order=("babies", "young", "adult", "old adult", "elder"),
    tissue_order=("Tonsil", "Spleen", "iLN", "tLN", "mLN"),
    age_bins=(5, 14, 34, 55),          # lower bound of Age bin: <5 babies, <14 young, <34 adult, <55 old adult, else elder
    age_col="Age",
    tissue_col="Tissue",
    donor_col="Donor",
    celltype_col="Subset",
    libsize_col="n_counts",
    lineage_cmaps={"GC B": "Reds", "naive/mem B": "Blues", "plasma": "Purples", "T": "Greens",
                   "NK/ILC": "GnBu", "myeloid": "Oranges", "stromal": "Greys"},
    gene_labels={"MS4A1": "MS4A1 (CD20)", "CR2": "CR2 (CD21)", "FCER2": "FCER2 (CD23)", "HIST1H3B": "HIST1H3B (H3C2)"},
    panel_n_bg=20000,                  # grey background cells in UMAP grids
    panel_cmap_num=3,                  # 1 RdBu_r, 2 blue-yellow-red, 3 grey -> dark red
    donor_min_cells=20,
    isg_quantile=0.97,                 # ISG score quantile defining hotspot candidates
    isg_link_factor=1.6,               # spots linked when closer than factor x median spot distance
    isg_pseudocount=0.05,              # log2FC pseudocount (cells per spot)
    isg_fdr=0.05,
    isg_n_top=8, isg_n_bottom=7,       # bars shown in the enrichment panel
    isg_maps=(("B_IFN", "abundance"), ("CXCL10", "gene"), ("DC_CCR7+", "abundance")),
    rho_vmax=0.8,
)


class SingleCellLymphoid(SingleCell):
    """Lymph node / lymphoid-organ analyses. Symbols in var['symbol'] by default (tutorial convention)."""

    default_params = {**DEFAULT_PARAMS, **LYMPHOID_PARAMS}

    def __init__(self, ROOT0, project: str = "lymphoid_tissue", symbol_col: str = "symbol",
                 sample_col: str = "sample", mt_prefix: str = "MT-", **kw):
        super().__init__(ROOT0, project, symbol_col=symbol_col, sample_col=sample_col, mt_prefix=mt_prefix, **kw)

    # =====================================================================================================
    # data: Visium section, reference re-indexing, symbol index, sample table
    # =====================================================================================================
    def read_visium(self, root_raw: Path, sample_id: str, count_file: str = "filtered_feature_bc_matrix.h5",
                    set_self: bool = False, verbose: bool = False) -> ad.AnnData:
        """One Space Ranger output (root_raw/sample_id/: count_file + spatial/), as sc.datasets.visium_sge reads it:
        obs_names = barcodes, obs[sample_col] = sample_id, var_names = Ensembl IDs, symbols in var[symbol_col]
        (Ensembl IDs give a correct match with the reference; symbols may be duplicated)."""
        path = Path(root_raw) / sample_id
        a = sc.read_visium(path, count_file=count_file, library_id=sample_id)
        a.obs[self.sample_col] = sample_id
        if self.symbol_col not in a.var.columns:
            a.var.reset_index(inplace=True, names=self.symbol_col)
        if "gene_ids" in a.var.columns:
            a.var.set_index("gene_ids", inplace=True)
        if set_self:
            self.adata = a
        if verbose:
            print(f"Reading {path}: {a.n_obs:,} spots x {a.n_vars:,} genes")
        return a

    def reindex_reference_var(self, adata=None, id_col: Optional[str] = None, drop_raw: bool = True) -> ad.AnnData:
        """Reference var: symbols (index) -> var[symbol_col]; index <- var[id_col] (Ensembl IDs, 'GeneID-2').
        Idempotent (skipped when var[symbol_col] exists). Deletes .raw (not used by cell2location)."""
        a = self._check_adata(adata)
        id_col = self.get_param("ref_id_col", id_col)
        if self.symbol_col not in a.var.columns:
            a.var[self.symbol_col] = a.var.index.astype(str)
            a.var.set_index(id_col, drop=True, inplace=True)
            cols = list(a.var.columns)
            a.var = a.var[[self.symbol_col] + [c for c in cols if c != self.symbol_col]]
        if drop_raw and a.raw is not None:
            del a.raw
        return a

    def symbol_index(self, adata) -> ad.AnnData:
        """Copy with var_names = gene symbols (made unique), the Ensembl ID kept in var['gene_id']."""
        a = adata.copy()
        if self.symbol_col in a.var.columns:
            a.var["gene_id"] = a.var_names.astype(str)
            a.var_names = a.var[self.symbol_col].astype(str).values
            a.var_names_make_unique()
        return a

    def calc_df_sample_stats(self, adata=None, by: Sequence[str] = ("Sex", "Tissue", "age_group", "Age", "Sample"),
                             col: Optional[str] = None) -> pd.DataFrame:
        """describe() of obs[col] (default library size) per sample, grouped by sex, tissue, age group, age."""
        a = self._check_adata(adata)
        if "age_group" in by and "age_group" not in a.obs:
            self.add_tissue_age_group(a)
        col = self.get_param("libsize_col", col)
        df_stats = a.obs.groupby(list(by), observed=True)[col].describe().round(2)
        return df_stats

    # =====================================================================================================
    # tissue x age groups
    # =====================================================================================================
    def age_group(self, s) -> str:
        """'0-4y' / '20-25y' ... -> age group by the lower bound of the bin (params['age_bins'])."""
        lo = int(str(s).split("-")[0])
        b = self.params["age_bins"]
        names = self.params["age_order"]
        for thr, name in zip(b, names):
            if lo < thr:
                return name
        return names[-1]

    def add_tissue_age_group(self, adata=None):
        """obs['age_group'] (in place); returns ordered (tissue, age_group) pairs that contain cells."""
        a = self._check_adata(adata)
        obs = a.obs
        tcol = self.params["tissue_col"]
        obs["age_group"] = obs[self.params["age_col"]].map(self.age_group)
        return [(t, g) for g in self.params["age_order"] for t in self.params["tissue_order"]
                if ((obs[tcol] == t) & (obs["age_group"] == g)).any()]

    def panel_lognorm(self, genes: Sequence[str], adata=None, libsize_col: Optional[str] = None):
        """Raw counts and log1p(CP10k) for the panel genes only. Returns (found, counts, log1p_cp10k)."""
        a = self._check_adata(adata)
        found, ids = self.resolve_gene_ids(genes, a)
        X = a[:, ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        lib = a.obs[self.get_param("libsize_col", libsize_col)].values[:, None]
        E = np.log1p(X / lib * self.params["target_sum"])
        return found, X, E

    def calc_df_gene_age_tissue_groups(self, genes: Sequence[str], adata=None) -> pd.DataFrame:
        """Per tissue x age group and gene: n_cells, pct_expr, mean_log1p_cp10k."""
        a = self._check_adata(adata)
        order = self.add_tissue_age_group(a)
        found, X, E = self.panel_lognorm(genes, a)
        obs = a.obs
        tcol = self.params["tissue_col"]
        rows = []
        for t, g in order:
            m = ((obs[tcol] == t) & (obs.age_group == g)).values
            for i, gene in enumerate(found):
                rows.append(dict(tissue=t, age_group=g, gene=gene, n_cells=int(m.sum()),
                                 pct_expr=round(100 * (X[m, i] > 0).mean(), 1),
                                 mean_log1p_cp10k=round(float(E[m, i].mean()), 3)))
        df_groups = pd.DataFrame(rows)
        return df_groups

    def _panel_cmap(self, cmap_num: int):
        if cmap_num == 1:
            return "RdBu_r"
        if cmap_num == 2:
            return cmap_byr
        return LinearSegmentedColormap.from_list("expr", ["#e6e6e6", "#fdbb84", "#e34a33", "#7f0000"])

    def plot_panel_umap_by_tissue_age(self, genes: Sequence[str], adata=None, fname: Optional[str] = None,
                                      q: Optional[float] = None, basis: str = "X_umap",
                                      n_bg: Optional[int] = None, cmap_num: Optional[int] = None, dpi: int = 200):
        """UMAP grid: rows = genes, columns = tissue x age groups; colour = log1p(CP10k) clipped per gene."""
        a = self._check_adata(adata)
        q = self.get_param("clip_quantile", q)
        labels = self.params["gene_labels"]
        order = self.add_tissue_age_group(a)
        found, _, E = self.panel_lognorm(genes, a)
        cap = np.array([self.percentile_cap(E[:, i], q) for i in range(E.shape[1])])
        C = np.clip(E, 0, cap)
        U = a.obsm[basis]
        obs = a.obs
        rng = np.random.default_rng(self.params["seed"])
        bg = rng.choice(len(U), min(self.get_param("panel_n_bg", n_bg), len(U)), replace=False)
        cmap = self._panel_cmap(self.get_param("panel_cmap_num", cmap_num))
        tcol, dcol = self.params["tissue_col"], self.params["donor_col"]
        nr, nc = len(found), len(order)
        fig, axes = plt.subplots(nr, nc, figsize=(1.35 * nc + 0.9, 1.2 * nr + 0.8), squeeze=False,
                                 gridspec_kw=dict(wspace=0.04, hspace=0.06))
        for j, (t, g) in enumerate(order):
            m = np.where(((obs[tcol] == t) & (obs.age_group == g)).values)[0]
            nd = obs[dcol].iloc[m].nunique() if dcol in obs else 0
            for i, gene in enumerate(found):
                ax = axes[i, j]
                ax.scatter(U[bg, 0], U[bg, 1], s=0.3, c="#f2f2f2", lw=0, rasterized=True)
                v = C[m, i]
                o = np.argsort(v)
                ax.scatter(U[m[o], 0], U[m[o], 1], s=0.6, c=v[o], cmap=cmap, vmin=0, vmax=cap[i], lw=0, rasterized=True)
                ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if i == 0:
                    ax.set_title(f"{t} · {g}\n{len(m) / 1000:.1f}k cells\n{nd} donor{'s' if nd != 1 else ''}", fontsize=7)
                if j == 0:
                    ax.set_ylabel(labels.get(gene, gene), fontsize=8, style="italic", rotation=0, ha="right", va="center")
        for i in range(nr):
            pos = axes[i, -1].get_position()
            cax = fig.add_axes([pos.x1 + 0.006, pos.y0 + pos.height * 0.15, 0.006, pos.height * 0.7])
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, cap[i]), cmap=cmap), cax=cax)
            cb.set_ticks([0, round(float(cap[i]), 1)])
            cb.ax.tick_params(labelsize=6, length=2)
        fig.text(0.5, 0.075, f"colour = log1p(CP10k) clipped at the {q}th percentile per gene, one scale per row. "
                 "Grey = all other cells.", ha="center", fontsize=7)
        self._savefig(fig, fname, dpi=dpi)
        return fig

    # =====================================================================================================
    # B-cell figures
    # =====================================================================================================
    def lineage_colors(self, lineage: Dict[str, Sequence[str]], cmaps: Optional[Dict[str, str]] = None) -> Dict[str, tuple]:
        """cell type -> RGBA, shades of one colormap per lineage."""
        cmaps = cmaps or self.params["lineage_cmaps"]
        col = {}
        for k, v in lineage.items():
            cm = plt.get_cmap(cmaps[k])
            for j, c in enumerate(v):
                col[c] = cm(0.45 + 0.45 * (j / max(len(v) - 1, 1)))
        return col

    def umap_cell_type_plot(self, adata_ref, genes, lineage, focus, cmaps=None, sizes=(8, 7, 6),
                            fname: Optional[str] = "celltype_umap_dotplot.png") -> pd.DataFrame:
        """Reference UMAP coloured by cell type + dot plot of the panel genes by cell type.
        Returns df_groups (per tissue x age group panel table)."""
        apply_figure_style(sizes=sizes)
        cmaps = cmaps or self.params["lineage_cmaps"]
        labels = self.params["gene_labels"]
        ct = adata_ref.obs[self.params["celltype_col"]].astype(str).values
        found, X, E = self.panel_lognorm(genes, adata_ref)
        df_pct = pd.DataFrame(X > 0, columns=found).groupby(ct).mean() * 100
        df_mean = pd.DataFrame(E, columns=found).groupby(ct).mean()
        n_ct = pd.Series(ct).value_counts()
        order = [c for v in lineage.values() for c in v]
        assert set(order) == set(ct), set(ct) ^ set(order)
        df_P = df_pct.loc[order, found]
        df_M = df_mean.loc[order, found]
        U = adata_ref.obsm["X_umap"]
        col = self.lineage_colors(lineage, cmaps)
        df_groups = self.calc_df_gene_age_tissue_groups(genes, adata_ref)

        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        ax = axes[0]
        o = np.random.default_rng(self.params["seed"]).permutation(len(U))
        ax.scatter(U[o, 0], U[o, 1], s=0.4, c=[col[c] for c in ct[o]], lw=0, rasterized=True)
        for c in focus:
            x, y = np.median(U[ct == c], 0)
            ax.text(x, y, c, fontsize=7, ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title("Reference UMAP coloured by cell type; labelled = focus types", loc="left", fontsize=8)
        h = [Line2D([], [], marker="o", ls="", color=plt.get_cmap(cmaps[k])(0.75), label=k) for k in lineage]
        ax.legend(handles=h, loc="lower left", frameon=False, fontsize=7, markerscale=1.2)

        ax2 = axes[1]
        yy, xx = np.meshgrid(np.arange(len(order)), np.arange(len(found)), indexing="ij")
        sca = ax2.scatter(xx.ravel(), yy.ravel(), s=(df_P.values.ravel() / 100) * 60, c=df_M.values.ravel(),
                          cmap="magma_r", vmin=0, vmax=np.nanmax(df_M.values), lw=0)
        ax2.set_xticks(range(len(found)))
        ax2.set_xticklabels([labels.get(g, g) for g in found], rotation=90, style="italic")
        ax2.set_yticks(range(len(order)))
        ax2.set_yticklabels([f"{c} (n={n_ct[c]:,})" for c in order], fontsize=6)
        for t, c in zip(ax2.get_yticklabels(), order):
            t.set_color(col[c] if c in focus else "#555555")
            t.set_fontweight("bold" if c in focus else "normal")
        ax2.set_xlim(-0.6, len(found) - 0.4)
        ax2.set_ylim(len(order) - 0.4, -0.6)
        ax2.set_title("Panel genes by reference cell type", loc="left", fontsize=8)
        fig.colorbar(sca, ax=ax2, fraction=0.03, pad=0.02).set_label("mean log1p(CP10k)", fontsize=7)
        for p in (25, 50, 100):
            ax2.scatter([], [], s=p / 100 * 60, c="#555555", label=f"{p}%")
        ax2.legend(title="% cells >0", loc="upper left", bbox_to_anchor=(1.18, 0.35), frameon=False, fontsize=6, title_fontsize=7)
        self._savefig(fig, fname)
        return df_groups

    def spatial_plot_cell_types(self, adata_vis, focus, lineage, cmaps=None, ncols: int = 4,
                                fname: Optional[str] = "celltype_spatial_maps.png"):
        """One spatial panel per cell type in focus (q05 abundance clipped at clip_quantile)."""
        img, _ = self.hires_image(adata=adata_vis)
        _, xy = self.spot_xy(adata=adata_vis)
        q = self.params["clip_quantile"]
        nrows = math.ceil(len(focus) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.2 * nrows), squeeze=False,
                                 gridspec_kw=dict(wspace=0.08, hspace=0.18))
        x0, x1 = xy[:, 0].min() - 40, xy[:, 0].max() + 40
        y0, y1 = xy[:, 1].min() - 40, xy[:, 1].max() + 40
        df_abund = self.calc_df_abundance(adata_vis)
        col = self.lineage_colors(lineage, cmaps)
        for ax in axes.ravel()[len(focus):]:
            ax.set_visible(False)
        for ax, c in zip(axes.ravel(), focus):
            v = df_abund[c].values
            vmax = np.percentile(v, q)
            ax.imshow(img, alpha=0.55)
            o = np.argsort(v)
            s_ = ax.scatter(xy[o, 0], xy[o, 1], c=np.clip(v[o], 0, vmax), cmap=cmap_byr, vmin=0, vmax=vmax, s=5.5, lw=0, alpha=0.9)
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.set_title(c, fontsize=8, color=col.get(c, "black"), fontweight="bold")
            cb = fig.colorbar(s_, ax=ax, fraction=0.04, pad=0.01)
            cb.set_ticks([0, round(vmax, 1)])
            cb.ax.tick_params(labelsize=6, length=2)
        self._savefig(fig, fname)
        return fig

    def spatial_plot_cell_abundance(self, adata_vis, adata_ref, genes, focus, lineage, cmaps=None,
                                    fname_fig: Optional[str] = "gene_celltype_spatial_correlation.png",
                                    fname_tsv: Optional[str] = "gene_celltype_spatial_correlation.tsv") -> pd.DataFrame:
        """Spearman rho across spots: Visium gene expression (log1p CP10k) vs mapped q05 abundance.
        Caveat: abundances were inferred from the same counts; read it as consistency, not validation."""
        found, _ = self.resolve_gene_ids(genes, adata_ref)
        gene_found, gene_ids = self.resolve_gene_ids(found, adata_vis)
        X = adata_vis[:, gene_ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        tot = np.asarray(adata_vis.X.sum(1)).ravel()
        df_abund = self.calc_df_abundance(adata_vis)
        E = np.log1p(X / tot[:, None] * self.params["target_sum"])
        df_rho = pd.DataFrame([[spearmanr(E[:, i], df_abund[c].values)[0] for c in focus] for i in range(len(gene_found))],
                              index=gene_found, columns=focus)
        vmax = self.params["rho_vmax"]
        labels = self.params["gene_labels"]
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(df_rho.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(df_rho.shape[0]):
            for j in range(df_rho.shape[1]):
                r = df_rho.values[i, j]
                ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(r) > 0.5 else "black")
        ax.set_xticks(range(len(focus))); ax.set_xticklabels(focus, rotation=60, ha="right")
        col = self.lineage_colors(lineage, cmaps)
        for t, c in zip(ax.get_xticklabels(), focus):
            t.set_color(col.get(c, "black"))
        ax.set_yticks(range(len(gene_found)))
        ax.set_yticklabels([labels.get(g, g) for g in gene_found], style="italic")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Spearman ρ across spots", fontsize=7)
        ax.set_title(f"Visium gene expression vs mapped cell-type abundance (n = {adata_vis.n_obs:,} spots)", loc="left", fontsize=8)
        self._savefig(fig, fname_fig)
        df_rho = df_rho.round(3)
        if fname_tsv:
            pdwritecsv(df_rho.reset_index(names='gene'), fname_tsv, self.root_result)
        return df_rho

    def dotplot_markers(self, adata_sub, markers: Dict[str, Sequence[str]], groupby: Optional[str] = None,
                        fname: Optional[str] = "bcell_subset_markers_dotplot.png", figsize=(15, 5)):
        """scanpy dot plot of marker genes (group label -> genes in var_names), scaled 0-1 per gene."""
        apply_figure_style(sizes=(8, 7, 6))
        groupby = groupby or self.params["celltype_col"]
        miss = [g for v in markers.values() for g in v if g not in adata_sub.var_names]
        if miss:
            print("missing:", miss)
        markers = {k: [g for g in v if g in adata_sub.var_names] for k, v in markers.items()}
        dp = sc.pl.dotplot(adata_sub, markers, groupby=groupby, standard_scale="var", cmap="RdBu_r", show=False,
                           figsize=figsize, var_group_rotation=30, colorbar_title="mean expression\n(scaled 0-1 per gene)",
                           size_title="% cells expressing")
        fig = plt.gcf()
        for ax in fig.axes:
            for t in ax.get_xticklabels():
                t.set_style("italic")
        n_groups = adata_sub.obs[groupby].nunique()
        fig.suptitle(f"Markers separating the {n_groups} subsets (within-subset comparison; {adata_sub.n_obs:,} cells)",
                     ha="left", fontsize=8, x=0.02, y=1.06)
        self._savefig(fig, fname, dpi=200)
        return dp

    def donor_consistency(self, adata_sub, gene, group, groups, donors, grp, Xn, df_cnt: pd.DataFrame,
                          min_cells: Optional[int] = None) -> Tuple[int, int]:
        """Count donors in which `gene` is higher in `group` than in every other group.
        df_cnt: donor x group cell counts; only pairs with >= min_cells cells are used.
        Returns (n_donors_where_group_is_highest, n_donors_tested)."""
        min_cells = self.get_param("donor_min_cells", min_cells)
        j = adata_sub.var_names.get_loc(gene)
        col = np.asarray(Xn[:, j].todense()).ravel() if sp.issparse(Xn) else Xn[:, j]
        ok = tot = 0
        for dn in df_cnt.index:
            if df_cnt.loc[dn, group] < min_cells:
                continue
            others = [o for o in groups if o != group and df_cnt.loc[dn, o] >= min_cells]
            if not others:
                continue
            mi = col[(donors == dn) & (grp == group)].mean()
            mo = max(col[(donors == dn) & (grp == o)].mean() for o in others)
            tot += 1
            ok += mi > mo
        return ok, tot

    # =====================================================================================================
    # ISG hotspot
    # =====================================================================================================
    def spot_expr(self, adata_vis, gene) -> np.ndarray:
        """log1p(CP10k) of one gene per spot, looked up by symbol."""
        j = np.where(self.symbols(adata_vis).values == gene)[0][0]
        x = adata_vis.X[:, j]
        x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
        return np.log1p(x / np.asarray(adata_vis.X.sum(1)).ravel() * self.params["target_sum"])

    def isg_hotspot(self, adata_vis, isg_genes: Sequence[str], q: Optional[float] = None,
                    link_factor: Optional[float] = None) -> Tuple[pd.Series, np.ndarray]:
        """ISG score per spot (mean z-score of log1p CP10k) and the largest spatially connected cluster of spots
        above the q quantile. Returns (score Series, hotspot bool array)."""
        q = self.get_param("isg_quantile", q)
        link_factor = self.get_param("isg_link_factor", link_factor)
        sym = self.symbols(adata_vis).values
        tot = np.asarray(adata_vis.X.sum(1)).ravel()
        cols = {}
        for g in isg_genes:
            j = np.where(sym == g)[0]
            if len(j) == 0:
                continue
            x = adata_vis.X[:, j[0]]
            x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
            cols[g] = np.log1p(x / tot * self.params["target_sum"])
        if not cols:
            raise KeyError("none of the ISG genes found")
        df_Z = pd.DataFrame(cols, index=adata_vis.obs_names)
        score = ((df_Z - df_Z.mean()) / df_Z.std()).mean(1)
        xy = np.asarray(adata_vis.obsm["spatial"], float)
        step = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
        idx = np.where(score.values >= np.quantile(score, q))[0]
        hs = np.zeros(adata_vis.n_obs, bool)
        pairs = np.array(list(cKDTree(xy[idx]).query_pairs(r=link_factor * step)), dtype=int).reshape(-1, 2)
        if len(pairs) == 0:
            warnings.warn("no adjacent high-score spots: hotspot = single top spot")
            hs[idx[np.argmax(score.values[idx])]] = True
            return score, hs
        A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
        _, lab = connected_components(A, directed=False)
        hs[idx[lab == np.bincount(lab).argmax()]] = True
        return score, hs

    def run_isg_hotspot(self, adata_vis, isg_genes: Sequence[str], fname_fig: Optional[str] = "ifn_hotspot_summary.png",
                        fname_tsv: Optional[str] = "ifn_hotspot_celltype_enrichment.tsv") -> pd.DataFrame:
        """ISG hotspot detection, cell-type enrichment in the hotspot (Mann-Whitney U, BH FDR) and summary figure.
        Returns df_hot (per cell type: mean_hot, mean_rest, log2fc, p, padj)."""
        from statsmodels.stats.multitest import multipletests
        P = self.params
        isg_score, hs = self.isg_hotspot(adata_vis, isg_genes)
        n_isg = int(self.symbols(adata_vis).isin(isg_genes).sum())
        df_abund = self.calc_df_abundance(adata_vis)
        img, _ = self.hires_image(adata=adata_vis)
        _, xy = self.spot_xy(adata=adata_vis)
        pc = P["isg_pseudocount"]
        rows = []
        for c in df_abund.columns:
            a_, b_ = df_abund.loc[hs, c], df_abund.loc[~hs, c]
            rows.append(dict(cell_type=c, mean_hot=a_.mean(), mean_rest=b_.mean(),
                             log2fc=np.log2((a_.mean() + pc) / (b_.mean() + pc)), p=mannwhitneyu(a_, b_).pvalue))
        df_hot = pd.DataFrame(rows).sort_values("log2fc", ascending=False)
        df_hot["padj"] = multipletests(df_hot.p, method="fdr_bh")[1]
        df_sel = pd.concat([df_hot.head(P["isg_n_top"]), df_hot.tail(P["isg_n_bottom"])]).set_index("cell_type").iloc[::-1]

        maps = [(f"ISG score (mean z of {n_isg} ISGs)", isg_score.values)]
        for feat, kind in P["isg_maps"]:
            if kind == "abundance" and feat in df_abund:
                maps.append((f"{feat} abundance (q05)", df_abund[feat].values))
            elif kind == "gene":
                maps.append((f"{feat} log1p(CP10k)", self.spot_expr(adata_vis, feat)))
        hull = ConvexHull(xy[hs]) if hs.sum() >= 3 else None
        x0, x1 = xy[:, 0].min() - 30, xy[:, 0].max() + 30
        y0, y1 = xy[:, 1].min() - 30, xy[:, 1].max() + 30
        fig = plt.figure(figsize=(14, 7.4))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.95], wspace=0.55, hspace=0.2)
        for k, (title, v) in enumerate(maps[:4]):
            ax = fig.add_subplot(gs[k // 2, k % 2])
            ax.imshow(img, alpha=0.5)
            vmin = np.percentile(v, 1) if v.min() < 0 else 0
            vmax = np.percentile(v, P["clip_quantile"])
            o = np.argsort(v)
            s_ = ax.scatter(xy[o, 0], xy[o, 1], c=np.clip(v[o], vmin, vmax), cmap="magma", vmin=vmin, vmax=vmax, s=4.5, lw=0)
            if hull is not None:
                hv = np.append(hull.vertices, hull.vertices[0])
                ax.plot(xy[hs][hv, 0], xy[hs][hv, 1], color="#00e5ff", lw=1.0, ls="--")
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.set_title(title, loc="left", fontsize=8)
            fig.colorbar(s_, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=6)
        ax = fig.add_subplot(gs[:, 2])
        ax.barh(range(len(df_sel)), df_sel.log2fc, height=0.7, color=["#b2182b" if x > 0 else "#2166ac" for x in df_sel.log2fc])
        ax.axvline(0, color="#888", lw=0.8)
        ax.set_yticks(range(len(df_sel))); ax.set_yticklabels(df_sel.index)
        for i, (fc, pa) in enumerate(zip(df_sel.log2fc, df_sel.padj)):
            if pa >= P["isg_fdr"]:
                ax.text(fc + 0.05, i, "n.s.", va="center", ha="left", fontsize=6, color="#555")
        ax.set_xlabel("log2 fold change, hotspot vs rest\n(mapped q05 abundance)")
        ax.set_title("Cell types enriched / depleted in the hotspot", loc="left", fontsize=8)
        fig.text(0.1, 0.03, f"Dashed cyan = hull of the hotspot ({hs.sum()} of {adata_vis.n_obs:,} spots).", fontsize=7)
        self._savefig(fig, fname_fig, dpi=200)
        if fname_tsv:
            pdwritecsv(df_hot.round(4), fname_tsv, self.root_result)
        return df_hot

    # =====================================================================================================
    # lineage helpers
    # =====================================================================================================
    @staticmethod
    def lineage_order(lineage: Dict[str, Sequence[str]]) -> List[str]:
        """Cell types in lineage order (rows of the lineage plots)."""
        return [c for v in lineage.values() for c in v]

    @staticmethod
    def lineage_of(lineage: Dict[str, Sequence[str]]) -> Dict[str, str]:
        """cell type -> lineage name."""
        return {c: k for k, v in lineage.items() for c in v}

    # =====================================================================================================
    # tissue x age within one tissue (mLN: the only tissue with several age groups)
    # =====================================================================================================
    def calc_df_tissue_age_expr(self, genes: Sequence[str], tissue: Optional[str] = None, adata=None,
                                order: Optional[Sequence[str]] = None
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Cells of one tissue: log1p(CP10k) of the panel genes (library size = row sums of X, i.e.
        sc.pp.normalize_total) and the mean / fraction > 0 per age group.
        Returns (df_mean, df_frac, df_cells: cells x genes + 'age_group')."""
        a = self._check_adata(adata)
        tissue = self.get_param("age_tissue", tissue)
        m = (a.obs[self.params["tissue_col"]] == tissue).values
        sub = a[m]
        found, ids = self.resolve_gene_ids(genes, sub)
        X = sub[:, ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        tot = np.asarray(sub.X.sum(1)).ravel()
        E = np.log1p(X / np.maximum(tot, 1)[:, None] * self.params["target_sum"])
        df_cells = pd.DataFrame(E, columns=found, index=sub.obs_names)
        df_cells["age_group"] = sub.obs[self.params["age_col"]].map(self.age_group).values
        present = set(df_cells.age_group)
        order = list(order or [g for g in self.params["age_order"] if g in present])
        df_mean = df_cells.groupby("age_group")[found].mean().reindex(order).round(3)
        df_frac = (df_cells[found] > 0).groupby(df_cells["age_group"]).mean().reindex(order).round(3)
        return df_mean, df_frac, df_cells

    def calc_df_panel_celltype_age(self, genes: Sequence[str], tissue: Optional[str] = None, adata=None,
                                   min_cells: Optional[int] = None, fname: Optional[str] = None
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """One tissue: per cell type x age group x gene: n_cells, n_donors, pct_expr, mean_log1p_cp10k
        (panel_lognorm, library size obs['n_counts']).
        Returns (df_long: all pairs, df_plot: cell types with >= min_cells in >= 2 age groups and pairs with
        >= min_cells, df_age: cells and donors per age group). fname -> df_long saved in root_result."""
        a = self._check_adata(adata)
        tissue = self.get_param("age_tissue", tissue)
        min_cells = self.get_param("panel_min_cells", min_cells)
        ct_col, dcol = self.params["celltype_col"], self.params["donor_col"]
        sub = a[(a.obs[self.params["tissue_col"]] == tissue).values]
        sub = sub.to_memory() if sub.isbacked else sub
        found, X, E = self.panel_lognorm(genes, sub)
        obs = sub.obs[[ct_col, dcol]].astype(str).copy()
        obs["age_group"] = sub.obs[self.params["age_col"]].map(self.age_group).values
        present = set(obs.age_group)
        age_order = [g for g in self.params["age_order"] if g in present]
        rows = []
        for (ct, ag), idx in obs.groupby([ct_col, "age_group"]).indices.items():
            n_don = obs[dcol].iloc[idx].nunique()
            for j, g in enumerate(found):
                rows.append(dict(cell_type=ct, age_group=ag, gene=g, n_cells=len(idx), n_donors=n_don,
                                 pct_expr=100 * (X[idx, j] > 0).mean(), mean_log1p_cp10k=E[idx, j].mean()))
        df_long = pd.DataFrame(rows)
        df_long["age_group"] = pd.Categorical(df_long.age_group, categories=age_order, ordered=True)
        ok = (df_long.drop_duplicates(["cell_type", "age_group"]).query("n_cells >= @min_cells")
              .groupby("cell_type").size())
        keep_ct = ok[ok >= 2].index
        df_plot = df_long[df_long.cell_type.isin(keep_ct) & (df_long.n_cells >= min_cells)].copy()
        df_age = pd.DataFrame({"n_cells": obs.age_group.value_counts().reindex(age_order),
                               "n_donors": obs.groupby("age_group")[dcol].nunique().reindex(age_order)})
        df_age.attrs["tissue"] = tissue
        if fname:
            pdwritecsv(df_long.round(4), fname, self.root_result)
        return df_long, df_plot, df_age

    def plot_dotplot_celltype_age(self, df_plot: pd.DataFrame, lineage: Optional[Dict[str, Sequence[str]]] = None,
                                  df_age: Optional[pd.DataFrame] = None, cmaps: Optional[Dict[str, str]] = None,
                                  cmap: str = "Reds", fname: Optional[str] = None, 
                                  fontsize_ticks:int=11, fontsize_title:int=14, fontsize_text:int=12,
                                  fontsize_bar:int=10, fontsize_perc_cells:int=10,
                                  fontsize_lineage:int=10, figsize: tuple = (12, 9)):
        """
        Dot plot: rows = cell types in lineage order (labels coloured by lineage), column blocks = genes,
        columns inside a block = age groups. Size = % cells > 0; colour = mean log1p(CP10k) / max per gene.
        """

        lineage = lineage or LYMPHOID_LINEAGE
        col = self.lineage_colors(lineage, cmaps)
        lin_of = self.lineage_of(lineage)
        genes = list(dict.fromkeys(df_plot.gene))
        age_order = list(df_plot.age_group.cat.categories) if hasattr(df_plot.age_group, "cat") \
            else list(dict.fromkeys(df_plot.age_group))
        ct_order = [c for c in self.lineage_order(lineage) if c in set(df_plot.cell_type)]
        na = len(age_order)
        fig, ax = plt.subplots(figsize=figsize)
        for gi, g in enumerate(genes):
            d = df_plot[df_plot.gene == g]
            vmax = max(d.mean_log1p_cp10k.max(), 1e-6)                    # one colour scale per gene
            for ai, ag in enumerate(age_order):
                s_ = d[d.age_group == ag]
                x = gi * (na + 1) + ai
                y = [ct_order.index(c) for c in s_.cell_type]
                ax.scatter(np.full(len(y), x), y, s=4 + 1.6 * s_.pct_expr, c=s_.mean_log1p_cp10k / vmax,
                           cmap=cmap, vmin=0, vmax=1, edgecolor="0.4", linewidth=0.3)
            ax.text(gi * (na + 1) + (na - 1) / 2, -1.3, g, ha="center", va="bottom", fontweight="bold", style="italic", fontsize=fontsize_text)
        xt = [gi * (na + 1) + ai for gi in range(len(genes)) for ai in range(na)]

        # Change the font size of the labels
        ax.tick_params(labelsize=fontsize_ticks) 
        # ax.tick_params(axis='x', labelsize=fontsize_ticks) 
        # ax.tick_params(axis='y', labelsize=fontsize_ticks) 

        ax.set_xticks(xt)
        ax.set_xticklabels([a_ for _ in genes for a_ in age_order], rotation=90)
        ax.set_yticks(range(len(ct_order)))
        ax.set_yticklabels(ct_order)
        for t, c in zip(ax.get_yticklabels(), ct_order):
            t.set_color(col.get(c, "black"))
        prev = None                                                         # lineage separators
        for i, c in enumerate(ct_order):
            if prev is not None and lin_of.get(c) != prev:
                ax.axhline(i - 0.5, color="0.85", lw=0.8)
            prev = lin_of.get(c)
        ax.set_ylim(len(ct_order) - 0.5, -1.6)
        ax.set_xlim(-0.8, xt[-1] + 0.8)
        for gi in range(1, len(genes)):
            ax.axvline(gi * (na + 1) - 1, color="0.85", lw=0.8)
        ax.tick_params(axis="both", length=0)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)

        size_h = [Line2D([], [], marker="o", ls="", color="0.6", markeredgecolor="0.4", markersize=np.sqrt(4 + 1.6 * p),
                         label=f"{p}%") for p in (10, 30, 60, 90)]
        lin_h = [Line2D([], [], marker="s", ls="", color=plt.get_cmap((cmaps or self.params["lineage_cmaps"])[k])(0.75),
                        label=k) for k in lineage if any(lin_of.get(c) == k for c in ct_order)]
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=ax, shrink=0.3, pad=0.02, anchor=(0, 0.95))
        cb.set_label("mean log1p(CP10k)\n(scaled to max per gene)", fontsize=fontsize_bar) 
        cb.ax.tick_params(labelsize=fontsize_ticks)
        leg = ax.legend(handles=size_h, title="% cells > 0", loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False, 
                        title_fontsize=fontsize_perc_cells , fontsize=fontsize_perc_cells)
        ax.add_artist(leg)
        ax.legend(handles=lin_h, title="lineage", loc="lower left", bbox_to_anchor=(1.01, 0.32), frameon=False,
                  title_fontsize=fontsize_lineage,  fontsize=fontsize_lineage)
        title = "naive / memory / GC panel by cell type (lineage) and age group"
        if df_age is not None:
            title = f"{df_age.attrs.get('tissue', '')}: {title}\n" + "   ".join(
                f"{a_}: {int(r.n_cells):,} cells, {int(r.n_donors)} donor(s)" for a_, r in df_age.iterrows())
        ax.set_title(title, pad=28, fontsize=fontsize_title)
        
        fig.tight_layout()

        self._savefig(fig, fname)

        return fig

    # =====================================================================================================
    # B-cell subsets: markers that separate one subset from all the other subsets
    # =====================================================================================================
    def prepare_subset_markers(self, adata_ref, groups: Optional[Sequence[str]] = None, groupby: Optional[str] = None,
                               min_gene_frac: Optional[float] = None, method: Optional[str] = None) -> ad.AnnData:
        """Cells of the given subsets (default params['bcell_types']), symbol var_names, counts in
        layers['counts'], X = log1p(CP10k), genes detected in >= min_gene_frac of these cells, then
        one-vs-rest rank_genes_groups (Wilcoxon, pts=True) within the subsets."""
        groups = list(groups or self.params["bcell_types"])
        groupby = groupby or self.params["celltype_col"]
        min_gene_frac = self.get_param("marker_min_gene_frac", min_gene_frac)
        a = adata_ref[adata_ref.obs[groupby].isin(groups).values]
        a = self.symbol_index(a.to_memory() if a.isbacked else a)
        a.obs[groupby] = pd.Categorical(a.obs[groupby].astype(str), categories=groups)
        a.layers["counts"] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=self.params["target_sum"])
        sc.pp.log1p(a)
        keep = np.asarray((a.layers["counts"] > 0).mean(0)).ravel() >= min_gene_frac
        a = a[:, keep].copy()
        sc.tl.rank_genes_groups(a, groupby, method=self.get_param("rank_method", method), pts=True, use_raw=False)
        return a

    def calc_df_group_pct_mean(self, adata_sub, groups: Optional[Sequence[str]] = None,
                               groupby: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Genes x groups: fraction of cells with counts > 0 (df_pct) and mean CP10k (df_mean, expm1 of X)."""
        groups = list(groups or self.params["bcell_types"])
        grp = adata_sub.obs[groupby or self.params["celltype_col"]].astype(str).values
        Xb = adata_sub.layers["counts"] > 0
        df_pct = pd.DataFrame({g: np.asarray(Xb[grp == g].mean(0)).ravel() for g in groups}, index=adata_sub.var_names)
        cols = {}
        for g in groups:
            Xg = adata_sub.X[grp == g]
            Xg = Xg.expm1() if sp.issparse(Xg) else np.expm1(Xg)
            cols[g] = np.asarray(Xg.mean(0)).ravel()
        df_mean = pd.DataFrame(cols, index=adata_sub.var_names)
        return df_pct, df_mean

    def calc_df_subset_markers(self, adata_sub, df_pct: pd.DataFrame, df_mean: pd.DataFrame,
                               groups: Optional[Sequence[str]] = None, groupby: Optional[str] = None,
                               fname: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Marker tiers for every subset vs the other subsets:
        unique            : padj < marker_padj, pct_in >= marker_min_pct_in, max % in every other subset <=
                            unique_max_pct_other, log2FC vs the next-highest subset (mean CP10k) >= unique_min_log2fc;
        enriched (shared) : same with shared_max_pct_other / shared_min_log2fc (and not unique).
        Candidates get donors_top = 'k/n' (donor_consistency: donors where the gene is highest in that subset).
        Returns (df_all: every gene x subset, df_markers: tiered genes sorted by subset, tier, log2fc).
        fname -> df_markers saved (csv, root_result)."""
        P = self.params
        groups = list(groups or P["bcell_types"])
        groupby = groupby or P["celltype_col"]
        rgg = adata_sub.uns["rank_genes_groups"]
        eps = P["marker_eps"]
        rows = []
        for g in groups:
            others = [o for o in groups if o != g]
            df_w = self.wtab(g, rgg).rename(columns={"lfc": "lfc_vs_rest"})
            df = pd.DataFrame({"pct_in": df_pct[g], "max_pct_other": df_pct[others].max(1),
                               "next_pct_type": df_pct[others].idxmax(1), "mean_cp10k": df_mean[g],
                               "max_mean_other": df_mean[others].max(1), "next_mean_type": df_mean[others].idxmax(1)})
            df["log2fc_vs_next"] = np.log2((df.mean_cp10k + eps) / (df.max_mean_other + eps))
            df = df.join(df_w)
            df["group"] = g
            rows.append(df.reset_index().rename(columns={"index": "gene"}))
        df_all = pd.concat(rows, ignore_index=True)
        base = (df_all.padj < P["marker_padj"]) & (df_all.pct_in >= P["marker_min_pct_in"])
        strict = base & (df_all.max_pct_other <= P["unique_max_pct_other"]) & (df_all.log2fc_vs_next >= P["unique_min_log2fc"])
        relax = base & (df_all.max_pct_other <= P["shared_max_pct_other"]) & (df_all.log2fc_vs_next >= P["shared_min_log2fc"]) & ~strict
        df_all["tier"] = np.where(strict, "unique", np.where(relax, "enriched (shared)", ""))
        obs = adata_sub.obs
        donors = obs[P["donor_col"]].astype(str).values
        grp = obs[groupby].astype(str).values
        df_cnt = pd.crosstab(donors, grp)
        for g in groups:
            if g not in df_cnt.columns:
                df_cnt[g] = 0
        cand = df_all[df_all.tier != ""]
        dc = [self.donor_consistency(adata_sub, r.gene, r.group, groups, donors, grp, adata_sub.X, df_cnt)
              for r in cand.itertuples()]
        df_all["donors_top"] = ""
        df_all.loc[cand.index, "donors_top"] = [f"{a_}/{b_}" for a_, b_ in dc]
        df_markers = df_all[df_all.tier != ""].copy()
        df_markers["group"] = pd.Categorical(df_markers.group, groups)
        df_markers = df_markers.sort_values(["group", "tier", "log2fc_vs_next"], ascending=[True, False, False])
        df_all.attrs["df_donor_counts"] = df_cnt.reindex(columns=groups)
        if fname:
            cols = ["group", "gene", "tier", "pct_in", "max_pct_other", "next_pct_type", "log2fc_vs_next",
                    "next_mean_type", "lfc_vs_rest", "padj", "donors_top"]
            pdwritecsv(df_markers[cols].round(3), fname, self.root_result, sep=",")
        return df_all, df_markers

    @staticmethod
    def print_subset_markers(df_markers: pd.DataFrame, groups: Sequence[str], n_unique: int = 12, n_enriched: int = 6):
        """Per subset: U = unique genes 'gene(% in / max % other, log2FC vs next, donors)', E = best shared genes."""
        for g in groups:
            df_u = df_markers[(df_markers.group == g) & (df_markers.tier == "unique")].head(n_unique)
            df_e = df_markers[(df_markers.group == g) & (df_markers.tier != "unique")].head(n_enriched)
            print(f"\n## {g}\n U:", ", ".join(f"{r.gene}({r.pct_in * 100:.0f}/{r.max_pct_other * 100:.0f}%,"
                                               f"{r.log2fc_vs_next:.1f},{r.donors_top})" for r in df_u.itertuples()))
            print(" E:", ", ".join(f"{r.gene}({r.pct_in * 100:.0f}/{r.max_pct_other * 100:.0f}%,"
                                   f"{r.log2fc_vs_next:.1f} vs {r.next_mean_type})" for r in df_e.itertuples()))

    def calc_df_best_vs_next(self, df_all: pd.DataFrame, group: str, n: int = 8) -> pd.DataFrame:
        """Best genes of one subset by log2FC vs the next-highest subset (padj and pct_in filters only):
        for subsets without unique markers."""
        P = self.params
        df_best = df_all[(df_all.group == group) & (df_all.padj < P["marker_padj"]) &
                         (df_all.pct_in >= P["marker_min_pct_in"])]
        df_best = df_best.sort_values("log2fc_vs_next", ascending=False).head(n)
        return df_best[["gene", "pct_in", "max_pct_other", "log2fc_vs_next", "next_mean_type", "lfc_vs_rest", "padj"]]

    def calc_gene_by_subset_tissue(self, gene: str, adata=None, groups: Optional[Sequence[str]] = None,
                                   gc_types: Optional[Sequence[str]] = None, min_gc: int = 20,
                                   fname_subset: Optional[str] = None, fname_gc: Optional[str] = None
                                   ) -> Dict[str, Any]:
        """Detection of one gene (raw counts > 0) in the B subsets by tissue, and within GC B cells per donor,
        study and chemistry (AICDA study). Returns dict:
        df_gc_share (GC B cells per tissue), df_subset (% positive and % single-UMI per subset),
        chem_tonsil_gc (% positive in tonsil GC B by Method), df_subset_tissue, df_gc_donor (>= min_gc GC cells),
        n_pos, pct_pos_tonsil."""
        a = self._check_adata(adata)
        groups = list(groups or self.params["bcell_types"])
        gc_types = list(gc_types or self.params["gc_types"])
        x = self.values(gene, a)                                         # raw UMI counts
        obs = a.obs
        P = self.params
        d = pd.DataFrame({"pos": x > 0, "single_UMI": x == 1, "Subset": obs[P["celltype_col"]].astype(str).values,
                          "Tissue": obs[P["tissue_col"]].astype(str).values, "Donor": obs[P["donor_col"]].astype(str).values,
                          "Study": obs["Study"].astype(str).values, "Method": obs["Method"].astype(str).values})
        out = dict(n_pos=int((x > 0).sum()), pct_pos_tonsil=round(100 * float((d.Tissue[x > 0] == "Tonsil").mean()), 1))
        d = d[d.Subset.isin(groups)].copy()
        d["GC"] = d.Subset.isin(gc_types)
        df_gc_share = d.groupby("Tissue").agg(GC_B=("GC", "sum"), B=("GC", "size"))
        df_gc_share["pct_GC"] = (100 * df_gc_share.GC_B / df_gc_share.B).round(1)
        out["df_gc_share"] = df_gc_share.sort_values("GC_B", ascending=False)
        df_sub = d.groupby("Subset").agg(n=("pos", "size"), n_pos=("pos", "sum"), pct=("pos", "mean"))
        df_sub["pct"] = (100 * df_sub.pct).round(2)
        df_sub["pct_single_UMI"] = (100 * d[d.pos].groupby("Subset").single_UMI.mean()).round(0)
        out["df_subset"] = df_sub.reindex(groups)
        out["chem_tonsil_gc"] = (100 * d[d.GC & (d.Tissue == "Tonsil")].groupby("Method").pos.mean()).round(1).to_dict()
        df_st = d.groupby(["Subset", "Tissue"]).agg(n_cells=("pos", "size"), pct_pos=("pos", "mean")).reset_index()
        df_st["pct_pos"] = (100 * df_st.pct_pos).round(2)
        df_gc = d[d.GC].groupby(["Tissue", "Donor", "Study", "Method"]).agg(n_gc=("pos", "size"),
                                                                            pct_pos=("pos", "mean")).reset_index()
        df_gc["pct_pos"] = (100 * df_gc.pct_pos).round(1)
        if fname_subset:
            pdwritecsv(df_st, fname_subset, self.root_result, sep=",")
        if fname_gc:
            pdwritecsv(df_gc, fname_gc, self.root_result, sep=",")
        out["df_subset_tissue"] = df_st
        out["df_gc_donor"] = df_gc[df_gc.n_gc >= min_gc].sort_values(["Tissue", "Donor"])
        return out

    # =====================================================================================================
    # NMF co-location factors: cell types grouped by lineage x factors
    # =====================================================================================================
    def plot_factor_dotplot(self, adata_vis, n_fact: int, lineage: Optional[Dict[str, Sequence[str]]] = None,
                            cmaps: Optional[Dict[str, str]] = None, size_max: float = 140, cmap: str = "RdPu",
                            min_show: float = 0.02, fname: Optional[str] = None, fname_tsv: Optional[str] = None
                            ) -> Tuple[Any, pd.DataFrame]:
        """Dot plot of the NMF cell-type loadings (calc_df_coloc_loadings, each cell type's weights summed to 1 over
        factors): rows = cell types in lineage order (labels coloured by lineage), columns = factors ordered by the
        row of their top cell type, labelled with their top-2 cell types. Size and colour = relative weight;
        weights < min_show are not drawn. Returns (fig, df_load in plotting order)."""
        lineage = lineage or LYMPHOID_LINEAGE
        df_load = self.calc_df_coloc_loadings(adata_vis, n_fact, normalize=True)
        rows = [c for c in self.lineage_order(lineage) if c in df_load.index]
        rows += [c for c in df_load.index if c not in rows]
        df_load = df_load.loc[rows]
        top_row = {f: rows.index(df_load[f].idxmax()) for f in df_load.columns}
        facts = sorted(df_load.columns, key=lambda f: (top_row[f], -df_load[f].max()))
        df_load = df_load[facts]
        top2 = {f: " / ".join(df_load[f].sort_values(ascending=False).index[:2]) for f in facts}
        col = self.lineage_colors(lineage, cmaps)
        lin_of = self.lineage_of(lineage)
        nr, nc = df_load.shape
        fig, ax = plt.subplots(figsize=(0.55 * nc + 4.2, 0.27 * nr + 2.6))
        yy, xx = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        V = df_load.values
        m = V.ravel() >= min_show
        sca = ax.scatter(xx.ravel()[m], yy.ravel()[m], s=V.ravel()[m] * size_max, c=V.ravel()[m], cmap=cmap,
                         vmin=0, vmax=1, edgecolors="0.4", linewidths=0.3)
        ax.set_xticks(range(nc))
        ax.set_xticklabels([f"{f.replace('fact_', 'F')}: {top2[f]}" for f in facts], rotation=90, fontsize=7)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rows)
        for t, c in zip(ax.get_yticklabels(), rows):
            t.set_color(col.get(c, "black"))
        prev = None
        for i, c in enumerate(rows):
            if prev is not None and lin_of.get(c) != prev:
                ax.axhline(i - 0.5, color="0.8", lw=0.8)
            prev = lin_of.get(c)
        ax.set_xlim(-0.6, nc - 0.4)
        ax.set_ylim(nr - 0.5, -0.5)
        ax.tick_params(length=0)
        cb = fig.colorbar(sca, ax=ax, fraction=0.03, pad=0.02, anchor=(0, 0.9), shrink=0.5)
        cb.set_label("relative weight of the cell type\n(sum over factors = 1)", fontsize=7)
        for p in (0.25, 0.5, 1.0):
            ax.scatter([], [], s=p * size_max, c="0.6", edgecolors="0.4", linewidths=0.3, label=f"{p:g}")
        ax.legend(title="weight", loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False, fontsize=7,
                  title_fontsize=7, labelspacing=1.1)
        ax.set_title(f"NMF co-location: {nr} cell types (by lineage) x {nc} factors", loc="left")
        self._savefig(fig, fname)
        if fname_tsv:
            pdwritecsv(df_load.round(4).reset_index(names="cell_type"), fname_tsv, self.root_result)
        return fig, df_load

    # aliases with the names used in ccc01 / ccc02 (cell2location_lib)
    def spatial_plot_bcell_types(self, adata_vis, focus, lineage, cmaps=None, ncols: int = 4,
                                 fname: Optional[str] = "celltype_spatial_maps.png"):
        return self.spatial_plot_cell_types(adata_vis, focus, lineage, cmaps=cmaps, ncols=ncols, fname=fname)

    def dotplot_bcell_markers(self, adata_sub, markers: Dict[str, Sequence[str]], groupby: Optional[str] = None,
                              fname: Optional[str] = "bcell_subset_markers_dotplot.png", figsize=(15, 5)):
        return self.dotplot_markers(adata_sub, markers, groupby=groupby, fname=fname, figsize=figsize)

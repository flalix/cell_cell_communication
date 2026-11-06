#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single_cell_lymphoid_lib.py - SingleCellLymphoid(SingleCell): human lymph node Visium + lymphoid-organ scRNA reference
(Kleshchevnikov et al., Fig 4; lymphoid_organ_sc.h5ad, 73,260 cells; V1_Human_Lymph_Node).

Methods ported from libs/cell2location_lib.py (that file is kept unchanged for ccc01/ccc02):
    tissue x age groups   : age_group, add_tissue_age_group, panel_lognorm, calc_df_gene_age_tissue_groups,
                            plot_panel_umap_by_tissue_age
    B-cell figures        : lineage_colors, umap_cell_type_plot, spatial_plot_cell_types, spatial_plot_cell_abundance,
                            dotplot_markers, donor_consistency
    ISG hotspot           : spot_expr, isg_hotspot, run_isg_hotspot

Parameters added to DEFAULT_PARAMS: LYMPHOID_PARAMS (age bins, tissue order, ISG cutoffs, donor cutoff, ...).
Every DataFrame variable starts with 'df_'.

@created on 2026/11/05
@author: Flavio Lichtenstein
"""

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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

import scanpy as sc

from libs.Basic import pdwritecsv
from libs.single_cell_lib import DEFAULT_PARAMS, SingleCell, cmap_byr
from libs.matplotlib_figure_style import apply_figure_style

__version__ = "1.0.0"

LYMPHOID_PARAMS: Dict[str, Any] = dict(
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
    gene_labels={"MS4A1": "MS4A1 (CD20)", "CR2": "CR2 (CD21)"},
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

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
    subregions            : add_region_ids, calc_df_subregion_ref_score, calc_df_ref_top, plotly_subregion_ref_heatmap,
                            plot_ref_umap_subregion_scores (the subregion pipeline itself is in SingleCell, section 8c)
    ISG hotspot           : spot_expr, isg_hotspot, run_isg_hotspot

Parameters added to DEFAULT_PARAMS: LYMPHOID_PARAMS (age bins, tissue order, model settings, marker cutoffs,
ISG cutoffs, ...). LYMPHOID_LINEAGE: the 34 reference cell types (Subset) grouped by lineage.
Every DataFrame variable starts with 'df_'.

@created on 2026/11/05
@updated on 2026/11/10  (mp_lymphoid01 / mp_lymphoid02 lymphoid notebooks)
@updated on 2026/11/11  (mp_lymphoid03 subregions)
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

from libs.Basic import pdreadcsv, pdwritecsv
from libs.single_cell_lib import DEFAULT_PARAMS, SingleCell, cmap_byr, cmap_lbwr
from libs.matplotlib_figure_style import apply_figure_style

__version__ = "1.2.0"

# the 34 reference cell types (obs['Subset']) grouped by lineage: row order of the lineage plots
LYMPHOID_LINEAGE: Dict[str, List[str]] = {
    "GC B":        ["B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "B_Cycling", "B_preGC"],
    "naive/mem B": ["B_naive", "B_activated", "B_IFN", "B_mem"],
    "plasma":      ["B_plasma"],
    "T":           ["T_CD4+_naive", "T_CD4+", "T_CD4+_TfH", "T_CD4+_TfH_GC", "T_TfR", "T_Treg", "T_TIM3+", "T_CD8+_naive", "T_CD8+_cytotoxic", "T_CD8+_CD161+"],
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
    bcell_types=("B_GC_DZ", "B_GC_LZ", "B_GC_prePB", "B_Cycling", "B_preGC", "B_naive", "B_activated", "B_IFN", "B_mem", "B_plasma"),
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
    # --- subregions (mp_lymphoid03; pipeline in SingleCell section 8c)
    subregion_skip=("R19",),           # low-quality strip, not subclustered
    region_table_first=("region_id", "signature_genes", "functional_description"),
    region_table_depth=True,           # median_UMIs, median_cells_per_spot in the region table
    subregion_table_first=("region_id", "sub_region_id", "subregion_name", "subregion_signature_genes", "n_other",
                           "other_genes_top_expressed", "not_present_genes", "functional_description"),
    region_id_fmt="R{:02d}",           # region_cluster 6 -> 'R06' (names of mp_lymphoid01 are not unique)
    gene_display_exclude=r"^RP[SL]\d|^MT-|^A[CLP]\d{6}\.\d|^LINC\d|-AS\d$",   # ribosomal / unnamed clones / lncRNA: not listed
    border_min_rel=1.25,               # regions are interleaved patches: 'border' only when the subregion is more on the
                                       # region edge than the region itself (border_frac / region_border_frac)
    subregion_map_title="{lib}: subregions of each lymph node region",
    ref_sig_n_genes=25,                # top subregion signature genes scored on the scRNA reference
    ref_sig_min_genes=3,               # subregions with fewer signature genes (found in the reference) are not scored
    ref_score_clip=99.0,               # percentile cap of the cell scores in the UMAP grid
    # --- dataset and figure style (mp_lymphoid01-04)
    visium_id="V1_Human_Lymph_Node",   # Space Ranger sample (one lymph node section, one donor)
    ref_symbol_col=None,               # reference LIANA (rank_aggregate): symbols are the var_names
    font_color="white",                # white text on a black background
    bg_color="black",
    fontsize_scale=1.4,                # x the font sizes set inside the matplotlib plot methods
)

# Per-analysis parameters (SingleCellLymphoid(..., analysis='lymph02')): only the values that differ from LYMPHOID_PARAMS.
LYMPHOID_ANALYSIS_PARAMS: Dict[str, Dict[str, Any]] = {
    "lymph01": dict(panel_cmap_num=2),         # gene panels in blue-yellow-red, as the Visium plots
    "lymph02": dict(coloc_n_facts=(12,)),      # NMF co-location: the 12-factor run of mp_lymphoid01 only
}


# Files of the lymphoid project: key -> (root, name); root 'data' | 'result' | 'figure' (folders of ROOT0/project)
# or 'root0' (ROOT0: inputs outside the project).
# 'inputs' are always registered; an analysis (SingleCellLymphoid(..., analysis='lymph03')) adds its outputs, which
# turns on the caching of its methods (force=False: read the file when it exists).
LYMPHOID_FILES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "inputs": dict(
        c2l_map=("result", "lymph_nodes_analysis/cell2location_map/sp.h5ad"),     # cell2location map (mp_lymphoid01)
        region_labels=("result", "lymph01_region_labels.tsv"),                    # regions of mp_lymphoid01
        reference=("data", "lymphoid_organ_sc.h5ad"),                             # scRNA reference
        region_info=("data", "lymph_node_regions.tsv"),                           # curated region table (editable)
        subregion_descriptions=("data", "lymph_node_subregion_descriptions.tsv"), # curated subregion descriptions
        signatures=("result", "lymph_nodes_analysis/reference_signatures/sc.h5ad"),  # mapping signatures (means)
        visium_raw=("root0", "notebooks_ccc_lymphoid_liana/data"),                # Space Ranger output (visium_sge)
        cell_type_annotation=("data", "cell_type_annotation.tsv"),                 # reference cell types (id, name, markers)
        c2l_ref_dir=("result", "lymph_nodes_analysis/reference_signatures"),       # sc.h5ad + model.pt (RegressionModel)
        c2l_map_dir=("result", "lymph_nodes_analysis/cell2location_map"),          # sp.h5ad + model.pt + posterior samples
    ),
    "lymph01": dict(
        # figures
        fig_genes_visium=("figure", "lymph01_visium_genes.png"),
        fig_mln_dotplot=("figure", "lymph01_mln_panel_by_celltype_age.png"),
        fig_panel_umap=("figure", "lymph01_bcell_genes_umap_by_tissue_age.png"),
        fig_ref_history=("figure", "lymph01_ref_training_history.png"),
        fig_ref_qc=("figure", "lymph01_ref_qc.png"),                               # plot_c2l_qc: _1, _2, ... per figure
        fig_map_history=("figure", "lymph01_map_training_history.png"),
        fig_map_qc=("figure", "lymph01_map_qc.png"),
        fig_abundance=("figure", "lymph01_celltype_abundance_8types.png"),
        fig_slide_3types=("figure", "lymph01_spatial_Tnaive_Bnaive_FDC.png"),
        fig_slide_4types=("figure", "lymph01_spatial_Tnaive_Bnaive_Mast_FDC.png"),
        fig_slide_tnaive=("figure", "lymph01_spatial_Tcd4naive_Tcd8naive.png"),
        fig_regions=("figure", "lymph01_umap_region_sample.png"),
        fig_regions_html=("figure", "lymph01_regions_spatial.html"),
        fig_umap_dotplot=("figure", "lymph01_celltype_umap_dotplot.png"),
        fig_median_mean=("figure", "lymph01_posterior_median_vs_mean.png"),
        fig_factor_dotplot=("figure", "lymph01_nmf_factors_lineage_dotplot_n{n}.png"),
        fig_genes_celltype=("figure", "lymph01_genes_per_cell_type.png"),
        # tables (region_labels: inputs, written here)
        tsv_mln_panel=("result", "lymph01_mln_panel_by_celltype_age.tsv"),
        tsv_factor_load=("result", "lymph01_nmf_factor_loadings_n{n}.tsv"),
    ),
    "lymph02": dict(
        # figures
        fig_bcell_dotplot=("figure", "lymph02_bcell_subset_markers_dotplot.png"),
        fig_genes_visium=("figure", "lymph02_visium_genes_{k}.png"),                 # {k}: figure number (8 genes each)
        fig_mln_dotplot=("figure", "lymph02_mln_panel_by_celltype_age.png"),
        fig_panel_umap=("figure", "lymph02_genes_umap_by_tissue_age.png"),
        fig_ref_history=("figure", "lymph02_ref_training_history.png"),
        fig_ref_qc=("figure", "lymph02_ref_qc.png"),
        fig_map_history=("figure", "lymph02_map_training_history.png"),
        fig_map_qc=("figure", "lymph02_map_qc.png"),
        fig_abundance=("figure", "lymph02_celltype_abundance_8types.png"),
        fig_slide_3types=("figure", "lymph02_spatial_Tnaive_Bnaive_FDC.png"),
        fig_isg=("figure", "lymph02_ifn_hotspot_summary.png"),
        fig_umap_dotplot=("figure", "lymph02_celltype_umap_dotplot.png"),
        fig_spatial_maps=("figure", "lymph02_celltype_spatial_maps.png"),
        fig_rho=("figure", "lymph02_gene_celltype_spatial_correlation.png"),
        fig_factor_dotplot=("figure", "lymph02_nmf_factors_lineage_dotplot_n{n}.png"),
        # tables
        csv_bcell_markers=("result", "lymph02_bcell_subset_unique_markers.csv"),
        tsv_bcell_markers_all=("result", "lymph02_bcell_subset_markers_all.tsv.gz"),      # cache: every gene x subset
        tsv_bcell_donor_counts=("result", "lymph02_bcell_subset_donor_counts.tsv"),       # cache: cells per donor x subset
        tsv_mln_panel=("result", "lymph02_mln_panel_by_celltype_age.tsv"),
        csv_aicda_subset=("result", "lymph02_aicda_by_subset_tissue.csv"),
        csv_aicda_gc=("result", "lymph02_aicda_gc_by_donor.csv"),
        tsv_isg=("result", "lymph02_ifn_hotspot_celltype_enrichment.tsv"),
        tsv_rho=("result", "lymph02_gene_celltype_spatial_correlation.tsv"),
        tsv_factor_load=("result", "lymph02_nmf_factor_loadings_n{n}.tsv"),
    ),
    "lymph03": dict(
        # tables (cached)
        region_rank_genes=("result", "lymph03_region_rank_genes.tsv.gz"),
        region_table=("result", "lymph03_region_table.tsv"),
        subregion_grid=("result", "lymph03_subcluster_resolution_grid.tsv"),
        subregion_adata=("result", "lymph03_subregions/{region}.h5ad"),            # run_subregions: one AnnData per region
        subregion_gene_sets=("result", "lymph03_subregion_gene_sets.tsv.gz"),
        subregion_table=("result", "lymph03_subregion_table.tsv"),
        subregion_ref_score=("result", "lymph03_subregion_ref_celltype_score.tsv"),
        subregion_ref_genes=("result", "lymph03_subregion_ref_genes.tsv"),
        # figures
        fig_region_heatmap=("figure", "lymph03_region_signature_heatmap.html"),
        fig_region_map=("figure", "lymph03_region_map_{lib}.html"),
        fig_region_sub=("figure", "lymph03_subregions_{region}.html"),
        fig_diagnostics=("figure", "lymph03_subregion_diagnostics.html"),
        fig_subregion_map=("figure", "lymph03_subregion_map_{lib}.html"),
        fig_ref_heatmap=("figure", "lymph03_subregion_ref_celltype_heatmap.html"),
        fig_ref_umap=("figure", "lymph03_ref_umap_subregion_scores.png"),
    ),
    "lymph04": dict(
        # subregions of lymph03 (read: same parameters; run_subregions reuses its cached subsets)
        subregion_grid=("result", "lymph03_subcluster_resolution_grid.tsv"),
        subregion_adata=("result", "lymph03_subregions/{region}.h5ad"),
        subregion_table_lymph03=("result", "lymph03_subregion_table.tsv"),
        # tables / caches
        spatial_weights=("result", "lymph04_spatial_weights.tsv"),
        liana_bivariate=("result", "lymph04_liana_bivariate.h5ad"),
        liana_global=("result", "lymph04_liana_bivariate_global.tsv"),
        lr_by_region=("result", "lymph04_lr_by_region.tsv"),
        lr_by_subregion=("result", "lymph04_lr_by_subregion.tsv"),
        celltype_lr_tissue=("result", "lymph04_celltype_lr_tissue.tsv.gz"),
        celltype_lr_by_group=("result", "lymph04_celltype_lr_by_group.tsv.gz"),
        celltype_links_by_group=("result", "lymph04_celltype_links_by_group.tsv"),
        liana_reference=("result", "lymph04_liana_reference_rank_aggregate.tsv"),
        liana_reference_supported=("result", "lymph04_liana_reference_supported.tsv"),
        ccc_by_region=("result", "lymph04_ccc_consensus_by_region.tsv"),
        ccc_by_subregion=("result", "lymph04_ccc_consensus_by_subregion.tsv"),
        # figures
        fig_lr_top=("figure", "lymph04_lr_top_local_scores_{lib}.png"),
        fig_lr_regions=("figure", "lymph04_lr_region_top_local_scores_{lib}.png"),
        fig_lr_region_hm=("figure", "lymph04_lr_by_region_heatmap.html"),
        fig_lr_sub_hm=("figure", "lymph04_lr_by_subregion_heatmap.html"),
        fig_sender_recv=("figure", "lymph04_celltype_lr_sender_receiver.png"),
        fig_links_region=("figure", "lymph04_celltype_links_by_region.html"),
        fig_links_sub=("figure", "lymph04_celltype_links_by_subregion.html"),
        fig_coloc=("figure", "lymph04_celltype_colocalisation.html"),
    ),
}


class SingleCellLymphoid(SingleCell):
    """Lymph node / lymphoid-organ analyses. Symbols in var['symbol'] by default (tutorial convention)."""

    default_params = {**DEFAULT_PARAMS, **LYMPHOID_PARAMS}
    analysis_files = LYMPHOID_FILES
    analysis_params = LYMPHOID_ANALYSIS_PARAMS

    def __init__(self, ROOT0, project: str = "lymphoid_tissue", symbol_col: str = "symbol",
                 sample_col: str = "sample", mt_prefix: str = "MT-", **kw):
        """analysis= key of LYMPHOID_FILES ('lymph01' ... 'lymph04') whose output files are registered (caching on) and of
        LYMPHOID_ANALYSIS_PARAMS (its parameter values); files= / params= add or override single entries."""
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
                                      q: Optional[float] = None, basis: str = "X_umap", figsize:tuple=(15, 30),
                                      n_bg: Optional[int] = None, cmap_num: Optional[int] = None, dpi: int = 200,
                                      font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                      bg_point_color: Optional[str] = None, force: bool = False, verbose: bool = False):
        """UMAP grid: rows = genes, columns = tissue x age groups; colour = log1p(CP10k) clipped per gene.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure); bg_point_color = the other cells (default light
        grey on a light background, dark grey on a dark one). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
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
        if bg_point_color is None:
            bg_point_color = "#f2f2f2" if self._luminance(bg_color) > 0.5 else "#3a3a3a"

        fig, axes = plt.subplots(nr, nc, figsize=figsize, squeeze=False, gridspec_kw=dict(wspace=0.04, hspace=0.06))
        
        for j, (t, g) in enumerate(order):
            m = np.where(((obs[tcol] == t) & (obs.age_group == g)).values)[0]
            nd = obs[dcol].iloc[m].nunique() if dcol in obs else 0
            for i, gene in enumerate(found):
                ax = axes[i, j]
                ax.scatter(U[bg, 0], U[bg, 1], s=0.3, c=bg_point_color, lw=0, rasterized=True)
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
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, dpi=dpi, force=force, verbose=verbose)
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

    def umap_cell_type_plot(self, adata_ref, genes, lineage, focus, cmaps=None, 
                            sizes=(8, 7, 6), figsize=(14, 8),
                            fname: Optional[str] = "celltype_umap_dotplot.png",
                            font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                            force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        Reference UMAP coloured by cell type + dot plot of the panel genes by cell type.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure; non-focus cell types in a muted grey that
        follows bg_color). force: rewrite an existing file; verbose: print the file status.
        Returns df_groups (per tissue x age group panel table).
        """
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        muted = self._muted(bg_color)
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

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax = axes[0]
        o = np.random.default_rng(self.params["seed"]).permutation(len(U))
        ax.scatter(U[o, 0], U[o, 1], s=0.4, c=[col[c] for c in ct[o]], lw=0, rasterized=True)
        for c in focus:
            x, y = np.median(U[ct == c], 0)
            ax.text(x, y, c, fontsize=7, ha="center", va="center", color=font_color,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=bg_color)])
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title("Reference UMAP coloured by cell type; labelled = focus types", loc="left", fontsize=8)
        h = [Line2D([], [], marker="o", ls="", color=plt.get_cmap(cmaps[k])(0.75), label=k) for k in lineage]
        ax.legend(handles=h, loc="lower left", frameon=False, fontsize=7, markerscale=1.2)

        ax2 = axes[1]
        yy, xx = np.meshgrid(np.arange(len(order)), np.arange(len(found)), indexing="ij")
        sca = ax2.scatter(xx.ravel(), yy.ravel(), s=(df_P.values.ravel() / 100) * 60, c=df_M.values.ravel(),
                          cmap=cmap_lbwr, vmin=0, vmax=np.nanmax(df_M.values), lw=0)   # cmap="magma_r"
        ax2.set_xticks(range(len(found)))
        ax2.set_xticklabels([labels.get(g, g) for g in found], rotation=90, style="italic")
        ax2.set_yticks(range(len(order)))
        ax2.set_yticklabels([f"{c} (n={n_ct[c]:,})" for c in order], fontsize=6)
        for t, c in zip(ax2.get_yticklabels(), order):
            t.set_color(col[c] if c in focus else muted)
            t.set_fontweight("bold" if c in focus else "normal")
        ax2.set_xlim(-0.6, len(found) - 0.4)
        ax2.set_ylim(len(order) - 0.4, -0.6)
        ax2.set_title("Panel genes by reference cell type", loc="left", fontsize=8)
        fig.colorbar(sca, ax=ax2, fraction=0.03, pad=0.02).set_label("mean log1p(CP10k)", fontsize=7)
        for p in (25, 50, 100):
            ax2.scatter([], [], s=p / 100 * 60, c=muted, label=f"{p}%")
        ax2.legend(title="% cells >0", loc="upper left", bbox_to_anchor=(1.18, 0.35), frameon=False, fontsize=6, title_fontsize=7)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return df_groups

    def spatial_plot_cell_types(self, adata_vis, focus, lineage, cmaps=None, ncols: int = 4, figsize:tuple=(20,16),
                                fname: Optional[str] = "celltype_spatial_maps.png",
                                font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                force: bool = False, verbose: bool = False):
        """
        One spatial panel per cell type in focus (q05 abundance clipped at clip_quantile).
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status.
        """
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        
        img, _ = self.hires_image(adata=adata_vis)
        _, xy = self.spot_xy(adata=adata_vis)
        q = self.params["clip_quantile"]
        nrows = math.ceil(len(focus) / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.18))
        
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

        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def spatial_plot_cell_abundance(self, adata_vis, adata_ref, genes, focus, lineage, cmaps=None, figsize: tuple = (20, 15),
                                    fname_fig: Optional[str] = "gene_celltype_spatial_correlation.png",
                                    fname_tsv: Optional[str] = "gene_celltype_spatial_correlation.tsv",
                                    font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = 2.4,
                                    force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        Spearman rho across spots: Visium gene expression (log1p CP10k) vs mapped q05 abundance.
        Caveat: abundances were inferred from the same counts; read it as consistency, not validation.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure; the cell values keep their colours).
        force: rewrite an existing figure; verbose: print the file status.
        """

        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)

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

        fig, ax = plt.subplots(figsize=figsize)

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
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname_fig, force=force, verbose=verbose)
        
        df_rho = df_rho.round(3)
        if fname_tsv:
            self._write_table(df_rho.reset_index(names='gene'), fname_tsv, force, verbose)
        return df_rho

    def dotplot_markers(self, adata_sub, markers: Dict[str, Sequence[str]], groupby: Optional[str] = None,
                        fname: Optional[str] = "bcell_subset_markers_dotplot.png", figsize=(15, 5), cmap=cmap_lbwr,
                        font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                        force: bool = False, verbose: bool = False):
        """scanpy dot plot of marker genes (group label -> genes in var_names), scaled 0-1 per gene.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        apply_figure_style(sizes=(8, 7, 6))
        groupby = groupby or self.params["celltype_col"]
        miss = [g for v in markers.values() for g in v if g not in adata_sub.var_names]
        if miss:
            print("missing:", miss)
        markers = {k: [g for g in v if g in adata_sub.var_names] for k, v in markers.items()}
        dp = sc.pl.dotplot(adata_sub, markers, groupby=groupby, standard_scale="var", cmap=cmap, show=False,
                           figsize=figsize, var_group_rotation=30, colorbar_title="mean expression\n(scaled 0-1 per gene)",
                           size_title="% cells expressing")
        fig = plt.gcf()
        for ax in fig.axes:
            for t in ax.get_xticklabels():
                t.set_style("italic")
        n_groups = adata_sub.obs[groupby].nunique()
        fig.suptitle(f"Markers separating the {n_groups} subsets (within-subset comparison; {adata_sub.n_obs:,} cells)",
                     ha="left", fontsize=8, x=0.02, y=1.06)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, dpi=200, force=force, verbose=verbose)
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
                        fname_tsv: Optional[str] = "ifn_hotspot_celltype_enrichment.tsv",
                        font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                        force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """ISG hotspot detection, cell-type enrichment in the hotspot (Mann-Whitney U, BH FDR) and summary figure.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing figure; verbose: print the file status.
        Returns df_hot (per cell type: mean_hot, mean_rest, log2fc, p, padj)."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
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
                ax.text(fc + 0.05, i, "n.s.", va="center", ha="left", fontsize=6, color=self._muted(bg_color))
        ax.set_xlabel("log2 fold change, hotspot vs rest\n(mapped q05 abundance)")
        ax.set_title("Cell types enriched / depleted in the hotspot", loc="left", fontsize=8)
        fig.text(0.1, 0.03, f"Dashed cyan = hull of the hotspot ({hs.sum()} of {adata_vis.n_obs:,} spots).", fontsize=7)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname_fig, dpi=200, force=force, verbose=verbose)
        if fname_tsv:
            self._write_table(df_hot.round(4), fname_tsv, force, verbose)
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
                                   , force: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """One tissue: per cell type x age group x gene: n_cells, n_donors, pct_expr, mean_log1p_cp10k
        (panel_lognorm, library size obs['n_counts']).
        Returns (df_long: all pairs, df_plot: cell types with >= min_cells in >= 2 age groups and pairs with
        >= min_cells, df_age: cells and donors per age group). fname -> df_long saved in root_result (a record: always
        computed, the file is kept when it exists unless force; verbose: print the file status)."""
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
        self._write_table(df_long.round(4), fname, force, verbose)     # kept when it exists, unless force
        return df_long, df_plot, df_age

    def plot_dotplot_celltype_age(self, df_plot: pd.DataFrame, lineage: Optional[Dict[str, Sequence[str]]] = None,
                                  df_age: Optional[pd.DataFrame] = None, cmaps: Optional[Dict[str, str]] = None,
                                  cmap=cmap_lbwr, fname: Optional[str] = None, 
                                  fontsize_ticks:int=11, fontsize_title:int=14, fontsize_text:int=12,
                                  fontsize_bar:int=10, fontsize_perc_cells:int=10,
                                  fontsize_lineage:int=10, figsize: tuple = (12, 9),
                                  font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                  force: bool = False, verbose: bool = False):
        """
        Dot plot: rows = cell types in lineage order (labels coloured by lineage), column blocks = genes,
        columns inside a block = age groups. Size = % cells > 0; colour = mean log1p(CP10k) / max per gene.
        cmap: colormap or name (default cmap_lbwr: light blue -> white -> light red, white = half of the gene's max).
        Font sizes: fontsize_*; colours: font_color, bg_color (style_figure). force: rewrite an existing file; verbose: print the file status.
        """
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)

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
        self.style_figure(fig, font_color, bg_color)
        fig.tight_layout()

        self._savefig(fig, fname, force=force, verbose=verbose)

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
                               fname: Optional[str] = None, fname_all: Optional[str] = None,
                               fname_donors: Optional[str] = None, force: bool = False,
                               verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Marker tiers for every subset vs the other subsets:
        unique            : padj < marker_padj, pct_in >= marker_min_pct_in, max % in every other subset <=
                            unique_max_pct_other, log2FC vs the next-highest subset (mean CP10k) >= unique_min_log2fc;
        enriched (shared) : same with shared_max_pct_other / shared_min_log2fc (and not unique).
        Candidates get donors_top = 'k/n' (donor_consistency: donors where the gene is highest in that subset).
        Returns (df_all: every gene x subset, df_markers: tiered genes sorted by subset, tier, log2fc);
        df_all.attrs['df_donor_counts'] = cells per donor x subset.

        Files (keys of self.files or names in root_result): fname = df_markers (csv, rounded, for reading);
        fname_all = df_all (full precision) and fname_donors = the donor counts (tsv) are the cache: when fname,
        fname_all and fname_donors all exist and not force, they are read and returned (df_markers rebuilt from
        df_all) instead of recomputed (the donor-consistency loop is the slow part). force: recompute and rewrite;
        verbose: print the file status. The cache holds the result of the parameters that wrote it: use force=True
        after changing marker_* / unique_* / shared_* / donor_min_cells or the subsets.
        """
        
        P = self.params

        groups = list(groups or P["bcell_types"])
        groupby = groupby or P["celltype_col"]
        paths = {k: self._resolve_path(f, None, self.root_result) for k, f in
                 (("markers", fname), ("all", fname_all), ("donors", fname_donors)) if f}
        
        if len(paths) == 3 and not force and all(p.exists() for p in paths.values()):
            txt = ["gene", "group", "tier", "donors_top", "next_pct_type", "next_mean_type"]
            df_all = pdreadcsv(paths["all"].name, paths["all"].parent, dtype={c: str for c in txt}, verbose=verbose)
            df_all[txt] = df_all[txt].fillna("")
            df_all.attrs["df_donor_counts"] = pdreadcsv(paths["donors"].name, paths["donors"].parent,
                                                        dtype={"donor": str}, index_col=0, verbose=verbose)
            return df_all, self._subset_markers_from_all(df_all, groups)

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
        df_markers = self._subset_markers_from_all(df_all, groups)
        df_cnt = df_cnt.reindex(columns=groups).rename_axis(index="donor", columns=None)
        df_all.attrs["df_donor_counts"] = df_cnt

        if fname:
            cols = ["group", "gene", "tier", "pct_in", "max_pct_other", "next_pct_type", "log2fc_vs_next",
                    "next_mean_type", "lfc_vs_rest", "padj", "donors_top"]
            pdwritecsv(df_markers[cols].round(3), paths["markers"].name, paths["markers"].parent, sep=",", verbose=verbose)
    
        if "all" in paths:                       # float32 -> float64: written in full, read back identical
            f32 = {c: "float64" for c, t in df_all.dtypes.items() if t == np.float32}
            pdwritecsv(df_all.astype(f32), paths["all"].name, paths["all"].parent, verbose=verbose)
    
        if "donors" in paths:
            pdwritecsv(df_cnt.reset_index(), paths["donors"].name, paths["donors"].parent, verbose=verbose)

        return df_all, df_markers

    @staticmethod
    def _subset_markers_from_all(df_all: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
        """Tiered genes of df_all (tier != ''), group in subset order, sorted by group, tier, log2fc_vs_next."""
        df_markers = df_all[df_all.tier != ""].copy()
        df_markers["group"] = pd.Categorical(df_markers.group, list(groups))
        return df_markers.sort_values(["group", "tier", "log2fc_vs_next"], ascending=[True, False, False])

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
                                   , force: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """Detection of one gene (raw counts > 0) in the B subsets by tissue, and within GC B cells per donor,
        study and chemistry (AICDA study). Returns dict:
        df_gc_share (GC B cells per tissue), df_subset (% positive and % single-UMI per subset),
        chem_tonsil_gc (% positive in tonsil GC B by Method), df_subset_tissue, df_gc_donor (>= min_gc GC cells),
        n_pos, pct_pos_tonsil. fname_subset / fname_gc (csv, root_result): kept when they exist unless force; verbose."""
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
        self._write_table(df_st, fname_subset, force, verbose, sep=",")     # kept when they exist, unless force
        self._write_table(df_gc, fname_gc, force, verbose, sep=",")
        out["df_subset_tissue"] = df_st
        out["df_gc_donor"] = df_gc[df_gc.n_gc >= min_gc].sort_values(["Tissue", "Donor"])
        return out

    # =====================================================================================================
    # NMF co-location factors: cell types grouped by lineage x factors
    # =====================================================================================================
    def plot_factor_dotplot(self, adata_vis, n_fact: int, lineage: Optional[Dict[str, Sequence[str]]] = None,
                            cmaps: Optional[Dict[str, str]] = None, figsize:tuple=(15,12), size_max: float = 140, cmap: str = "RdPu",
                            min_show: float = 0.02, fname: Optional[str] = None, fname_tsv: Optional[str] = None,
                            font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                            force: bool = False, verbose: bool = False
                            ) -> Tuple[Any, pd.DataFrame]:
        """
        Dot plot of the NMF cell-type loadings (calc_df_coloc_loadings, each cell type's weights summed to 1 over
        factors): rows = cell types in lineage order (labels coloured by lineage), columns = factors ordered by the
        row of their top cell type, labelled with their top-2 cell types. Size and colour = relative weight;
        weights < min_show are not drawn. Style (None = params): font_color, bg_color, fontsize_scale (style_figure); force: rewrite
        an existing figure. Returns (fig, df_load in plotting order).
        """
        
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
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
        
        fig, ax = plt.subplots(figsize=figsize)
        
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
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        if fname_tsv:
            self._write_table(df_load.round(4).reset_index(names="cell_type"), fname_tsv, force, verbose)
        return fig, df_load

    # =====================================================================================================
    # subregions (mp_lymphoid03): region ids; subregion signatures scored on the scRNA reference
    # =====================================================================================================
    def add_region_ids(self, df_regions: pd.DataFrame, adata=None, label_col: str = "region_label",
                       cluster_col: str = "region_cluster", name_col: str = "region_name") -> pd.Series:
        """obs[label_col] = region id ('R06', params['region_id_fmt']) and obs[name_col] from df_regions (index = barcode,
        columns region_cluster, region_name: lymph01_region_labels.tsv). Returns the id -> name map."""
        a = self._check_adata(adata)
        d = df_regions.loc[a.obs_names]
        ids = d[cluster_col].astype(int).map(self.params["region_id_fmt"].format)
        a.obs[label_col] = pd.Categorical(ids.values, categories=sorted(ids.unique()))
        a.obs[name_col] = d[name_col].astype(str).values
        return pd.Series(dict(zip(ids.values, d[name_col].astype(str).values))).sort_index()

    def calc_df_subregion_ref_score(self, adata_ref, df_gene_sets: pd.DataFrame, df_sub_table: Optional[pd.DataFrame] = None,
                                    status: Optional[str] = "supported", n_genes: Optional[int] = None,
                                    min_genes: Optional[int] = None, groupby: Optional[str] = None,
                                    force: bool = False, verbose: bool = False
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """Score the signature of each subregion on the scRNA reference.

        Genes: the top n_genes subregion_signature genes (by Wilcoxon score) found in the reference; subregions with
        < min_genes are skipped (with df_sub_table, only subregions whose status starts with `status`).
        Cell score = mean over the genes of the per-gene z-score of log1p(CP10k) (library size obs['n_counts']), so
        highly expressed genes do not dominate. Returns
          df_ref_score : subregion x cell type (params['celltype_col']) mean cell score
          df_cell_score: cells x subregions (for UMAPs)
          genes_used   : {subregion: genes}.
        Cached in files 'subregion_ref_score' (rounded to 6 decimals) and 'subregion_ref_genes'; when both exist and
        not force, df_cell_score is recomputed from the cached genes (no Wilcoxon / selection step)."""
        groupby = groupby or self.params["celltype_col"]
        cache = ("subregion_ref_score", "subregion_ref_genes")
        if not force and all(k in self.files and self.file_path(k).exists() for k in cache):
            return self._read_ref_score(adata_ref, verbose)
        n_genes = self.get_param("ref_sig_n_genes", n_genes)
        min_genes = self.get_param("ref_sig_min_genes", min_genes)
        groupby = groupby or self.params["celltype_col"]
        d = df_gene_sets[df_gene_sets.gene_set == "subregion_signature"].sort_values("scores", ascending=False)
        if df_sub_table is not None and status:
            keep = df_sub_table.loc[df_sub_table.status.str.startswith(status), "sub_region_id"]
            d = d[d.sub_region_id.isin(set(keep))]
        sym = set(self.symbols(adata_ref).astype(str))
        genes_used = {}
        for s_id, g in d.groupby("sub_region_id", sort=False):
            gl = [x for x in g.gene if x in sym][:n_genes]
            if len(gl) >= min_genes:
                genes_used[s_id] = gl
        order = [s_id for s_id in df_gene_sets.sub_region_id.unique() if s_id in genes_used]
        genes_used = {k: genes_used[k] for k in order}
        df_cell_score = self._calc_df_cell_score(adata_ref, genes_used)
        ct = adata_ref.obs[groupby].astype(str).values
        df_ref_score = df_cell_score.groupby(ct).mean().T.round(6)      # 6 decimals: ref_cell_types text (2 decimals) not double-rounded
        if all(k in self.files for k in cache):
            self.write_tsv(df_ref_score.reset_index(names="sub_region_id"), "subregion_ref_score", verbose=verbose)
            self.write_tsv(pd.DataFrame({"sub_region_id": list(genes_used),
                                         "genes": [",".join(v) for v in genes_used.values()]}),
                           "subregion_ref_genes", verbose=verbose)
            return self._read_ref_score(adata_ref, verbose)         # read back: same values as a cached run
        return df_ref_score, df_cell_score, genes_used

    def _read_ref_score(self, adata_ref, verbose: bool = False):
        """Cached calc_df_subregion_ref_score: df_ref_score and genes_used from their files, df_cell_score recomputed."""
        df_ref_score = self.read_tsv("subregion_ref_score", verbose=verbose).set_index("sub_region_id")
        df_ref_score.index.name = None
        d = self.read_tsv("subregion_ref_genes", verbose=verbose)
        genes_used = {r.sub_region_id: r.genes.split(",") for r in d.itertuples()}
        return df_ref_score, self._calc_df_cell_score(adata_ref, genes_used), genes_used

    def _calc_df_cell_score(self, adata_ref, genes_used: Dict[str, List[str]]) -> pd.DataFrame:
        """Cells x subregions: mean over the subregion's genes of the per-gene z-score of log1p(CP10k)."""
        union = sorted({x for v in genes_used.values() for x in v})
        found, _, E = self.panel_lognorm(union, adata_ref)
        E = E.astype(np.float32)
        sd = E.std(0)
        Z = (E - E.mean(0)) / np.where(sd > 0, sd, 1)
        col = {g: i for i, g in enumerate(found)}
        S = np.column_stack([Z[:, [col[g] for g in gl]].mean(1) for gl in genes_used.values()])
        return pd.DataFrame(S, index=adata_ref.obs_names, columns=list(genes_used))

    def add_ref_cell_types(self, df_sub_table: pd.DataFrame, df_ref_score: pd.DataFrame, n: int = 3,
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """df_sub_table['ref_cell_types'] = top-n reference cell types per subregion ('' = not scored). The subregion
        table file ('subregion_table') is rewritten when the column changed, or always with force."""
        df = df_sub_table.copy()
        new = df.sub_region_id.map(self.calc_df_ref_top(df_ref_score, n=n)).fillna("").astype(str)
        changed = "ref_cell_types" not in df or not new.equals(df["ref_cell_types"].fillna("").astype(str))
        df["ref_cell_types"] = new.values
        if "subregion_table" in self.files and (changed or force):
            self.write_tsv(df, "subregion_table", verbose=verbose)
        elif verbose:
            print("ref_cell_types unchanged: subregion table not rewritten")
        return df

    @staticmethod
    def calc_df_ref_top(df_ref_score: pd.DataFrame, n: int = 3) -> pd.Series:
        """Top-n reference cell types per subregion: 'B_GC_LZ (1.42), FDC (1.10), ...'."""
        return df_ref_score.apply(lambda r: ", ".join(f"{c} ({v:.2f})" for c, v in r.sort_values(ascending=False)
                                                      .head(n).items()), axis=1).rename("ref_cell_types")

    def plotly_subregion_ref_heatmap(self, df_ref_score: pd.DataFrame, lineage: Optional[Dict[str, Sequence[str]]] = None,
                                     df_sub_table: Optional[pd.DataFrame] = None, height: Optional[int] = None,
                                     width: Optional[int] = None, fontsize_title: int = 17, fontsize_xticks: int = 9,
                                     fontsize_yticks: int = 9, fontsize_colorbar: int = 12, fontsize_lineage: int = 10,
                                     fontsize_hover: int = 13, colorscale: str = "RdBu_r", lineage_line_color: str = "#444444",
                                     lineage_line_width: float = 1.0, font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                     fname: Optional[str] = "fig_ref_heatmap",
                                     force: bool = False, verbose: bool = False):
        """Heatmap subregion (rows, grouped by region) x reference cell type (columns, lineage order): mean cell score
        of the subregion signature. Hover: signature genes, status.

        Font sizes: fontsize_title, fontsize_xticks (cell types), fontsize_yticks (subregions), fontsize_colorbar,
        fontsize_lineage (lineage labels on top), fontsize_hover. Colours: colorscale (diverging, symmetric at the 99th
        percentile of |score|), lineage_line_color / lineage_line_width (separators), font_color, bg_color."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        cols = [c for c in (self.lineage_order(lineage) if lineage else df_ref_score.columns) if c in df_ref_score.columns]
        Z = df_ref_score[cols]
        info = {}
        if df_sub_table is not None:
            info = {r.sub_region_id: f"{r.status}<br>up: {r.subregion_signature_genes[:70]}" for r in df_sub_table.itertuples()}
        cd = np.array([[info.get(s_id, "")] * len(cols) for s_id in Z.index])
        vmax = float(np.nanpercentile(np.abs(Z.values), 99))
        fig = go.Figure(go.Heatmap(z=Z.values, x=cols, y=list(Z.index), colorscale=colorscale, zmid=0, zmin=-vmax, zmax=vmax,
                                   customdata=cd,
                                   colorbar=dict(title=dict(text="mean<br>score (z)", font=dict(size=fontsize_colorbar)),
                                                 tickfont=dict(size=fontsize_colorbar)),
                                   hovertemplate="%{y} signature in %{x}<br>score %{z:.2f}<br>%{customdata}<extra></extra>"))
        if lineage:
            x0 = 0
            for k, v in lineage.items():
                n_ = len([c for c in v if c in cols])
                fig.add_vline(x=x0 + n_ - 0.5, line_color=lineage_line_color, line_width=lineage_line_width)
                fig.add_annotation(x=x0 + (n_ - 1) / 2, y=1.0, yref="paper", yanchor="bottom", text=k, showarrow=False,
                                   font=dict(size=fontsize_lineage, color=font_color))
                x0 += n_
        fig.update_layout(title=dict(text="Subregion signatures scored on the scRNA reference (mean per-gene z of log1p CP10k)",
                                     font=dict(size=fontsize_title, color=font_color)),
                          height=height or max(420, 18 * len(Z) + 200), width=width, template="plotly_white",
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          hoverlabel=dict(font=dict(size=fontsize_hover)),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=fontsize_yticks, color=font_color)),
                          xaxis=dict(tickfont=dict(size=fontsize_xticks, color=font_color)),
                          margin=dict(t=110))
        self._save_figure(fig, fname, force, verbose)
        return fig

    def plot_ref_umap_subregion_scores(self, adata_ref, df_cell_score: pd.DataFrame, subregions: Sequence[str],
                                       df_ref_score: Optional[pd.DataFrame] = None, ncols: int = 4, q: Optional[float] = None,
                                       panel_size: float = 3.2, point_size: float = 0.3, fontsize_title: float = 8,
                                       fontsize_suptitle: float = 9, fontsize_colorbar: float = 6,
                                       cmap_colors: Sequence[str] = ("#e6e6e6", "#fdbb84", "#d7301f", "#7f0000"),
                                       font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                       fname: Optional[str] = "fig_ref_umap",
                                       force: bool = False, verbose: bool = False):
        """Reference UMAP (obsm['X_umap']) coloured by the signature score of each subregion (one panel per subregion,
        own scale clipped at the q-th percentile, values < 0 grey). Title: subregion + top reference cell type.

        Sizes: panel_size (inches per panel), point_size. Font sizes: fontsize_title (panel titles), fontsize_suptitle,
        fontsize_colorbar (tick labels). Colours: cmap_colors (low -> high; the first is the colour of scores <= 0),
        font_color, bg_color (figure and panel background)."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        q = self.get_param("ref_score_clip", q)
        U = adata_ref.obsm["X_umap"]
        subregions = [s_id for s_id in subregions if s_id in df_cell_score.columns]
        nrows = int(math.ceil(len(subregions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False,
                                 facecolor=bg_color)
        o = np.random.default_rng(self.params["seed"]).permutation(len(U))
        cmap = LinearSegmentedColormap.from_list("score", list(cmap_colors))
        for ax, s_id in zip(axes.ravel(), subregions):
            ax.set_facecolor(bg_color)
            v = df_cell_score[s_id].values
            cap = np.nanpercentile(v, q)
            vv = np.clip(v, 0, cap)
            oo = o[np.argsort(vv[o], kind="stable")]           # high scores on top
            sca = ax.scatter(U[oo, 0], U[oo, 1], s=point_size, c=vv[oo], cmap=cmap, vmin=0, vmax=cap, lw=0, rasterized=True)
            top = ""
            if df_ref_score is not None and s_id in df_ref_score.index:
                top = df_ref_score.loc[s_id].idxmax()
            ax.set_title(f"{s_id}" + (f" - {top}" if top else ""), fontsize=fontsize_title, loc="left", color=font_color)
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            cb = fig.colorbar(sca, ax=ax, fraction=0.04, pad=0.01)
            cb.ax.tick_params(labelsize=fontsize_colorbar, colors=font_color)
            cb.outline.set_edgecolor(font_color)
        for ax in axes.ravel()[len(subregions):]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.suptitle("Reference UMAP coloured by subregion signature score (mean per-gene z; < 0 grey)",
                     fontsize=fontsize_suptitle, color=font_color, x=0.01, y=1.0, ha="left", va="bottom")
        self._save_figure(fig, fname, force, verbose)
        return fig

    def reference_signatures_symbols(self, fname: str = "sc.h5ad", root: Optional[Path] = None,
                                     summary: str = "means") -> pd.DataFrame:
        """Reference signatures used by the cell2location map (genes x 34 cell types) from the trained reference
        (results/lymph_nodes_analysis/reference_signatures/sc.h5ad, varm '<summary>_per_cluster_mu_fg'), indexed
        by gene symbol (var[symbol_col]; duplicated symbols: first kept). With 'means' and q05 abundances,
        calc_celltype_expr reproduces the per-cell-type layers of sp.h5ad. fname: file in (root or root_result), or a
        key of self.files (e.g. 'signatures')."""
        path = self._resolve_path(fname, root, self.root_result)
        a = ad.read_h5ad(path, backed="r")
        df_sig, _ = self.get_reference_signatures(a, prefer=(summary,))
        df_sig.index = a.var.loc[df_sig.index, self.symbol_col].astype(str).values
        a.file.close()
        return df_sig[~df_sig.index.duplicated()]

    # aliases with the names used in ccc01 / ccc02 (cell2location_lib)
    def spatial_plot_bcell_types(self, adata_vis, focus, lineage, cmaps=None, ncols: int = 4, figsize:tuple=(20,16),
                                 fname: Optional[str] = "celltype_spatial_maps.png",
                                 font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                 force: bool = False, verbose: bool = False):
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        return self.spatial_plot_cell_types(adata_vis, focus, lineage, cmaps=cmaps, ncols=ncols, figsize=figsize,
                                            fname=fname, font_color=font_color, bg_color=bg_color, fontsize_scale=fontsize_scale,
                                            force=force, verbose=verbose)

    def dotplot_bcell_markers(self, adata_sub, markers: Dict[str, Sequence[str]], groupby: Optional[str] = None,
                              fname: Optional[str] = "bcell_subset_markers_dotplot.png", figsize=(15, 5), cmap=cmap_lbwr,
                              font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                              force: bool = False, verbose: bool = False):
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        return self.dotplot_markers(adata_sub, markers, groupby=groupby, fname=fname, figsize=figsize, cmap=cmap,
                                    font_color=font_color, bg_color=bg_color, fontsize_scale=fontsize_scale, force=force, verbose=verbose)

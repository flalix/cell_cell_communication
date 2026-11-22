#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single_cell_heart_dvlp_lib.py - SingleCellHeart(SingleCell)
developing human heart: 38 Visium sections (16 hearts, PCW 6-12) + scRNA reference (76,991 cells, PCW 5.5-14).

Paper: Lazar E, Mauron R, ..., Lundeberg J. Spatiotemporal gene expression and cellular dynamics of the developing human
heart. Nat Genet 57, 2756-2771 (2025). doi:10.1038/s41588-025-02352-6. Code: github.com/rmauron/HDCA_heart_dev;
data: Mendeley doi:10.17632/fhtb99mdzd.1 (Space Ranger, metadata) and doi:10.17632/w65jtfsvpr.1 (Seurat RDS).

Methods (the generic Scanpy / Squidpy / cell2location / LIANA ones are in SingleCell):
    build data            : read_sections_table, build_visium_h5ad, build_reference_h5ad, assemble_bundle,
                            read_stereoscope, paper_cluster_names
    sections / ages       : section_table, representative_sections, age_order, add_section_obs
    reference             : ref_cell_types, filter_reference, calc_df_ref_age_composition, lineage_of / lineage_order,
                            lineage_colors, celltype_colors
    figures (reference)   : plot_umap_celltype_dotplot, plot_umap_genes_by_age
    figures (Visium)      : plot_sections_gallery, plot_spatial_genes_by_age, plot_spatial_celltypes_by_age,
                            plot_regions_by_age
    abundance             : get_abundance (cell2location q05 or the paper's stereoscope proportions), abundance_source
    regions               : add_region_ids, calc_df_region_summary, calc_df_region_vs_paper, plot_region_vs_paper,
                            calc_df_region_age, plot_region_age, calc_df_celltype_age_trend, plot_celltype_age_trend,
                            plot_factor_dotplot
Parameters added to DEFAULT_PARAMS: HEART_PARAMS. HEART_LINEAGE: the 31 high-level cell types grouped by lineage.
HEART_GENES: 20 heart-development genes (symbol -> role). Every DataFrame variable starts with 'df_'.

@created on 2026/11/15
@author: Flavio Lichtenstein & Claude-Science Opus 5.5
"""

import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import anndata as ad
import scanpy as sc

from libs.Basic import pdreadcsv, pdwritecsv
from libs.single_cell_lib import DEFAULT_PARAMS, REGION_COLORS, SingleCell, cmap_byr, cmap_lbwr

__version__ = "1.0.0"

# the 31 high-level (coarse-grained) scRNA cell types of the paper (obs['cell_type']), grouped by lineage
HEART_LINEAGE: Dict[str, List[str]] = {
    "cardiomyocyte": ["Mat_vCM", "MetAct_vCM_1", "MetAct_vCM_2", "Immat_CM", "Prol_CM", "Mat_aCM", "MetAct_aCM"],
    "fibroblast":    ["Int_FB", "OFT_FB", "AnnFibr_FB", "PDE4Chigh_FB", "Prol_FB"],
    "epicardial":    ["EpC", "EPDC"],
    "mesenchymal":   ["Valve_MC", "Peric_MC"],
    "mural":         ["OFT_SMC", "CA_SMC", "PC"],
    "endothelial":   ["Endoc_EC", "EndocCush_EC", "MicroVasc_EC", "MacroVasc_EC", "PDE4Chigh_EC", "LEC"],
    "TMSB10high":    ["TMSB10high_C_1", "TMSB10high_C_2"],
    "neural":        ["NB-N", "SCP-GC"],
    "immune":        ["MyC", "LyC"],
}

# 20 genes of heart development shown on the reference UMAP and on the Visium sections (symbol -> role)
HEART_GENES: Dict[str, str] = {
    "NKX2-5": "core cardiac TF (first + second heart field)",
    "GATA4":  "core cardiac TF, septation and valves",
    "TBX5":   "left ventricle / atria; septation, conduction",
    "TBX20":  "cardiac TF, chamber and valve development",
    "HAND1":  "left ventricle (first heart field)",
    "HAND2":  "right ventricle / outflow tract (second heart field)",
    "ISL1":   "second heart field progenitors, sinoatrial node",
    "MYH6":   "atrial myosin (alpha-MHC)",
    "MYH7":   "ventricular myosin (beta-MHC), compact myocardium",
    "MYL7":   "atrial myosin light chain (MLC2a)",
    "MYL2":   "ventricular myosin light chain (MLC2v)",
    "NPPA":   "atrial / trabecular natriuretic peptide (ANP)",
    "IRX4":   "ventricular identity TF",
    "PITX2":  "left-right asymmetry, left atrium",
    "BMP10":  "right atrium; trabeculation",
    "TBX3":   "conduction system (SAN, AVN, bundle)",
    "SHOX2":  "sinoatrial node pacemaker",
    "HCN4":   "pacemaker channel (SAN, AVN)",
    "WT1":    "epicardium, epicardium-derived cells",
    "TCF21":  "epicardium-derived fibroblast progenitors",
}

HEART_PARAMS: Dict[str, Any] = dict(
    # --- Visium layout: obs['section'] = Space Ranger id (library), var_names = Ensembl IDs, symbols in var['symbol']
    age_col="age_label",               # 'w6' ... 'w12' (section table); numeric: 'age_pcw'
    age_group_col="age_group",         # 'w6-w7', 'w8-w9', 'w10-w12' (authors, ST_export_data.Rmd)
    heart_col="heart",
    age_order=("w6", "w7", "w7.5", "w8", "w9", "w10", "w10.5", "w11", "w12"),
    age_group_order=("w6-w7", "w8-w9", "w10-w12"),
    library_main="V11A20-401_A1",      # w8, 4 chambers: the section used for single-section figures

    # --- scRNA reference
    ref_labels_key="cell_type",        # 31 high-level types (+ 4 HL_excl_* removed by the authors)
    ref_batch_key="sample_id",         # 21 10x libraries
    ref_covariate_keys=(),
    ref_exclude_regex=r"^HL_excl",     # clusters removed by the authors (red blood cells, low quality)
    ref_age_col="age_group",           # scRNA age groups ('5-6', '7-8', '9-11', '12-14' PCW)
    ref_age_order=("5-6", "7-8", "9-11", "12-14"),
    ref_symbol_col="symbol",           # reference LIANA: symbols in var['symbol'] (var_names = Ensembl IDs)
    celltype_col="cell_type",
    libsize_col="n_counts",

    # --- cell2location (runs: results/c2l_{c2l_tag}/; GPU: src/programs/run_c2l_heart_dvlp.py)
    c2l_tag="gpu",                     # results/c2l_<tag>/: run src/programs/run_c2l_heart_dvlp.py --tag gpu first
    c2l_N_cells_per_location=20,       # dense fetal myocardium; estimate from H&E nuclei counts if possible
    c2l_detection_alpha=20,
    c2l_max_epochs=6000,               # minibatch (7 x 10000 spots per epoch): 6000 epochs = 42,000 steps (tutorial full data: 30,000)
    c2l_batch_size=10000,              # spots per step; full data (None) and 20000 run out of memory on a 24 GB GPU (~1.1 GB per 1000 spots)
    c2l_train_if_missing=False,        # notebooks: train when the cache is missing (long; normally the program does it)
    ref_max_epochs=250,
    abundance_source="auto",           # 'c2l' | 'stereoscope' | 'auto' (c2l map when it exists, else the paper's)

    # --- regions on cell-abundance space (69k spots, 31 types)
    region_n_neighbors=30,
    region_resolution=1.0,             # 20 regions on the stereoscope abundances (paper: 23 expression clusters)
    region_leiden_flavor="igraph",
    region_top_n=2,
    region_id_fmt="R{:02d}",
    region_min_overlap=0.30,           # region -> paper cluster label when >= this share of its (shared) spots

    # --- subregions (heart03) and LIANA (heart04): SingleCell defaults; ComBat by section
    subregion_skip=("R28",),            # c2l partition: R28 = low-signal spots (618 UMIs); stereoscope partition: use ("R18",)
    sub_min_batch_spots=10,            # ComBat by section: sections with fewer spots in a region are pooled           # 20-spot outlier cluster of the stereoscope regions (heart_regions.tsv)
    region_table_first=("region_id", "region_name", "signature_genes", "functional_description"),
    region_table_depth=True,
    subregion_table_first=("region_id", "sub_region_id", "subregion_name", "subregion_signature_genes", "n_other",
                           "other_genes_top_expressed", "not_present_genes", "functional_description"),
    gene_display_exclude=r"^RP[SL]\d|^MT-|^A[CLP]\d{6}\.\d|^LINC\d|-AS\d$",
    border_min_rel=1.25,
    subregion_map_title="{lib}: subregions of each heart region",
    
    # --- plotting
    lineage_cmaps={"cardiomyocyte": "Reds", "fibroblast": "Oranges", "epicardial": "YlOrBr", "mesenchymal": "Purples",
                   "mural": "RdPu", "endothelial": "Blues", "TMSB10high": "Greys", "neural": "Greens",
                   "immune": "GnBu"},
    panel_n_bg=20000,
    dpi_maps=150,                      # H&E grids (sections x genes / cell types / regions): 300 dpi -> 15-25 MB files
    font_color="white",
    bg_color="black",
    fontsize_scale=1.4,
)

HEART_ANALYSIS_PARAMS: Dict[str, Dict[str, Any]] = {
    "heart01": dict(),
    "heart02": dict(coloc_n_facts=(12,)),
}

# key -> (root, name); root 'data' | 'result' | 'figure' | 'root0'. {tag} = params['c2l_tag'] (filled in __init__).
HEART_FILES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "inputs": dict(
        visium=("data", "hdca_heart_visium_38sections.h5ad"),          # complete Visium AnnData (raw counts)
        reference=("data", "hdca_heart_sc_reference.h5ad"),            # complete scRNA AnnData (raw counts)
        sections=("data", "heart_sections.tsv"),                       # 38 sections: age, heart, chambers, ...
        visium_raw=("data", "visium"),                                 # Space Ranger outputs (one folder per section)
        bundle_sc=("data", "bundles/sc_reference"),                    # export of HDCA_heart_sc_full_extended.rds
        bundle_st=("data", "bundles/st_clustering"),                   # export of HDCA_heart_clustering.rds
        st_annotations=("data", "metadata/HDCA_heart_ST_annotations.csv"),
        hl_annotations=("data", "metadata/HDCA_heart_SC_annotations_HL_240115.csv"),
        dl_annotations=("data", "metadata/HDCA_heart_SC_annotations_DL_240204.csv"),
        stereoscope_hl=("data", "metadata/W.2023-09-25133054.715121_hl.tsv"),
        stereoscope_dl=("data", "metadata/W.2023-10-28105405.918149_dl.tsv"),
        hires_bg=("data", "hires_bg/{lib}_bg_{color}.png"),
        c2l_ref_dir=("result", "c2l_{tag}/reference_signatures"),
        c2l_map_dir=("result", "c2l_{tag}/cell2location_map"),
        c2l_map=("result", "c2l_{tag}/cell2location_map/sp.h5ad"),
        signatures=("result", "c2l_{tag}/reference_signatures/sc.h5ad"),
        region_labels=("result", "heart01_region_labels.tsv"),
        regions_cache=("result", "heart01_regions_cache.tsv.gz"),        # region_cluster + UMAP of cluster_regions / umap_regions (+ .json fingerprint)
        region_info=("data", "heart_regions.tsv"),                     # curated region table (editable)
        mean_signatures=("result", "heart_reference_mean_counts_per_celltype.tsv.gz"),   # fallback signatures
        subregion_descriptions=("data", "heart_subregion_descriptions.tsv"),
        lineage_table=("root0", "tables/heart_lineage.tsv"),            # HL_ID (lineage), HL_SUBID (cell type), description, summary
    ),
    "heart01": dict(
        tsv_ref_celltypes=("result", "heart01_reference_celltypes.tsv"),  # cell types of the filtered reference
        tsv_gene_panel=("result", "heart01_gene_panel.tsv"),              # HEART_GENES: role, presence in reference / Visium
        tsv_gene_best_celltype=("result", "heart01_gene_best_celltype.tsv"),  # per gene: cell type of highest mean expression
        fig_sections=("figure", "heart01_sections_he_gallery.png"),
        fig_sections_rep=("figure", "heart01_sections_he_by_age.png"),
        fig_qc_visium=("figure", "heart01_qc_visium_sections.png"),
        fig_qc_ref=("figure", "heart01_qc_reference_libraries.png"),
        fig_umap_dotplot=("figure", "heart01_reference_umap_celltype_dotplot.png"),
        fig_umap_genes_age=("figure", "heart01_reference_umap_genes_by_age.png"),
        fig_ref_age_comp=("figure", "heart01_reference_celltype_by_age.png"),
        fig_ref_age_comp_row=("figure", "heart01_reference_celltype_by_age_row_norm.png"),
        fig_ref_age_counts=("figure", "heart01_reference_celltype_by_age_counts.png"),
        fig_genes_age=("figure", "heart01_visium_genes_by_age_{k}.png"),
        fig_ref_history=("figure", "heart01_ref_training_history.png"),
        fig_ref_qc=("figure", "heart01_ref_qc.png"),
        fig_map_history=("figure", "heart01_map_training_history.png"),
        fig_map_qc=("figure", "heart01_map_qc.png"),
        fig_abund_age=("figure", "heart01_celltype_abundance_by_age.png"),
        fig_abund_c2l_vs_stereo=("figure", "heart01_c2l_vs_stereoscope.png"),
        fig_regions=("figure", "heart01_umap_region_section.png"),
        fig_regions_age=("figure", "heart01_regions_by_age.png"),
        fig_regions_age_html=("figure", "heart01_regions_by_age.html"),
        fig_regions_3plot_html=("figure", "heart01_umap_region_section.html"),
        fig_regions_html=("figure", "heart01_regions_{lib}.html"),
        fig_factor_dotplot=("figure", "heart01_nmf_factors_lineage_dotplot_n{n}.png"),
        tsv_section_qc=("result", "heart01_section_qc.tsv"),
        tsv_ref_age_comp=("result", "heart01_reference_celltype_by_age.tsv"),
        tsv_ref_age_comp_row=("result", "heart01_reference_celltype_by_age_row_norm.tsv"),
        tsv_ref_age_counts=("result", "heart01_reference_celltype_by_age_counts.tsv"),
        tsv_c2l_vs_stereo=("result", "heart01_c2l_vs_stereoscope_correlation.tsv"),
        tsv_region_names=("result", "heart01_region_names.tsv"),        # per region: automatic + curated name, n spots
        tsv_factor_load=("result", "heart01_nmf_factor_loadings_n{n}.tsv"),
    ),
    "heart02": dict(
        fig_region_vs_paper=("figure", "heart02_regions_vs_paper_clusters.png"),
        fig_region_age=("figure", "heart02_region_share_by_age.png"),
        fig_celltype_trend=("figure", "heart02_celltype_abundance_trend.png"),
        fig_region_celltype=("figure", "heart02_region_celltype_heatmap.png"),
        fig_region_genes=("figure", "heart02_region_gene_dotplot.png"),
        fig_regions_age=("figure", "heart02_regions_by_age.png"),
        fig_regions_age_html=("figure", "heart02_regions_by_age.html"),
        fig_conduction=("figure", "heart02_conduction_system_by_age.png"),
        fig_factor_dotplot=("figure", "heart02_nmf_factors_lineage_dotplot_n{n}.png"),
        fig_region_map=("figure", "heart02_region_map_{lib}.html"),
        tsv_region_summary=("result", "heart02_region_summary.tsv"),
        tsv_region_vs_paper=("result", "heart02_region_vs_paper_clusters.tsv"),
        tsv_region_age=("result", "heart02_region_share_by_age.tsv"),
        tsv_celltype_trend=("result", "heart02_celltype_abundance_by_section.tsv"),
        tsv_factor_load=("result", "heart02_nmf_factor_loadings_n{n}.tsv"),
    ),
    "heart03": dict(
        region_rank_genes=("result", "heart03_region_rank_genes.tsv.gz"),
        region_table=("result", "heart03_region_table.tsv"),
        subregion_grid=("result", "heart03_subcluster_resolution_grid.tsv"),
        subregion_adata=("result", "heart03_subregions/{region}.h5ad"),
        subregion_gene_sets=("result", "heart03_subregion_gene_sets.tsv.gz"),
        subregion_table=("result", "heart03_subregion_table.tsv"),
        subregion_ref_score=("result", "heart03_subregion_ref_celltype_score.tsv"),
        subregion_ref_genes=("result", "heart03_subregion_ref_genes.tsv"),
        tsv_subregion_age=("result", "heart03_subregion_share_by_age.tsv"),
        fig_region_heatmap=("figure", "heart03_region_signature_heatmap.html"),
        fig_region_map=("figure", "heart03_region_map_{lib}.html"),
        fig_region_sub=("figure", "heart03_subregions_{region}.html"),
        fig_diagnostics=("figure", "heart03_subregion_diagnostics.html"),
        fig_subregion_map=("figure", "heart03_subregion_map_{lib}.html"),
        fig_subregion_age=("figure", "heart03_subregion_share_by_age.png"),
        fig_ref_heatmap=("figure", "heart03_subregion_ref_celltype_heatmap.png"),
    ),
    "heart04": dict(
        subregion_grid=("result", "heart03_subcluster_resolution_grid.tsv"),
        subregion_adata=("result", "heart03_subregions/{region}.h5ad"),
        subregion_table_heart03=("result", "heart03_subregion_table.tsv"),
        region_table_heart03=("result", "heart03_region_table.tsv"),
        spatial_weights=("result", "heart04_spatial_weights.tsv"),
        liana_bivariate=("result", "heart04_liana_bivariate.h5ad"),
        liana_global=("result", "heart04_liana_bivariate_global.tsv"),
        lr_by_region=("result", "heart04_lr_by_region.tsv"),
        lr_by_subregion=("result", "heart04_lr_by_subregion.tsv"),
        lr_by_age=("result", "heart04_lr_by_age_group.tsv"),
        celltype_lr_tissue=("result", "heart04_celltype_lr_tissue.tsv.gz"),
        celltype_lr_by_group=("result", "heart04_celltype_lr_by_group.tsv.gz"),
        celltype_links_by_group=("result", "heart04_celltype_links_by_group.tsv"),
        liana_reference=("result", "heart04_liana_reference_rank_aggregate.tsv"),
        liana_reference_supported=("result", "heart04_liana_reference_supported.tsv"),
        ccc_by_region=("result", "heart04_ccc_consensus_by_region.tsv"),
        ccc_by_subregion=("result", "heart04_ccc_consensus_by_subregion.tsv"),
        fig_lr_top=("figure", "heart04_lr_top_local_scores_{lib}.png"),
        fig_lr_regions=("figure", "heart04_lr_region_top_local_scores_{lib}.png"),
        fig_lr_region_hm=("figure", "heart04_lr_by_region_heatmap.html"),
        fig_lr_sub_hm=("figure", "heart04_lr_by_subregion_heatmap.html"),
        fig_lr_age_hm=("figure", "heart04_lr_by_age_group_heatmap.html"),
        fig_sender_recv=("figure", "heart04_celltype_lr_sender_receiver.png"),
        fig_links_region=("figure", "heart04_celltype_links_by_region.html"),
        fig_links_sub=("figure", "heart04_celltype_links_by_subregion.html"),
        fig_coloc=("figure", "heart04_celltype_colocalisation.html"),
    ),
}


class SingleCellHeart(SingleCell):
    """Developing human heart (Lazar et al. 2025): 38 Visium sections + scRNA reference.
    obs['section'] = Space Ranger id (= library_id of uns['spatial']); var_names = Ensembl IDs, symbols in var['symbol'].
    """

    default_params = {**DEFAULT_PARAMS, **HEART_PARAMS}
    analysis_files = HEART_FILES
    analysis_params = HEART_ANALYSIS_PARAMS

    def __init__(self, ROOT0, project: str = "heart_dvlp", symbol_col: str = "symbol", sample_col: str = "section",
                 mt_prefix: str = "MT-", **kw):
        """analysis= 'heart01' ... 'heart04' (its output files are registered: caching on); params= / files= override.
        The cell2location folders carry params['c2l_tag'] ('cpu' sandbox run, 'gpu' full run)."""
        super().__init__(ROOT0, project, symbol_col=symbol_col, sample_col=sample_col, mt_prefix=mt_prefix, **kw)
        tag = self.params["c2l_tag"]
        self.files = {k: (r, n.replace("{tag}", tag)) for k, (r, n) in self.files.items()}
        self.create_file_dirs()

    def create_file_dirs(self) -> List[Path]:
        """As SingleCell.create_file_dirs, without the folders whose name still has the unfilled {tag} (the base
        constructor runs before the c2l_tag substitution of __init__)."""
        keep = self.files
        self.files = {k: v for k, v in keep.items() if "{tag}" not in v[1]}
        try:
            return super().create_file_dirs()
        finally:
            self.files = keep

    def _cached_any(self, fname: Optional[str], compute, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached table given as a key of self.files or as a file name with an extension in root_result (the notebooks
        pass sch.fname(key)); None = computed, not written."""
        if fname is None:
            return compute()
        if fname in self.files:
            return self._cached_table(fname, compute, force=force, verbose=verbose)
        return self._cached_frames({"t": fname}, lambda: {"t": compute()}, force=force, verbose=verbose)["t"]

    # =====================================================================================================
    # building the complete h5ad files (src/programs/heart_dvlp_build_h5ad.py)
    # =====================================================================================================
    def read_sections_table(self, verbose: bool = False) -> pd.DataFrame:
        """heart_sections.tsv: one row per Visium section (section_id, section_num = the paper's section number, age_label,
        age_pcw, pcw, age_group, heart, LV/RV/LA/RA, chambers, n_chambers, comment, in_paper_clustering, n_spots,
        representative). Ordered by age, then section number."""
        df_sec = self.read_tsv("sections", verbose=verbose)
        for c in ("LV", "RV", "LA", "RA", "chambers", "chambers_partial", "comment"):
            if c in df_sec:
                df_sec[c] = df_sec[c].fillna("").astype(str)
        return df_sec.sort_values(["age_pcw", "section_num"]).reset_index(drop=True)

    def section_table(self, adata=None) -> pd.DataFrame:
        """Section table with the spot counts and QC medians of adata (when given)."""
        df_sec = self.read_sections_table()
        if adata is not None:
            obs = adata.obs
            g = obs.groupby(self.sample_col, observed=True)
            extra = pd.DataFrame({"n_spots_adata": g.size()})
            for c in ("total_counts", "n_genes_by_counts"):
                if c in obs:
                    extra[f"median_{c}"] = g[c].median()
            df_sec = df_sec.join(extra, on="section_id")
        return df_sec

    def representative_sections(self, df_sections: Optional[pd.DataFrame] = None) -> List[str]:
        """One section per age label (most whole chambers, then most spots), in age order."""
        df_sec = df_sections if df_sections is not None else self.read_sections_table()
        return df_sec.loc[df_sec["representative"].astype(bool), "section_id"].tolist()

    def age_order(self, present: Optional[Sequence[str]] = None) -> List[str]:
        order = list(self.params["age_order"])
        return [a for a in order if present is None or a in set(present)]

    def add_section_obs(self, adata, df_sections: Optional[pd.DataFrame] = None,
                        cols: Sequence[str] = ("section_num", "age_label", "age_pcw", "pcw", "age_group", "heart",
                                               "chambers", "n_chambers", "in_paper_clustering")) -> ad.AnnData:
        """Section metadata -> obs (categoricals in age order)."""
        df_sec = (df_sections if df_sections is not None else self.read_sections_table()).set_index("section_id")
        sec = adata.obs[self.sample_col].astype(str)
        for c in cols:
            adata.obs[c] = sec.map(df_sec[c]).values
        adata.obs[self.sample_col] = pd.Categorical(sec, categories=list(df_sec.index))
        adata.obs["age_label"] = pd.Categorical(adata.obs["age_label"], categories=self.age_order(adata.obs["age_label"].unique()))
        adata.obs["age_group"] = pd.Categorical(adata.obs["age_group"], categories=list(self.params["age_group_order"]))
        adata.obs["heart"] = adata.obs["heart"].astype("category")
        return adata

    def paper_cluster_names(self) -> Dict[str, str]:
        """The paper's 23 spatial clusters: cluster number -> short name (HDCA_heart_ST_annotations.csv)."""
        df = pd.read_csv(self.file_path("st_annotations"), sep=";")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        return dict(zip(df["cluster"].astype(str), df["cluster_name"].astype(str).str.strip()))

    def paper_cluster_full_names(self) -> Dict[str, str]:
        df = pd.read_csv(self.file_path("st_annotations"), sep=";")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        return dict(zip(df["cluster_name"].astype(str).str.strip(), df["cluster_full_name"].astype(str).str.strip()))

    def read_stereoscope(self, level: str = "hl") -> pd.DataFrame:
        """The paper's stereoscope proportions (69,114 spots x 33 high-level or 74 deep-level types; rows sum to 1).
        Index = '<barcode>_<section_num>'; columns = cell-type names."""
        df_w = pd.read_csv(self.file_path(f"stereoscope_{level}"), sep="\t", index_col=0)
        df_ann = pd.read_csv(self.file_path(f"{level}_annotations"), sep=";").iloc[:, :2]
        df_ann.columns = ["cluster", "cell_type"]
        key = df_ann["cluster"].astype(str).str.strip()
        if level == "hl":                                 # '4_35' in the table, 'HL_4-35' / '4_35' in the annotations
            key = key.str.replace(r"^HL_", "", regex=True).str.replace("-", "_")
        names = dict(zip(key, df_ann["cell_type"].astype(str).str.strip()))
        df_w.columns = [names.get(str(c).replace("-", "_") if level == "hl" else str(c), str(c)) for c in df_w.columns]
        return df_w

    @staticmethod
    def assemble_bundle(bundle_dir, verbose: bool = True) -> ad.AnnData:
        """AnnData from a bundle of src/programs/export_seurat_rds.R: X = raw counts (CSR float32), obs = meta.data,
        obsm['X_<reduction>']."""
        from scipy.io import mmread
        d = Path(bundle_dir)
        X = sp.csr_matrix(mmread(d / "counts.mtx").T.tocsr(), dtype=np.float32)
        barcodes = pd.read_csv(d / "barcodes.csv")["barcode"].astype(str)
        features = pd.read_csv(d / "features.csv")["feature"].astype(str)
        obs = pd.read_csv(d / "metadata.csv", index_col=0, low_memory=False)
        obs.index = obs.index.astype(str)
        assert obs.index.equals(pd.Index(barcodes)), "metadata rows are not in matrix order"
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(features)))
        for f in sorted(d.glob("reduction_*.csv")):
            emb = pd.read_csv(f, index_col=0)
            emb.index = emb.index.astype(str)
            a.obsm["X_" + f.stem.replace("reduction_", "")] = emb.reindex(a.obs_names).to_numpy(dtype=np.float32)
        if verbose:
            print(f"{d.name}: {a.n_obs:,} x {a.n_vars:,}; obsm {list(a.obsm)}")
        return a

    def visium_features(self, section: Optional[str] = None) -> pd.DataFrame:
        """Space Ranger features (gene_ids, symbol) of one section (all sections share the GRCh38 reference)."""
        import h5py
        df_sec = self.read_sections_table()
        section = section or df_sec.section_id.iloc[0]
        with h5py.File(self.file_path("visium_raw") / section / "filtered_feature_bc_matrix.h5", "r") as h:
            ids = h["matrix/features/id"][:].astype(str)
            names = h["matrix/features/name"][:].astype(str)
        return pd.DataFrame({"gene_ids": ids, "symbol": names})

    def make_lowres_images(self, sections: Optional[Sequence[str]] = None, verbose: bool = False) -> List[Path]:
        """The Mendeley Space Ranger outputs have no tissue_lowres_image.png (scanpy read_visium needs it): derived
        from the hires image with the scale-factor ratio (lowres_scalef / hires_scalef), Lanczos resampling; existing
        files are kept. Returns the files written."""
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        sections = sections or self.read_sections_table().section_id.tolist()
        written = []
        for s in sections:
            d = self.file_path("visium_raw") / s / "spatial"
            lo = d / "tissue_lowres_image.png"
            if lo.exists():
                continue
            sf = json.loads((d / "scalefactors_json.json").read_text())
            hi = Image.open(d / "tissue_hires_image.png")
            r = sf["tissue_lowres_scalef"] / sf["tissue_hires_scalef"]
            hi.resize((round(hi.width * r), round(hi.height * r)), Image.LANCZOS).save(lo)
            written.append(lo)
            if verbose:
                print(f"{s}: lowres {round(hi.width * r)} x {round(hi.height * r)} (ratio {r:.4f})")
        return written

    def build_visium_h5ad(self, df_sections: Optional[pd.DataFrame] = None, fname: str = "visium",
                          force: bool = False, verbose: bool = True) -> ad.AnnData:
        """
        Complete Visium AnnData of the 38 sections (read_visium_samples) + section metadata (obs), the paper's
        spatial cluster of the 17 clustered sections (obs['paper_cluster'], 'paper_cluster_name'), the paper's
        stereoscope proportions (obsm['stereoscope_hl'] 33 types, obsm['stereoscope_dl'] 74 states; obs['in_paper_qc']
        = spot in the paper's 69,114). X = raw counts. Written to files[fname] (read when it exists and not force).
        """

        path = self.file_path(fname)
        if path.exists() and not force:
            return self.read_h5ad(fname, set_self=False, verbose=verbose)
        
        df_sec = df_sections if df_sections is not None else self.read_sections_table()
        
        self.make_lowres_images(df_sec.section_id.tolist(), verbose=verbose)
        
        a = self.read_visium_samples(self.file_path("visium_raw"), df_sec.section_id.tolist(), set_self=False)
        
        a.X = sp.csr_matrix(a.X, dtype=np.float32)
        
        a.obs["barcode"] = [n[len(s) + 1:] for n, s in zip(a.obs_names, a.obs[self.sample_col].astype(str))]   # '<section>_<barcode>'
        self.add_section_obs(a, df_sec)
        a.obs["spot_id_paper"] = a.obs["barcode"].astype(str) + "_" + a.obs["section_num"].astype(int).astype(str)
        
        # paper clusters (HDCA_heart_clustering.rds: 17 sections, 25,309 spots)
        df_st = pd.read_csv(self.file_path("bundle_st") / "metadata.csv", index_col=0)
        
        key = df_st["section.name"].astype(str) + "_" + df_st.index.str.rsplit("_", n=1).str[0]
        names = self.paper_cluster_names()
        
        cl = pd.Series(df_st["seurat_clusters"].astype(str).values, index=key.values)
        
        a.obs["paper_cluster"] = pd.Categorical(cl.reindex(a.obs_names).values)
        a.obs["paper_cluster_name"] = pd.Categorical(a.obs["paper_cluster"].astype(str).map(names).where(
            a.obs["paper_cluster"].notna()))
        
        # stereoscope proportions (all 38 sections)
        for level in ("hl", "dl"):
            df_w = self.read_stereoscope(level)
            a.obsm[f"stereoscope_{level}"] = df_w.reindex(a.obs["spot_id_paper"].values).fillna(0.0).set_index(a.obs_names)
            if level == "hl":
                a.obs["in_paper_qc"] = a.obs["spot_id_paper"].isin(df_w.index).values
        
        a.uns["dataset"] = dict(paper="Lazar et al. 2025, Nat Genet 57:2756, doi:10.1038/s41588-025-02352-6",
                                source="Mendeley doi:10.17632/fhtb99mdzd.1 (Space Ranger outputs)",
                                n_sections=int(len(df_sec)), n_hearts=int(df_sec["heart"].nunique()))
        
        a.write_h5ad(path, compression="gzip")
        
        if verbose:
            print(f"Writing {path}: {a.n_obs:,} spots x {a.n_vars:,} genes, {a.obs[self.sample_col].nunique()} sections; "
                  f"paper clusters on {int(a.obs['paper_cluster'].notna().sum()):,} spots")
        
        return a

    def build_reference_h5ad(self, fname: str = "reference", force: bool = False, verbose: bool = True) -> ad.AnnData:
        """
        Complete scRNA AnnData from the export of HDCA_heart_sc_full_extended.rds (76,991 cells): X = raw counts,
        var_names = Ensembl IDs (symbols mapped through the Visium features; symbols without a unique Ensembl ID, e.g.
        the transgene markers of the Cell Ranger reference, are dropped), var['symbol']; obs: cell_type (35 labels,
        4 HL_excl_* removed by the authors -> obs['excluded']), cluster, sample_id (10x library), heart_id, age,
        age_group, QC; obsm['X_umap'] = the paper's UMAP (umap_oi), obsm['X_umap_df'], 'X_pca', 'X_harmony_oi'.
        """

        path = self.file_path(fname)
        if path.exists() and not force:
            return self.read_h5ad(fname, set_self=False, verbose=verbose)
        
        a = self.assemble_bundle(self.file_path("bundle_sc"), verbose=verbose)
        df_f = self.visium_features()
        df_u = df_f[~df_f.symbol.duplicated(keep=False)]
        sym2id = dict(zip(df_u.symbol, df_u.gene_ids))
        keep = a.var_names.isin(list(sym2id))
        n0 = a.n_vars
        a = a[:, keep].copy()
        a.var[self.symbol_col] = a.var_names.astype(str)
        a.var_names = [sym2id[s] for s in a.var[self.symbol_col]]
        ren = {"cell.types": "cell_type", "high_level_clusters": "cluster", "sampleID": "sample_id",
               "orig.ident": "heart_id", "age_groups": "age_group", "nCount_RNA": "n_counts",
               "nFeature_RNA": "n_genes"}
        a.obs = a.obs.rename(columns=ren)
        a.obs["excluded"] = a.obs["cell_type"].astype(str).str.match(self.params["ref_exclude_regex"]).values
        a.obs["lineage"] = a.obs["cell_type"].astype(str).map(self.lineage_of(HEART_LINEAGE)).fillna("excluded")
        for c in ("cell_type", "cluster", "sample_id", "heart_id", "lineage", "Phase"):
            a.obs[c] = a.obs[c].astype(str).astype("category")
        a.obs["age_group"] = pd.Categorical(a.obs["age_group"].astype(str), categories=list(self.params["ref_age_order"]))
        for c in a.obs.columns:
            if a.obs[c].dtype == object:
                a.obs[c] = a.obs[c].astype(str)
        a.obsm["X_umap"] = a.obsm.pop("X_umap_oi")
        a.uns["dataset"] = dict(paper="Lazar et al. 2025, Nat Genet 57:2756, doi:10.1038/s41588-025-02352-6",
                                source="Mendeley doi:10.17632/w65jtfsvpr.1, HDCA_heart_sc_full_extended.rds",
                                n_features_rds=int(n0), n_features_mapped=int(a.n_vars))
        a.write_h5ad(path, compression="gzip")
        if verbose:
            print(f"Writing {path}: {a.n_obs:,} cells x {a.n_vars:,} genes ({n0:,} in the RDS; "
                  f"{n0 - a.n_vars:,} without a unique Ensembl ID dropped)")
        return a

    # =====================================================================================================
    # reference helpers
    # =====================================================================================================
    @staticmethod
    def lineage_order(lineage: Dict[str, Sequence[str]]) -> List[str]:
        return [c for v in lineage.values() for c in v]

    @staticmethod
    def lineage_of(lineage: Dict[str, Sequence[str]]) -> Dict[str, str]:
        return {c: k for k, v in lineage.items() for c in v}

    def celltype_colors(self, lineage: Optional[Dict[str, Sequence[str]]] = None,
                        cmaps: Optional[Dict[str, str]] = None) -> Dict[str, tuple]:
        """Cell type -> colour: shades of its lineage colormap (params['lineage_cmaps'])."""
        lineage = lineage or HEART_LINEAGE
        cmaps = cmaps or self.params["lineage_cmaps"]
        out = {}
        for lin, types in lineage.items():
            cm = plt.get_cmap(cmaps.get(lin, "Greys"))
            for i, t in enumerate(types):
                out[t] = cm(0.40 + 0.42 * (i / max(len(types) - 1, 1)))      # 0.40-0.82: visible on black and white
        return out

    def lineage_colors(self, lineage: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, tuple]:
        lineage = lineage or HEART_LINEAGE
        return {lin: plt.get_cmap(self.params["lineage_cmaps"].get(lin, "Greys"))(0.75) for lin in lineage}

    def filter_reference(self, adata_ref, exclude_regex: Optional[str] = None) -> ad.AnnData:
        """Cells of the 31 high-level types (the 4 HL_excl_* clusters removed, as by the authors)."""
        rx = self.get_param("ref_exclude_regex", exclude_regex)
        keep = ~adata_ref.obs[self.params["celltype_col"]].astype(str).str.match(rx).values
        out = adata_ref[keep].copy()
        out.obs[self.params["celltype_col"]] = out.obs[self.params["celltype_col"]].astype(str).astype("category")
        return out

    def calc_df_ref_age_composition(self, adata_ref, groupby: Optional[str] = None, age_col: Optional[str] = None,
                                    normalize: str = "col", fname: Optional[str] = None, force: bool = False,
                                    verbose: bool = False) -> pd.DataFrame:
        """Cell types (rows) x scRNA age groups (columns) of the reference cells, + lineage.
        normalize='none': number of cells.
        normalize='col': % of the cells of the age group (each column sums to 100): composition of each age.
        normalize='row': the 'col' percentages rescaled so each cell type sums to 100 over the ages: in which ages a
        cell type is frequent, corrected for the unequal number of cells sequenced per age (row % of raw counts
        would mostly follow that sampling depth)."""
        if normalize not in ("none", "col", "row"):
            raise ValueError("normalize: 'none', 'col' or 'row'")
        groupby = self.get_param("celltype_col", groupby)
        age_col = self.get_param("ref_age_col", age_col)

        def _compute():
            df_ct = pd.crosstab(adata_ref.obs[groupby].astype(str), adata_ref.obs[age_col])
            order = [c for c in self.lineage_order(HEART_LINEAGE) if c in df_ct.index]
            df_ct = df_ct.reindex(order)
            df_pct = df_ct.astype(int) if normalize == "none" else 100 * df_ct / df_ct.sum(0)
            if normalize == "row":
                df_pct = 100 * df_pct.div(df_pct.sum(1), axis=0)
            df_pct = df_pct.round(3)
            df_pct.insert(0, "lineage", [self.lineage_of(HEART_LINEAGE).get(c, "") for c in df_pct.index])
            return df_pct.reset_index(names="cell_type")

        df = self._cached_any(fname, _compute, force=force, verbose=verbose)
        return df.set_index("cell_type")

    # =====================================================================================================
    # cell2location mapping with optional spot minibatches (38 sections, 69k spots: GPU memory)
    # =====================================================================================================
    def fit_or_load_mapping(self, adata_vis, df_inf_aver, run_dir, N_cells_per_location=None, detection_alpha=None,
                            batch_key=None, max_epochs=None, accelerator=None, fname: str = "sp.h5ad",
                            expected_layers: bool = False, read_layers: bool = False, batch_size=None,
                            force: bool = False, verbose: bool = True):
        """SingleCell.fit_or_load_mapping with expected_layers / read_layers False by default (38 sections x 31 types
        of dense per-cell-type layers would be ~100 GB) and batch_size (params['c2l_batch_size']; None = full data
        per step, as SingleCell). A cached map is loaded by SingleCell.fit_or_load_mapping."""
        batch_size = self.get_param("c2l_batch_size", batch_size)
        filename = Path(run_dir) / fname
        if batch_size is not None and filename.exists() and not force:      # load; posterior re-export in minibatches
            import cell2location
            accelerator = self.get_param("accelerator", accelerator)
            a = self.read_h5ad_slots(fname, root=Path(run_dir))
            mod = cell2location.models.Cell2location.load(str(run_dir), a, accelerator=accelerator)
            a = self.ensure_posterior_samples(mod, a, quantiles=list(self.params["c2l_quantiles"]),
                                              accelerator=accelerator, batch_size=batch_size)
            self.abundance_to_obs(a)
            if verbose:
                print(f"Cell2location map read from {filename}; model loaded (posterior in batches of {batch_size})")
            return mod, a, False
        if batch_size is None or (filename.exists() and not force):
            return super().fit_or_load_mapping(adata_vis, df_inf_aver, run_dir, N_cells_per_location=N_cells_per_location,
                                               detection_alpha=detection_alpha, batch_key=batch_key, max_epochs=max_epochs,
                                               accelerator=accelerator, fname=fname, expected_layers=expected_layers,
                                               read_layers=read_layers, force=force, verbose=verbose)
        import cell2location
        from libs.Basic import create_dir
        run_dir = create_dir(run_dir)
        accelerator = self.get_param("accelerator", accelerator)
        batch_key = batch_key or self.sample_col
        genes = np.intersect1d(adata_vis.var_names, df_inf_aver.index)
        a = adata_vis[:, genes].copy()
        cell2location.models.Cell2location.setup_anndata(adata=a, batch_key=batch_key)
        mod = cell2location.models.Cell2location(
            a, cell_state_df=df_inf_aver.loc[genes, :].copy(),
            N_cells_per_location=self.get_param("c2l_N_cells_per_location", N_cells_per_location),
            detection_alpha=self.get_param("c2l_detection_alpha", detection_alpha))
        mod.train(max_epochs=self.get_param("c2l_max_epochs", max_epochs), batch_size=batch_size, train_size=1,
                  accelerator=accelerator)
        a = mod.export_posterior(a, use_quantiles=True, add_to_obsm=list(self.params["c2l_quantiles"]),
                                 sample_kwargs={"batch_size": batch_size, "accelerator": accelerator})
        mod.save(str(run_dir), overwrite=True)
        self.abundance_to_obs(a)
        a.write_h5ad(filename)
        if verbose:
            print(f"Cell2location trained ({len(genes):,} genes, batch_size {batch_size}); model and map saved to {run_dir}")
        return mod, a, True

    # =====================================================================================================
    # images: 38 hires H&E (~1.8 GB) are not copied into every expression subset
    # =====================================================================================================
    def prepare_expression(self, adata_map, label_col: str = "region_label") -> ad.AnnData:
        """SingleCell.prepare_expression without the H&E arrays in uns['spatial'] (scale factors kept): the 38 images
        would be copied into every region subset of run_subregions (~1.8 GB each). Images are read from disk when a
        method needs them (hires_image)."""
        a = super().prepare_expression(adata_map, label_col=label_col)
        a.uns["spatial"] = {lib: {k: v for k, v in d.items() if k != "images"} for lib, d in a.uns.get("spatial", {}).items()}
        return a

    def hires_image(self, library_id: Optional[str] = None, adata=None, img_key: Optional[str] = None):
        """(image, scale factor) of one section; when adata has no image array (prepare_expression), the Space Ranger
        PNG is read from files['visium_raw']/<section>/spatial/."""
        a = self._check_adata(adata)
        img_key = self.get_param("img_key", img_key)
        lib = library_id or self.library_id(a)
        d = a.uns["spatial"][lib]
        sf = d["scalefactors"][f"tissue_{img_key}_scalef"]
        if img_key in d.get("images", {}):
            return d["images"][img_key], sf
        cache = self.__dict__.setdefault("_disk_image_cache", {})
        if (lib, img_key) not in cache:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            path = self.file_path("visium_raw") / lib / "spatial" / f"tissue_{img_key}_image.png"
            cache[(lib, img_key)] = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        return cache[(lib, img_key)], sf

    # =====================================================================================================
    # subclustering: ComBat needs >= 2 spots per batch (a region spans up to 38 sections)
    # =====================================================================================================
    def subcluster_adaptive(self, adata_sub, batch_key: Optional[str] = None, **kw):
        """SingleCell.subcluster_adaptive with the sections that have < params['sub_min_batch_spots'] spots in the
        subset pooled into one batch ('pooled_small_sections'; when the pool has < 2 spots it joins the largest section).
        obs['combat_batch'] holds the batch used."""
        bk = batch_key or self.sample_col
        if bk not in adata_sub.obs:
            return super().subcluster_adaptive(adata_sub, batch_key=batch_key, **kw)
        a = adata_sub.copy()
        b = a.obs[bk].astype(str)
        counts = b.value_counts()
        small = set(counts.index[counts < self.params["sub_min_batch_spots"]])
        nb = b.where(~b.isin(small), "pooled_small_sections")
        if (nb == "pooled_small_sections").sum() == 1 or (nb == "pooled_small_sections").sum() and nb.nunique() == 1:
            nb = nb.replace("pooled_small_sections", counts.index[0])
        a.obs["combat_batch"] = pd.Categorical(nb)
        return super().subcluster_adaptive(a, batch_key="combat_batch", **kw)

    # =====================================================================================================
    # expression helpers
    # =====================================================================================================
    def lognorm_genes(self, genes: Sequence[str], adata, libsize: Optional[np.ndarray] = None,
                      target_sum: Optional[float] = None) -> Tuple[List[str], np.ndarray]:
        """log1p(CP<target_sum>) of the given symbols from raw counts in X (library size = row sums of X, or libsize).
        Returns (symbols found, cells x genes array)."""
        target_sum = self.get_param("target_sum", target_sum)
        found, ids = self.resolve_gene_ids(list(genes), adata)
        X = adata[:, ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        if libsize is None:
            libsize = np.asarray(adata.X.sum(1)).ravel()
        Y = np.log1p(X / np.maximum(libsize, 1)[:, None] * target_sum)
        return found, Y.astype(np.float32)

    def _spot_size_points(self, ax, lib: str, adata, factor: float = 1.0) -> float:
        """scatter size (points^2) of a Visium spot on ax (after the limits are set)."""
        d = adata.uns["spatial"][lib]["scalefactors"]
        diam_px = d["spot_diameter_fullres"] * d[f"tissue_{self.params['img_key']}_scalef"]
        x0, x1 = ax.get_xlim()
        w_pt = ax.get_window_extent().width * 72 / ax.figure.dpi
        return (factor * diam_px * w_pt / abs(x1 - x0)) ** 2

    def _section_panel(self, ax, lib: str, adata, values: Optional[np.ndarray] = None, colors=None,
                       vmin: float = 0, vmax: Optional[float] = None, cmap=None, pad: float = 30,
                       img_alpha: float = 1.0, show_image: bool = True, spot_factor: float = 1.0, title: str = ""):
        """One section on its H&E (glass = params['image_bg_color']), spots coloured by values (continuous) or colors
        (list of colours). Returns the scatter."""
        m, xy = self.spot_xy(lib, adata)
        if show_image:
            img = self._map_image(lib, adata)
            ax.imshow(img, alpha=img_alpha)
        x0, y0 = xy.min(0) - pad
        x1, y1 = xy.max(0) + pad
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        ax.axis("off")
        s = self._spot_size_points(ax, lib, adata, spot_factor)
        if colors is not None:
            sca = ax.scatter(xy[:, 0], xy[:, 1], c=np.asarray(colors, dtype=object)[m] if len(colors) == adata.n_obs
                             else colors, s=s, linewidths=0, rasterized=True)
        else:
            v = np.asarray(values)[m]
            sca = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=s, cmap=cmap or cmap_byr, vmin=vmin, vmax=vmax, linewidths=0,
                             rasterized=True)
        if title:
            ax.set_title(title)
        return sca

    def section_plot_title(self, lib: str) -> str:
        """Title of a section in the interactive maps: section | age (age group) | heart, e.g.
        'V10A13-160_A1 | w6 (w6-w7) | XDD363'."""
        df_sec = self.read_sections_table().set_index("section_id")
        if lib not in df_sec.index:
            return str(lib)
        r = df_sec.loc[lib]
        return f"{lib} | {r.age_label} ({r.age_group}) | {r.heart}"

    def section_title(self, lib: str, df_sections: Optional[pd.DataFrame] = None, short: bool = False) -> str:
        df_sec = (df_sections if df_sections is not None else self.read_sections_table()).set_index("section_id")
        r = df_sec.loc[lib]
        if short:
            return f"{r.age_label} | {r.heart}"
        return f"{r.age_label} | {r.heart} | {lib}\n{r.chambers or '-'}"

    # =====================================================================================================
    # figures: sections
    # =====================================================================================================
    def plot_sections_gallery(self, adata, sections: Optional[Sequence[str]] = None, ncols: int = 8,
                              panel_size: float = 2.6, fname: Optional[str] = None,
                              font_color: Optional[str] = None, bg_color: Optional[str] = None,
                              fontsize_scale: Optional[float] = None, force: bool = False, verbose: bool = False):
        """H&E of every section (glass recoloured), cropped to the tissue spots; title = age | heart | section, chambers."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        df_sec = self.read_sections_table()
        sections = list(sections) if sections is not None else df_sec.section_id.tolist()
        nrows = math.ceil(len(sections) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * 1.25 * nrows), squeeze=False,
                                 gridspec_kw=dict(hspace=0.35, wspace=0.08))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, lib in zip(axes.ravel(), sections):
            m, xy = self.spot_xy(lib, adata)
            img = self._map_image(lib, adata)
            ax.imshow(img)
            x0, y0 = xy.min(0) - 40
            x1, y1 = xy.max(0) + 40
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            r = df_sec.set_index("section_id").loc[lib]
            ax.set_title(f"{r.age_label} | {r.heart}\n{lib} | {r.chambers or '-'} | {int(m.sum()):,}", fontsize=7)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return fig

    def _plot_spatial_grid(self, rows: Dict[str, np.ndarray], adata, sections: Sequence[str],
                           row_titles: Optional[Dict[str, str]] = None, q: Optional[float] = None, cmap=None,
                           panel_size: float = 2.4, img_alpha: float = 0.35, cbar_label: str = "",
                           df_sections: Optional[pd.DataFrame] = None, font_color: Optional[str] = None):
        """rows = {label: values over adata spots}; grid rows x sections; one colour scale per row (shared by the
        sections: 0 .. q-th percentile over the shown spots)."""
        q = self.get_param("clip_quantile", q)
        df_sec = df_sections if df_sections is not None else self.read_sections_table()
        sec = adata.obs[self.sample_col].astype(str).values
        shown = np.isin(sec, list(sections))
        nr, nc = len(rows), len(sections)
        fig, axes = plt.subplots(nr, nc + 1, figsize=(panel_size * nc + 0.5, panel_size * nr), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1] * nc + [0.06], wspace=0.04, hspace=0.12))
        for i, (lab, v) in enumerate(rows.items()):
            v = np.asarray(v, dtype=float)
            vmax = self.percentile_cap(v[shown], q)
            sca = None
            for j, lib in enumerate(sections):
                ax = axes[i, j]
                sca = self._section_panel(ax, lib, adata, values=v, vmin=0, vmax=vmax, cmap=cmap, img_alpha=img_alpha)
                if i == 0:
                    ax.set_title(self.section_title(lib, df_sec, short=True), fontsize=9)
            axes[i, 0].text(-0.06, 0.5, (row_titles or {}).get(lab, lab), transform=axes[i, 0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=12, color=self.get_param("font_color", font_color),
                            fontstyle="italic" if lab in HEART_GENES else "normal")
            cb = fig.colorbar(sca, cax=axes[i, nc])
            cb.ax.tick_params(labelsize=7)
            if cbar_label:
                cb.set_label(cbar_label, fontsize=7)
        return fig

    def plot_spatial_genes_by_age(self, genes: Sequence[str], adata, sections: Optional[Sequence[str]] = None,
                                  nframe: int = 5, panel_size: float = 2.4, img_alpha: float = 0.35, cmap=None,
                                  q: Optional[float] = None, fname: Optional[str] = None,
                                  font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                  fontsize_scale: Optional[float] = None, force: bool = False,
                                  verbose: bool = False) -> List[Any]:
        """Genes (rows) x sections ordered by age (default: one representative section per age label); colour =
        log1p(CP10k) of the spot, one scale per gene shared by the sections (0 .. 99.2th percentile). nframe genes per
        figure; fname may contain '{k}' (figure number)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        sections = list(sections) if sections is not None else self.representative_sections()
        sub = adata[adata.obs[self.sample_col].isin(sections)]
        found, Y = self.lognorm_genes(genes, sub)
        figs = []
        for k, i in enumerate(range(0, len(found), nframe)):
            rows = {}
            for g in found[i:i + nframe]:
                v = np.zeros(adata.n_obs, dtype=np.float32)
                v[np.flatnonzero(adata.obs_names.isin(sub.obs_names))] = Y[:, found.index(g)]
                rows[g] = v
            titles = {g: g for g in rows}
            fig = self._plot_spatial_grid(rows, adata, sections, row_titles=titles, q=q, cmap=cmap,
                                          panel_size=panel_size, img_alpha=img_alpha, cbar_label="log1p(CP10k)",
                                          font_color=font_color)
            self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
            if fname is not None:
                self._savefig(fig, fname.format(k=k + 1) if "{k}" in fname else fname, dpi=self.params["dpi_maps"],
                              force=force, verbose=verbose)
            figs.append(fig)
        return figs

    def plot_spatial_celltypes_by_age(self, celltypes: Sequence[str], adata, df_abund: pd.DataFrame,
                                      sections: Optional[Sequence[str]] = None, panel_size: float = 2.4,
                                      img_alpha: float = 0.35, cmap="magma", q: Optional[float] = None,
                                      cbar_label: str = "cells per spot (q05)", fname: Optional[str] = None,
                                      font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                      fontsize_scale: Optional[float] = None, force: bool = False,
                                      verbose: bool = False):
        """Cell-type abundances (rows) x sections ordered by age; one scale per cell type."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        sections = list(sections) if sections is not None else self.representative_sections()
        df_abund = df_abund.reindex(adata.obs_names).fillna(0)
        rows = {c: df_abund[c].values for c in celltypes if c in df_abund.columns}
        fig = self._plot_spatial_grid(rows, adata, sections, q=q, cmap=cmap, panel_size=panel_size,
                                      img_alpha=img_alpha, cbar_label=cbar_label, font_color=font_color)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return fig

    def plot_regions_by_age(self, adata, key: str = "region_label", sections: Optional[Sequence[str]] = None,
                            names: Optional[Dict[str, str]] = None, ncols: Optional[int] = None,
                            panel_size: float = 3.0, colors: Optional[Dict[str, str]] = None,
                            legend_ncol: int = 4, fname: Optional[str] = None,
                            font_color: Optional[str] = None, bg_color: Optional[str] = None,
                            bottom: float = 0.1, wspace: float = 0.02, hspace: float = 0.12, 
                            fontsize_legend:int=10, bbox_to_anchor: tuple = (0.5, -0.02),
                            fontsize_scale: Optional[float] = None, force: bool = False, verbose: bool = False):
        """
        Categorical key (regions, paper clusters) on the sections ordered by age, one shared legend.
        """

        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        sections = list(sections) if sections is not None else self.representative_sections()

        df_sec = self.read_sections_table()
        labs = adata.obs[key].astype(str).values
        cats = [c for c in (adata.obs[key].cat.categories if hasattr(adata.obs[key], "cat") else sorted(set(labs)))
                if c != "nan"]
        
        colors = colors or self._region_palette(cats)
        ncols = ncols or len(sections)
        nrows = math.ceil(len(sections) / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows + 1.6), squeeze=False)

        for ax in axes.ravel():
            ax.axis("off")
        col = np.array([colors.get(l, "#555555") if l != "nan" else "#00000000" for l in labs], dtype=object)
        for ax, lib in zip(axes.ravel(), sections):
            self._section_panel(ax, lib, adata, colors=col, img_alpha=0.5, title=self.section_title(lib, df_sec, short=True))
        present = set(labs[np.isin(adata.obs[self.sample_col].astype(str).values, sections)])
        handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=colors[c], mec="none",
                          label=f"{c} - {names[c]}" if names and c in names else c) for c in cats if c in present]
        fig.legend(handles=handles, loc="lower center", ncol=legend_ncol, frameon=False, fontsize=fontsize_legend,
                   bbox_to_anchor=bbox_to_anchor)
        
        fig.subplots_adjust(bottom=bottom, wspace=wspace, hspace=hspace)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return fig

    def plotly_regions_by_age(self, adata, key: str = "region_label", sections: Optional[Sequence[str]] = None,
                              **kwargs):
        """
        Interactive version of plot_regions_by_age: plotly_regions_on_sections on the sections ordered by age
        (default representative_sections), panel titles = age | section.
        """
        return self.plotly_regions_on_sections(adata, key=key, sections=sections, **kwargs)

    def plot_umap_celltype_dotplot(self, adata_ref, genes: Sequence[str], lineage: Optional[Dict[str, Sequence[str]]] = None,
                                   groupby: Optional[str] = None, figsize=(26, 12), point_size: float = 1.5,
                                   fname: Optional[str] = None, font_color: Optional[str] = None,
                                   bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                   fontsize_x:int=12, fontsize_y:int=12,
                                   force: bool = False, verbose: bool = False) -> pd.DataFrame:
        
        """
        Left: the paper's UMAP coloured by cell type (shades of the lineage colour, labels at the medians).
        Right: dot plot cell types (rows, grouped by lineage) x genes: size = % cells > 0, colour = mean log1p(CP10k)
        scaled to the maximum per gene. Returns the long table (cell_type, gene, frac, mean, rel).
        """
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        lineage = lineage or HEART_LINEAGE
        groupby = self.get_param("celltype_col", groupby)
        
        order = [c for c in self.lineage_order(lineage) if c in set(adata_ref.obs[groupby].astype(str))]
        df_frac, df_mean, df_rel = self.calc_df_dotplot(list(genes), groupby, adata_ref, order=order)
        
        colors = self.celltype_colors(lineage)
        fig = plt.figure(figsize=figsize)
        
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.08)
        
        ax0 = fig.add_subplot(gs[0])
        U = adata_ref.obsm["X_umap"]
        labs = adata_ref.obs[groupby].astype(str).values
        self.plot_umap_labeled(ax0, U, labs, {k: mpl.colors.to_hex(v) for k, v in colors.items()},
                               title=f"scRNA reference: {adata_ref.n_obs:,} cells, {len(order)} cell types",
                               point_size=point_size, fontsize=8, seed=self.params["seed"])
        
        ax1 = fig.add_subplot(gs[1])
        n_r, n_c = len(order), df_frac.shape[1]
        yy, xx = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
        sca = ax1.scatter(xx.ravel(), yy.ravel(), s=df_frac.values.ravel() * 160, c=df_rel.values.ravel(), cmap=cmap_lbwr,
                          vmin=0, vmax=1, edgecolors="#444444", linewidths=0.3)
        ax1.set_xticks(range(n_c))
        ax1.set_xticklabels(df_frac.columns, rotation=90, fontstyle="italic", fontsize=fontsize_x)
        ax1.set_yticks(range(n_r))
        ax1.set_yticklabels(order, fontsize=fontsize_y)
        for t, c in zip(ax1.get_yticklabels(), order):
            t.set_color(mpl.colors.to_hex(self.lineage_colors(lineage)[self.lineage_of(lineage)[c]]))
        ax1.set_ylim(n_r - 0.5, -0.5)
        ax1.set_xlim(-0.6, n_c - 0.4)
        ax1.yaxis.tick_right()
        b = 0
        for lin, types in lineage.items():
            k = len([t for t in types if t in order])
            if k:
                ax1.axhline(b - 0.5, color="#777777", lw=0.5)
                b += k
        cb = fig.colorbar(sca, ax=ax1, orientation="horizontal", fraction=0.025, pad=0.12, shrink=0.5, anchor=(0.0, 1.0))
        cb.set_label("mean expression / max over cell types")
        for f in (0.25, 0.5, 1.0):
            ax1.scatter([], [], s=f * 160, c="#bbbbbb", label=f"{int(f * 100)}%")
        ax1.legend(title="% cells > 0", loc="upper left", bbox_to_anchor=(0.62, -0.1), ncol=3, frameon=False)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, force=force, verbose=verbose)
        df_long = pd.concat({"frac": df_frac.stack(), "mean": df_mean.stack(), "rel": df_rel.stack()}, axis=1)
        df_long.index.names = ["cell_type", "gene"]
        return df_long.reset_index()

    def plot_umap_genes_by_age(self, genes: Sequence[str], adata_ref, age_col: Optional[str] = None,
                               panel_size: float = 2.4, point_size: float = 0.6, q: Optional[float] = None, cmap=None,
                               fname: Optional[str] = None, font_color: Optional[str] = None,
                               bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                               force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Genes (rows) x scRNA age groups (columns) on the paper's UMAP: grey = all cells (subsample), colour = cells of
        the age group, log1p(CP10k) (library size obs['n_counts']), one scale per gene shared across the age groups.
        Returns n cells, % expressing and mean per gene x age group."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        q = self.get_param("clip_quantile", q)
        age_col = self.get_param("ref_age_col", age_col)
        ages = [a for a in self.params["ref_age_order"] if a in set(adata_ref.obs[age_col].astype(str))]
        lib = adata_ref.obs[self.params["libsize_col"]].values.astype(float)
        found, Y = self.lognorm_genes(genes, adata_ref, libsize=lib)
        U = adata_ref.obsm["X_umap"]
        rng = np.random.default_rng(self.params["seed"])
        bg = rng.choice(adata_ref.n_obs, min(self.params["panel_n_bg"], adata_ref.n_obs), replace=False)
        age = adata_ref.obs[age_col].astype(str).values
        fig, axes = plt.subplots(len(found), len(ages) + 1, squeeze=False,
                                 figsize=(panel_size * len(ages) + 0.5, panel_size * len(found)),
                                 gridspec_kw=dict(width_ratios=[1] * len(ages) + [0.06], wspace=0.04, hspace=0.08))
        rows = []
        for i, g in enumerate(found):
            v = Y[:, i]
            vmax = self.percentile_cap(v, q)
            for j, a in enumerate(ages):
                ax = axes[i, j]
                ax.scatter(U[bg, 0], U[bg, 1], s=point_size * 0.5, c="#3a3a3a", linewidths=0, rasterized=True)
                m = np.flatnonzero(age == a)
                o = m[np.argsort(v[m])]
                sca = ax.scatter(U[o, 0], U[o, 1], s=point_size, c=np.clip(v[o], 0, vmax), cmap=cmap or cmap_byr,
                                 vmin=0, vmax=vmax, linewidths=0, rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if i == 0:
                    ax.set_title(f"PCW {a}\n({len(m):,} cells)")
                rows.append(dict(gene=g, age_group=a, n_cells=len(m), pct_expr=round(100 * float((v[m] > 0).mean()), 2),
                                 mean_log1p_cp10k=round(float(v[m].mean()), 4)))
            axes[i, 0].set_ylabel(g, fontstyle="italic")
            fig.colorbar(sca, cax=axes[i, -1]).ax.tick_params(labelsize=7)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return pd.DataFrame(rows)

    def annot_color(self, im, val, dark_lum: float = 0.30) -> str:
        """Text colour for a heatmap cell: black, white when the cell colour (im.cmap(im.norm(val))) is dark
        (luminance < dark_lum). NaN cells (no colour) -> black."""
        if val is None or not np.isfinite(val):
            return "black"
        return "white" if self._luminance(im.cmap(im.norm(val))) < dark_lum else "black"

    def plot_ref_age_composition(self, df_pct: pd.DataFrame, normalize: str = "col", figsize=(7, 12),
                                 fname: Optional[str] = None, font_color: Optional[str] = None,
                                 bg_color: Optional[str] = None, cmap="Reds", fontsize_scale: Optional[float] = None,
                                 force: bool = False, verbose: bool = False):
        """
        Heatmap of calc_df_ref_age_composition; rows coloured by lineage. normalize: the one used for df_pct
        ('none': cell counts; 'col': columns sum to 100; 'row': rows sum to 100) - sets the colour-bar label, the title
        and the number format of the cells.
        """

        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)

        df_pct2 = df_pct.drop(columns=["lineage"])

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(df_pct2.values, cmap=cmap, aspect="auto")

        ax.set_xticks(range(df_pct2.shape[1]))
        ax.set_xticklabels([f"PCW {c}" for c in df_pct2.columns])
        ax.set_yticks(range(df_pct2.shape[0]))
        ax.set_yticklabels(df_pct2.index)

        
        lc = self.lineage_colors()
        for t, lin in zip(ax.get_yticklabels(), df_pct["lineage"]):
            t.set_color(mpl.colors.to_hex(lc.get(lin, "#888888")))

        for i in range(df_pct2.shape[0]):
            for j in range(df_pct2.shape[1]):
                val = df_pct2.values[i, j]
                ax.text(j, i, f"{val:,.0f}" if normalize == "none" else f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color=self.annot_color(im, val))
        
        label, title = {"none": ("number of cells", "scRNA reference: cells per cell type and age"),
                        "col": ("% of cells in the age group", "scRNA reference: cell-type composition by age"),
                        "row": ("% of the cell type over ages (age-size corrected)",
                                "scRNA reference: age profile of each cell type")}[normalize]
        
        fig.colorbar(im, ax=ax, fraction=0.04).set_label(label)
        ax.set_title(title)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    # =====================================================================================================
    # cell-type abundance: cell2location (q05) or the paper's stereoscope proportions
    # =====================================================================================================
    def abundance_source(self, adata=None, source: Optional[str] = None) -> str:
        """'c2l' when adata has the cell2location obsm (or, for adata=None, the map file exists), else 'stereoscope'
        (params['abundance_source'] = 'auto'); an explicit 'c2l' / 'stereoscope' is returned as is."""
        source = self.get_param("abundance_source", source)
        if source != "auto":
            return source
        if adata is not None:
            return "c2l" if self.params["c2l_obsm_key"] in adata.obsm else "stereoscope"
        return "c2l" if self.file_path("c2l_map").exists() else "stereoscope"

    def get_abundance(self, adata, source: Optional[str] = None, drop_excluded: bool = True) -> pd.DataFrame:
        """Spots x cell types. 'c2l': q05 cell abundance (cells per spot); 'stereoscope': the paper's proportions
        (obsm['stereoscope_hl'], rows sum to 1; the HL_excl_* columns dropped when drop_excluded)."""
        source = self.abundance_source(adata, source)
        if source == "c2l":
            df_abund = self.calc_df_abundance(adata)
        else:
            if "stereoscope_hl" not in adata.obsm:          # e.g. adata = the cell2location map (sp.h5ad)
                self.add_paper_slots(adata)
            df_abund = pd.DataFrame(adata.obsm["stereoscope_hl"]).copy()
            df_abund.index = adata.obs_names
        if drop_excluded:
            df_abund = df_abund.loc[:, ~df_abund.columns.str.match(self.params["ref_exclude_regex"])]
        order = [c for c in self.lineage_order(HEART_LINEAGE) if c in df_abund.columns]
        df_abund = df_abund[order + [c for c in df_abund.columns if c not in order]]
        df_abund.attrs["source"] = source
        return df_abund

    def add_paper_slots(self, adata, keys: Sequence[str] = ("stereoscope_hl", "stereoscope_dl"),
                        obs_cols: Sequence[str] = ("paper_cluster", "paper_cluster_name", "in_paper_qc")) -> ad.AnnData:
        """Copy the paper's stereoscope proportions (obsm) and spatial clusters / QC flag (obs) from the Visium h5ad
        (files['visium']) into adata when missing, aligned on obs_names. Needed when adata is the cell2location map
        (sp.h5ad: src/programs/run_c2l_heart_dvlp.py drops the stereoscope obsm before training). Only these slots are
        read from disk (h5py), not X."""
        import h5py
        from anndata.io import read_elem
        need_obsm = [k for k in keys if k not in adata.obsm]
        need_obs = [c for c in obs_cols if c not in adata.obs]
        if not need_obsm and not need_obs:
            return adata
        with h5py.File(self.file_path("visium"), "r") as f:
            obs_names = pd.Index(read_elem(f["obs"]).index.astype(str))
            pos = obs_names.get_indexer(adata.obs_names.astype(str))
            if (pos < 0).any():
                raise KeyError(f"{int((pos < 0).sum())} spots of adata are not in {self.file_path('visium')}")
            for k in need_obsm:
                if k in f["obsm"]:
                    df = read_elem(f["obsm"][k])
                    df = pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df
                    adata.obsm[k] = df.iloc[pos].set_axis(adata.obs_names)
            if need_obs:
                df_obs = read_elem(f["obs"])
                for c in need_obs:
                    if c in df_obs:
                        adata.obs[c] = df_obs[c].iloc[pos].values
        return adata

    def calc_df_c2l_vs_stereoscope(self, adata, fname: Optional[str] = None, force: bool = False,
                                   verbose: bool = False) -> pd.DataFrame:
        """Per cell type: Spearman rho across spots between the cell2location share (q05 / spot total) and the
        paper's stereoscope proportion (spots in the paper's QC), and the mean of both."""
        from scipy.stats import spearmanr

        def _compute():
            df_c = self.get_abundance(adata, "c2l")
            df_s = self.get_abundance(adata, "stereoscope")
            m = adata.obs["in_paper_qc"].values.astype(bool)
            df_c = df_c[m].div(df_c[m].sum(1).replace(0, np.nan), axis=0).fillna(0)
            df_s = df_s[m]
            rows = []
            for c in df_c.columns:
                if c in df_s.columns:
                    rho = spearmanr(df_c[c].values, df_s[c].values).statistic
                    rows.append(dict(cell_type=c, lineage=self.lineage_of(HEART_LINEAGE).get(c, ""), spearman=round(float(rho), 4),
                                     mean_share_c2l=round(float(df_c[c].mean()), 5),
                                     mean_prop_stereoscope=round(float(df_s[c].mean()), 5)))
            return pd.DataFrame(rows)

        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    # =====================================================================================================
    # regions (Leiden on the abundances: SingleCell.cluster_regions) and their comparison with the paper
    # =====================================================================================================
    def add_region_ids(self, adata, key: Optional[str] = None, df_abund: Optional[pd.DataFrame] = None,
                       top_n: Optional[int] = None) -> Dict[str, str]:
        """obs['region_label'] = 'R00' ... (= region_cluster), obs['region_name'] = top cell types (region_names).
        Returns {region_label: name}."""
        key = self.get_param("region_key", key)
        fmt = self.params["region_id_fmt"]
        names_cl = self.region_names(adata, key=key, df_abund=df_abund, top_n=top_n)
        cl = adata.obs[key].astype(str)
        labels = {c: fmt.format(int(c)) for c in sorted(cl.unique(), key=int)}
        adata.obs["region_label"] = pd.Categorical(cl.map(labels), categories=list(labels.values()))
        names = {labels[c]: names_cl[c] for c in labels}
        adata.obs["region_name"] = adata.obs["region_label"].map(names).astype("category")
        return names

    def plot_lr_auroc_by_age_group(self, df_lr_age: pd.DataFrame, df_age_top: pd.DataFrame, vmin: float = 0.35,
                                   vmax: float = 0.65, fname: Optional[str] = "heart04_lr_by_age_group_top.png",
                                   fontsize_x:int=10, fontsize_y:int=10, figsize: Optional[tuple] = None,
                                   force: bool = False, verbose: bool = False):
        """
        Heatmap: AUROC (stage vs the other stages) of the top-ranked LR pairs of every developmental stage
        (df_age_top) in all stages (df_lr_age: interaction, group, auroc); 0.5 = as in the other stages. Rows ordered
        by the stage of maximal AUROC.
        """

        order_age = [g for g in self.params["age_group_order"] if g in set(df_lr_age.group)]
        lr_union = list(dict.fromkeys(df_age_top.sort_values(["group", "auroc"], ascending=[True, False]).interaction))
        M = df_lr_age[df_lr_age.interaction.isin(lr_union)].pivot_table(index="interaction", columns="group",
                                                                         values="auroc")[order_age]
        M = M.loc[M.idxmax(1).map({g: i for i, g in enumerate(order_age)}).sort_values(kind="stable").index]
        
        fig, ax = plt.subplots(figsize=figsize or (6, 0.32 * len(M) + 1.5))
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        
        ax.set_xticks(range(len(order_age)))
        ax.set_xticklabels(order_age, fontsize=fontsize_x)
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(M.index, fontsize=fontsize_y)
        fig.colorbar(im, ax=ax, fraction=0.05).set_label("AUROC (stage vs other stages)")
        ax.set_title("Top-ranked LR pairs per developmental stage")

        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])

        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def read_lineage_table(self) -> Optional[pd.DataFrame]:
        """tables/heart_lineage.tsv: HL_ID (lineage), HL_SUBID (cell type), description, functional_summary; indexed by
        HL_SUBID. None when the file is missing."""
        if "lineage_table" not in self.files or not self.file_path("lineage_table").exists():
            return None
        return self.read_tsv("lineage_table").drop_duplicates("HL_SUBID").set_index("HL_SUBID")

    def calc_df_gene_best_celltype(self, df_dot: pd.DataFrame, genes: Optional[Sequence[str]] = None,
                                   lineage: Optional[Dict[str, Sequence[str]]] = None, force: bool = False,
                                   verbose: bool = False) -> pd.DataFrame:
        """Per gene: the cell type with the highest mean expression in the dot plot table (df_dot of
        plot_umap_celltype_dotplot: gene, cell_type, frac, mean), its % expressing (frac), mean, the cell type's
        lineage and the gene's role (HEART_GENES). Ordered as genes (default HEART_GENES). Index = gene.
        Cached in tsv_gene_best_celltype: force=True after changing the reference or the panel."""
        genes = list(genes or HEART_GENES)
        lin = self.lineage_of(lineage or HEART_LINEAGE)

        def compute():
            d = df_dot.loc[df_dot.groupby("gene")["mean"].idxmax(), ["gene", "cell_type", "frac", "mean"]]
            d = d.set_index("gene").loc[[g for g in genes if g in set(d.gene)]].reset_index()
            d.insert(1, "role", d.gene.map(HEART_GENES).fillna(""))
            d.insert(3, "lineage", d.cell_type.map(lin).fillna(""))
            return d.round({"frac": 3, "mean": 3})
        return self._cached_any("tsv_gene_best_celltype", compute, force=force, verbose=verbose).set_index("gene")

    def calc_df_gene_panel(self, adata_ref, adata_vis, genes: Optional[Dict[str, str]] = None, force: bool = False,
                           verbose: bool = False) -> pd.DataFrame:
        """The gene panel (default HEART_GENES: symbol -> role): gene, role, in_reference, gene_id_reference,
        in_visium, gene_id_visium (var_names; '' when absent). Cached in tsv_gene_panel: force=True after changing
        the panel. Prints how many genes are found in each dataset."""
        genes = dict(genes or HEART_GENES)

        def compute():
            ids = {}
            for name, a in (("reference", adata_ref), ("visium", adata_vis)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    found, gid = self.resolve_gene_ids(list(genes), a)
                ids[name] = dict(zip(found, gid))
            df = pd.DataFrame({"gene": list(genes), "role": list(genes.values())})
            for name in ("reference", "visium"):
                df[f"in_{name}"] = df.gene.isin(list(ids[name]))
                df[f"gene_id_{name}"] = df.gene.map(ids[name]).fillna("")
            return df
        df = self._cached_any("tsv_gene_panel", compute, force=force, verbose=verbose)
        df["in_reference"] = df["in_reference"].astype(bool)
        df["in_visium"] = df["in_visium"].astype(bool)
        print(f"{int(df.in_reference.sum())} of {len(df)} in the reference; {int(df.in_visium.sum())} in the Visium data"
              + (f"; missing: {df.loc[~(df.in_reference & df.in_visium), 'gene'].tolist()}" if not (df.in_reference & df.in_visium).all() else ""))
        return df

    def calc_df_reference_celltypes(self, adata_ref, lineage: Optional[Dict[str, Sequence[str]]] = None,
                                    key: str = "cell_type", force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cell types of the (filtered) scRNA reference: cell_type, lineage, n_cells, perc (% of the reference cells),
        plus description and functional_summary from tables/heart_lineage.tsv when present. Sorted by n_cells.
        Cached in tsv_ref_celltypes: force=True after changing the reference filters."""
        lineage = lineage or HEART_LINEAGE

        def compute():
            df = adata_ref.obs[key].astype(str).value_counts().rename_axis("cell_type").reset_index(name="n_cells")
            df.insert(1, "lineage", df.cell_type.map(self.lineage_of(lineage)).fillna(""))
            df["perc"] = (100 * df.n_cells / df.n_cells.sum()).round(2)
            df_lin = self.read_lineage_table()
            if df_lin is not None:
                for c in ("description", "functional_summary"):
                    if c in df_lin:
                        df[c] = df.cell_type.map(df_lin[c]).fillna("")
            return df
        return self._cached_any("tsv_ref_celltypes", compute, force=force, verbose=verbose)

    def calc_df_section_qc(self, adata, df_sections: pd.DataFrame, cols: Sequence[str] = ("age_label", "heart"),
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Per-section QC (calc_df_qc: n, median / q25 / q75 of genes, counts and % MT) joined with the columns cols
        of df_sections (by section_id). Cached in tsv_section_qc: force=True after changing the QC filters.
        Index = section."""
        def compute():
            return self.calc_df_qc(adata=adata).join(df_sections.set_index("section_id")[list(cols)]).reset_index()
        df = self._cached_any("tsv_section_qc", compute, force=force, verbose=verbose)
        return df.set_index(df.columns[0])

    # ---- regions: Leiden + UMAP with a file cache (69,115 spots: kNN + UMAP take minutes) ----------------
    def _regions_cache_paths(self):
        path = self.file_path("regions_cache")
        return path, path.with_name(path.name.replace(".tsv.gz", "") + ".json")

    def _regions_cache_read(self):
        path, path_js = self._regions_cache_paths()
        if not (path.exists() and path_js.exists()):
            return None, {}
        return pd.read_csv(path, sep="\t", index_col=0, dtype={"region_cluster": str}), json.load(open(path_js))

    def _regions_cache_write(self, df, meta, verbose: bool = False):
        path, path_js = self._regions_cache_paths()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep="\t")
        json.dump(meta, open(path_js, "w"), indent=1)
        if verbose:
            print(f"Region cache written: {path}")

    def cluster_regions(self, adata=None, df_abund: Optional[pd.DataFrame] = None, n_neighbors: Optional[int] = None,
                        resolution: Optional[float] = None, key_added: Optional[str] = None,
                        flavor: Optional[str] = None, cache: bool = True, force: bool = False,
                        verbose: bool = True) -> pd.Series:
        """SingleCell.cluster_regions (kNN on the abundances, Leiden) with a file cache (files['regions_cache'] +
        .json). The cache is read when the inputs are the same: abundance source, cell types, checksum of the
        abundances, barcodes, n_neighbors, resolution, flavor, seed; otherwise (e.g. stereoscope -> c2l) the clustering
        is recomputed and the cache rewritten. A cache hit does not build the kNN graph (umap_regions builds it only
        when its own cache misses). force=True: recompute; cache=False: no file."""
        import hashlib
        a = self._check_adata(adata)
        key_added = self.get_param("region_key", key_added)
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(a)
        X = df_abund.loc[a.obs_names]
        fp = dict(source=str(df_abund.attrs.get("source", "")), n_obs=int(a.n_obs),
                  barcodes_md5=hashlib.md5("\n".join(map(str, a.obs_names)).encode()).hexdigest(),
                  celltypes=[str(c) for c in X.columns], abund_sum=round(float(np.nansum(X.values)), 3),
                  n_neighbors=int(self.get_param("region_n_neighbors", n_neighbors)),
                  resolution=float(self.get_param("region_resolution", resolution)),
                  flavor=str(self.get_param("region_leiden_flavor", flavor)), seed=int(self.params["seed"]))
        self._regions_graph = dict(abund=X.values, n_neighbors=fp["n_neighbors"], fp=fp)
        if cache and not force:
            df, meta = self._regions_cache_read()
            if df is not None and meta.get("cluster") == fp and df.index.equals(pd.Index(a.obs_names.astype(str))):
                lab = df["region_cluster"].astype(str)
                a.obs[key_added] = pd.Categorical(lab.values, categories=sorted(lab.unique(), key=int))
                a.obsm["_abund"] = X.values
                a.uns.pop("abund", None)                     # a graph left from another clustering: not reused
                for k in ("abund_connectivities", "abund_distances"):
                    a.obsp.pop(k, None)
                if verbose:
                    print(f"Regions read from cache ({lab.nunique()} clusters, {fp['source']}): "
                          f"{self._regions_cache_paths()[0]} (force=True to recompute)")
                return a.obs[key_added]
        res = super().cluster_regions(a, df_abund=df_abund, n_neighbors=fp["n_neighbors"],
                                      resolution=fp["resolution"], key_added=key_added, flavor=fp["flavor"])
        if cache:
            df = pd.DataFrame({"region_cluster": a.obs[key_added].astype(str).values}, index=a.obs_names.astype(str))
            df.index.name = "barcode"
            self._regions_cache_write(df, dict(cluster=fp, umap=None), verbose=verbose)   # old UMAP: other graph
        return res

    def umap_regions(self, adata=None, min_dist: Optional[float] = None, spread: Optional[float] = None,
                     neighbors_key: str = "abund", cache: bool = True, force: bool = False,
                     verbose: bool = True) -> np.ndarray:
        """SingleCell.umap_regions (obsm['X_umap'] on the abundance kNN graph) with the cache of cluster_regions:
        read when the clustering inputs of this session (cluster_regions called first) and min_dist / spread are
        the same; otherwise the kNN graph is built if missing (cache hit of cluster_regions), the UMAP computed and
        stored in the cache. force=True: recompute; cache=False: no file."""
        a = self._check_adata(adata)
        g = getattr(self, "_regions_graph", None)
        fp = dict(min_dist=float(self.get_param("region_umap_min_dist", min_dist)),
                  spread=float(self.get_param("region_umap_spread", spread)), seed=int(self.params["seed"]))
        df, meta = self._regions_cache_read() if (cache and g is not None) else (None, {})
        ok = (df is not None and g is not None and meta.get("cluster") == g["fp"]
              and df.index.equals(pd.Index(a.obs_names.astype(str))))
        if ok and not force and meta.get("umap") == fp and {"umap_1", "umap_2"} <= set(df.columns):
            a.obsm["X_umap"] = df[["umap_1", "umap_2"]].values.astype(np.float32)
            if verbose:
                print(f"UMAP read from cache: {self._regions_cache_paths()[0]} (force=True to recompute)")
            return a.obsm["X_umap"]
        if neighbors_key not in a.uns:
            if g is None:
                raise ValueError("no kNN graph: run cluster_regions first")
            a.obsm["_abund"] = g["abund"]
            sc.pp.neighbors(a, use_rep="_abund", n_neighbors=g["n_neighbors"], key_added=neighbors_key,
                            random_state=self.params["seed"])
        res = super().umap_regions(a, min_dist=fp["min_dist"], spread=fp["spread"], neighbors_key=neighbors_key)
        if cache and ok:
            df = df[["region_cluster"]].copy()
            df["umap_1"], df["umap_2"] = res[:, 0], res[:, 1]
            meta["umap"] = fp
            self._regions_cache_write(df, meta, verbose=verbose)
        return res

    def save_region_labels(self, adata, abundance_source: str = "",
                           cols: Sequence[str] = ("region_cluster", "region_label", "region_name", "section"),
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Region labels of adata.obs (barcode + cols + abundance_source) -> region_labels.tsv, the input of the later
        notebooks (add_region_labels_from_table). File present and not force: kept, and the share of spots whose
        label agrees with it is printed (re-clustering is not bit-reproducible across machines / library versions).
        Returns the table of the current labels."""
        df = adata.obs[list(cols)].reset_index(names="barcode")
        df["abundance_source"] = abundance_source
        path = self.file_path("region_labels")
        if path.exists() and not force:
            df_old = pdreadcsv(path.name, path.parent)
            same = (df_old.set_index("barcode").region_label.reindex(df.barcode).astype(str).values
                    == df.region_label.astype(str).values)
            print(f"File exists: {path} (same labels: {same.mean():.1%}; force=True to rewrite)")
        else:
            self.write_tsv(df, "region_labels", verbose=verbose)
        return df

    def add_region_labels_from_table(self, df_regions: pd.DataFrame, adata) -> Dict[str, str]:
        """Region labels of heart01 (region_labels.tsv: barcode, region_cluster, region_label, region_name) -> obs."""
        df = df_regions.set_index("barcode") if "barcode" in df_regions else df_regions
        df = df.loc[adata.obs_names]
        cats = sorted(df["region_label"].astype(str).unique())
        adata.obs["region_cluster"] = pd.Categorical(df["region_cluster"].astype(str).values)
        adata.obs["region_label"] = pd.Categorical(df["region_label"].astype(str).values, categories=cats)
        adata.obs["region_name"] = df["region_name"].astype(str).values
        return dict(zip(df["region_label"].astype(str), df["region_name"].astype(str)))

    def read_region_info(self, source: Optional[str] = None, adata=None, verbose: bool = False) -> Optional[pd.DataFrame]:
        """Curated region table (heart_regions.tsv: region_id, region_name, compartment, abundance_source, paper_cluster,
        description). Only the rows of the abundance source of the regions are returned (the names were written for
        that partition); None when there are none (new source, e.g. after the GPU cell2location run: write new rows)."""
        source = self.abundance_source(adata, source)
        df = self.read_tsv("region_info", verbose=verbose)
        df = df[df["abundance_source"] == source].reset_index(drop=True)
        if df.empty:
            warnings.warn(f"heart_regions.tsv has no rows for abundance source {source!r}: automatic region names are used")
            return None
        return df

    def calc_region_names(self, adata, df_abund: Optional[pd.DataFrame] = None, key: str = "region_label",
                          fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """
        Region ids and names of the clustered spots (after cluster_regions):
        - names_auto = add_region_ids(adata, df_abund): obs['region_label'] R00 ... and the automatic names (top cell types);
        - df_region_info = read_region_info(adata=adata): the curated rows of heart_regions.tsv for this abundance
          source (None: no rows, e.g. the first cell2location run);
        - names = curated names (region_id -> region_name), else names_auto; written to obs['region_name'];
        - df_cur: per region (index region_label) the automatic and curated names, n_spots and abundance_source.
        df_cur is always recomputed (cheap); fname (e.g. sch.fname('tsv_region_names')): written when the file is
        missing or force=True; a file present that differs from df_cur is kept and reported (force=True to rewrite).
        Returns names_auto, df_region_info, names, df_cur.
        """

        names_auto = self.add_region_ids(adata, df_abund=df_abund)

        df_region_info = self.read_region_info(adata=adata)

        names = (dict(zip(df_region_info.region_id, df_region_info.region_name)) if df_region_info is not None
                 else names_auto)
        
        adata.obs["region_name"] = adata.obs[key].map(names).astype(str)

        print(adata.obs[key].nunique(), "regions")

        df_cur = pd.DataFrame({"automatic (top cell types)": names_auto, "curated (heart_regions.tsv)": names}).join(
            adata.obs[key].value_counts().rename("n_spots"))
        df_cur["n_spots"] = df_cur["n_spots"].fillna(0).astype(int)
        df_cur.index.name = key
        df_cur["abundance_source"] = self.abundance_source(adata)

        if fname is not None:
            path = self.file_path(fname) if fname in self.files else self.root_result / fname
            if path.exists() and not force:
                df_old = pd.read_csv(path, sep="\t", index_col=0)
                same = df_old.astype(str).reindex_like(df_cur.astype(str)).equals(df_cur.astype(str))
                print(f"File exists: {path} ({'same' if same else 'DIFFERENT'} table; force=True to rewrite)")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                df_cur.to_csv(path, sep="\t")
                if verbose:
                    print(f"Region names written: {path}")
                    
        return names_auto, df_region_info, names, df_cur

    def calc_df_region_summary(self, adata, df_abund: pd.DataFrame, names: Optional[Dict[str, str]] = None,
                               key: str = "region_label", n_top: int = 3, fname: Optional[str] = None,
                               force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Per region: n spots, n sections, ages present, median UMIs, the top cell types by enrichment
        (region mean / overall mean) and by share, the best matching paper cluster (see calc_df_region_vs_paper)."""

        def _compute():
            obs = adata.obs
            g = obs.groupby(key, observed=True)
            df_mean = df_abund.loc[obs.index].groupby(obs[key].values, observed=True).mean()
            df_share = df_mean.div(df_mean.sum(1), axis=0)
            df_enr = df_mean / df_abund.loc[obs.index].mean(0)
            df_vs = self.calc_df_region_vs_paper(adata, key=key)
            rows = []
            for r, sub in g:
                top_e = df_enr.loc[r].sort_values(ascending=False).head(n_top)
                top_s = df_share.loc[r].sort_values(ascending=False).head(n_top)
                row = dict(region_label=r, region_name=(names or {}).get(r, ""), n_spots=len(sub),
                           n_sections=sub[self.sample_col].nunique(),
                           ages=", ".join(a for a in self.age_order() if a in set(sub["age_label"].astype(str))),
                           median_UMIs=float(np.median(sub["total_counts"])) if "total_counts" in sub else np.nan,
                           top_share=", ".join(f"{c} ({v:.0%})" for c, v in top_s.items()),
                           top_enriched=", ".join(f"{c} ({v:.1f}x)" for c, v in top_e.items()))
                if r in df_vs.index:
                    row.update(paper_best=df_vs.loc[r, "paper_best"], paper_share=df_vs.loc[r, "paper_share"],
                               n_shared=int(df_vs.loc[r, "n_shared"]))
                rows.append(row)
            return pd.DataFrame(rows)

        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def calc_df_region_vs_paper(self, adata, key: str = "region_label", paper_key: str = "paper_cluster_name",
                                min_overlap: Optional[float] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        How do our regions (adata.obs[key]) match the paper's 23 spatial clusters (adata.obs[paper_key])?

        Only spots that the paper clustered are used (17 sections; the other spots have paper_key = NaN).
        Returns one row per region (index = key) with:
            n_shared     number of the region's spots that the paper also clustered
            paper_best   the paper cluster holding most of these spots, if its share >= min_overlap
                         (params['region_min_overlap']); otherwise 'mixed: A / B' (the two largest)
            paper_share  share of the region's spots in paper_best (0-1)
            EN, LV_C ... one column per paper cluster: share of the region's spots in it (each row sums to 1)
        and in df.attrs the agreement of the two partitions over all these spots:
            ARI (adjusted Rand index), NMI (normalized mutual information), n_spots.

        File: tsv_region_vs_paper (the table) + the same name with .attrs.json added (the attrs).
        File present and not force: read it, so the table comes back exactly as computed (index and attrs).
        The file belongs to the region labels of the run that wrote it: force=True after re-clustering.
        """

        # ---- the files ----
        key_file = "tsv_region_vs_paper"
        if key_file not in self.files:
            raise KeyError(f"file key not registered: {key_file}")

        filename = self.file_path(key_file)                                   # the table (tsv)
        filename_attrs = filename.with_name(filename.name + ".attrs.json")      # ARI, NMI, n_spots

        # ---- already computed: read the table and its attrs ----
        if filename.exists() and not force:
            df = pdreadcsv(filename.name, filename.parent, verbose=verbose)
            df = df.set_index(key)                      # the region ids back as the index (R00, R01, ...)
            df.index = df.index.astype(str)
            if filename_attrs.exists():
                with open(filename_attrs) as fh:
                    df.attrs = json.load(fh)
            return df

        # ---- 1. the spots both partitions have: those the paper clustered ----
        min_overlap = self.get_param("region_min_overlap", min_overlap)
        obs = adata.obs[adata.obs[paper_key].notna()]
        region = obs[key].astype(str)                   # our region of each spot
        paper = obs[paper_key].astype(str)              # the paper's cluster of each spot

        # ---- 2. counts: rows = our regions, columns = paper clusters, cell = number of spots in both ----
        df_count = pd.crosstab(region, paper)

        # ---- 3. shares: each row divided by its total, so a row tells how a region splits over the paper clusters ----
        n_shared = df_count.sum(axis=1)
        df_share = df_count.div(n_shared, axis=0)

        # ---- 4. the best matching paper cluster of each region ----
        rows = []
        for reg in df_share.index:
            shares = df_share.loc[reg].sort_values(ascending=False)    # this region's paper clusters, largest first
            first, second = shares.index[0], (shares.index[1] if len(shares) > 1 else "")
            share_first = shares.iloc[0]
            if share_first >= min_overlap:
                paper_best = first                                      # one paper cluster dominates
            else:
                paper_best = f"mixed: {first} / {second}"               # no cluster reaches min_overlap
            rows.append(dict(n_shared=int(n_shared[reg]), paper_best=paper_best, paper_share=round(float(share_first), 3)))
        
        df_best = pd.DataFrame(rows, index=df_share.index)

        # ---- 5. the table: summary columns first, then the share of every paper cluster ----
        df = pd.concat([df_best, df_share.round(4)], axis=1)
        df.index.name = key

        # ---- 6. agreement of the two partitions over all shared spots (1 = identical; ~0 = unrelated) ----
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        df.attrs["ARI"] = float(adjusted_rand_score(region, paper))
        df.attrs["NMI"] = float(normalized_mutual_info_score(region, paper))
        df.attrs["n_spots"] = int(len(obs))

        # ---- 7. save: the table (index as a column) and the attrs ----
        self.write_tsv(df.reset_index(), key_file, verbose=verbose)
        with open(filename_attrs, "w") as fh:
            json.dump(df.attrs, fh, indent=1)

        return df

    def plot_region_vs_paper(self, df_vs: pd.DataFrame, names: Optional[Dict[str, str]] = None, figsize=(16, 10),
                             fname: Optional[str] = None, font_color: Optional[str] = None, cmap=cmap_byr,
                             bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                             force: bool = False, verbose: bool = False):
        """Heatmap: regions (rows) x paper clusters (columns), share of the region's spots (row sums = 1)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)

        M = df_vs.drop(columns=["n_shared", "paper_best", "paper_share"])
        col_order = list(dict.fromkeys(list(M.idxmax(1)) + list(M.columns)))     # diagonal-like order
        M = M[col_order]
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        
        full = self.paper_cluster_full_names()
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([f"{c} ({full.get(c, '')})" for c in M.columns], rotation=70, ha="right")
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([f"{r} {names.get(r, '') if names else ''} (n={int(df_vs.loc[r, 'n_shared'])})" for r in M.index])
        
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M.values[i, j] >= 0.1:
                    ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color=self.annot_color(im, M.values[i, j]))
        
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("share of the region's spots")
        
        ax.set_title(f"Regions (this study) vs the paper's 23 spatial clusters; {df_vs.attrs.get('n_spots', 0):,} spots of 17 "
                     f"sections; ARI {df_vs.attrs.get('ARI', np.nan):.2f}, NMI {df_vs.attrs.get('NMI', np.nan):.2f}")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        
        return fig

    def calc_df_region_age(self, adata, key: str = "region_label", fname: Optional[str] = None,
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Share of each region per section, then mean over the sections of each age label (each section weighs the
        same); long table: age_label, region, share_mean, share_sd, n_sections."""

        def _compute():
            obs = adata.obs
            df_ct = pd.crosstab(obs[self.sample_col].astype(str), obs[key].astype(str), normalize="index")
            age = obs.groupby(self.sample_col, observed=True)["age_label"].first().astype(str)
            df_ct["age_label"] = age.reindex(df_ct.index).values
            rows = []
            for a, sub in df_ct.groupby("age_label"):
                v = sub.drop(columns="age_label")
                for r in v.columns:
                    rows.append(dict(age_label=a, region=r, share_mean=round(float(v[r].mean()), 5),
                                     share_sd=round(float(v[r].std(ddof=0)), 5), n_sections=len(v)))
            df = pd.DataFrame(rows)
            df["age_label"] = pd.Categorical(df["age_label"], categories=self.age_order(df["age_label"].unique()))
            return df.sort_values(["age_label", "region"]).reset_index(drop=True)

        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def plot_region_age(self, df_region_age: pd.DataFrame, names: Optional[Dict[str, str]] = None,
                        colors: Optional[Dict[str, str]] = None, figsize=(14, 6), fname: Optional[str] = None,
                        font_color: Optional[str] = None, bg_color: Optional[str] = None,
                        fontsize_scale: Optional[float] = None, force: bool = False, verbose: bool = False):
        """
        Stacked bars: region shares per age label (mean over the sections of the age).
        """

        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)

        M = df_region_age.pivot_table(index="age_label", columns="region", values="share_mean", observed=True)
        M = M.reindex(self.age_order(M.index.astype(str)))

        n_sec = df_region_age.groupby("age_label", observed=True)["n_sections"].first().reindex(M.index)
        colors = colors or self._region_palette(list(M.columns))

        fig, ax = plt.subplots(figsize=figsize)

        bottom = np.zeros(len(M))
        for r in M.columns:
            ax.bar(range(len(M)), M[r].values, bottom=bottom, color=colors.get(r, "#888888"), width=0.8,
                   label=f"{r} {names.get(r, '') if names else ''}", edgecolor="black", linewidth=0.3)
            bottom += M[r].fillna(0).values
        
        ax.set_xticks(range(len(M)))
        ax.set_xticklabels([f"{a}\n(n={int(n)})" for a, n in zip(M.index, n_sec)])
        ax.set_ylabel("share of spots (mean over sections)")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
        ax.set_title("Regions by age (n = sections per age)")

        self.style_figure(fig, font_color, bg_color, fontsize_scale)

        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def calc_df_celltype_age_trend(self, adata, df_abund: pd.DataFrame, fname: Optional[str] = None,
                                   force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Mean cell-type share per section (abundance / spot total, averaged over the section's spots), with the
        section's age; long table: section, age_label, age_pcw, heart, cell_type, lineage, share."""

        def _compute():
            df_p = df_abund.loc[adata.obs_names]
            df_p = df_p.div(df_p.sum(1).replace(0, np.nan), axis=0).fillna(0)
            df_m = df_p.groupby(adata.obs[self.sample_col].astype(str).values).mean()
            df_sec = self.read_sections_table().set_index("section_id")
            df_l = df_m.stack().rename("share").reset_index()
            df_l.columns = ["section", "cell_type", "share"]
            for c in ("age_label", "age_pcw", "heart"):
                df_l[c] = df_l["section"].map(df_sec[c])
            df_l["lineage"] = df_l["cell_type"].map(self.lineage_of(HEART_LINEAGE)).fillna("")
            df_l["share"] = df_l["share"].round(6)
            return df_l[["section", "age_label", "age_pcw", "heart", "lineage", "cell_type", "share"]]

        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def plot_celltype_age_trend(self, df_trend: pd.DataFrame, lineage: Optional[Dict[str, Sequence[str]]] = None,
                                ncols: int = 3, panel_size=(7.5, 3.8), fname: Optional[str] = None,
                                font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                fontsize_scale: Optional[float] = None, figsize:tuple=(),
                                force: bool = False, verbose: bool = False):
        """
        One panel per lineage: share of each cell type vs PCW (points = sections, line = mean per age).
        """

        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        lineage = lineage or HEART_LINEAGE

        colors = self.celltype_colors(lineage)
        lins = [l for l in lineage if l in set(df_trend.lineage)]
        nrows = math.ceil(len(lins) / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)

        for ax in axes.ravel()[len(lins):]:
            ax.axis("off")

        for ax, lin in zip(axes.ravel(), lins):
            for i, df_ct in enumerate(lineage[lin]):
                d = df_trend[df_trend.cell_type == df_ct]
                if d.empty:
                    continue
                col = colors[df_ct]
                jitter = (i - len(lineage[lin]) / 2) * 0.04
                ax.scatter(d.age_pcw + jitter, d.share, s=10, color=col, alpha=0.6, linewidths=0)
                mm = d.groupby("age_pcw")["share"].mean()
                ax.plot(mm.index, mm.values, color=col, lw=1.6, label=df_ct)
            ax.set_title(lin)
            ax.set_xlabel("PCW")
            ax.set_ylabel("mean share per spot")
            ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        
        fig.subplots_adjust(wspace=0.85, hspace=0.45, right=0.88)
        
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, force=force, verbose=verbose)
        
        return fig

    def plot_region_celltype_heatmap(self, adata, df_abund: pd.DataFrame, key: str = "region_label", cmap="RdBu_r",
                                     names: Optional[Dict[str, str]] = None, figsize=(15, 9), fname: Optional[str] = None,
                                     font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                     fontsize_scale: Optional[float] = None, force: bool = False,
                                     verbose: bool = False) -> pd.DataFrame:
        """
        Regions (rows) x cell types (columns, by lineage): log2 enrichment (region mean / overall mean).
        Returns the enrichment table.
        """

        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)

        df_a = df_abund.loc[adata.obs_names]
        df_enr = df_a.groupby(adata.obs[key].astype(str).values).mean() / df_a.mean(0)
        L = np.log2(df_enr.clip(lower=1 / 16))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(L.values, cmap=cmap, vmin=-3, vmax=3, aspect="auto")

        ax.set_xticks(range(L.shape[1]))
        ax.set_xticklabels(L.columns, rotation=80, ha="right")
        lc = self.lineage_colors()

        for t, c in zip(ax.get_xticklabels(), L.columns):
            t.set_color(mpl.colors.to_hex(lc.get(self.lineage_of(HEART_LINEAGE).get(c, ""), "#888888")))
        
        ax.set_yticks(range(L.shape[0]))
        ax.set_yticklabels([f"{r} {names.get(r, '') if names else ''}" for r in L.index])
        
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("log2 enrichment (region / all spots)")
        
        ax.set_title(f"Cell types in each region ({df_abund.attrs.get('source', '')} abundances)")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        
        self._savefig(fig, fname, force=force, verbose=verbose)
        return df_enr

    # =====================================================================================================
    # signatures by gene symbol (LIANA cell-type split) and subregion scores on the reference
    # =====================================================================================================
    def calc_df_mean_signatures(self, adata_ref, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Mean raw counts per cell of every gene in each of the 31 cell types (cell2location's 'hard-coded'
        signature; genes x cell types, index = Ensembl IDs, column 'symbol'). Cached in files['mean_signatures']."""
        def _compute():
            a = self.filter_reference(adata_ref)
            lab = a.obs[self.params["celltype_col"]].astype(str).values
            cats = sorted(set(lab))
            G = sp.csr_matrix((np.ones(len(lab)), (np.searchsorted(cats, lab), np.arange(len(lab)))))
            M = np.asarray((G @ a.X).todense()) / np.asarray(G.sum(1))
            df = pd.DataFrame(M.T, index=a.var_names, columns=cats).round(6)
            df.insert(0, "symbol", a.var[self.symbol_col].astype(str).values)
            return df.reset_index(names="gene_id")
        return self._cached_table("mean_signatures", _compute, force=force, verbose=verbose).set_index("gene_id")

    def reference_signatures_symbols(self, adata_ref=None, summary: str = "means", source: Optional[str] = None
                                     ) -> pd.DataFrame:
        """Signatures (genes x 31 cell types) indexed by gene symbol (duplicates: first kept): the cell2location reference
        model of params['c2l_tag'] (files['signatures'], varm '<summary>_per_cluster_mu_fg') when it exists (or
        source='c2l'), else the mean counts per cell type (calc_df_mean_signatures; needs adata_ref).
        attrs['source'] = 'c2l' | 'mean_counts'."""
        path = self.file_path("signatures")
        if source == "c2l" or (source is None and path.exists()):
            a = ad.read_h5ad(path, backed="r")
            df_sig, _ = self.get_reference_signatures(a, prefer=(summary,))
            df_sig.index = a.var.loc[df_sig.index, self.symbol_col].astype(str).values
            a.file.close()
            df_sig.attrs["source"] = "c2l"
        else:
            if adata_ref is None:
                raise ValueError("no cell2location reference model: pass adata_ref for the mean-count signatures")
            df = self.calc_df_mean_signatures(adata_ref)
            df_sig = df.drop(columns="symbol")
            df_sig.index = df["symbol"].astype(str).values
            df_sig.attrs["source"] = "mean_counts"
        df = df_sig[~df_sig.index.duplicated()]
        df.attrs["source"] = df_sig.attrs["source"]
        return df

    def calc_df_subregion_ref_score(self, adata_ref, df_gene_sets: pd.DataFrame, df_sub_table: Optional[pd.DataFrame] = None,
                                    status: Optional[str] = "supported", n_genes: int = 25, min_genes: int = 3,
                                    force: bool = False, verbose: bool = False
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """Score each subregion's signature on the scRNA reference (as SingleCellLymphoid): the top n_genes
        subregion_signature genes (by Wilcoxon score) found in the reference (>= min_genes; with df_sub_table only
        subregions whose status starts with `status`); cell score = mean per-gene z-score of log1p(CP10k)
        (library size obs['n_counts']). Returns (df_ref_score subregions x cell types, df_cell_score cells x
        subregions, genes_used). Cached in files 'subregion_ref_score' / 'subregion_ref_genes'."""
        groupby = self.params["celltype_col"]
        cache = ("subregion_ref_score", "subregion_ref_genes")
        if not force and all(k in self.files and self.file_path(k).exists() for k in cache):
            df_ref_score = self.read_tsv("subregion_ref_score", verbose=verbose).set_index("sub_region_id")
            df_ref_score.index.name = None
            d = self.read_tsv("subregion_ref_genes", verbose=verbose)
            genes_used = {r.sub_region_id: r.genes.split(",") for r in d.itertuples()}
            return df_ref_score, self._calc_df_cell_score(adata_ref, genes_used), genes_used
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
        df_ct = adata_ref.obs[groupby].astype(str).values
        df_ref_score = df_cell_score.groupby(df_ct).mean().T.round(6)
        lin = [c for c in self.lineage_order(HEART_LINEAGE) if c in df_ref_score.columns]
        df_ref_score = df_ref_score[lin]
        if all(k in self.files for k in cache):
            self.write_tsv(df_ref_score.reset_index(names="sub_region_id"), "subregion_ref_score", verbose=verbose)
            self.write_tsv(pd.DataFrame({"sub_region_id": list(genes_used),
                                         "genes": [",".join(v) for v in genes_used.values()]}),
                           "subregion_ref_genes", verbose=verbose)
        return df_ref_score, df_cell_score, genes_used

    def _calc_df_cell_score(self, adata_ref, genes_used: Dict[str, List[str]]) -> pd.DataFrame:
        union = sorted({x for v in genes_used.values() for x in v})
        found, E = self.lognorm_genes(union, adata_ref, libsize=adata_ref.obs[self.params["libsize_col"]].values)
        sd = E.std(0)
        Z = (E - E.mean(0)) / np.where(sd > 0, sd, 1)
        col = {g: i for i, g in enumerate(found)}
        S = np.column_stack([Z[:, [col[g] for g in gl if g in col]].mean(1) for gl in genes_used.values()])
        return pd.DataFrame(S, index=adata_ref.obs_names, columns=list(genes_used))

    @staticmethod
    def calc_df_ref_top(df_ref_score: pd.DataFrame, n: int = 3) -> pd.Series:
        """Top-n reference cell types per subregion: 'Valve_MC (1.42), EndocCush_EC (1.10), ...'."""
        return df_ref_score.apply(lambda r: ", ".join(f"{c} ({v:.2f})" for c, v in r.sort_values(ascending=False)
                                                      .head(n).items()), axis=1).rename("ref_cell_types")

    def add_ref_cell_types(self, df_sub_table: pd.DataFrame, df_ref_score: pd.DataFrame, n: int = 3,
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """df_sub_table['ref_cell_types'] = top-n reference cell types ('' = not scored); the subregion table file is
        rewritten when the column changed (or with force)."""
        df = df_sub_table.copy()
        new = df.sub_region_id.map(self.calc_df_ref_top(df_ref_score, n=n)).fillna("").astype(str)
        changed = "ref_cell_types" not in df or not new.equals(df["ref_cell_types"].fillna("").astype(str))
        df["ref_cell_types"] = new.values
        if "subregion_table" in self.files and (changed or force):
            self.write_tsv(df, "subregion_table", verbose=verbose)
        return df

    def plot_subregion_ref_heatmap(self, df_ref_score: pd.DataFrame, figsize=(16, 14), vmax: float = 1.5,
                                   fname: Optional[str] = None, font_color: Optional[str] = None,
                                   bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                   force: bool = False, verbose: bool = False):
        """Subregions (rows) x reference cell types (columns, by lineage): mean signature score."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(df_ref_score.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(df_ref_score.shape[1]))
        ax.set_xticklabels(df_ref_score.columns, rotation=80, ha="right")
        ax.set_yticks(range(df_ref_score.shape[0]))
        ax.set_yticklabels(df_ref_score.index, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("mean signature z-score of the cell type")
        ax.set_title("Subregion signatures on the scRNA reference")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def plot_group_age_heatmap(self, df_age: pd.DataFrame, group_col: str = "region", value: str = "share_mean",
                               min_share: float = 0.0, figsize=(9, 16), title: str = "", fname: Optional[str] = None,
                               cmap='Reds', font_color: Optional[str] = None, bg_color: Optional[str] = None,
                               fontsize_scale: Optional[float] = None, force: bool = False, verbose: bool = False):
        """
        Heatmap groups (rows) x age labels (columns) of calc_df_region_age (any categorical key), each row scaled
        to its maximum (the text gives the share in %).
        """

        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        
        M = df_age.pivot_table(index=group_col, columns="age_label", values=value, observed=True)
        M = M[[a for a in self.age_order() if a in M.columns]]
        M = M[M.max(1) >= min_share]
        R = M.div(M.max(1), axis=0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(R.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(M.columns)
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(M.index, fontsize=7)
        
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M.values[i, j] > 0:
                    ax.text(j, i, f"{100 * M.values[i, j]:.1f}", ha="center", va="center", fontsize=5,
                            color=self.annot_color(im, R.values[i, j]))
        
        fig.colorbar(im, ax=ax, fraction=0.04).set_label("share / row maximum (text: % of the section's spots)")
        ax.set_title(title or f"{group_col} share by age")
        
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        
        return fig

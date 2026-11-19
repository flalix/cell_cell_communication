#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single_cell_pdac_lib.py - SingleCellPDAC(SingleCell)
pancreatic ductal adenocarcinoma (PDAC) and neural invasion (NI): 21 Visium sections (17 patients; FF, FFPE and
CytAssist) + sc/snRNA reference (199,184 scRNA cells, 32,005 FFPE snRNA nuclei).

Paper: Chen MM, Gao Q, Ning H, ..., Chen R, Zhang Z. Integrated single-cell and spatial transcriptomics uncover distinct
cellular subtypes involved in neural invasion in pancreatic cancer. Cancer Cell 43(9), 1656-1676 (2025).
doi:10.1016/j.ccell.2025.06.020. PMID 40680743. Code: github.com/PDAC-zhanglab/PDAC (R, figure scripts).
Data: GEO SuperSeries GSE278694 = GSE278687 (Visium, 21 sections) + GSE278688 (scRNA) + GSE278689 (snRNA)
+ GSE300435 (scTCR). Raw reads: GSA HRA008568.

Methods (the generic Scanpy / Squidpy / cell2location / LIANA ones are in SingleCell):
    build data          : read_geo_samples, unpack_visium, build_sections_table, read_sections_table,
                          build_visium_h5ad, build_reference_h5ad
    sections            : section_table, representative_sections, section_title, add_section_obs
    (more sections are documented in the class body)

Conventions (as single_cell_heart_dvlp_lib): obs['section'] = section id ('PA01', 'PA04-1', ...) = library_id of
uns['spatial']; var_names = Ensembl IDs, symbols in var['symbol'] (the GEO sc/sn features are symbols: mapped to the
Ensembl IDs of the Space Ranger features). Every DataFrame variable starts with 'df_'.

@created on 2026/11/16
@author: Flavio Lichtenstein & Claude-Science Opus 5.5
"""

import gzip
import io
import json
import math
import re
import shutil
import tarfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import anndata as ad
import scanpy as sc

from libs.Basic import pdreadcsv, pdwritecsv
from libs.single_cell_lib import DEFAULT_PARAMS, REGION_COLORS, SingleCell, cmap_byr, cmap_lbwr
from libs.region_cache_lib import RegionCacheMixin

__version__ = "0.1.0"

# ---------------------------------------------------------------------------------------------------------------------
# the 14 major cell types of the GEO metadata (obs['major_type'], 'all_celltype'), grouped by compartment
# ---------------------------------------------------------------------------------------------------------------------
PDAC_MAJOR_LINEAGE: Dict[str, List[str]] = {
    "epithelial": ["Ductal", "Acinar", "Endocrine"],
    "stromal":    ["CAF", "Stellate", "Endothelial", "Schwann"],
    "myeloid":    ["Myeloid", "Platelet"],
    "lymphoid":   ["CD4T", "CD8T", "NK", "B", "Plasma"],
}

# major-type markers of the paper (STAR Methods, 'Unsupervised clustering and annotation')
PDAC_MAJOR_MARKERS: Dict[str, List[str]] = {
    "CD8T": ["CD3D", "CD3E", "CD8A"], "CD4T": ["CD3D", "CD3E", "CD4"], "B": ["CD79A", "MS4A1"],
    "NK": ["KLRF1", "NCAM1"], "Myeloid": ["AIF1", "CD68", "CST3"], "Platelet": ["PPBP"],
    "CAF": ["LUM", "DCN", "COL1A1"], "Stellate": ["RGS5"], "Endothelial": ["PLVAP", "VWF", "PECAM1"],
    "Schwann": ["SOX10", "PLP1"], "Acinar": ["PRSS1", "PRSS3"], "Endocrine": ["CHGB"], "Ductal": ["KRT19", "MMP7"],
    "Plasma": ["MZB1", "JCHAIN"],
}

# 26 PDAC / neural-invasion genes shown on the reference UMAP and on the Visium sections (symbol -> role)
PDAC_GENES: Dict[str, str] = {
    "KRT19":   "ductal epithelium",
    "CFTR":    "non-malignant ductal (D01)",
    "CEACAM5": "classical malignant (D08)",
    "CEACAM6": "basal-like malignant (D09)",
    "GABRP":   "neural-reactive malignant (D04)",
    "APOA2":   "neuroendocrine-like malignant (D02)",
    "PRSS1":   "acinar",
    "CHGB":    "endocrine (islets)",
    "LUM":     "fibroblasts",
    "MMP11":   "myCAF (F01)",
    "DPT":     "progenitor CAF (F02)",
    "NRP2":    "endoneurial fibroblasts (F07 Fibro-NRP2)",
    "RGS5":    "stellate / pericytes",
    "PLVAP":   "endothelium",
    "SOX10":   "Schwann cells",
    "S100B":   "Schwann cells / nerve",
    "MPZ":     "myelinating Schwann cells",
    "TGFBI":   "TGFBI+ Schwann cells (S02), invading edge",
    "NLRP3":   "NLRP3+ TAM (M07)",
    "SPP1":    "SPP1+ macrophages (invaded nerves)",
    "LYVE1":   "LYVE1+ macrophages (non-invaded nerves)",
    "MS4A1":   "B cells / TLS",
    "CXCL13":  "Tfh-like T cells / TLS",
    "CXCL12":  "CXCL12+ fibroblasts around non-invaded nerves",
    "CD3E":    "T cells",
    "MZB1":    "plasma cells",
}

# neural-invasion-related genes of malignant cells (STAR Methods, 'Signature score calculation', Figure 2M)
PDAC_NI_GENES: List[str] = ["MMP2", "MMP9", "MMP12", "TGFBR1", "TGFBR3", "NGFR", "NTRK1", "NTRK2", "CCR2", "CXCR2",
                            "IL6R", "IL6ST", "IL1B", "CXCL12", "BDNF", "NTF3", "NTF4", "CSF2", "GFRA1", "RET", "MUC1",
                            "PLXND1", "CCR7", "CXCR3", "CX3CR1", "MAPK1", "MAPK3", "NGF", "ARTN", "GFRA3", "CXCR4",
                            "L1CAM", "SMAD2", "SMAD3"]

# nerve-spot markers (Schwann cells, perineurium, axon-associated) used with the Schwann abundance to call nerves
PDAC_NERVE_GENES: List[str] = ["S100B", "PLP1", "MPZ", "SOX10", "NGFR", "CDH19", "SCN7A", "PMP22", "NRXN1", "GFRA3"]

# re-derived subclusters (mp_pdac01, src/programs/pdac_reference_subclusters.py): the paper's labels (text, figure
# legends and the authors' R code) carry their signature gene; the other genes are canonical markers of the state.
# 'method': integration of sc + sn cells of the compartment ('scvi': batch = sample, covariate = technology; 'pca':
# small compartments). Clusters are labelled by the marker set with the highest cluster score (see annotate_clusters).
PDAC_SUBCLUSTERS: Dict[str, Dict[str, Any]] = {
    "ductal": dict(major_types=["Ductal"], method="scvi", resolution=1.0, markers={
        "D01_Ductal-CFTR":    ["CFTR", "SLC4A4", "CA2", "BICC1", "KRT7"],
        "D02_Ductal-APOA2":   ["APOA2", "APOA1", "ALB", "APOC3", "FGB"],
        "D03_Ductal-TRPM5":   ["TRPM5", "POU2F3", "SH2D6", "AVIL"],
        "D04_Ductal-GABRP":   ["GABRP", "KCNN4", "SLC2A1", "LAMB3"],
        "D05_Ductal-PRSS1":   ["PRSS1", "CPA1", "CELA3A", "CTRB2"],
        "D06_Ductal-MYO1A":   ["MYO1A", "CDH17", "VIL1", "MUC13"],
        "D07_Ductal-MKI67":   ["MKI67", "TOP2A", "CENPF", "NUSAP1"],
        "D08_Ductal-CEACAM5": ["CEACAM5", "TFF1", "AGR2", "MUC5AC", "LGALS4"],
        "D09_Ductal-CEACAM6": ["CEACAM6", "KRT17", "S100A2", "LAMC2", "KRT6A", "CTSB"]}),
    "fibroblast": dict(major_types=["CAF"], method="scvi", resolution=1.0, markers={
        "F01_myCAF-MMP11":        ["MMP11", "COL11A1", "CTHRC1", "POSTN", "COL10A1"],
        "F02_Progenitor-CAF-DPT": ["DPT", "CFD", "PI16", "C7", "MFAP5"],
        "F03_tCAF-MME":           ["MME", "TNC", "SFRP2", "WNT5A"],
        "F04_hspCAF-HSPA1A":      ["HSPA1A", "HSPA1B", "DNAJB1", "HSPH1"],
        "F05_dCAF-MKI67":         ["MKI67", "TOP2A", "CENPF"],
        "F06_Fibro-MALAT1":       ["MALAT1", "NEAT1"],
        "F07_Fibro-NRP2":         ["NRP2", "SLC2A1", "CLDN1", "ITGB4", "SFRP5"],
        "F08_pro-mCAF-IGFBP5":    ["IGFBP5", "IGFBP3", "PLA2G2A"]}),
    "myeloid": dict(major_types=["Myeloid"], method="scvi", resolution=2.0, markers={
        "M02_cDC-CLEC9A":                 ["CLEC9A", "XCR1", "IDO1"],
        "M03_cDC-CD1C":                   ["CD1C", "FCER1A", "CLEC10A"],
        "M04_cDC-LAMP3":                  ["LAMP3", "CCR7", "FSCN1", "CCL19"],
        "M05_Mono-CD16":                  ["FCGR3A", "CDKN1C", "LST1", "MS4A7"],
        "M06_Mono-CD14":                  ["CD14", "S100A8", "S100A9", "VCAN", "FCN1"],
        "M07_Macro-NLRP3":                ["NLRP3", "IL1B", "EREG", "AREG", "PTGS2"],
        "M08_Macro-LYVE1":                ["LYVE1", "F13A1", "FOLR2", "SELENOP"],
        "M09_Macro-APOC1":                ["APOC1", "APOE", "C1QC", "GPNMB"],
        "M10_Macro-SPP1":                 ["SPP1", "FN1", "VEGFA", "MARCO"],
        "M11_Neutrophil-CXCL8":           ["CXCL8", "FCGR3B", "CSF3R", "G0S2"],
        "M12_Myeloid-proliferating-MKI67": ["MKI67", "TOP2A", "CENPF"],
        "M13_Myeloid-str-HSPA1A":         ["HSPA1A", "HSPA1B", "DNAJB1"],
        "M14_Mast-KIT":                   ["KIT", "TPSAB1", "CPA3", "MS4A2"]}),
    "schwann": dict(major_types=["Schwann"], method="pca", resolution=0.6, markers={
        "S01_Schwann-ABCA8":    ["ABCA8", "MPZ", "EGR2", "MBP"],
        "S02_Schwann-TGFBI":    ["TGFBI", "FN1", "COL4A1", "SPARC"],
        "S03_Schwann-SERPINA3": ["SERPINA3", "CHI3L1", "SAA1", "CXCL14"]}),
}

# the 12 spatial niches of the paper (paper code myUtiliity.R: cols_niche) with the cell types that define them
PAPER_NICHES: Dict[str, str] = {
    "N01_TLS": "#A1D57B", "N02_Endocrine": "grey", "N03_Peri-vascular": "#A2D2E2", "N04_Plasma-enriched": "#4994C4",
    "N05_CAF-enriched": "#A77EBF", "N06_TAM-enriched": "#ecb0c1", "N07_Malignant_ductal_1": "#984f31",
    "N08_Malignant_ductal_2": "#FF7A74", "N09_Malignant_ductal_3": "#b49273", "N10_Malignant_ductal_4": "#e18a3b",
    "N11_Non-malignant-pancreas": "#B1D5C8", "N12_Nerve": "#FFEE6F",
}

PDAC_PARAMS: Dict[str, Any] = dict(
    # --- Visium layout: obs['section'] = section id (library), var_names = Ensembl IDs, symbols in var['symbol']
    patient_col="patient",
    ni_col="ni_status",                # 'high' | 'low' (GEO sample titles; paper: 16 high, 5 low sections)
    ni_order=("low", "high"),
    technique_col="technique",         # 'FF' (polyA, 36,601 genes) | 'FFPE' (v1 probes) | 'CytAssist' (v2 probes)
    technique_order=("FF", "FFPE", "CytAssist"),
    library_main="PA98",               # the paper's representative section (Figures 1D-1E)
    sections_rep=("PA98", "PA11-2", "PA40", "PA06", "PA72", "PA12"),   # 4 high-NI + 2 low-NI (paper Fig. 3G/3J, 4C)
    shared_genes_only=True,            # joint analyses on the genes measured by the three Visium panels
    # --- reference (sc + sn): obs['cell_type'] = major type or re-derived subcluster
    ref_labels_key="cell_type",
    ref_batch_key="sample_id",         # 10x library (sc) / FFPE sample (sn)
    ref_covariate_keys=("technology",),   # 'sc5p' (5' v2) vs 'snFlex' (fixed RNA profiling)
    ref_tissues=("Tumor", "Adjacent_normal"),   # PBMC excluded from the deconvolution reference
    celltype_col="cell_type",
    major_col="major_type",
    libsize_col="n_counts",
    ref_symbol_col="symbol",
    ref_max_cells_per_type=3000,       # cell2location / RCTD reference: at most this many cells per type (seed)
    ref_min_cells_per_type=25,         # types with fewer cells are merged back into their major type
    # --- deconvolution: cell2location (GPU, src/programs/run_c2l_pdac.py) and RCTD (CPU, src/programs/run_rctd_pdac.R)
    c2l_tag="gpu",
    c2l_N_cells_per_location=10,       # tumour stroma: ~5-15 nuclei per 55-um spot on the H&E
    c2l_detection_alpha=20,
    c2l_max_epochs=6000,
    c2l_batch_size=10000,
    c2l_train_if_missing=False,
    ref_max_epochs=250,
    abundance_source="auto",           # 'c2l' | 'rctd' | 'auto' (c2l map when it exists, else RCTD weights)
    # --- regions on cell-abundance space (paper: 12 niches from Louvain res 2 + manual assignment)
    region_n_neighbors=30,
    region_resolution=1.0,
    region_leiden_flavor="igraph",
    region_top_n=2,
    region_id_fmt="R{:02d}",
    subregion_skip=(),
    sub_min_batch_spots=10,
    region_table_first=("region_id", "region_name", "niche", "signature_genes", "functional_description"),
    region_table_depth=True,
    subregion_table_first=("region_id", "sub_region_id", "subregion_name", "subregion_signature_genes", "n_other",
                           "other_genes_top_expressed", "not_present_genes", "functional_description"),
    gene_display_exclude=r"^RP[SL]\d|^MT-|^A[CLP]\d{6}\.\d|^LINC\d|-AS\d$",
    border_min_rel=1.25,
    subregion_map_title="{lib}: subregions of each PDAC region",
    # --- nerves (calls, invasion, rings; mp_pdac03)
    nerve_min_schwann_share=0.10,      # nerve spot: Schwann share >= this ...
    nerve_min_marker_z=1.0,            # ... and nerve-marker score (z within the section) >= this
    nerve_min_size=3,                  # connected nerve components with fewer spots are dropped
    nerve_invaded_min_malignant=0.20,  # invaded: malignant-ductal share in the nerve or its first ring >= this
    nerve_n_rings=5,                   # juxta-neural 1-2, peri-neural 3-5 (paper Fig. S5A)
    # --- plotting
    panel_n_bg=20000,
    dpi_maps=150,
    font_color="white",
    bg_color="black",
    fontsize_scale=1.4,
    image_bg_color=None,               # H&E as is: the glass mask of DEFAULT_PARAMS blackens the pale CytAssist H&E
)

PDAC_ANALYSIS_PARAMS: Dict[str, Dict[str, Any]] = {"pdac01": dict(), "pdac02": dict(coloc_n_facts=(12,))}

# key -> (root, name); root 'data' | 'result' | 'figure' | 'root0'. {tag} = params['c2l_tag'] (filled in __init__).
PDAC_FILES: Dict[str, Dict[str, Tuple[str, str]]] = {
    "inputs": dict(
        visium=("data", "pdac_visium_21sections.h5ad"),               # complete Visium AnnData (raw counts)
        reference=("data", "pdac_sc_sn_reference.h5ad"),              # complete sc + sn AnnData (raw counts)
        sections=("data", "pdac_sections.tsv"),                       # 21 sections: patient, NI, technique, ...
        visium_raw=("data", "visium"),                                # Space Ranger outputs (one folder per section)
        geo_dir=("data", "geo"),                                      # GEO downloads
        geo_samples=("data", "geo/GSE278694_samples_soft.txt"),       # GEO SOFT of the SuperSeries samples
        sc_dir=("data", "geo/GSE278688"),
        sn_dir=("data", "geo/GSE278689"),
        st_tar_dir=("data", "geo/GSE278687/raw"),
        subcluster_labels=("result", "pdac01_reference_subclusters.tsv"),   # re-derived subclusters (cell -> label)
        rctd_dir=("result", "rctd"),
        rctd_weights=("result", "rctd/pdac_rctd_weights.tsv.gz"),
        c2l_ref_dir=("result", "c2l_{tag}/reference_signatures"),
        c2l_map_dir=("result", "c2l_{tag}/cell2location_map"),
        c2l_map=("result", "c2l_{tag}/cell2location_map/sp.h5ad"),
        signatures=("result", "c2l_{tag}/reference_signatures/sc.h5ad"),
        region_labels=("result", "pdac01_region_labels.tsv"),
        regions_cache=("result", "pdac01_regions_cache.tsv.gz"),         # cluster_regions / umap_regions cache (+ .json)
        region_niches=("result", "pdac01_region_niches.tsv"),            # niche assignment of mp_pdac01
        region_info=("data", "pdac_regions.tsv"),                     # curated region table (editable)
        lineage_table=("root0", "tables/pdac_lineage.tsv"),          # HL_ID (major type), HL_SUBID, description, summary
        nerve_overrides=("data", "pdac_nerve_annotation.tsv"),       # optional manual nerve calls (H&E)
        mean_signatures=("result", "pdac_reference_mean_counts_per_celltype.tsv.gz"),
        subregion_descriptions=("data", "pdac_subregion_descriptions.tsv"),
    ),
    "pdac01": dict(
        fig_factor_dotplot=("figure", "pdac01_nmf_factors_dotplot_n{n}.png"),   # NMF co-location (RCTD or c2l)
        tsv_factor_load=("result", "pdac01_nmf_factor_loadings_n{n}.tsv"),
        tsv_region_names=("result", "pdac01_region_names.tsv"),         # calc_region_names: automatic + curated name, n spots
        fig_sections=("figure", "pdac01_sections_he_gallery.png"),
        fig_qc_visium=("figure", "pdac01_qc_visium_sections.png"),
        tsv_section_qc=("result", "pdac01_section_qc.tsv"),
        fig_qc_ref=("figure", "pdac01_qc_reference_samples.png"),
        reference_umap=("result", "pdac01_reference_umap.tsv"),
        fig_umap_dotplot=("figure", "pdac01_reference_umap_celltype_dotplot.png"),
        fig_umap_sub_dotplot=("figure", "pdac01_reference_umap_subcluster_dotplot.png"),
        tsv_ref_comp=("result", "pdac01_reference_celltype_by_tissue.tsv"),
        tsv_gene_panel=("result", "pdac01_gene_panel.tsv"),
        tsv_qc_ref=("result", "pdac01_qc_reference_samples.tsv"),
        tsv_gene_best_major=("result", "pdac01_gene_best_major_type.tsv"),
        tsv_gene_best_celltype=("result", "pdac01_gene_best_celltype.tsv"),
        fig_regions_sections_html=("figure", "pdac01_regions_on_sections.html"),
        tsv_sections_summary=("result", "pdac01_sections_by_ni_technique.tsv"),
        tsv_ref_major_types=("result", "pdac01_reference_major_types.tsv"),              # PDAC_GENES: role, presence in reference / Visium
        fig_subclusters=("figure", "pdac01_reference_subclusters_{comp}.png"),
        tsv_subcluster_clusters=("result", "pdac01_reference_subcluster_clusters.tsv"),
        fig_genes=("figure", "pdac01_visium_genes_{k}.png"),
        fig_ref_history=("figure", "pdac01_ref_training_history.png"),
        fig_ref_qc=("figure", "pdac01_ref_qc.png"),
        fig_map_history=("figure", "pdac01_map_training_history.png"),
        fig_map_qc=("figure", "pdac01_map_qc.png"),
        fig_abund=("figure", "pdac01_celltype_abundance_{source}.png"),
        tsv_c2l_vs_rctd=("result", "pdac01_c2l_vs_rctd_correlation.tsv"),
        fig_c2l_vs_rctd=("figure", "pdac01_c2l_vs_rctd.png"),
        fig_regions=("figure", "pdac01_umap_region_section.png"),
        fig_regions_3plot_html=("figure", "pdac01_umap_region_section.html"),
        fig_regions_sections=("figure", "pdac01_regions_on_sections.png"),
        fig_regions_html=("figure", "pdac01_regions_{lib}.html"),
        tsv_niches=("result", "pdac01_region_niches.tsv"),
    ),
    "pdac02": dict(
        tsv_region_summary=("result", "pdac02_region_summary.tsv"),
        fig_region_celltype=("figure", "pdac02_region_celltype_heatmap.png"),
        fig_niche_celltype=("figure", "pdac02_niche_celltype_heatmap.png"),
        tsv_niche_by_section=("result", "pdac02_niche_share_by_section.tsv"),
        tsv_niche_ni_test=("result", "pdac02_niche_high_vs_low_NI.tsv"),
        tsv_niche_ni_test_cyt=("result", "pdac02_niche_high_vs_low_NI_CytAssist.tsv"),
        fig_niche_ni=("figure", "pdac02_niche_share_by_NI.png"),
        fig_niche_ni_cyt=("figure", "pdac02_niche_share_by_NI_CytAssist.png"),
        fig_section_pca=("figure", "pdac02_section_pca_niches.png"),
        tsv_celltype_by_section=("result", "pdac02_celltype_share_by_section.tsv"),
        tsv_celltype_ni_test=("result", "pdac02_celltype_high_vs_low_NI.tsv"),
        tsv_celltype_ni_test_cyt=("result", "pdac02_celltype_high_vs_low_NI_CytAssist.tsv"),
        fig_celltype_ni=("figure", "pdac02_celltype_share_by_NI.png"),
        tsv_malignant_diversity=("result", "pdac02_malignant_composition_by_section.tsv"),
        fig_malignant_comp=("figure", "pdac02_malignant_composition_shannon.png"),
        fig_malignant_pure=("figure", "pdac02_malignant_pure_spots_{sub}.png"),
        fig_niches_sections=("figure", "pdac02_niches_on_sections.png"),
        fig_region_genes=("figure", "pdac02_region_gene_dotplot.png"),
        fig_ni_genes=("figure", "pdac02_ni_genes_by_region.png"),
        fig_cooccurrence=("figure", "pdac02_celltype_cooccurrence.png"),
        tsv_cooccurrence=("result", "pdac02_celltype_cooccurrence.tsv"),
        fig_region_map=("figure", "pdac02_region_map_{lib}.html"),
        fig_factor_dotplot=("figure", "pdac02_nmf_factors_dotplot_n{n}.png"),
        tsv_factor_load=("result", "pdac02_nmf_factor_loadings_n{n}.tsv"),
    ),
    "pdac03": dict(
        region_rank_genes=("result", "pdac03_region_rank_genes.tsv.gz"),
        region_table=("result", "pdac03_region_table.tsv"),
        subregion_grid=("result", "pdac03_subcluster_resolution_grid.tsv"),
        subregion_adata=("result", "pdac03_subregions/{region}.h5ad"),
        subregion_gene_sets=("result", "pdac03_subregion_gene_sets.tsv.gz"),
        subregion_table=("result", "pdac03_subregion_table.tsv"),
        subregion_ref_score=("result", "pdac03_subregion_ref_celltype_score.tsv"),
        subregion_ref_genes=("result", "pdac03_subregion_ref_genes.tsv"),
        tsv_subregion_ni=("result", "pdac03_subregion_share_by_section.tsv"),
        fig_region_heatmap=("figure", "pdac03_region_signature_heatmap.html"),
        fig_region_map=("figure", "pdac03_region_map_{lib}.html"),
        fig_region_sub=("figure", "pdac03_subregions_{region}.html"),
        fig_diagnostics=("figure", "pdac03_subregion_diagnostics.html"),
        fig_subregion_map=("figure", "pdac03_subregion_map_{lib}.html"),
        fig_ref_heatmap=("figure", "pdac03_subregion_ref_celltype_heatmap.png"),
        nerve_spots=("result", "pdac03_nerve_spots.tsv"),
        nerve_table=("result", "pdac03_nerve_bundles.tsv"),
        fig_nerves=("figure", "pdac03_nerves_on_sections.png"),
        fig_nerve_rings=("figure", "pdac03_nerve_rings_{lib}.png"),
        tsv_ring_niche=("result", "pdac03_niche_by_nerve_ring.tsv"),
        fig_ring_niche=("figure", "pdac03_niche_by_nerve_ring.png"),
        tsv_juxta=("result", "pdac03_juxtaneural_invaded_vs_noninvaded.tsv"),
        fig_juxta=("figure", "pdac03_juxtaneural_invaded_vs_noninvaded.png"),
        tsv_ring_malignant=("result", "pdac03_malignant_by_nerve_ring.tsv"),
        fig_ring_malignant=("figure", "pdac03_malignant_by_nerve_ring.png"),
        tsv_schwann_distance=("result", "pdac03_schwann_by_distance_to_malignant.tsv"),
        fig_schwann_distance=("figure", "pdac03_schwann_by_distance_to_malignant.png"),
    ),
    "pdac04": dict(
        subregion_grid=("result", "pdac03_subcluster_resolution_grid.tsv"),
        subregion_adata=("result", "pdac03_subregions/{region}.h5ad"),
        subregion_table_pdac03=("result", "pdac03_subregion_table.tsv"),
        nerve_spots=("result", "pdac03_nerve_spots.tsv"),
        spatial_weights=("result", "pdac04_spatial_weights.tsv"),
        liana_bivariate=("result", "pdac04_liana_bivariate_consensus.h5ad"),
        liana_global=("result", "pdac04_liana_bivariate_global_consensus.tsv"),
        liana_bivariate_cellchat=("result", "pdac04_liana_bivariate_cellchatdb.h5ad"),
        liana_global_cellchat=("result", "pdac04_liana_bivariate_global_cellchatdb.tsv"),
        lr_by_region=("result", "pdac04_lr_by_region.tsv"),
        lr_by_subregion=("result", "pdac04_lr_by_subregion.tsv"),
        lr_by_ni=("result", "pdac04_lr_by_NI.tsv"),
        lr_by_nerve=("result", "pdac04_lr_by_nerve_zone.tsv"),
        celltype_lr_tissue=("result", "pdac04_celltype_lr_tissue.tsv.gz"),
        celltype_lr_by_group=("result", "pdac04_celltype_lr_by_group.tsv.gz"),
        celltype_links_by_group=("result", "pdac04_celltype_links_by_group.tsv"),
        liana_reference=("result", "pdac04_liana_reference_rank_aggregate.tsv"),
        liana_reference_supported=("result", "pdac04_liana_reference_supported.tsv"),
        ccc_by_region=("result", "pdac04_ccc_consensus_by_region.tsv"),
        ccc_by_nerve=("result", "pdac04_ccc_consensus_by_nerve_zone.tsv"),
        tsv_tgfb=("result", "pdac04_tgfb_to_schwann.tsv"),
        fig_lr_top=("figure", "pdac04_lr_top_local_scores_{lib}.png"),
        fig_lr_regions=("figure", "pdac04_lr_region_top_local_scores_{lib}.png"),
        fig_lr_region_hm=("figure", "pdac04_lr_by_region_heatmap.html"),
        fig_lr_sub_hm=("figure", "pdac04_lr_by_subregion_heatmap.html"),
        fig_lr_nerve_hm=("figure", "pdac04_lr_by_nerve_zone_heatmap.html"),
        fig_lr_ni=("figure", "pdac04_lr_by_NI_top.png"),
        fig_lr_nerve=("figure", "pdac04_lr_nerve_local_scores_{lib}.png"),
        fig_sender_recv=("figure", "pdac04_celltype_lr_sender_receiver.png"),
        fig_tgfb=("figure", "pdac04_tgfb_to_schwann.png"),
        fig_links_region=("figure", "pdac04_celltype_links_by_region.html"),
        fig_links_nerve=("figure", "pdac04_celltype_links_by_nerve_zone.html"),
        fig_coloc=("figure", "pdac04_celltype_colocalisation.html"),
    ),
}


class SingleCellPDAC(RegionCacheMixin, SingleCell):
    """PDAC and neural invasion (Chen et al. 2025): 21 Visium sections + sc/snRNA reference.
    obs['section'] = section id (= library_id of uns['spatial']); var_names = Ensembl IDs, symbols in var['symbol'].
    """

    default_params = {**DEFAULT_PARAMS, **PDAC_PARAMS}
    analysis_files = PDAC_FILES
    analysis_params = PDAC_ANALYSIS_PARAMS

    def __init__(self, ROOT0, project: str = "pdac", symbol_col: str = "symbol", sample_col: str = "section",
                 mt_prefix: str = "MT-", **kw):
        """analysis= 'pdac01' ... 'pdac04' (its output files are registered: caching on); params= / files= override.
        The cell2location folders carry params['c2l_tag'] ('gpu' full run; 'cpu' code test)."""
        super().__init__(ROOT0, project, symbol_col=symbol_col, sample_col=sample_col, mt_prefix=mt_prefix, **kw)
        tag = self.params["c2l_tag"]
        self.files = {k: (r, n.replace("{tag}", tag)) for k, (r, n) in self.files.items()}
        self.create_file_dirs()

    def create_file_dirs(self) -> List[Path]:
        """As SingleCell.create_file_dirs, without the folders whose name still has the unfilled {tag}."""
        keep = self.files
        self.files = {k: v for k, v in keep.items() if "{tag}" not in v[1]}
        try:
            return super().create_file_dirs()
        finally:
            self.files = keep

    def _cached_any(self, fname: Optional[str], compute, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached table given as a key of self.files or as a file name with an extension in root_result. File present
        and not force -> read; else compute() -> written and read back. df.attrs (JSON-serialisable values: counts,
        settings) are kept in <file>.attrs.json and restored on a cached read."""
        if fname is None:
            return compute()
        holder = {}

        def _compute():
            df = compute()
            holder["attrs"] = dict(df.attrs)
            return df
        if fname in self.files:
            df = self._cached_table(fname, _compute, force=force, verbose=verbose)
        else:
            df = self._cached_frames({"t": fname}, lambda: {"t": _compute()}, force=force, verbose=verbose)["t"]
        path = self._resolve_table(fname)
        if path is None:
            return df
        side = path.parent / (path.name + ".attrs.json")
        if "attrs" in holder:
            df.attrs.update(holder["attrs"])
            if holder["attrs"] or side.exists():
                side.write_text(json.dumps(holder["attrs"], default=str, indent=1))
        elif side.exists():
            df.attrs.update(json.loads(side.read_text()))
        return df

    # =================================================================================================================
    # 1. building the complete h5ad files (src/programs/pdac_build_h5ad.py)
    # =================================================================================================================
    def read_geo_samples(self) -> pd.DataFrame:
        """GEO SOFT of the SuperSeries samples (files['geo_samples']) -> one row per GSM: gsm, title, series, kind
        ('ST' | 'scRNA' | 'snRNA' | 'TCR'), patient ('PA01'), section_id ('PA04-1'), preservation / technique and NI
        status of the Visium sections (title 'PA#04_1, FFPE, high'), tissue, supplementary files."""
        txt = self.file_path("geo_samples").read_text()
        rows = []
        for b in txt.split("^SAMPLE = ")[1:]:
            def g(k):
                return [x.strip() for x in re.findall(r"!Sample_" + k + r" = (.*)", b)]
            gsm = b.split("\n")[0].strip()
            title = g("title")[0]
            series = [s for s in g("series_id") if s != "GSE278694"]
            ser = series[0] if series else ""
            kind = {"GSE278687": "ST", "GSE278688": "scRNA", "GSE278689": "snRNA", "GSE300435": "TCR"}.get(ser, "")
            parts = [p.strip() for p in title.split(",")]
            m = re.match(r"PA#(\d+)(?:_(\d+))?", parts[0])
            patient = f"PA{int(m.group(1)):02d}" if m else ""
            section = (patient + (f"-{m.group(2)}" if m and m.group(2) else "")) if kind == "ST" else ""
            files = [f.split("/")[-1] for k in ("supplementary_file_1", "supplementary_file_2") for f in g(k)
                     if f and f != "NONE"]
            tissue = (g("characteristics_ch1") or [""])[0].replace("tissue: ", "")
            rows.append(dict(gsm=gsm, title=title, series=ser, kind=kind, patient=patient, section_id=section,
                             technique=parts[1] if kind == "ST" else "", ni_status=parts[2] if kind == "ST" else "",
                             tissue=tissue, files=";".join(files)))
        return pd.DataFrame(rows)

    def unpack_visium(self, force: bool = False, verbose: bool = True) -> List[str]:
        """GSE278687_RAW.tar members (files['st_tar_dir']) -> files['visium_raw']/<section>/
        filtered_feature_bc_matrix.h5 + spatial/ (the GEO spatial.tar.gz files are gzipped twice). Returns sections."""
        df_geo = self.read_geo_samples()
        df_st = df_geo[df_geo.kind == "ST"]
        src, dst = self.file_path("st_tar_dir"), self.file_path("visium_raw")
        dst.mkdir(parents=True, exist_ok=True)
        done = []
        for r in df_st.itertuples():
            tag = r.section_id.replace("-", "-")
            d = dst / r.section_id
            h5 = d / "filtered_feature_bc_matrix.h5"
            if h5.exists() and (d / "spatial" / "scalefactors_json.json").exists() and not force:
                done.append(r.section_id)
                continue
            d.mkdir(exist_ok=True)
            f_h5 = src / f"{r.gsm}_{tag}_filtered_feature_bc_matrix.h5"
            f_sp = src / f"{r.gsm}_{tag}_spatial.tar.gz"
            shutil.copyfile(f_h5, h5)
            raw = gzip.decompress(f_sp.read_bytes())
            while raw[:2] == b"\x1f\x8b":                       # gzipped twice
                raw = gzip.decompress(raw)
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                tf.extractall(d, filter="data")
            done.append(r.section_id)
            if verbose:
                print(f"{r.section_id}: {sorted(p.name for p in (d / 'spatial').iterdir())}")
        return done

    def build_sections_table(self, force: bool = False, verbose: bool = True) -> pd.DataFrame:
        """files['sections'] (pdac_sections.tsv): one row per Visium section: section_id, gsm, patient, technique
        (FF / FFPE / CytAssist), panel (n genes of the Space Ranger reference or probe set), ni_status, n_spots,
        median_umis, median_genes, in_scRNA / in_snRNA (the patient has sc / sn data), representative, comment.
        Read when it exists and not force (it is editable: comment, representative)."""
        path = self.file_path("sections")
        if path.exists() and not force:
            return self.read_sections_table(verbose=verbose)
        df_geo = self.read_geo_samples()
        df_st = df_geo[df_geo.kind == "ST"].copy()
        sc_pat = set(df_geo.loc[df_geo.kind == "scRNA", "patient"])
        sn_pat = set(df_geo.loc[df_geo.kind == "snRNA", "patient"])
        rows = []
        for r in df_st.itertuples():
            a = sc.read_10x_h5(self.file_path("visium_raw") / r.section_id / "filtered_feature_bc_matrix.h5")
            tot = np.asarray(a.X.sum(1)).ravel()
            ng = np.asarray((a.X > 0).sum(1)).ravel()
            rows.append(dict(section_id=r.section_id, gsm=r.gsm, patient=r.patient, technique=r.technique,
                             panel_genes=a.n_vars, ni_status=r.ni_status, n_spots=a.n_obs,
                             median_umis=int(np.median(tot)), median_genes=int(np.median(ng)),
                             in_scRNA=r.patient in sc_pat, in_snRNA=r.patient in sn_pat,
                             representative=r.section_id in set(self.params["sections_rep"]),
                             comment="low depth (median < 2,000 UMIs)" if np.median(tot) < 2000 else ""))
        df_sec = pd.DataFrame(rows)
        df_sec["technique"] = pd.Categorical(df_sec.technique, categories=list(self.params["technique_order"]))
        df_sec["ni_status"] = pd.Categorical(df_sec.ni_status, categories=list(self.params["ni_order"]))
        df_sec = df_sec.sort_values(["ni_status", "technique", "section_id"]).reset_index(drop=True)
        self.write_tsv(df_sec, "sections", verbose=verbose)
        return self.read_sections_table()

    def read_sections_table(self, verbose: bool = False) -> pd.DataFrame:
        """pdac_sections.tsv ordered by NI status (low, high), technique, section id."""
        df_sec = self.read_tsv("sections", str_cols=("comment",), verbose=verbose)
        df_sec["technique"] = pd.Categorical(df_sec.technique, categories=list(self.params["technique_order"]))
        df_sec["ni_status"] = pd.Categorical(df_sec.ni_status, categories=list(self.params["ni_order"]))
        return df_sec.sort_values(["ni_status", "technique", "section_id"]).reset_index(drop=True)

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
        """The sections of params['sections_rep'] in their given order (4 high-NI, 2 low-NI)."""
        df_sec = df_sections if df_sections is not None else self.read_sections_table()
        rep = set(df_sec.loc[df_sec["representative"].astype(bool), "section_id"])
        return [s for s in self.params["sections_rep"] if s in rep]

    def section_order(self) -> List[str]:
        return self.read_sections_table().section_id.tolist()

    def section_title(self, lib: str, df_sections: Optional[pd.DataFrame] = None, short: bool = False) -> str:
        df_sec = (df_sections if df_sections is not None else self.read_sections_table()).set_index("section_id")
        r = df_sec.loc[lib]
        if short:
            return f"{lib} | {r.ni_status}-NI | {r.technique}"
        return f"{lib} | {r.ni_status}-NI | {r.technique}\n{int(r.n_spots):,} spots, {int(r.median_umis):,} UMIs"

    def add_section_obs(self, adata, df_sections: Optional[pd.DataFrame] = None,
                        cols: Sequence[str] = ("patient", "ni_status", "technique", "gsm")) -> ad.AnnData:
        """Section metadata -> obs (categoricals: NI status, technique, section in table order)."""
        df_sec = (df_sections if df_sections is not None else self.read_sections_table()).set_index("section_id")
        sec = adata.obs[self.sample_col].astype(str)
        for c in cols:
            adata.obs[c] = sec.map(df_sec[c].astype(str)).values
        adata.obs[self.sample_col] = pd.Categorical(sec, categories=list(df_sec.index))
        adata.obs["ni_status"] = pd.Categorical(adata.obs["ni_status"], categories=list(self.params["ni_order"]))
        adata.obs["technique"] = pd.Categorical(adata.obs["technique"], categories=list(self.params["technique_order"]))
        adata.obs["patient"] = adata.obs["patient"].astype("category")
        return adata

    def build_visium_h5ad(self, fname: str = "visium", force: bool = False, verbose: bool = True) -> ad.AnnData:
        """Complete Visium AnnData of the 21 sections: X = raw counts (float32 CSR), union of the three panels
        (var_names = Ensembl IDs; var['symbol']; var['in_FF'], var['in_FFPE'], var['in_CytAssist'], var['n_panels'],
        var['in_all_panels']: a gene absent from a section's panel is 0 in that section, not 'not expressed'),
        obs: section, barcode, patient, ni_status, technique, gsm, in_tissue; uns['spatial'] one entry per section
        (hires + lowres H&E, scale factors). Written to files[fname] (read when it exists and not force)."""
        path = self.file_path(fname)
        if path.exists() and not force:
            return self.read_h5ad(fname, set_self=False, verbose=verbose)
        df_sec = self.read_sections_table()
        sections = df_sec.section_id.tolist()
        panels = {}
        ads = []
        for s in sections:
            a = sc.read_visium(self.file_path("visium_raw") / s, count_file="filtered_feature_bc_matrix.h5", library_id=s)
            a.var["symbol"] = a.var_names.astype(str)
            a.var_names = a.var.pop("gene_ids").astype(str).values
            a = a[:, ~a.var_names.duplicated()].copy()
            a.obs["barcode"] = a.obs_names.astype(str)
            a.obs_names = s + "_" + a.obs_names
            a.obs[self.sample_col] = s
            a.X = sp.csr_matrix(a.X, dtype=np.float32)
            tech = df_sec.set_index("section_id").loc[s, "technique"]
            panels.setdefault(tech, set()).update(a.var_names)
            ads.append(a)
            if verbose:
                print(f"{s}: {a.n_obs:,} spots x {a.n_vars:,} genes ({tech})")
        sym = pd.concat([a.var["symbol"] for a in ads])
        sym = sym[~sym.index.duplicated()]
        out = ad.concat(ads, join="outer", uns_merge="unique", fill_value=0)
        out.X = sp.csr_matrix(out.X, dtype=np.float32)
        out.var["symbol"] = sym.reindex(out.var_names).values
        for t in self.params["technique_order"]:
            out.var[f"in_{t}"] = out.var_names.isin(list(panels.get(t, ())))
        out.var["n_panels"] = out.var[[f"in_{t}" for t in self.params["technique_order"]]].sum(1).astype(int)
        out.var["in_all_panels"] = out.var["n_panels"] == len(self.params["technique_order"])
        self.add_section_obs(out, df_sec)
        out.uns["dataset"] = dict(paper="Chen et al. 2025, Cancer Cell 43:1656, doi:10.1016/j.ccell.2025.06.020",
                                  source="GEO GSE278687 (Space Ranger filtered matrices + spatial folders)",
                                  n_sections=int(len(df_sec)), n_patients=int(df_sec.patient.nunique()))
        out.write_h5ad(path, compression="gzip")
        if verbose:
            print(f"Writing {path}: {out.n_obs:,} spots x {out.n_vars:,} genes ({int(out.var.in_all_panels.sum()):,} in all "
                  f"panels), {out.obs[self.sample_col].nunique()} sections")
        return out

    @staticmethod
    def _read_mtx_csr(path: Path) -> sp.csr_matrix:
        """MatrixMarket (genes x cells, GEO layout) -> cells x genes CSR float32."""
        from scipy.io import mmread
        M = mmread(str(path))
        return sp.csr_matrix(M.T, dtype=np.float32)

    def _read_geo_mtx(self, root: Path, prefix: str, verbose: bool = True) -> ad.AnnData:
        X = self._read_mtx_csr(root / f"{prefix}_matrix.mtx.gz")
        df_f = pd.read_csv(root / f"{prefix}_features.tsv.gz", sep="\t", header=None)
        bc = pd.read_csv(root / f"{prefix}_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).values
        df_m = pd.read_csv(root / f"{prefix}_metadata.csv.gz", index_col=0)
        a = ad.AnnData(X=X, obs=pd.DataFrame(index=bc), var=pd.DataFrame(index=df_f[0].astype(str).values))
        a.obs = a.obs.join(df_m)
        if verbose:
            print(f"{prefix}: {a.n_obs:,} cells x {a.n_vars:,} genes; metadata for {int(a.obs['all_celltype'].notna().sum()):,}")
        return a

    def build_reference_h5ad(self, fname: str = "reference", force: bool = False, verbose: bool = True) -> ad.AnnData:
        """Complete sc + sn AnnData (GSE278688 + GSE278689): X = raw UMI counts of 231,189 cells (199,184 scRNA 5' v2,
        32,005 FFPE snRNA Flex) x the GEO features, renamed to the Ensembl IDs of the Visium (FF, 36,601 genes)
        Space Ranger features (symbols without an Ensembl ID are dropped; var['symbol']). obs: sample_id (10x library /
        FFPE sample), patient, tissue (Tumor / Adjacent_normal / PBMC), technology (sc5p / snFlex), major_type (the
        14 GEO types = 'all_celltype'), cell_type (= major_type until mp_pdac01 adds the re-derived subclusters),
        n_counts, n_genes, pct_counts_mt. The GEO matrices have no ribosomal genes (removed by the authors)."""
        path = self.file_path(fname)
        if path.exists() and not force:
            return self.read_h5ad(fname, set_self=False, verbose=verbose)
        a_sc = self._read_geo_mtx(self.file_path("sc_dir"), "GSE278688_sc", verbose=verbose)
        a_sc.obs["technology"] = "sc5p"
        a_sc.obs["sample_id"] = [re.sub(r"_out$", "", b.rsplit("_", 1)[0]).replace("_", "-") for b in a_sc.obs_names]
        a_sn = self._read_geo_mtx(self.file_path("sn_dir"), "GSE278689_sn", verbose=verbose)
        a_sn.obs["technology"] = "snFlex"
        a_sn.obs["sample_id"] = [b.rsplit("_", 1)[0] for b in a_sn.obs_names]
        a = ad.concat([a_sc, a_sn], join="inner")
        a.obs = a.obs.rename(columns={"patients": "patient"})
        a.obs["major_type"] = a.obs["all_celltype"].astype(str)
        a.obs["cell_type"] = a.obs["major_type"]
        for c in ("tissue", "patient", "technology", "sample_id", "major_type", "cell_type", "all_celltype"):
            a.obs[c] = a.obs[c].astype(str).astype("category")
        # symbols -> Ensembl IDs of the Space Ranger features (FF section: whole transcriptome)
        df_sec = self.read_sections_table()
        ff = df_sec.loc[df_sec.technique.astype(str) == "FF", "section_id"].iloc[0]
        import h5py
        with h5py.File(self.file_path("visium_raw") / ff / "filtered_feature_bc_matrix.h5", "r") as h:
            ids = h["matrix/features/id"][:].astype(str)
            names = h["matrix/features/name"][:].astype(str)
        df_feat = pd.DataFrame({"gene_id": ids, "symbol": names})
        # GEO symbols are Cell Ranger names made unique by Seurat ('.1' suffixes): match the same way
        df_feat["symbol_unique"] = ad.utils.make_index_unique(pd.Index(df_feat.symbol), join=".").values
        sym2id = dict(zip(df_feat.symbol_unique, df_feat.gene_id))
        keep = np.array([s in sym2id for s in a.var_names])
        n0 = a.n_vars
        a = a[:, keep].copy()
        a.var["symbol"] = a.var_names.astype(str)
        a.var_names = [sym2id[s] for s in a.var["symbol"]]
        a.var_names.name = None
        a.X = sp.csr_matrix(a.X, dtype=np.float32)
        a.obs["n_counts"] = np.asarray(a.X.sum(1)).ravel()
        a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
        mt = a.var["symbol"].str.upper().str.startswith(self.mt_prefix.upper()).values
        a.obs["pct_counts_mt"] = 100 * np.asarray(a.X[:, mt].sum(1)).ravel() / np.maximum(a.obs["n_counts"].values, 1)
        a.uns["dataset"] = dict(paper="Chen et al. 2025, Cancer Cell 43:1656, doi:10.1016/j.ccell.2025.06.020",
                                source="GEO GSE278688 (scRNA, 5' v2) + GSE278689 (FFPE snRNA, Flex)",
                                genes_dropped_no_ensembl=int(n0 - a.n_vars))
        a.write_h5ad(path, compression="gzip")
        if verbose:
            print(f"Writing {path}: {a.n_obs:,} cells x {a.n_vars:,} genes ({n0 - a.n_vars} GEO symbols without an "
                  f"Ensembl ID dropped)")
        return a

    # =================================================================================================================
    # 2. reference: tissue cells, re-derived subclusters (scVI + Leiden + marker scores)
    # =================================================================================================================
    def ref_tissue_cells(self, adata_ref, tissues: Optional[Sequence[str]] = None) -> np.ndarray:
        """Boolean mask of the reference cells of params['ref_tissues'] (PBMC excluded)."""
        tissues = list(self.get_param("ref_tissues", tissues))
        return adata_ref.obs["tissue"].astype(str).isin(tissues).values

    def subcluster_compartment(self, adata_ref, compartment: str, n_hvg: int = 2000, n_latent: int = 20,
                               max_epochs: Optional[int] = None, resolution: Optional[float] = None,
                               accelerator: str = "cpu", verbose: bool = True) -> Tuple[ad.AnnData, pd.DataFrame]:
        """Tissue cells (sc + sn) of one compartment of PDAC_SUBCLUSTERS: HVGs (seurat_v3 on counts, batch =
        technology), integration ('scvi': SCVI on counts, batch = sample_id, covariate = technology; 'pca': log-norm
        PCA with ComBat by technology), kNN (15), Leiden (resolution), UMAP, and the marker-set scores of the
        compartment (score_genes on log1p CP10k). Returns (adata_c: obs leiden + score_<label>, obsm X_umap,
        df_cluster from annotate_clusters)."""
        spec = PDAC_SUBCLUSTERS[compartment]
        resolution = spec["resolution"] if resolution is None else resolution
        seed = self.params["seed"]
        m = self.ref_tissue_cells(adata_ref) & adata_ref.obs[self.params["major_col"]].astype(str).isin(spec["major_types"]).values
        a = adata_ref[m].to_memory() if adata_ref.isbacked else adata_ref[m].copy()
        a.var_names = a.var["symbol"].astype(str).values
        a.var_names_make_unique()
        a.layers["counts"] = a.X.copy()
        sc.pp.filter_genes(a, min_cells=max(3, int(0.005 * a.n_obs)))
        sc.pp.normalize_total(a, target_sum=self.params["target_sum"])
        sc.pp.log1p(a)
        n_hvg = min(n_hvg if a.n_obs >= 1000 else 1000, a.n_vars - 1)
        tech_n = a.obs.technology.value_counts()
        try:                                     # seurat_v3 per technology fails (loess) on very small batches
            sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", layer="counts",
                                        batch_key="technology" if (tech_n >= 200).sum() > 1 else None)
        except ValueError:
            sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat")
        if spec["method"] == "scvi" and a.n_obs >= 1000:
            import scvi
            scvi.settings.seed = seed
            ah = a[:, a.var.highly_variable].copy()
            ah.obs["sample_id"] = ah.obs["sample_id"].astype(str).astype("category")
            scvi.model.SCVI.setup_anndata(ah, layer="counts", batch_key="sample_id",
                                          categorical_covariate_keys=["technology"])
            mod = scvi.model.SCVI(ah, n_latent=n_latent, n_layers=2, gene_likelihood="nb")
            mod.train(max_epochs=max_epochs or min(400, int(np.round(20000 / ah.n_obs * 400))), accelerator=accelerator,
                      early_stopping=True, enable_progress_bar=False)
            a.obsm["X_int"] = mod.get_latent_representation()
            a.uns["integration"] = f"scVI (n_latent {n_latent}; batch sample_id, covariate technology)"
        else:
            ah = a[:, a.var.highly_variable].copy()
            if ah.obs.technology.nunique() > 1:
                sc.pp.combat(ah, key="technology")
            sc.pp.scale(ah, max_value=10)
            sc.pp.pca(ah, n_comps=min(15, ah.n_obs - 1), random_state=seed)
            a.obsm["X_int"] = ah.obsm["X_pca"]
            a.uns["integration"] = "PCA (15) after ComBat by technology"
        sc.pp.neighbors(a, use_rep="X_int", n_neighbors=15, random_state=seed)
        sc.tl.leiden(a, resolution=resolution, flavor="igraph", n_iterations=2, directed=False, random_state=seed)
        sc.tl.umap(a, random_state=seed)
        for lab, genes in spec["markers"].items():
            g = [x for x in genes if x in a.var_names]
            if g:
                sc.tl.score_genes(a, g, score_name=f"score_{lab}", random_state=seed, ctrl_size=50)
        df_cluster = self.annotate_clusters(a, compartment)
        if verbose:
            print(f"{compartment}: {a.n_obs:,} cells, {a.obs.leiden.nunique()} clusters, {a.uns['integration']}")
        return a, df_cluster

    def annotate_clusters(self, a, compartment: str, min_score: float = 0.05, min_anchor_frac: float = 0.10,
                          rescue_z: float = 1.0, generic_z: float = 1.5) -> pd.DataFrame:
        """Label every Leiden cluster with a paper subcluster: per cluster the mean marker score of each label (S),
        z-scored per label over the clusters (Z). Admissible labels: S >= min_score and the anchor gene (the gene in the
        label, e.g. TGFBI) detected in >= min_anchor_frac of the cluster's cells; generic states (MALAT1, HSPA1A,
        MKI67 labels) also need Z >= generic_z. label = the admissible label with the highest S, otherwise
        '<major>-other'. Rescue: a label that wins no cluster takes the cluster where its Z is highest (>= rescue_z,
        admissible) when that cluster's label is shared with another cluster. Writes obs['subcluster']. Returns df_cluster: leiden, n_cells, label, S / Z of the label,
        second label, anchor %, technology / tissue mix."""
        spec = PDAC_SUBCLUSTERS[compartment]
        labs = [l for l in spec["markers"] if f"score_{l}" in a.obs]
        S = a.obs.groupby("leiden", observed=True)[[f"score_{l}" for l in labs]].mean()
        S.columns = labs
        Z = (S - S.mean(0)) / S.std(0).replace(0, 1)
        anchor = {l: l.split("-")[-1] for l in labs}
        frac = {}
        for l, g in anchor.items():
            if g in a.var_names:
                x = a[:, g].layers["counts"] if "counts" in a.layers else a[:, g].X     # log1p > 0 <=> counts > 0
                x = (np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()) > 0
                frac[l] = pd.Series(x, index=a.obs_names).groupby(a.obs["leiden"], observed=True).mean()
        F = pd.DataFrame(frac).reindex(S.index).fillna(0)
        other = f"{spec['major_types'][0]}-other"
        ok_lab = {cl: [l for l in labs if S.loc[cl, l] >= min_score and F.loc[cl].get(l, 0) >= min_anchor_frac]
                  for cl in S.index}
        generic = [l for l in labs if re.search(r"MALAT1|HSPA1A|MKI67", l)]   # states, not identities
        for cl in S.index:                       # a generic-state label needs a clear relative enrichment
            ok_lab[cl] = [l for l in ok_lab[cl] if l not in generic or Z.loc[cl, l] >= generic_z]
        assign = {cl: (S.loc[cl, ok].idxmax() if ok else other) for cl, ok in ok_lab.items()}
        # rescue: a paper label that won no cluster takes its best cluster (top Z for the label, Z >= rescue_z,
        # label admissible there) when that cluster's current label is shared with another cluster
        for l in labs:
            if l in assign.values():
                continue
            cand = Z[l][[cl for cl in S.index if l in ok_lab[cl]]].sort_values(ascending=False)
            for cl, zv in cand.items():
                cur = assign[cl]
                if zv >= rescue_z and (cur == other or sum(v == cur for v in assign.values()) > 1):
                    assign[cl] = l
                    break
        rows = []
        for cl in S.index:
            ok = ok_lab[cl]
            lab = assign[cl]
            z = Z.loc[cl, ok].sort_values(ascending=False) if ok else pd.Series(dtype=float)
            z = pd.concat([z[[lab]], z.drop(lab)]) if lab in z.index else z
            sub = a.obs[a.obs.leiden == cl]
            rows.append(dict(compartment=compartment, leiden=cl, n_cells=len(sub), label=lab,
                             score=round(float(S.loc[cl, lab]), 3) if lab in S.columns else np.nan,
                             z=round(float(Z.loc[cl, lab]), 2) if lab in Z.columns else np.nan,
                             second=z.index[1] if len(z) > 1 else "",
                             anchor_pct=round(100 * float(F.loc[cl].get(lab, 0)), 1) if lab != other else np.nan,
                             pct_sn=round(100 * float((sub.technology.astype(str) == "snFlex").mean()), 1),
                             pct_normal=round(100 * float((sub.tissue.astype(str) == "Adjacent_normal").mean()), 1),
                             n_patients=int(sub.patient.nunique()),
                             best_raw=S.loc[cl].idxmax()))
        a.obs["subcluster"] = a.obs["leiden"].map(assign).astype(str).astype("category")
        return pd.DataFrame(rows).sort_values(["label", "n_cells"], ascending=[True, False]).reset_index(drop=True)

    def read_subcluster_labels(self, verbose: bool = False) -> pd.DataFrame:
        """files['subcluster_labels']: cell, compartment, leiden, subcluster, umap1, umap2 (re-derived subclusters)."""
        return self.read_tsv("subcluster_labels", verbose=verbose)

    def add_subclusters(self, adata_ref, df_labels: Optional[pd.DataFrame] = None,
                        min_cells: Optional[int] = None) -> ad.AnnData:
        """obs['cell_type'] = re-derived subcluster where present (files['subcluster_labels']; '<major>-other' = cells of
        a subclustered compartment that match no paper subcluster), else the major type; labels with
        < ref_min_cells_per_type cells fall back to the major type. obs['subcluster'] keeps the raw label."""
        min_cells = self.get_param("ref_min_cells_per_type", min_cells)
        df = df_labels if df_labels is not None else self.read_subcluster_labels()
        lab = pd.Series(df.subcluster.astype(str).values, index=df.cell.astype(str).values)
        sub = lab.reindex(adata_ref.obs_names).fillna("").astype(str)
        adata_ref.obs["subcluster"] = sub.values
        major = adata_ref.obs[self.params["major_col"]].astype(str)
        n = sub[sub != ""].value_counts()
        ok = sub.isin(n.index[n >= min_cells]) & (sub != "")
        adata_ref.obs["cell_type"] = pd.Categorical(np.where(ok.values, sub.values, major.values))
        return adata_ref

    def reference_for_deconvolution(self, adata_ref, max_cells: Optional[int] = None, verbose: bool = True
                                    ) -> ad.AnnData:
        """Deconvolution reference: tissue cells (params['ref_tissues']), obs['cell_type'] with the re-derived
        subclusters (add_subclusters), at most ref_max_cells_per_type cells per type (seeded sample; rare types kept
        whole). In memory, raw counts, var_names = Ensembl IDs."""
        max_cells = self.get_param("ref_max_cells_per_type", max_cells)
        m = self.ref_tissue_cells(adata_ref)
        a = adata_ref[m].to_memory() if adata_ref.isbacked else adata_ref[m].copy()
        if self.file_path("subcluster_labels").exists():
            self.add_subclusters(a)
        lab = a.obs["cell_type"].astype(str)
        n = lab.value_counts()
        rare = sorted(n.index[n < self.params["ref_min_cells_per_type"]])
        if rare:                                   # RCTD: CELL_MIN_INSTANCE = 25; c2l: unstable signatures
            a = a[~lab.isin(rare).values].copy()
            lab = a.obs["cell_type"].astype(str)
            if verbose:
                print(f"dropped (< {self.params['ref_min_cells_per_type']} tissue cells): {rare}")
        rng = np.random.default_rng(self.params["seed"])
        keep = np.sort(np.concatenate([ix if len(ix) <= max_cells else rng.choice(ix, max_cells, replace=False)
                                       for ix in lab.groupby(lab.values).indices.values()]))
        out = a[keep].copy()
        out.obs["cell_type"] = out.obs["cell_type"].astype(str).astype("category")
        if verbose:
            print(f"reference for deconvolution: {out.n_obs:,} of {int(m.sum()):,} tissue cells, "
                  f"{out.obs.cell_type.nunique()} cell types")
        return out

    def shared_genes(self, adata_vis, adata_ref=None) -> List[str]:
        """Genes (Ensembl IDs) measured by the three Visium panels (var['in_all_panels']) and, with adata_ref, present
        in the reference."""
        g = adata_vis.var_names[adata_vis.var["in_all_panels"].values.astype(bool)]
        if adata_ref is not None:
            g = g[g.isin(adata_ref.var_names)]
        return list(g)

    def export_rctd_inputs(self, adata_vis, adata_ref_dec, genes: Optional[Sequence[str]] = None,
                           force: bool = False, verbose: bool = True) -> Path:
        """Inputs of src/programs/run_rctd_pdac.R in files['rctd_dir']/inputs/: reference (counts.mtx genes x cells,
        genes.tsv = symbols, cells.tsv: barcode, cell_type, nUMI) and one folder per section (counts.mtx, genes.tsv,
        spots.tsv: barcode, x, y, nUMI). Genes: shared_genes (the three panels and the reference). nUMI = total counts
        of all genes (not only the exported ones), as in Load10X_Spatial / spacexr."""
        from scipy.io import mmwrite
        root = self.file_path("rctd_dir") / "inputs"
        if (root / "reference" / "counts.mtx").exists() and not force:
            if verbose:
                print(f"RCTD inputs exist: {root}")
            return root
        genes = list(genes) if genes is not None else self.shared_genes(adata_vis, adata_ref_dec)
        sym = adata_ref_dec.var.loc[genes, "symbol"].astype(str).values
        d = root / "reference"
        d.mkdir(parents=True, exist_ok=True)
        R = adata_ref_dec[:, genes].X
        mmwrite(str(d / "counts.mtx"), sp.csr_matrix(R).T.astype(np.int32))
        pd.Series(sym).to_csv(d / "genes.tsv", sep="\t", index=False, header=False)
        pd.DataFrame({"barcode": adata_ref_dec.obs_names, "cell_type": adata_ref_dec.obs["cell_type"].astype(str).values,
                      "nUMI": np.asarray(adata_ref_dec.X.sum(1)).ravel().round().astype(int)}).to_csv(
            d / "cells.tsv", sep="\t", index=False)
        for s in adata_vis.obs[self.sample_col].cat.categories:
            m = (adata_vis.obs[self.sample_col] == s).values
            a = adata_vis[m]
            ds = root / s
            ds.mkdir(exist_ok=True)
            mmwrite(str(ds / "counts.mtx"), sp.csr_matrix(a[:, genes].X).T.astype(np.int32))
            pd.Series(sym).to_csv(ds / "genes.tsv", sep="\t", index=False, header=False)
            xy = a.obsm["spatial"]
            pd.DataFrame({"barcode": a.obs_names, "x": xy[:, 0], "y": xy[:, 1],
                          "nUMI": np.asarray(a.X.sum(1)).ravel().round().astype(int)}).to_csv(ds / "spots.tsv", sep="\t",
                                                                                           index=False)
        if verbose:
            print(f"RCTD inputs: {len(genes):,} genes, {adata_ref_dec.n_obs:,} reference cells, "
                  f"{adata_vis.obs[self.sample_col].nunique()} sections -> {root}")
        return root

    def read_rctd_weights(self, adata=None) -> pd.DataFrame:
        """files['rctd_weights'] (spots x cell types, rows sum to 1; spots RCTD dropped, e.g. < 100 UMIs, are NaN).
        With adata: reindexed to adata.obs_names."""
        df = pd.read_csv(self.file_path("rctd_weights"), sep="\t", index_col=0)
        return df.reindex(adata.obs_names) if adata is not None else df

    # =================================================================================================================
    # 3. cell-type abundance: cell2location (q05 cells per spot) or RCTD (proportions)
    # =================================================================================================================
    def abundance_source(self, adata=None, source: Optional[str] = None) -> str:
        """'c2l' when adata has the cell2location obsm (adata=None: the map file exists), else 'rctd'
        (params['abundance_source'] = 'auto'); an explicit 'c2l' / 'rctd' is returned as is."""
        source = self.get_param("abundance_source", source)
        if source != "auto":
            return source
        if adata is not None and self.params["c2l_obsm_key"] in adata.obsm:
            return "c2l"
        if adata is None and self.file_path("c2l_map").exists():
            return "c2l"
        return "rctd"

    def celltype_order(self, cols: Sequence[str]) -> List[str]:
        """Cell types ordered by compartment (major lineage), subclusters next to their major type."""
        order = []
        sub_by_major = {}
        for comp, spec in PDAC_SUBCLUSTERS.items():
            for mj in spec["major_types"]:
                sub_by_major.setdefault(mj, []).extend(list(spec["markers"]) + [f"{mj}-other"])
        for lin, majors in PDAC_MAJOR_LINEAGE.items():
            for mj in majors:
                order += [mj] + sub_by_major.get(mj, [])
        cols = list(cols)
        return [c for c in order if c in cols] + [c for c in cols if c not in order]

    def celltype_lineage(self, cols: Sequence[str]) -> Dict[str, List[str]]:
        """{lineage: [cell types of cols]} in celltype_order, lineages as PDAC_MAJOR_LINEAGE (epithelial, stromal,
        myeloid, lymphoid; unmapped types -> 'other'): the lineage argument of plot_factor_dotplot for the fine
        RCTD / c2l types (e.g. scp.celltype_lineage(df_abund.columns))."""
        d: Dict[str, List[str]] = {}
        for c in self.celltype_order(cols):
            d.setdefault(self.lineage_of_celltype(c) or "other", []).append(c)
        return d

    def lineage_of_celltype(self, ct: str) -> str:
        ct = ct[: -len("-other")] if ct.endswith("-other") else ct
        for lin, majors in PDAC_MAJOR_LINEAGE.items():
            if ct in majors:
                return lin
        for comp, spec in PDAC_SUBCLUSTERS.items():
            if ct in spec["markers"]:
                return self.lineage_of_celltype(spec["major_types"][0])
        return ""

    def major_of_celltype(self, ct: str) -> str:
        if ct.endswith("-other"):
            return ct[: -len("-other")]
        for comp, spec in PDAC_SUBCLUSTERS.items():
            if ct in spec["markers"]:
                return spec["major_types"][0]
        return ct

    def get_abundance(self, adata, source: Optional[str] = None, fill_na: bool = True) -> pd.DataFrame:
        """Spots x cell types. 'c2l': q05 cell abundance (cells per spot); 'rctd': RCTD full-mode weights (rows sum
        to 1; spots dropped by RCTD -> 0 when fill_na, obs['rctd_ok'] False). attrs['source']."""
        source = self.abundance_source(adata, source)
        if source == "c2l":
            df_abund = self.calc_df_abundance(adata)
        else:
            df_abund = self.read_rctd_weights(adata)
            adata.obs["rctd_ok"] = df_abund.notna().all(1).values
            if fill_na:
                df_abund = df_abund.fillna(0.0)
        df_abund = df_abund[self.celltype_order(df_abund.columns)]
        df_abund.attrs["source"] = source
        return df_abund

    def collapse_to_major(self, df_abund: pd.DataFrame) -> pd.DataFrame:
        """Sum the subclusters of each major type (e.g. D01..D09 -> Ductal)."""
        maj = [self.major_of_celltype(c) for c in df_abund.columns]
        out = df_abund.T.groupby(maj, sort=False).sum().T
        out = out[[c for c in self.celltype_order(out.columns)]]
        out.attrs["source"] = df_abund.attrs.get("source", "")
        return out

    def calc_df_c2l_vs_rctd(self, adata, fname: Optional[str] = None, force: bool = False,
                            verbose: bool = False) -> pd.DataFrame:
        """Per cell type: Spearman rho across spots between the cell2location share (q05 / spot total) and the RCTD
        weight (spots kept by RCTD), and the mean of both."""
        from scipy.stats import spearmanr

        def _compute():
            df_c = self.get_abundance(adata, "c2l")
            df_r = self.get_abundance(adata, "rctd", fill_na=False)
            ok = df_r.notna().all(1).values
            df_c = df_c[ok].div(df_c[ok].sum(1).replace(0, np.nan), axis=0).fillna(0)
            df_r = df_r[ok]
            rows = []
            for c in df_c.columns:
                if c in df_r.columns:
                    rows.append(dict(cell_type=c, lineage=self.lineage_of_celltype(c),
                                     spearman=round(float(spearmanr(df_c[c].values, df_r[c].values).statistic), 4),
                                     mean_share_c2l=round(float(df_c[c].mean()), 5),
                                     mean_weight_rctd=round(float(df_r[c].mean()), 5)))
            return pd.DataFrame(rows)

        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    # =================================================================================================================
    # 4. cell2location mapping with spot minibatches; images from disk; subclustering batches
    #    (ported from single_cell_heart_dvlp_lib, independent of it)
    # =================================================================================================================
    def fit_or_load_mapping(self, adata_vis, df_inf_aver, run_dir, N_cells_per_location=None, detection_alpha=None,
                            batch_key=None, max_epochs=None, accelerator=None, fname: str = "sp.h5ad",
                            expected_layers: bool = False, read_layers: bool = False, batch_size=None,
                            force: bool = False, verbose: bool = True):
        """SingleCell.fit_or_load_mapping with expected_layers / read_layers False by default and spot minibatches
        (params['c2l_batch_size']; None = full data per step). A cached map is loaded."""
        batch_size = self.get_param("c2l_batch_size", batch_size)
        filename = Path(run_dir) / fname
        if batch_size is not None and filename.exists() and not force:
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
            print(f"Cell2location trained ({len(genes):,} genes, batch_size {batch_size}); saved to {run_dir}")
        return mod, a, True

    def prepare_expression(self, adata_map, label_col: str = "region_label") -> ad.AnnData:
        """SingleCell.prepare_expression without the H&E arrays in uns['spatial'] (scale factors kept); images are
        read from disk when needed (hires_image). With params['shared_genes_only'] only the genes of the three panels
        are kept (var['in_all_panels'], when present)."""
        a = adata_map
        if self.params["shared_genes_only"] and "in_all_panels" in a.var:
            a = a[:, a.var["in_all_panels"].values.astype(bool)]
        a = super().prepare_expression(a, label_col=label_col)
        a.uns["spatial"] = {lib: {k: v for k, v in d.items() if k != "images"} for lib, d in a.uns.get("spatial", {}).items()}
        return a

    def hires_image(self, library_id: Optional[str] = None, adata=None, img_key: Optional[str] = None):
        """(image, scale factor) of one section; when adata has no image array, the Space Ranger PNG is read from
        files['visium_raw']/<section>/spatial/."""
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

    def subcluster_adaptive(self, adata_sub, batch_key: Optional[str] = None, **kw):
        """SingleCell.subcluster_adaptive with the sections that have < params['sub_min_batch_spots'] spots in the
        subset pooled into one batch (obs['combat_batch'])."""
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

    def lognorm_genes(self, genes: Sequence[str], adata, libsize: Optional[np.ndarray] = None,
                      target_sum: Optional[float] = None) -> Tuple[List[str], np.ndarray]:
        """log1p(CP<target_sum>) of the given symbols from raw counts in X (library size = row sums, or libsize)."""
        target_sum = self.get_param("target_sum", target_sum)
        found, ids = self.resolve_gene_ids(list(genes), adata)
        X = adata[:, ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        if libsize is None:
            libsize = np.asarray(adata.X.sum(1)).ravel()
        Y = np.log1p(X / np.maximum(libsize, 1)[:, None] * target_sum)
        return found, Y.astype(np.float32)

    # =================================================================================================================
    # 5. figures: sections (H&E panels), genes / cell types / regions on the sections
    # =================================================================================================================
    def _spot_size_points(self, ax, lib: str, adata, factor: float = 1.0) -> float:
        d = adata.uns["spatial"][lib]["scalefactors"]
        diam_px = d["spot_diameter_fullres"] * d[f"tissue_{self.params['img_key']}_scalef"]
        x0, x1 = ax.get_xlim()
        w_pt = ax.get_window_extent().width * 72 / ax.figure.dpi
        return (factor * diam_px * w_pt / abs(x1 - x0)) ** 2

    def _section_panel(self, ax, lib: str, adata, values: Optional[np.ndarray] = None, colors=None,
                       vmin: float = 0, vmax: Optional[float] = None, cmap=None, pad: float = 30,
                       img_alpha: float = 1.0, show_image: bool = True, spot_factor: float = 1.0, title: str = ""):
        """One section on its H&E (glass = params['image_bg_color']), spots coloured by values or colors."""
        m, xy = self.spot_xy(lib, adata)
        if show_image:
            ax.imshow(self._map_image(lib, adata), alpha=img_alpha)
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

    def plot_sections_gallery(self, adata, sections: Optional[Sequence[str]] = None, ncols: int = 7,
                              panel_size: float = 3.0, fname: Optional[str] = None, font_color: Optional[str] = None,
                              bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                              force: bool = False, verbose: bool = False):
        """H&E of every section, cropped to the tissue spots; title = section | NI | technique, spots, UMIs."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        df_sec = self.read_sections_table()
        sections = list(sections) if sections is not None else df_sec.section_id.tolist()
        nrows = math.ceil(len(sections) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * 1.2 * nrows), squeeze=False,
                                 gridspec_kw=dict(hspace=0.3, wspace=0.06))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, lib in zip(axes.ravel(), sections):
            m, xy = self.spot_xy(lib, adata)
            ax.imshow(self._map_image(lib, adata))
            x0, y0 = xy.min(0) - 40
            x1, y1 = xy.max(0) + 40
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_title(self.section_title(lib, df_sec), fontsize=8)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return fig

    def _plot_spatial_grid(self, rows: Dict[str, np.ndarray], adata, sections: Sequence[str],
                           row_titles: Optional[Dict[str, str]] = None, q: Optional[float] = None, cmap=None,
                           panel_size: float = 2.4, img_alpha: float = 0.35, cbar_label: str = "",
                           italic: bool = False, font_color: Optional[str] = None):
        """rows = {label: values over adata spots}; grid rows x sections; one colour scale per row (0 .. q-th
        percentile over the shown spots)."""
        q = self.get_param("clip_quantile", q)
        df_sec = self.read_sections_table()
        sec = adata.obs[self.sample_col].astype(str).values
        shown = np.isin(sec, list(sections))
        nr, nc = len(rows), len(sections)
        fig, axes = plt.subplots(nr, nc + 1, figsize=(panel_size * nc + 0.5, panel_size * nr), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1] * nc + [0.06], wspace=0.04, hspace=0.12))
        for i, (lab, v) in enumerate(rows.items()):
            v = np.asarray(v, dtype=float)
            vmax = max(self.percentile_cap(v[shown], q), 1e-6)
            sca = None
            for j, lib in enumerate(sections):
                sca = self._section_panel(axes[i, j], lib, adata, values=v, vmin=0, vmax=vmax, cmap=cmap,
                                          img_alpha=img_alpha)
                if i == 0:
                    axes[i, j].set_title(self.section_title(lib, df_sec, short=True), fontsize=9)
            axes[i, 0].text(-0.06, 0.5, (row_titles or {}).get(lab, lab), transform=axes[i, 0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=11, color=self.get_param("font_color", font_color),
                            fontstyle="italic" if italic else "normal")
            cb = fig.colorbar(sca, cax=axes[i, nc])
            cb.ax.tick_params(labelsize=7)
            if cbar_label:
                cb.set_label(cbar_label, fontsize=7)
        return fig

    def plot_spatial_genes(self, genes: Sequence[str], adata, sections: Optional[Sequence[str]] = None,
                           nframe: int = 6, panel_size: float = 2.4, img_alpha: float = 0.35, cmap=None,
                           q: Optional[float] = None, fname: Optional[str] = None, font_color: Optional[str] = None,
                           bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                           force: bool = False, verbose: bool = False) -> List[Any]:
        """Genes (rows) x sections (default: the representative sections); log1p(CP10k) of the spot, one scale per
        gene shared by the sections. nframe genes per figure; fname may contain '{k}' (figure number)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        sections = list(sections) if sections is not None else self.representative_sections()
        sub = adata[adata.obs[self.sample_col].isin(sections)]
        found, Y = self.lognorm_genes(genes, sub)
        pos = np.flatnonzero(adata.obs_names.isin(sub.obs_names))
        figs = []
        for k, i in enumerate(range(0, len(found), nframe)):
            rows = {}
            for g in found[i:i + nframe]:
                v = np.zeros(adata.n_obs, dtype=np.float32)
                v[pos] = Y[:, found.index(g)]
                rows[g] = v
            fig = self._plot_spatial_grid(rows, adata, sections, q=q, cmap=cmap, panel_size=panel_size,
                                          img_alpha=img_alpha, cbar_label="log1p(CP10k)", italic=True,
                                          font_color=font_color)
            self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
            if fname is not None:
                self._savefig(fig, fname.format(k=k + 1) if "{k}" in fname else fname, dpi=self.params["dpi_maps"],
                              force=force, verbose=verbose)
            figs.append(fig)
        return figs

    def plot_spatial_celltypes(self, celltypes: Sequence[str], adata, df_abund: pd.DataFrame,
                               sections: Optional[Sequence[str]] = None, panel_size: float = 2.4,
                               img_alpha: float = 0.35, cmap="magma", q: Optional[float] = None,
                               cbar_label: str = "", fname: Optional[str] = None, font_color: Optional[str] = None,
                               bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                               force: bool = False, verbose: bool = False):
        """Cell-type abundances (rows) x sections; one scale per cell type."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        sections = list(sections) if sections is not None else self.representative_sections()
        df_abund = df_abund.reindex(adata.obs_names).fillna(0)
        rows = {c: df_abund[c].values for c in celltypes if c in df_abund.columns}
        cbar_label = cbar_label or ("cells per spot (q05)" if df_abund.attrs.get("source") == "c2l" else "RCTD weight")
        fig = self._plot_spatial_grid(rows, adata, sections, q=q, cmap=cmap, panel_size=panel_size,
                                      img_alpha=img_alpha, cbar_label=cbar_label, font_color=font_color)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return fig

    def plot_regions_on_sections(self, adata, key: str = "region_label", sections: Optional[Sequence[str]] = None,
                                 names: Optional[Dict[str, str]] = None, ncols: Optional[int] = None,
                                 panel_size: float = 3.4, colors: Optional[Dict[str, str]] = None,
                                 legend_ncol: int = 4, fname: Optional[str] = None, font_color: Optional[str] = None,
                                 bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None, bottom_scale:float=3.,
                                 force: bool = False, verbose: bool = False):
        
        """Categorical key (regions, niches, nerve rings) on the sections, one shared legend."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        sections = list(sections) if sections is not None else self.representative_sections()
        df_sec = self.read_sections_table()
        labs = adata.obs[key].astype(str).values
        cats = [c for c in (adata.obs[key].cat.categories if hasattr(adata.obs[key], "cat") else sorted(set(labs)))
                if c not in ("nan", "")]
        colors = colors or self._region_palette(cats)
        ncols = ncols or min(len(sections), 6)
        nrows = math.ceil(len(sections) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows + 1.6), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        col = np.array([colors.get(l, "#555555") if l not in ("nan", "") else "#00000000" for l in labs], dtype=object)
        for ax, lib in zip(axes.ravel(), sections):
            self._section_panel(ax, lib, adata, colors=col, img_alpha=0.5, title=self.section_title(lib, df_sec, short=True))
        present = set(labs[np.isin(adata.obs[self.sample_col].astype(str).values, sections)])
        handles = [Line2D([], [], ls="", marker="o", ms=7, mfc=colors[c], mec="none",
                          label=f"{c} - {names[c]}" if names and c in names else c) for c in cats if c in present]
        fig.legend(handles=handles, loc="lower center", ncol=legend_ncol, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))
        
        fig.subplots_adjust(bottom=bottom_scale / (panel_size * nrows + 1.6) + 0.05, wspace=0.02, hspace=0.12)
        
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        
        return fig


    # =================================================================================================================
    # 6. reference figures: global UMAP (scVI on a per-type subsample), cell types x genes, compartments
    # =================================================================================================================
    def calc_reference_umap(self, adata_ref, n_per_type: int = 3000, n_hvg: int = 3000, n_latent: int = 30,
                            accelerator: str = "cpu", fname: Optional[str] = "reference_umap", force: bool = False,
                            verbose: bool = True) -> pd.DataFrame:
        """UMAP of the tissue cells (Figure 1B analogue): <= n_per_type cells per cell type (seeded), scVI on n_hvg
        HVGs (batch sample_id, covariate technology), kNN 15, UMAP. Cached table: cell, umap1, umap2, cell_type,
        major_type, technology, tissue, patient (files[fname] when registered)."""
        def _compute():
            import scvi
            a = self.reference_for_deconvolution(adata_ref, max_cells=n_per_type, verbose=verbose)
            a.var_names = a.var["symbol"].astype(str).values
            a.var_names_make_unique()
            sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", batch_key="technology")
            ah = a[:, a.var.highly_variable].copy()
            ah.obs["sample_id"] = ah.obs["sample_id"].astype(str).astype("category")
            scvi.settings.seed = self.params["seed"]
            scvi.model.SCVI.setup_anndata(ah, batch_key="sample_id", categorical_covariate_keys=["technology"])
            mod = scvi.model.SCVI(ah, n_latent=n_latent, n_layers=2, gene_likelihood="nb")
            mod.train(max_epochs=min(300, int(np.round(20000 / ah.n_obs * 400)) + 50), accelerator=accelerator,
                      early_stopping=True, enable_progress_bar=False)
            a.obsm["X_scvi"] = mod.get_latent_representation()
            sc.pp.neighbors(a, use_rep="X_scvi", n_neighbors=15, random_state=self.params["seed"])
            sc.tl.umap(a, random_state=self.params["seed"])
            return pd.DataFrame({"cell": a.obs_names, "umap1": a.obsm["X_umap"][:, 0].round(4),
                                 "umap2": a.obsm["X_umap"][:, 1].round(4),
                                 **{c: a.obs[c].astype(str).values for c in
                                    ("cell_type", "major_type", "technology", "tissue", "patient")}})
        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def celltype_colors(self, celltypes: Sequence[str]) -> Dict[str, str]:
        """Major types: one colour per lineage shade; subclusters: shades of their major type's colormap."""
        cmaps = {"epithelial": "Oranges", "stromal": "Purples", "myeloid": "Greens", "lymphoid": "Blues"}
        out = {}
        groups: Dict[str, List[str]] = {}
        for c in self.celltype_order(celltypes):
            groups.setdefault(self.major_of_celltype(c), []).append(c)
        majors = [m_ for m_ in groups]
        lin_major: Dict[str, List[str]] = {}
        for mj in majors:
            lin_major.setdefault(self.lineage_of_celltype(mj), []).append(mj)
        for lin, mjs in lin_major.items():
            cmap = plt.get_cmap(cmaps.get(lin, "Greys"))
            for i, mj in enumerate(mjs):
                base = 0.45 + 0.5 * (i + 1) / (len(mjs) + 1)
                cts = groups[mj]
                for k, c in enumerate(cts):
                    shade = base if len(cts) == 1 else 0.3 + 0.65 * (k + 1) / (len(cts) + 1)
                    out[c] = mpl.colors.to_hex(cmap(shade))
        sch = [c for c in out if c.startswith("S0") or c == "Schwann"]     # Schwann: the paper's nerve yellow
        yellows = ["#FFEE6F", "#F3C300", "#C9A400", "#FFF7B0"]
        for k, c in enumerate(sch):
            out[c] = yellows[k % len(yellows)]
        return out

    def plot_umap_celltype_dotplot(self, df_umap: pd.DataFrame, adata_ref, genes: Sequence[str],
                                   groupby: str = "major_type", figsize=(26, 11), point_size: float = 1.5,
                                   fname: Optional[str] = None, font_color: Optional[str] = None,
                                   bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                   force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Left: reference UMAP (calc_reference_umap) coloured by groupby, labels at the medians. Right: dot plot
        groupby (rows, by lineage) x genes over the UMAP cells (size = % cells > 0; colour = mean log1p(CP10k) / max
        over the rows). Returns the long table (group, gene, frac, mean, rel)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        a = adata_ref[df_umap.cell.values]
        a = a.to_memory() if a.isbacked else a.copy()
        a.obs[groupby] = df_umap[groupby].astype(str).values
        a.obsm["X_umap"] = df_umap[["umap1", "umap2"]].values
        order = self.celltype_order(a.obs[groupby].unique())
        df_frac, df_mean, df_rel = self.calc_df_dotplot(list(genes), groupby, a, order=order)
        colors = self.celltype_colors(order)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.1)
        ax0 = fig.add_subplot(gs[0])
        self.plot_umap_labeled(ax0, a.obsm["X_umap"], a.obs[groupby].values, colors,
                               title=f"sc + sn reference (tissue): {a.n_obs:,} cells shown, {len(order)} {groupby}",
                               point_size=point_size, fontsize=8, seed=self.params["seed"])
        ax1 = fig.add_subplot(gs[1])
        n_r, n_c = len(order), df_frac.shape[1]
        yy, xx = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
        sca = ax1.scatter(xx.ravel(), yy.ravel(), s=df_frac.values.ravel() * 160, c=df_rel.values.ravel(), cmap=cmap_lbwr,
                          vmin=0, vmax=1, edgecolors="#444444", linewidths=0.3)
        ax1.set_xticks(range(n_c))
        ax1.set_xticklabels(df_frac.columns, rotation=90, fontstyle="italic")
        ax1.set_yticks(range(n_r))
        ax1.set_yticklabels(order)
        for t, c in zip(ax1.get_yticklabels(), order):
            t.set_color(colors[c])
        ax1.set_ylim(n_r - 0.5, -0.5)
        ax1.yaxis.tick_right()
        cb = fig.colorbar(sca, ax=ax1, orientation="horizontal", fraction=0.025, pad=0.12, shrink=0.5, anchor=(0.0, 1.0))
        cb.set_label("mean expression / max over rows")
        for f in (0.25, 0.5, 1.0):
            ax1.scatter([], [], s=f * 160, c="#bbbbbb", label=f"{int(f * 100)}%")
        ax1.legend(title="% cells > 0", loc="upper left", bbox_to_anchor=(0.62, -0.12), ncol=3, frameon=False)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, force=force, verbose=verbose)
        df_long = pd.concat({"frac": df_frac.stack(), "mean": df_mean.stack(), "rel": df_rel.stack()}, axis=1)
        df_long.index.names = [groupby, "gene"]
        return df_long.reset_index()

    def plot_compartment_subclusters(self, compartment: str, n_genes: int = 4, figsize=(22, 8), point_size: float = 3,
                                     fname: Optional[str] = None, font_color: Optional[str] = None,
                                     bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                     force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Re-derived subclusters of one compartment (results/pdac01_subclusters/<compartment>.h5ad): UMAP by
        subcluster, UMAP by technology, dot plot subclusters x marker genes (the first n_genes of each paper marker
        set). Returns the cluster table (annotate_clusters)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        a = ad.read_h5ad(self.root_result / "pdac01_subclusters" / f"{compartment}.h5ad")
        df_cl = self.annotate_clusters(a, compartment)
        spec = PDAC_SUBCLUSTERS[compartment]
        order = [l for l in spec["markers"] if l in set(a.obs.subcluster.astype(str))]
        order += sorted(set(a.obs.subcluster.astype(str)) - set(order))
        genes = list(dict.fromkeys(g for l in spec["markers"] for g in spec["markers"][l][:n_genes] if g in a.var_names))
        colors = self.celltype_colors(order)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6], wspace=0.15)
        ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        U = a.obsm["X_umap"]
        self.plot_umap_labeled(ax0, U, a.obs.subcluster.astype(str).values, colors,
                               title=f"{compartment}: {a.n_obs:,} cells", point_size=point_size, fontsize=8,
                               seed=self.params["seed"])
        tc = {"sc5p": "#4C9BE8", "snFlex": "#E8834C"}
        for t, col in tc.items():
            mk = (a.obs.technology.astype(str) == t).values
            ax1.scatter(U[mk, 0], U[mk, 1], s=point_size, c=col, linewidths=0, rasterized=True,
                        label=f"{t} ({int(mk.sum()):,})")
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.legend(frameon=False, markerscale=4, loc="best")
        ax1.set_title("technology")
        a.obs["_g"] = a.obs.subcluster.astype(str)
        a_cp = a.copy()                                  # X is log1p(CP10k): expm1 -> CP10k (renormalises to itself)
        a_cp.X = a_cp.X.expm1() if sp.issparse(a_cp.X) else np.expm1(a_cp.X)
        df_frac, df_mean, df_rel = self.calc_df_dotplot(genes, "_g", a_cp, order=order)
        n_r, n_c = len(order), len(genes)
        yy, xx = np.meshgrid(np.arange(n_r), np.arange(n_c), indexing="ij")
        sca = ax2.scatter(xx.ravel(), yy.ravel(), s=df_frac.values.ravel() * 150, c=df_rel.values.ravel(), cmap=cmap_lbwr,
                          vmin=0, vmax=1, edgecolors="#444444", linewidths=0.3)
        ax2.set_xticks(range(n_c)); ax2.set_xticklabels(df_frac.columns, rotation=90, fontstyle="italic")
        ax2.set_yticks(range(n_r)); ax2.set_yticklabels([f"{o} ({int((a.obs._g == o).sum())})" for o in order])
        ax2.set_ylim(n_r - 0.5, -0.5); ax2.yaxis.tick_right()
        fig.colorbar(sca, ax=ax2, fraction=0.02, pad=0.25).set_label("mean / max over subclusters")
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return df_cl

    def read_lineage_table(self) -> pd.DataFrame:
        """tables/pdac_lineage.tsv (as tables/heart_lineage.tsv): HL_ID = GEO major type, HL_SUBID = deconvolution cell
        type (re-derived subcluster or major type), description, functional_summary. Indexed by HL_SUBID."""
        return self.read_tsv("lineage_table").set_index("HL_SUBID")

    def calc_df_ref_composition(self, adata_ref, by: str = "tissue", fname: Optional[str] = None,
                                force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """% of cells of each cell type (obs['cell_type'] with subclusters) per level of `by` (columns) + lineage."""
        def _compute():
            df_ct = pd.crosstab(adata_ref.obs["cell_type"].astype(str), adata_ref.obs[by].astype(str))
            df_ct = df_ct.reindex(self.celltype_order(df_ct.index))
            df_pct = (100 * df_ct / df_ct.sum(0)).round(3)
            df_pct.insert(0, "lineage", [self.lineage_of_celltype(c) for c in df_pct.index])
            df_pct.insert(1, "n_cells", df_ct.sum(1).values)
            if self.file_path("lineage_table").exists():
                desc = self.read_lineage_table()["description"]
                df_pct.insert(1, "description", [desc.get(c, "") for c in df_pct.index])
            return df_pct.reset_index(names="cell_type")
        return self._cached_any(fname, _compute, force=force, verbose=verbose).set_index("cell_type")

    # =================================================================================================================
    # 7. regions -> the paper's niches (rules on the cell-type shares)
    # =================================================================================================================
    def niche_groups(self, celltypes: Sequence[str]) -> Dict[str, List[str]]:
        """Cell types that define each niche of the paper (STAR Methods; Figure 1F). 'malignant' (D02-D09,
        Ductal-other) is split into the four malignant niches by the dominant subtype (assign_niches)."""
        ct = list(celltypes)
        pick = lambda f: [c for c in ct if f(c)]
        return {
            "N01_TLS": pick(lambda c: c in ("B", "CD4T", "CD8T")),
            "N02_Endocrine": pick(lambda c: c == "Endocrine"),
            "N03_Peri-vascular": pick(lambda c: c in ("Endothelial", "Stellate")),
            "N04_Plasma-enriched": pick(lambda c: c == "Plasma"),
            "N05_CAF-enriched": pick(lambda c: c.startswith("F0") and c != "F07_Fibro-NRP2" or c == "CAF"),
            "N06_TAM-enriched": pick(lambda c: re.match(r"M(0[6-9]|1[0-3])_", c) is not None or c == "Myeloid"),
            "malignant": pick(lambda c: re.match(r"D0[2-9]_", c) is not None or c == "Ductal-other"),
            "N11_Non-malignant-pancreas": pick(lambda c: c in ("Acinar", "D01_Ductal-CFTR")),
            "N12_Nerve": pick(lambda c: c.startswith("S0") or c in ("Schwann", "F07_Fibro-NRP2")),
        }

    @staticmethod
    def shares(df_abund: pd.DataFrame) -> pd.DataFrame:
        """Abundances -> shares per spot (rows sum to 1; empty rows 0)."""
        out = df_abund.div(df_abund.sum(1).replace(0, np.nan), axis=0).fillna(0)
        out.attrs = dict(df_abund.attrs)
        return out

    def calc_df_sections_summary(self, df_sections: pd.DataFrame, force: bool = False,
                                 verbose: bool = False) -> pd.DataFrame:
        """Sections by NI status x technique: n_sections, n_patients, n_spots, median of the section median UMIs.
        Cached in tsv_sections_summary."""
        def compute():
            return (df_sections.groupby(["ni_status", "technique"], observed=True)
                    .agg(n_sections=("section_id", "size"), n_patients=("patient", "nunique"),
                         n_spots=("n_spots", "sum"), median_umis=("median_umis", "median")).reset_index())
        return self._cached_any("tsv_sections_summary", compute, force=force, verbose=verbose).set_index(["ni_status", "technique"])

    def calc_df_ref_major_types(self, adata_ref, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """sc + sn reference: cells of each GEO major type (columns) per technology x tissue (rows), with totals
        (crosstab margins 'All'). Cached in tsv_ref_major_types."""
        def compute():
            obs = adata_ref.obs
            df = pd.crosstab([obs.technology.astype(str), obs.tissue.astype(str)], obs.major_type.astype(str), margins=True)
            return df.reset_index()
        return self._cached_any("tsv_ref_major_types", compute, force=force, verbose=verbose).set_index(["technology", "tissue"])

    def calc_df_gene_best_group(self, df_dot: pd.DataFrame, groupby: str = "cell_type",
                                genes: Optional[Sequence[str]] = None, fname: Optional[str] = None,
                                force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Per gene: the group (groupby column of the dot table of plot_umap_celltype_dotplot: major_type or
        cell_type) with the highest mean expression, its % expressing (frac) and mean, the group's lineage and the
        gene's role (PDAC_GENES). Ordered as genes (default PDAC_GENES). Index = gene. fname: cache (key)."""
        genes = list(genes or PDAC_GENES)

        def compute():
            d = df_dot.loc[df_dot.groupby("gene")["mean"].idxmax(), ["gene", groupby, "frac", "mean"]]
            d = d.set_index("gene").loc[[g for g in genes if g in set(d.gene)]].reset_index()
            d.insert(1, "role", d.gene.map(PDAC_GENES).fillna(""))
            d.insert(3, "lineage", [self.lineage_of_celltype(c) for c in d[groupby]])
            return d.round({"frac": 3, "mean": 3})
        return self._cached_any(fname, compute, force=force, verbose=verbose).set_index("gene")

    def calc_df_gene_panel(self, adata_ref, adata_vis, genes: Optional[Dict[str, str]] = None, force: bool = False,
                           verbose: bool = False) -> pd.DataFrame:
        """The gene panel (default PDAC_GENES: symbol -> role): gene, role, in_reference, gene_id_reference,
        in_visium, gene_id_visium (var_names; '' when absent). Cached in tsv_gene_panel: force=True after changing
        the panel. Prints how many genes are found in each dataset."""
        genes = dict(genes or PDAC_GENES)

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

    def calc_df_section_qc(self, adata, df_sections: pd.DataFrame, cols: Sequence[str] = ("ni_status", "technique"),
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Per-section QC (calc_df_qc: n, median / q25 / q75 of genes, counts and % MT) joined with the columns cols
        of df_sections (by section_id). Cached in tsv_section_qc: force=True after changing the QC filters.
        Index = section."""
        def compute():
            return self.calc_df_qc(adata=adata).join(df_sections.set_index("section_id")[list(cols)]).reset_index()
        df = self._cached_any("tsv_section_qc", compute, force=force, verbose=verbose)
        return df.set_index(df.columns[0])

    def annotate_niches(self, adata, df_abund: pd.DataFrame, abundance_source: str = "", key: str = "region_label",
                        cluster_key: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        assign_niches on the regions of adata.obs[key] -> obs 'region_name' and 'niche' (the niche name of the
        region), plus n_spots, n_sections and abundance_source per region. obs[key] is first set from the Leiden
        clusters obs[cluster_key] (params['region_key'], 'region_cluster'): '0' -> 'R00' (params['region_id_fmt']),
        the same ids as calc_region_names; when obs has no cluster_key, the existing obs[key] is used. Written to
        tsv_niches when the file is missing or force (the same rule as save_region_labels, so the niche table and
        the region labels written to disk always come from the same clustering). Returns the table of the current
        regions.
        """
        cluster_key = self.get_param("region_key", cluster_key)

        if cluster_key in adata.obs:
            fmt = self.params["region_id_fmt"]
            cl = adata.obs[cluster_key].astype(str)
            labels = {c: fmt.format(int(c)) for c in sorted(cl.unique(), key=int)}
            adata.obs[key] = pd.Categorical(cl.map(labels), categories=list(labels.values()))
        elif key not in adata.obs:
            raise KeyError(f"obs has neither {cluster_key!r} (run cluster_regions) nor {key!r}")
        
        df_niche = self.assign_niches(adata, df_abund, key=key)
        names = dict(zip(df_niche[key], df_niche.niche))

        adata.obs["region_name"] = adata.obs[key].map(names).astype(str)
        adata.obs["niche"] = adata.obs["region_name"]

        counts = adata.obs.groupby(key, observed=True).agg(n_spots=("section", "size"), n_sections=("section", "nunique"))

        df_niche = df_niche.merge(counts, left_on=key, right_index=True)
        df_niche["abundance_source"] = abundance_source

        if "tsv_niches" in self.files and (force or not self.file_path("tsv_niches").exists()):
            self.write_tsv(df_niche, "tsv_niches", verbose=verbose)
        elif verbose and "tsv_niches" in self.files:
            print(f"File exists: {self.file_path('tsv_niches')} (force=True to rewrite)")
            
        return df_niche

    def save_region_labels(self, adata, abundance_source: str = "",
                           cols: Sequence[str] = ("region_cluster", "region_label", "region_name", "niche", "section"),
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Region labels of adata.obs (barcode + cols + abundance_source) -> region_labels.tsv, the input of
        mp_pdac02-04. File present and not force: kept, and the share of spots whose label agrees with it is printed
        (re-clustering is not bit-reproducible across machines / library versions). Returns the current labels."""
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

    def assign_niches(self, adata, df_abund: pd.DataFrame, key: str = "region_label", min_share: float = 0.10,
                      min_score: float = 0.10) -> pd.DataFrame:
        """Each region -> one of the paper's niches (rules on the RCTD / cell2location shares). Per region: mean share of
        every niche group (niche_groups) and its enrichment (region / all spots); score = share x log2(enrichment) for
        the groups with share >= min_share and enrichment > 1; niche = the group with the highest score, 'mixed' when
        the best score < min_score, 'N00_unassigned' for regions of spots without deconvolution (all zero, e.g. RCTD
        drops spots < 100 UMIs). Malignant regions are named after their dominant subtype,
        'N07-10_Malignant (<subtype>)' (the paper's N07-N10 numbering is not public, except N10 = D02_Ductal-APOA2).
        Returns one row per region: region_label, niche, niche_group, niche_share, niche_enrichment, niche_score,
        dominant (top cell types), malignant_top."""
        df_s = self.shares(df_abund.loc[adata.obs_names])
        groups = self.niche_groups(df_s.columns)
        lab = adata.obs[key].astype(str).values
        G = pd.DataFrame({g: df_s[c].sum(1) if c else 0.0 for g, c in groups.items()})
        Gm = G.groupby(lab).mean()
        En = Gm / G.mean(0).replace(0, np.nan)
        Sc = Gm * np.log2(En.clip(lower=1)).fillna(0)
        Sc = Sc.where(Gm >= min_share, 0.0)
        Cm = df_s.groupby(lab).mean()
        rows = []
        for r in Gm.index:
            mal = Cm.loc[r, groups["malignant"]].sort_values(ascending=False) if groups["malignant"] else pd.Series(dtype=float)
            if Gm.loc[r].sum() <= 1e-9:
                g, niche = "unassigned", "N00_unassigned (no deconvolution)"
            elif Sc.loc[r].max() < min_score:
                g = "mixed"
                niche = "mixed (" + " / ".join(Gm.loc[r].sort_values(ascending=False).index[:2]) + ")"
            else:
                g = Sc.loc[r].idxmax()
                niche = f"N07-10_Malignant ({mal.index[0]})" if g == "malignant" else g
            rows.append(dict(region_label=r, niche=niche, niche_group=g,
                             niche_share=round(float(Gm.loc[r, g]), 3) if g in Gm.columns else 0.0,
                             niche_enrichment=round(float(En.loc[r, g]), 2) if g in En.columns else np.nan,
                             niche_score=round(float(Sc.loc[r, g]), 3) if g in Sc.columns else 0.0,
                             dominant=", ".join(f"{c} ({v:.0%})" for c, v in Cm.loc[r].sort_values(ascending=False)[:3].items()),
                             malignant_top=mal.index[0] if len(mal) else ""))
        return pd.DataFrame(rows)

    def niche_colors(self, niches: Sequence[str]) -> Dict[str, str]:
        """The paper's niche colours (myUtiliity.R); malignant niches take the paper's four malignant colours, then
        three extra ones (more than four malignant subtypes are found here); mixed = slate blue, unassigned = dark grey."""
        mal_cols = [PAPER_NICHES[k] for k in PAPER_NICHES if "Malignant" in k] + ["#C71585", "#7FFFD4", "#B8860B"]
        out, k = {}, 0
        for n in sorted(niches):
            base = n.split(" (")[0]
            if base in PAPER_NICHES:
                out[n] = PAPER_NICHES[base]
            elif "Malignant" in n:
                out[n] = mal_cols[k % len(mal_cols)]; k += 1
            elif n.startswith("mixed"):
                out[n] = "#5C6BC0"           # slate blue: distinct from the paper's grey of N02_Endocrine
            else:
                out[n] = "#3A3A3A"
        return out

    def unassigned_regions(self) -> List[str]:
        """Regions of mp_pdac01 named N00_unassigned (spots without deconvolution): skipped by subregions / LR."""
        d = self.read_tsv("region_niches")
        return d.loc[d.niche.str.startswith("N00"), "region_label"].astype(str).tolist()

    def read_region_info(self, source: Optional[str] = None, verbose: bool = False) -> pd.DataFrame:
        """Region table (region_id, region_name, compartment, description, abundance_source): the curated
        files['region_info'] (pdac/data/pdac_regions.tsv) rows of the abundance source when present, otherwise built from
        the niche assignment of mp_pdac01 (files['region_niches']: name = niche, compartment = niche group,
        description = dominant cell types)."""
        if self.file_path("region_info").exists():
            df = self.read_tsv("region_info", verbose=verbose)
            if source is not None and "abundance_source" in df:
                df = df[df.abundance_source == source]
            if len(df):
                return df.reset_index(drop=True)
        d = self.read_tsv("region_niches", verbose=verbose)
        return pd.DataFrame({"region_id": d.region_label, "region_name": d.niche, "compartment": d.niche_group,
                             "description": "dominant: " + d.dominant.astype(str),
                             "abundance_source": d.get("abundance_source", "")})

    # =================================================================================================================
    # 8. sections as replicates: shares per section, NI groups (high vs low), CytAssist-only sensitivity
    # =================================================================================================================
    def calc_df_share_by_section(self, adata, key: str, fname: Optional[str] = None, force: bool = False,
                                 verbose: bool = False) -> pd.DataFrame:
        """Share of each level of obs[key] (regions, niches) per section; long table: section, patient, ni_status,
        technique, group, share."""
        def _compute():
            obs = adata.obs
            ct = pd.crosstab(obs[self.sample_col].astype(str), obs[key].astype(str), normalize="index")
            df = ct.stack().rename("share").reset_index()
            df.columns = ["section", "group", "share"]
            df_sec = self.read_sections_table().set_index("section_id")
            for c in ("patient", "ni_status", "technique"):
                df[c] = df.section.map(df_sec[c].astype(str))
            df["share"] = df["share"].round(6)
            return df[["section", "patient", "ni_status", "technique", "group", "share"]]
        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def calc_df_celltype_by_section(self, adata, df_abund: pd.DataFrame, fname: Optional[str] = None,
                                    force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Mean cell-type share per section (abundance / spot total averaged over the section's spots); long table:
        section, patient, ni_status, technique, lineage, cell_type, share."""
        def _compute():
            df_p = self.shares(df_abund.loc[adata.obs_names])
            df_m = df_p.groupby(adata.obs[self.sample_col].astype(str).values).mean()
            df = df_m.stack().rename("share").reset_index()
            df.columns = ["section", "cell_type", "share"]
            df_sec = self.read_sections_table().set_index("section_id")
            for c in ("patient", "ni_status", "technique"):
                df[c] = df.section.map(df_sec[c].astype(str))
            df["lineage"] = df.cell_type.map(self.lineage_of_celltype)
            df["share"] = df["share"].round(6)
            return df[["section", "patient", "ni_status", "technique", "lineage", "cell_type", "share"]]
        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def calc_df_ni_test(self, df_long: pd.DataFrame, value: str = "share", group_col: str = "group",
                        techniques: Optional[Sequence[str]] = None, fname: Optional[str] = None,
                        force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """High vs low NI per group (sections as replicates): medians, log2 ratio of means, two-sided Mann-Whitney U
        p-value and BH FDR (all sections, and with techniques= a subset, e.g. ('CytAssist',): all low-NI sections are
        CytAssist, so the subset compares like with like). fname: cache (key or file name), None = not written."""
        return self._cached_any(fname, lambda: self._ni_test(df_long, value, group_col, techniques), force=force,
                                verbose=verbose)

    def _ni_test(self, df_long: pd.DataFrame, value: str, group_col: str,
                 techniques: Optional[Sequence[str]]) -> pd.DataFrame:
        from scipy.stats import mannwhitneyu
        d = df_long if techniques is None else df_long[df_long.technique.isin(list(techniques))]
        rows = []
        for g, sub in d.groupby(group_col, sort=False):
            lo = sub.loc[sub.ni_status == "low", value].values
            hi = sub.loc[sub.ni_status == "high", value].values
            if len(lo) < 2 or len(hi) < 2:
                continue
            p = mannwhitneyu(hi, lo, alternative="two-sided").pvalue if (np.ptp(np.r_[lo, hi]) > 0) else 1.0
            rows.append({group_col: g, "n_low": len(lo), "n_high": len(hi), "median_low": np.median(lo),
                         "median_high": np.median(hi),
                         "log2_ratio_high_low": np.log2((hi.mean() + 1e-4) / (lo.mean() + 1e-4)), "p": p})
        df = pd.DataFrame(rows)
        if len(df):
            df["fdr"] = self._bh(df.p.values)
        df.attrs["sections"] = "all" if techniques is None else "+".join(techniques)
        return df.round(5)

    def plot_ni_groups(self, df_long: pd.DataFrame, groups: Sequence[str], value: str = "share",
                       group_col: str = "group", df_test: Optional[pd.DataFrame] = None, ncols: int = 6,
                       panel_size=(2.6, 3.2), title: str = "", ylabel: str = "share of the section",
                       fname: Optional[str] = None, font_color: Optional[str] = None, bg_color: Optional[str] = None,
                       fontsize_scale: Optional[float] = None, force: bool = False, verbose: bool = False):
        """One panel per group: section values for low vs high NI (box + points; marker = technique), Mann-Whitney
        p in the title (Figure 2E / 3B / 3D analogue; the paper's ridge plots summarise the same 5 vs 16 sections)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        groups = [g for g in groups if g in set(df_long[group_col])]
        nrows = math.ceil(len(groups) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        col = {"low": "#4C9BE8", "high": "#E8504C"}
        mk = {"FF": "s", "FFPE": "^", "CytAssist": "o"}
        rng = np.random.default_rng(0)
        p_of = df_test.set_index(group_col)["p"] if df_test is not None and len(df_test) else {}
        for ax, g in zip(axes.ravel(), groups):
            sub = df_long[df_long[group_col] == g]
            for i, ni in enumerate(self.params["ni_order"]):
                v = sub[sub.ni_status == ni]
                ax.boxplot([v[value].values], positions=[i], widths=0.55, showfliers=False,
                           medianprops=dict(color=col[ni]), boxprops=dict(color="#888888"),
                           whiskerprops=dict(color="#888888"), capprops=dict(color="#888888"))
                for t, vt in v.groupby("technique"):
                    ax.scatter(i + rng.uniform(-0.15, 0.15, len(vt)), vt[value], s=18, c=col[ni], marker=mk.get(t, "o"),
                               edgecolors="none", alpha=0.9)
            ax.set_xticks([0, 1]); ax.set_xticklabels([f"{n}-NI" for n in self.params["ni_order"]])
            p = p_of.get(g, np.nan) if hasattr(p_of, "get") else np.nan
            ax.set_title(f"{g}\np = {p:.3g}" if np.isfinite(p) else g, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
        handles = [Line2D([], [], ls="", marker=v, color="#999999", label=k) for k, v in mk.items()]
        fig.legend(handles=handles, loc="upper right", frameon=False, title="technique")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def plot_section_pca(self, df_long: pd.DataFrame, value: str = "share", group_col: str = "group",
                         figsize=(7, 6), title: str = "", fname: Optional[str] = None, font_color: Optional[str] = None,
                         bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None, force: bool = False,
                         verbose: bool = False) -> pd.DataFrame:
        """PCA of the sections on their group shares (Figure 3C analogue); colour = NI, marker = technique.
        Returns the section scores (PC1, PC2) with metadata."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        M = df_long.pivot_table(index="section", columns=group_col, values=value).fillna(0)
        X = (M - M.mean(0)) / M.std(0).replace(0, 1)
        U, S, Vt = np.linalg.svd(X.values, full_matrices=False)
        pcs = U[:, :2] * S[:2]
        ev = S ** 2 / (S ** 2).sum()
        meta = df_long.drop_duplicates("section").set_index("section").loc[M.index]
        df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1]}, index=M.index).join(meta[["patient", "ni_status", "technique"]])
        fig, ax = plt.subplots(figsize=figsize)
        col = {"low": "#4C9BE8", "high": "#E8504C"}
        mk = {"FF": "s", "FFPE": "^", "CytAssist": "o"}
        for (ni, t), d in df.groupby(["ni_status", "technique"]):
            ax.scatter(d.PC1, d.PC2, c=col[ni], marker=mk[t], s=70, label=f"{ni}-NI, {t}", edgecolors="none")
        for s, r in df.iterrows():
            ax.annotate(s, (r.PC1, r.PC2), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(title or f"Sections on {group_col} shares")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return df.round(4)

    @staticmethod
    def shannon(p: np.ndarray) -> float:
        p = np.asarray(p, float)
        p = p[p > 0] / p[p > 0].sum() if (p > 0).any() else p
        return float(-(p * np.log(p)).sum()) if len(p) else 0.0

    def calc_df_malignant_diversity(self, adata, df_abund: pd.DataFrame, fname: Optional[str] = None,
                                    force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Per section (index): share of each malignant subtype (D02-D09) among malignant cells, the Shannon index,
        ni_status and technique (Figure 2J analogue). fname: cache, None = not written."""
        df = self._cached_any(fname, lambda: self._malignant_diversity(adata, df_abund).reset_index(names="section"),
                              force=force, verbose=verbose)
        return df.set_index("section")

    def _malignant_diversity(self, adata, df_abund: pd.DataFrame) -> pd.DataFrame:
        df_s = self.shares(df_abund.loc[adata.obs_names])
        mal = [c for c in df_s.columns if re.match(r"D0[2-9]_", c)]
        M = df_s[mal].groupby(adata.obs[self.sample_col].astype(str).values).sum()
        M = M.div(M.sum(1), axis=0)
        M["shannon"] = [self.shannon(r) for r in M[mal].values]
        df_sec = self.read_sections_table().set_index("section_id")
        M["ni_status"] = df_sec.loc[M.index, "ni_status"].astype(str).values
        M["technique"] = df_sec.loc[M.index, "technique"].astype(str).values
        return M.round(4)

    def calc_df_diversity_test(self, df_div: pd.DataFrame, techniques: Sequence[Optional[str]] = (None, "CytAssist"),
                               ) -> pd.DataFrame:
        """Shannon index of the malignant subtypes, low vs high NI: medians and two-sided Mann-Whitney p, over all
        sections (None) and the CytAssist subset."""
        from scipy.stats import mannwhitneyu
        rows = []
        for t in techniques:
            d = df_div if t is None else df_div[df_div.technique == t]
            lo, hi = d[d.ni_status == "low"].shannon, d[d.ni_status == "high"].shannon
            rows.append(dict(sections=t or "all", n_low=len(lo), n_high=len(hi), median_low=lo.median(),
                             median_high=hi.median(),
                             p=mannwhitneyu(lo, hi).pvalue if len(lo) > 1 and len(hi) > 1 else np.nan))
        return pd.DataFrame(rows).round(4)

    def plot_malignant_diversity(self, df_div: pd.DataFrame, figsize=(16, 8), fname: Optional[str] = None,
                                 force: bool = False, verbose: bool = False):
        """Top: Shannon index per section (blue = low NI, red = high NI); bottom: stacked malignant-subtype shares.
        Sections ordered by NI status, then Shannon (Figure 2J analogue)."""
        mal = [c for c in df_div.columns if re.match(r"D0[2-9]_", c)]
        order = df_div.sort_values(["ni_status", "shannon"]).index
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[1, 3]), sharex=True)
        x = np.arange(len(order))
        axes[0].bar(x, df_div.loc[order, "shannon"],
                    color=["#4C9BE8" if v == "low" else "#E8504C" for v in df_div.loc[order, "ni_status"]])
        axes[0].set_ylabel("Shannon")
        bottom = np.zeros(len(order))
        col = self.celltype_colors(mal)
        for c in mal:
            axes[1].bar(x, df_div.loc[order, c], bottom=bottom, color=col[c], label=c, width=0.85)
            bottom += df_div.loc[order, c].values
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{s}\n{df_div.loc[s, 'ni_status']}" for s in order], rotation=90)
        axes[1].set_ylabel("share of malignant cells")
        axes[1].legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)
        fig.suptitle("Malignant subtypes per section (Figure 2J analogue)")
        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])
        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def calc_df_celltype_correlation(self, df_ct_sec: pd.DataFrame) -> pd.DataFrame:
        """Spearman correlation (cell types x cell types) of the cell-type shares over the sections
        (df_ct_sec: calc_df_celltype_by_section)."""
        M = df_ct_sec.pivot_table(index="section", columns="cell_type", values="share")
        R = M.corr(method="spearman")
        R.index.name, R.columns.name = "cell_type_1", "cell_type_2"
        return R

    def calc_df_cooccurrence(self, R: pd.DataFrame, min_rho: float = 0.6, fname: Optional[str] = None,
                             force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cell-type pairs with Spearman rho > min_rho (the paper's threshold) from calc_df_celltype_correlation,
        sorted by rho. fname: cache, None = not written."""
        def _compute():
            pairs = R.where(np.triu(np.ones(R.shape, bool), 1)).stack().rename("rho").reset_index()
            pairs.columns = ["cell_type_1", "cell_type_2", "rho"]
            return pairs[pairs.rho > min_rho].sort_values("rho", ascending=False).reset_index(drop=True)
        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    def plot_cooccurrence(self, R: pd.DataFrame, figsize=(14, 12), fname: Optional[str] = None, force: bool = False,
                          verbose: bool = False):
        """Heatmap of the cell-type Spearman correlation over sections (lineage order)."""
        order = self.celltype_order(R.index)
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(R.loc[order, order].values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.03).set_label(f"Spearman rho over {R.attrs.get('n_sections', 'the')} sections")
        ax.set_title("Co-occurrence of cell types across sections")
        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])
        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def calc_df_region_summary(self, adata, df_share: pd.DataFrame, names: Dict[str, str],
                               fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """One row per region: niche, n_spots, n_sections, low / high NI spots, technique mix, median UMIs, the three
        cell types of highest mean share and of highest enrichment (mean share / overall mean share)."""
        def _compute():
            obs = adata.obs
            df_m = df_share.groupby(obs.region_label.values, observed=True).mean()
            df_e = df_m / df_share.mean(0)
            rows = []
            for r, sub in obs.groupby("region_label", observed=True):
                tech = sub.technique.value_counts(normalize=True)
                rows.append(dict(region_label=r, niche=names.get(r, ""), n_spots=len(sub), n_sections=sub.section.nunique(),
                                 n_low_NI=int((sub.ni_status == "low").sum()), n_high_NI=int((sub.ni_status == "high").sum()),
                                 techniques=", ".join(f"{t} {v:.0%}" for t, v in tech.items() if v > 0.02),
                                 median_UMIs=float(sub.total_counts.median()),
                                 top_share=", ".join(f"{c} ({v:.0%})" for c, v in df_m.loc[r].sort_values(ascending=False)[:3].items()),
                                 top_enriched=", ".join(f"{c} ({v:.1f}x)" for c, v in df_e.loc[r].sort_values(ascending=False)[:3].items())))
            return pd.DataFrame(rows)
        return self._cached_any(fname, _compute, force=force, verbose=verbose)

    # =================================================================================================================
    # 9. nerves: spots, invasion, juxta- (1-2) and peri-neural (3-5) rings (paper Figures 4, 5, S5A, 6A)
    # =================================================================================================================
    def nerve_marker_score(self, adata, genes: Optional[Sequence[str]] = None) -> np.ndarray:
        """Mean z-score (within each section) of log1p(CP10k) of the nerve genes (PDAC_NERVE_GENES)."""
        found, Y = self.lognorm_genes(list(genes or PDAC_NERVE_GENES), adata)
        sec = adata.obs[self.sample_col].astype(str).values
        Z = np.zeros_like(Y)
        for s in np.unique(sec):
            m = sec == s
            mu, sd = Y[m].mean(0), Y[m].std(0)
            Z[m] = (Y[m] - mu) / np.where(sd > 0, sd, 1)
        return Z.mean(1)

    def _section_graphs(self, adata):
        """Per section: (indices into adata, spot adjacency of the 6 Visium neighbours)."""
        sec = adata.obs[self.sample_col].astype(str).values
        xy_all = np.asarray(adata.obsm["spatial"], float)
        out = {}
        for s in pd.unique(sec):
            idx = np.flatnonzero(sec == s)
            A, _ = self.spot_neighbor_graph(xy_all[idx], radius_factor=self.params["coherence_radius_factor"])
            out[s] = (idx, A)
        return out

    def call_nerves(self, adata, df_abund: pd.DataFrame, fname: Optional[str] = None, force: bool = False,
                    verbose: bool = False) -> pd.DataFrame:
        """Nerve spots, nerve bundles, invasion status and rings (written to obs and returned per spot):
          nerve spot  : Schwann share (S01-S03 / spot total) >= nerve_min_schwann_share and nerve-marker score
                        (nerve_marker_score) >= nerve_min_marker_z; connected bundles of >= nerve_min_size spots
                        (6-neighbour graph) are kept -> obs['nerve_id'] '<section>_n<k>';
          invasion    : malignant share (D02-D09) averaged over the bundle and its first ring >= nerve_invaded_min_malignant
                        -> 'invaded', else 'non-invaded' (proxy of the pathologists' call on the H&E);
          rings       : graph distance (spots) to the nearest bundle, 1 .. nerve_n_rings -> obs['nerve_ring'] (0 = nerve),
                        obs['nerve_zone'] 'nerve' | 'juxta-neural' (1-2) | 'peri-neural' (3-5), obs['ring_status']
                        = invasion status of the nearest bundle.
        files['nerve_overrides'] (section, barcode, nerve 0/1, invaded 0/1/''; e.g. from the H&E) replaces the calls
        of the listed spots. Cached table: barcode, section, schwann_share, nerve_score, malignant_share, nerve_id,
        nerve_status, nerve_ring, nerve_zone, ring_status."""
        from scipy.sparse.csgraph import connected_components

        def _compute():
            P = self.params
            df_s = self.shares(df_abund.loc[adata.obs_names])
            sch = df_s[[c for c in df_s.columns if c.startswith("S0") or c == "Schwann"]].sum(1).values
            mal = df_s[[c for c in df_s.columns if re.match(r"D0[2-9]_", c)]].sum(1).values
            score = self.nerve_marker_score(adata)
            is_nerve = (sch >= P["nerve_min_schwann_share"]) & (score >= P["nerve_min_marker_z"])
            inv_override = np.full(adata.n_obs, np.nan)
            path = self.file_path("nerve_overrides")
            if path.exists():
                df_o = self.read_tsv("nerve_overrides")
                key = df_o.section.astype(str) + "_" + df_o.barcode.astype(str)
                pos = pd.Series(np.arange(adata.n_obs), index=adata.obs_names).reindex(key).values
                okp = ~np.isnan(pos)
                pos = pos[okp].astype(int)
                is_nerve[pos] = df_o.nerve.values[okp].astype(bool)
                inv = pd.to_numeric(df_o["invaded"], errors="coerce").values[okp]
                inv_override[pos] = inv
                if verbose:
                    print(f"nerve overrides: {len(pos)} spots from {path}")
            nerve_id = np.full(adata.n_obs, "", dtype=object)
            status = np.full(adata.n_obs, "", dtype=object)
            ring = np.full(adata.n_obs, np.nan)
            ring_status = np.full(adata.n_obs, "", dtype=object)
            for s, (idx, A) in self._section_graphs(adata).items():
                nv = is_nerve[idx]
                if not nv.any():
                    continue
                sub = A[nv][:, nv]
                k, comp = connected_components(sub, directed=False)
                nidx = idx[nv]
                sizes = np.bincount(comp, minlength=k)
                keep = sizes[comp] >= P["nerve_min_size"]
                local = np.full(len(idx), -1)
                local[np.flatnonzero(nv)[keep]] = comp[keep]
                # rings by BFS from all kept bundle spots; nearest bundle label propagated
                dist = np.full(len(idx), np.inf)
                owner = np.full(len(idx), -1)
                frontier = np.flatnonzero(local >= 0)
                dist[frontier] = 0
                owner[frontier] = local[frontier]
                A_csr = A.tocsr()
                for d in range(1, P["nerve_n_rings"] + 1):
                    nxt = []
                    for i in frontier:
                        for j in A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i + 1]]:
                            if dist[j] == np.inf:
                                dist[j] = d
                                owner[j] = owner[i]
                                nxt.append(j)
                    frontier = np.array(nxt, dtype=int)
                    if not len(frontier):
                        break
                # invasion per bundle: malignant share over the bundle and its first ring
                comp_status = {}
                for c in np.unique(local[local >= 0]):
                    m_ = (owner == c) & (dist <= 1)
                    ov = inv_override[idx[local == c]]
                    if np.isfinite(ov).any():
                        comp_status[c] = "invaded" if np.nanmax(ov) > 0 else "non-invaded"
                    else:
                        comp_status[c] = ("invaded" if mal[idx[m_]].mean() >= P["nerve_invaded_min_malignant"]
                                          else "non-invaded")
                for i_loc, gi in enumerate(idx):
                    if np.isfinite(dist[i_loc]):
                        ring[gi] = dist[i_loc]
                        ring_status[gi] = comp_status[owner[i_loc]]
                        if dist[i_loc] == 0:
                            nerve_id[gi] = f"{s}_n{owner[i_loc]}"
                            status[gi] = comp_status[owner[i_loc]]
            zone = np.where(ring == 0, "nerve", np.where(ring <= 2, "juxta-neural", np.where(ring <= 5, "peri-neural", "")))
            zone = np.where(np.isnan(ring), "", zone)
            return pd.DataFrame({"barcode": adata.obs_names, "section": adata.obs[self.sample_col].astype(str).values,
                                 "schwann_share": sch.round(4), "nerve_score": score.round(3),
                                 "malignant_share": mal.round(4), "nerve_id": nerve_id, "nerve_status": status,
                                 "nerve_ring": ring, "nerve_zone": zone, "ring_status": ring_status})

        df = self._cached_any(fname, _compute, force=force, verbose=verbose)
        df = df.set_index("barcode").reindex(adata.obs_names)
        for c in ("nerve_id", "nerve_status", "nerve_zone", "ring_status"):
            adata.obs[c] = df[c].fillna("").astype(str).values
        adata.obs["nerve_ring"] = df["nerve_ring"].values
        return df.reset_index(names="barcode")

    # =================================================================================================================
    # TGF-beta / HGF -> Schwann subsets (paper Figure 6)
    # =================================================================================================================
    TGFB_LIGANDS = ("TGFB1^", "TGFB2^", "TGFB3^", "HGF^")

    def _tgfb_rows(self, df_ct: pd.DataFrame) -> pd.DataFrame:
        tgf = [i for i in df_ct.interaction.unique() if i.startswith(self.TGFB_LIGANDS)]
        return df_ct[df_ct.interaction.isin(tgf) & df_ct.receiver.str.startswith("S0")]

    def calc_df_tgfb_to_schwann(self, df_ct: pd.DataFrame, top: int = 5, fname: Optional[str] = None,
                                force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """TGF-beta 1-3 / HGF pairs received by the Schwann subsets (df_ct: calc_df_celltype_lr): the top senders
        (score) per interaction and receiver. fname: cache, None = not written."""
        return self._cached_any(fname, lambda: self._tgfb_rows(df_ct).sort_values("score", ascending=False)
                                .groupby(["interaction", "receiver"]).head(top), force=force, verbose=verbose)

    def plot_tgfb_to_schwann(self, df_ct: pd.DataFrame, n_senders: int = 15, figsize=(7, 6),
                             fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """Heatmap senders x Schwann subsets of the summed TGF-beta / HGF pair scores (None when no such pair)."""
        d = self._tgfb_rows(df_ct)
        if not len(d):
            print("no TGF-beta / HGF pair received by Schwann subsets")
            return None
        M = d.pivot_table(index="sender", columns="receiver", values="score", aggfunc="sum")
        M = M.loc[M.sum(1).sort_values(ascending=False).index[:n_senders]]
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M.values, cmap="magma", aspect="auto")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(M.columns, rotation=30, ha="right")
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(M.index)
        fig.colorbar(im, ax=ax, fraction=0.04).set_label("sum of TGF-beta / HGF pair scores")
        ax.set_title("Senders of TGF-beta / HGF to Schwann subsets")
        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])
        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def calc_df_nerve_table(self, df_nerve: pd.DataFrame, fname: Optional[str] = None, force: bool = False,
                            verbose: bool = False) -> pd.DataFrame:
        """One row per nerve bundle: section, n_spots, status, mean Schwann / malignant share, ring-1 malignant share,
        and the section's NI status. fname: cache, None = not written."""
        return self._cached_any(fname, lambda: self._nerve_table(df_nerve), force=force, verbose=verbose)

    def _nerve_table(self, df_nerve: pd.DataFrame) -> pd.DataFrame:
        d = df_nerve[df_nerve.nerve_id != ""]
        df_sec = self.read_sections_table().set_index("section_id")
        out = d.groupby("nerve_id").agg(section=("section", "first"), n_spots=("barcode", "size"),
                                        status=("nerve_status", "first"), schwann_share=("schwann_share", "mean"),
                                        malignant_share=("malignant_share", "mean")).reset_index()
        out["ni_status"] = out.section.map(df_sec["ni_status"].astype(str))
        return out.sort_values(["section", "n_spots"], ascending=[True, False]).round(3)

    def calc_df_ring_profile(self, adata, df_values: pd.DataFrame, max_ring: Optional[int] = None,
                             fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Mean of every column of df_values (spots x features: cell-type shares or niche indicators) per ring
        (0 = nerve, 1 .. 5) and ring_status (invaded / non-invaded). Long table: ring_status, ring, feature, mean,
        n_spots (Figure 4B analogue). fname: cache, None = not written."""
        return self._cached_any(fname, lambda: self._ring_profile(adata, df_values, max_ring), force=force,
                                verbose=verbose)

    def _ring_profile(self, adata, df_values: pd.DataFrame, max_ring: Optional[int] = None) -> pd.DataFrame:
        max_ring = max_ring or self.params["nerve_n_rings"]
        r = adata.obs["nerve_ring"].values
        st = adata.obs["ring_status"].astype(str).values
        ok = np.isfinite(r) & (r <= max_ring) & (st != "")
        D = df_values.loc[adata.obs_names[ok]].reset_index(drop=True)
        D.columns.name = "feature"
        keys = [pd.Series(st[ok], name="ring_status"), pd.Series(r[ok].astype(int), name="ring")]
        M = D.groupby(keys).mean()
        n = D.groupby(keys).size().rename("n_spots")
        df = M.stack().rename("mean").reset_index()
        return df.merge(n.reset_index(), on=["ring_status", "ring"])

    def calc_df_juxta_enrichment(self, adata, df_share: pd.DataFrame, zone: str = "juxta-neural",
                                 fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cell types in the juxta-neural spots (rings 1-2) of invaded vs non-invaded nerves: means, log2 ratio,
        two-sided Mann-Whitney p and BH FDR over spots (Figure 4A analogue; spots are not independent: rank by effect).
        attrs: n_invaded, n_non_invaded. fname: cache, None = not written."""
        return self._cached_any(fname, lambda: self._juxta_enrichment(adata, df_share, zone), force=force,
                                verbose=verbose)

    def plot_juxta_enrichment(self, df_juxta: pd.DataFrame, max_fdr: float = 0.05, fname: Optional[str] = None,
                              force: bool = False, verbose: bool = False):
        """Bars of log2 (invaded / non-invaded) of the cell types with FDR < max_fdr (Figure 4A analogue)."""
        d = df_juxta[df_juxta.fdr < max_fdr]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(d))))
        ax.barh(range(len(d)), d.log2_ratio, color=["#E8504C" if v > 0 else "#4C9BE8" for v in d.log2_ratio])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(d.cell_type)
        ax.invert_yaxis()
        ax.axvline(0, color="#888888", lw=0.8)
        ax.set_xlabel("log2 (invaded / non-invaded), juxta-neural rings 1-2")
        ax.set_title(f"Juxta-neural cell types (Figure 4A analogue; FDR < {max_fdr} over spots)")
        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])
        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def plot_ring_heatmap(self, df_ring: pd.DataFrame, title: str = "", figsize=(14, 5), fname: Optional[str] = None,
                          force: bool = False, verbose: bool = False):
        """Heatmap of a calc_df_ring_profile table: features (rows) x (ring_status, ring) columns, mean share."""
        M = df_ring.pivot_table(index="feature", columns=["ring_status", "ring"], values="mean") if len(df_ring) else pd.DataFrame()
        if M.empty:
            M = pd.DataFrame(np.zeros((1, 1)), index=["no nerves called"], columns=pd.MultiIndex.from_tuples([("", 0)]))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(M.values, cmap="magma", aspect="auto")
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels([f"{s}\nring {r}" for s, r in M.columns], fontsize=7)
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(M.index)
        fig.colorbar(im, ax=ax, fraction=0.03).set_label("mean share")
        ax.set_title(title)
        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])
        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def calc_df_schwann_by_distance(self, adata, df_abund: pd.DataFrame, df_share: pd.DataFrame, n_rings: int = 5,
                                    fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Nerve spots (ring 0): mean share of each Schwann subset (S0*, renormalised to the Schwann total) by radius
        (spots) to the nearest malignant niche, plus n_spots (Figure 6A analogue). Sets obs['dist_malignant']
        (always computed: later cells use it). Index = radius. fname: cache, None = not written."""
        dist = self.calc_df_distance_to_malignant(adata, df_abund, n_rings=n_rings)
        adata.obs["dist_malignant"] = dist

        def _compute():
            schw = [c for c in df_share.columns if c.startswith("S0")]
            nv = adata.obs.nerve_ring.values == 0
            S = df_share.loc[nv, schw]
            S = S.div(S.sum(1).replace(0, np.nan), axis=0).fillna(0)
            df = S.groupby(np.asarray(dist)[nv]).mean()
            df["n_spots"] = pd.Series(np.asarray(dist)[nv]).value_counts().sort_index().values
            return df.reset_index(names="radius")
        return self._cached_any(fname, _compute, force=force, verbose=verbose).set_index("radius")

    def plot_schwann_distance(self, df_sd: pd.DataFrame, figsize=(7, 4.5), fname: Optional[str] = None,
                              force: bool = False, verbose: bool = False):
        """Share of each Schwann subset vs radius to the nearest malignant niche (Figure 6A analogue)."""
        schw = [c for c in df_sd.columns if c.startswith("S0")]
        col = self.celltype_colors(schw)
        fig, ax = plt.subplots(figsize=figsize)
        for c in schw:
            ax.plot(df_sd.index, df_sd[c], marker="o", color=col[c], label=c)
        ax.set_xlabel("radius (spots) to the nearest malignant niche")
        ax.set_ylabel("share of Schwann cells")
        ax.legend(frameon=False)
        ax.set_title("Schwann subsets vs distance to tumour (Figure 6A analogue)")
        self.style_figure(fig, self.params["font_color"], self.params["bg_color"], self.params["fontsize_scale"])
        self._save_figure(fig, fname, force=force, verbose=verbose)
        return fig

    def _juxta_enrichment(self, adata, df_share: pd.DataFrame, zone: str) -> pd.DataFrame:
        from scipy.stats import mannwhitneyu
        z = adata.obs["nerve_zone"].astype(str).values == zone
        st = adata.obs["ring_status"].astype(str).values
        a_inv = df_share.loc[adata.obs_names[z & (st == "invaded")]]
        a_non = df_share.loc[adata.obs_names[z & (st == "non-invaded")]]
        rows = []
        for c in df_share.columns:
            x, y = a_inv[c].values, a_non[c].values
            if len(x) < 5 or len(y) < 5:
                continue
            p = mannwhitneyu(x, y).pvalue if np.ptp(np.r_[x, y]) > 0 else 1.0
            rows.append(dict(cell_type=c, lineage=self.lineage_of_celltype(c), mean_invaded=x.mean(),
                             mean_non_invaded=y.mean(),
                             log2_ratio=np.log2((x.mean() + 1e-4) / (y.mean() + 1e-4)), p=p))
        df = pd.DataFrame(rows, columns=["cell_type", "lineage", "mean_invaded", "mean_non_invaded", "log2_ratio", "p"])
        df["fdr"] = self._bh(df.p.values) if len(df) else []
        df.attrs.update(n_invaded=len(a_inv), n_non_invaded=len(a_non))
        return df.sort_values("log2_ratio", ascending=False).round(6)

    def plot_ring_profile(self, df_ring: pd.DataFrame, features: Sequence[str], ncols: int = 4, panel_size=(3.6, 3),
                          ylabel: str = "mean share", title: str = "", fname: Optional[str] = None,
                          font_color: Optional[str] = None, bg_color: Optional[str] = None,
                          fontsize_scale: Optional[float] = None, force: bool = False, verbose: bool = False):
        """One panel per feature: mean vs ring (5 -> 0 = nerve), invaded (dashed) vs non-invaded (solid)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        features = [f for f in features if f in set(df_ring.feature)]
        if not features:
            print("no nerve rings (no nerve called with the current thresholds): nothing to plot")
            return None
        nrows = math.ceil(len(features) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
        for ax in axes.ravel()[len(features):]:
            ax.axis("off")
        sty = {"invaded": dict(ls="--", color="#E8504C"), "non-invaded": dict(ls="-", color="#4C9BE8")}
        for ax, f in zip(axes.ravel(), features):
            d = df_ring[df_ring.feature == f]
            for st, dd in d.groupby("ring_status"):
                dd = dd.sort_values("ring", ascending=False)
                ax.plot(dd.ring, dd["mean"], marker="o", ms=4, label=st, **sty.get(st, {}))
            ax.invert_xaxis()
            ax.set_xticks(range(0, int(d.ring.max()) + 1))
            ax.set_title(f, fontsize=9)
            ax.set_xlabel("ring (0 = nerve)", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
        axes.ravel()[0].legend(frameon=False, fontsize=8)
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def calc_df_distance_to_malignant(self, adata, df_abund: pd.DataFrame, n_rings: int = 5,
                                      malignant_key: str = "niche", malignant_prefix: str = "N0") -> np.ndarray:
        """Graph distance of every spot to the nearest malignant-niche spot (niche names containing 'Malignant'),
        capped at n_rings (Figure 6A: 'radius 1-5', farther spots grouped into 5)."""
        lab = adata.obs[malignant_key].astype(str).values
        is_mal = np.char.find(lab.astype(str), "Malignant") >= 0
        out = np.full(adata.n_obs, float(n_rings))
        for s, (idx, A) in self._section_graphs(adata).items():
            A = A.tocsr()
            dist = np.full(len(idx), np.inf)
            frontier = np.flatnonzero(is_mal[idx])
            dist[frontier] = 0
            for d in range(1, n_rings + 1):
                nxt = []
                for i in frontier:
                    for j in A.indices[A.indptr[i]:A.indptr[i + 1]]:
                        if dist[j] == np.inf:
                            dist[j] = d
                            nxt.append(j)
                frontier = np.array(nxt, dtype=int)
            out[idx] = np.minimum(dist, n_rings)
        return out


    def plot_region_celltype_heatmap(self, adata, df_share: pd.DataFrame, key: str = "region_label",
                                     names: Optional[Dict[str, str]] = None, figsize=(18, 9), fname: Optional[str] = None,
                                     font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                     fontsize_scale: Optional[float] = None, force: bool = False,
                                     verbose: bool = False) -> pd.DataFrame:
        """Groups (rows) x cell types (columns, by lineage): log2 enrichment (group mean share / mean over all spots;
        clipped at 1/16). Returns the enrichment table (Figure 1F analogue)."""
        font_color = self.get_param("font_color", font_color)
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        df_a = df_share.loc[adata.obs_names]
        df_a = df_a[self.celltype_order(df_a.columns)]
        df_enr = df_a.groupby(adata.obs[key].astype(str).values).mean() / df_a.mean(0).replace(0, np.nan)
        df_enr = df_enr.fillna(0)
        L = np.log2(df_enr.clip(lower=1 / 16))
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(L.values, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_xticks(range(L.shape[1]))
        ax.set_xticklabels(L.columns, rotation=80, ha="right")
        col = self.celltype_colors(L.columns)
        for t, c in zip(ax.get_xticklabels(), L.columns):
            t.set_color(col.get(c, "#888888"))
        ax.set_yticks(range(L.shape[0]))
        ax.set_yticklabels([f"{r} {names.get(r, '')}" if names else r for r in L.index])
        fig.colorbar(im, ax=ax, fraction=0.03).set_label(f"log2 enrichment ({key} / all spots)")
        ax.set_title(f"Cell types in each {key} ({df_share.attrs.get('source', '')} shares)")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return df_enr


    # =================================================================================================================
    # 10. signatures by gene symbol (LIANA cell-type split) and subregion scores on the reference
    # =================================================================================================================
    def calc_df_mean_signatures(self, adata_ref_dec, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Mean raw counts per cell of every gene in each cell type of the deconvolution reference (genes x cell types,
        index = Ensembl IDs, column 'symbol'). Cached in files['mean_signatures']."""
        def _compute():
            a = adata_ref_dec
            lab = a.obs[self.params["celltype_col"]].astype(str).values
            cats = sorted(set(lab))
            G = sp.csr_matrix((np.ones(len(lab)), (np.searchsorted(cats, lab), np.arange(len(lab)))))
            M = np.asarray((G @ a.X).todense()) / np.asarray(G.sum(1))
            df = pd.DataFrame(M.T, index=a.var_names, columns=cats).round(6)
            df.insert(0, "symbol", a.var[self.symbol_col].astype(str).values)
            return df.reset_index(names="gene_id")
        return self._cached_table("mean_signatures", _compute, force=force, verbose=verbose).set_index("gene_id")

    def reference_signatures_symbols(self, adata_ref_dec=None, summary: str = "means", source: Optional[str] = None
                                     ) -> pd.DataFrame:
        """Signatures (genes x cell types) indexed by gene symbol: the cell2location reference model (files['signatures'],
        varm '<summary>_per_cluster_mu_fg') when it exists (or source='c2l'), else the mean counts per cell type
        (calc_df_mean_signatures; needs adata_ref_dec). attrs['source'] = 'c2l' | 'mean_counts'."""
        path = self.file_path("signatures")
        if source == "c2l" or (source is None and path.exists()):
            a = ad.read_h5ad(path, backed="r")
            df_sig, _ = self.get_reference_signatures(a, prefer=(summary,))
            df_sig.index = a.var.loc[df_sig.index, self.symbol_col].astype(str).values
            a.file.close()
            src = "c2l"
        else:
            if adata_ref_dec is None:
                raise ValueError("no cell2location reference model: pass adata_ref_dec for the mean-count signatures")
            df = self.calc_df_mean_signatures(adata_ref_dec)
            df_sig = df.drop(columns="symbol")
            df_sig.index = df["symbol"].astype(str).values
            src = "mean_counts"
        out = df_sig[~df_sig.index.duplicated()]
        out = out[self.celltype_order(out.columns)]
        out.attrs["source"] = src
        return out

    def calc_df_subregion_ref_score(self, adata_ref, df_gene_sets: pd.DataFrame, df_sub_table: Optional[pd.DataFrame] = None,
                                    status: Optional[str] = "supported", n_genes: int = 25, min_genes: int = 3,
                                    force: bool = False, verbose: bool = False
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """Each subregion's signature on the reference (symbols as var_names): top n_genes subregion_signature genes
        found in the reference (>= min_genes; only subregions whose status starts with `status`); cell score = mean
        per-gene z-score of log1p(CP10k) (library size obs['n_counts']). Returns (df_ref_score subregions x cell types,
        df_cell_score, genes_used). Cached in files 'subregion_ref_score' / 'subregion_ref_genes'."""
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
        sym = set(adata_ref.var_names.astype(str))
        genes_used = {}
        for s_id, g in d.groupby("sub_region_id", sort=False):
            gl = [x for x in g.gene if x in sym][:n_genes]
            if len(gl) >= min_genes:
                genes_used[s_id] = gl
        order = [s_id for s_id in df_gene_sets.sub_region_id.unique() if s_id in genes_used]
        genes_used = {k: genes_used[k] for k in order}
        df_cell_score = self._calc_df_cell_score(adata_ref, genes_used)
        ct = adata_ref.obs[groupby].astype(str).values
        df_ref_score = df_cell_score.groupby(ct).mean().T.round(6)
        df_ref_score = df_ref_score[self.celltype_order(df_ref_score.columns)]
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
        return df_ref_score.apply(lambda r: ", ".join(f"{c} ({v:.2f})" for c, v in r.sort_values(ascending=False)
                                                      .head(n).items()), axis=1).rename("ref_cell_types")

    def add_ref_cell_types(self, df_sub_table: pd.DataFrame, df_ref_score: pd.DataFrame, n: int = 3,
                           force: bool = False, verbose: bool = False) -> pd.DataFrame:
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
        ax.set_title("Subregion signatures on the sc / sn reference")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig


    # =================================================================================================================
    # 11. LIANA bivariate per section (liana densifies the spot kernel: 81,846^2 x 8 B = 53 GB; per section <= 0.2 GB)
    # =================================================================================================================
    def liana_bivariate(self, adata=None, resource_name: Optional[str] = None, local_name: Optional[str] = None,
                        global_name: Optional[str] = None, n_perms: Optional[int] = None,
                        nz_prop: Optional[float] = None, fname: Optional[str] = "liana_bivariate",
                        fname_global: Optional[str] = "liana_global", force: bool = False,
                        verbose: bool = False, **kw) -> Tuple[pd.DataFrame, ad.AnnData]:
        """SingleCell.liana_bivariate run section by section (same signature, cache and outputs).
        The spatial kernel is block-diagonal (spots of different sections are never neighbours), so the local scores
        equal those of a joint run; local permutations shuffle spots within a section. The LR pairs are fixed once:
        all subunits of ligand and receptor detected in >= nz_prop of all spots (min over subunits). Global score:
        spot-weighted mean of the per-section Moran's R; p = Stouffer combination (weights sqrt(n spots)) of the
        per-section permutation p-values; var['n_sections_sig'] = sections with R > 0 and p < liana_fdr.
        Per-section global table: <fname_global stem>_by_section.tsv in root_result."""
        import liana as li
        from scipy.stats import norm
        a = self._check_adata(adata)
        path = self._resolve_table(fname)
        res_name = self.get_param("liana_resource", resource_name)
        local_name = self.get_param("liana_local", local_name)
        g = self.get_param("liana_global", global_name)
        n_perms = self.get_param("liana_n_perms", n_perms)
        nz_prop = self.get_param("liana_nz_prop", nz_prop)
        computed = False
        if path is not None and path.exists() and not force:
            adata_lr = ad.read_h5ad(path)
            if verbose:
                print(f"Reading {path}")
        else:
            if "spatial_connectivities" not in a.obsp:
                self.spatial_weights(a)
            res = li.rs.select_resource(res_name)
            Xb = a.X if sp.issparse(a.X) else sp.csr_matrix(a.X)
            frac = pd.Series(np.asarray((Xb > 0).mean(0)).ravel(), index=a.var_names)
            ent_prop = lambda e: min(frac.get(x, 0.0) for x in str(e).split("_"))
            pl, pr = res.ligand.map(ent_prop), res.receptor.map(ent_prop)
            res_keep = res[(pl >= nz_prop) & (pr >= nz_prop)].drop_duplicates().reset_index(drop=True)
            secs = list(pd.unique(a.obs[self.sample_col].astype(str)))
            parts, glob = [], []
            for s in secs:
                sub = a[(a.obs[self.sample_col].astype(str) == s).values]
                lr = li.mt.bivariate(sub, resource=res_keep, local_name=local_name, global_name=g, n_perms=n_perms,
                                     mask_negatives=False, add_categories=True, nz_prop=0, use_raw=False, verbose=False, **kw)
                parts.append(lr)
                v = lr.var[[g, f"{g}_pvals"]].copy()
                v["section"], v["n_spots"] = s, sub.n_obs
                glob.append(v.reset_index(names="interaction"))
                if verbose:
                    print(f"{s}: {sub.n_obs:,} spots, {lr.n_vars} pairs")
            inter = list(dict.fromkeys(i for p in parts for i in p.var_names))
            X, CATS, PV = [], [], []
            for p in parts:
                idx = pd.Index(p.var_names).get_indexer(inter)
                ok = idx >= 0
                def _take(M, fill):
                    M = M.toarray() if sp.issparse(M) else np.asarray(M)
                    out = np.full((p.n_obs, len(inter)), fill, dtype=np.float32)
                    out[:, ok] = M[:, idx[ok]]
                    return out
                X.append(_take(p.X, 0.0)); CATS.append(_take(p.layers["cats"], 0.0)); PV.append(_take(p.layers["pvals"], 1.0))
            df_g = pd.concat(glob, ignore_index=True)
            W_ = df_g.pivot_table(index="interaction", columns="section", values="n_spots").reindex(inter)
            R_ = df_g.pivot_table(index="interaction", columns="section", values=g).reindex(inter)
            P_ = df_g.pivot_table(index="interaction", columns="section", values=f"{g}_pvals").reindex(inter)
            w = W_.fillna(0).values
            r_pool = np.nansum(R_.values * w, 1) / np.maximum(w.sum(1), 1)
            pmin = 1.0 / (n_perms + 1)
            z = norm.isf(np.clip(P_.values, pmin, 1 - 1e-12))
            sw = np.sqrt(w)
            Z = np.nansum(z * sw, 1) / np.sqrt(np.maximum(np.nansum(sw ** 2 * ~np.isnan(z), 1), 1e-12))
            first = {}
            for p in parts:
                for i, row in p.var.iterrows():
                    first.setdefault(i, row)
            var = pd.DataFrame([first[i] for i in inter], index=pd.Index(inter, name="interaction"))
            var[g] = r_pool
            var[f"{g}_pvals"] = norm.sf(Z)
            var["n_sections"] = (~np.isnan(R_.values)).sum(1)
            var["n_sections_sig"] = ((R_.values > 0) & (P_.values < self.params["liana_fdr"])).sum(1)
            var["ligand_props"] = var.ligand.map(ent_prop).values
            var["receptor_props"] = var.receptor.map(ent_prop).values
            obs_names = np.concatenate([p.obs_names.values for p in parts])
            adata_lr = ad.AnnData(X=np.vstack(X), obs=pd.DataFrame(index=obs_names), var=var.drop(columns=["mean", "std"], errors="ignore"))
            adata_lr.layers["cats"] = sp.csr_matrix(np.vstack(CATS))
            adata_lr.layers["pvals"] = sp.csr_matrix(1 - np.vstack(PV))          # stored as 1 - p (sparse); restored below
            adata_lr.uns["pvals_stored_as"] = "1 - p"
            adata_lr = adata_lr[a.obs_names].copy()
            adata_lr.obs = a.obs.copy()
            adata_lr.obsm["spatial"] = a.obsm["spatial"]
            adata_lr.uns["spatial"] = a.uns.get("spatial", {})
            adata_lr.uns["liana_bivariate"] = dict(resource=res_name, by_section=True, n_pairs=len(inter), n_perms=n_perms)
            computed = True
            if path is not None:
                adata_lr.write_h5ad(path, compression="gzip")
                df_g.to_csv(path.with_name(path.stem + "_by_section.tsv"), sep="\t", index=False)
                if verbose:
                    print(f"Saving {path}")
        if adata_lr.uns.get("pvals_stored_as") == "1 - p":
            P = adata_lr.layers["pvals"]
            adata_lr.layers["pvals"] = 1 - (P.toarray() if sp.issparse(P) else np.asarray(P))
            adata_lr.uns["pvals_stored_as"] = "p"
        df_inter = adata_lr.var.sort_values(g, ascending=False).copy()
        df_inter[f"{g}_fdr"] = self._bh(df_inter[f"{g}_pvals"].values)
        df_inter.index.name = "interaction"
        if fname_global is not None and fname_global in self.files and \
                (computed or force or not self.file_path(fname_global).exists()):
            self.write_tsv(df_inter.reset_index(), fname_global, verbose=verbose)
        return df_inter, adata_lr

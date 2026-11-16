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

# 24 PDAC / neural-invasion genes shown on the reference UMAP and on the Visium sections (symbol -> role)
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
        region_info=("data", "pdac_regions.tsv"),                     # curated region table (editable)
        nerve_overrides=("data", "pdac_nerve_annotation.tsv"),       # optional manual nerve calls (H&E)
        mean_signatures=("result", "pdac_reference_mean_counts_per_celltype.tsv.gz"),
        subregion_descriptions=("data", "pdac_subregion_descriptions.tsv"),
    ),
}


class SingleCellPDAC(SingleCell):
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
        """Cached table given as a key of self.files or as a file name with an extension in root_result."""
        if fname is None:
            return compute()
        if fname in self.files:
            return self._cached_table(fname, compute, force=force, verbose=verbose)
        return self._cached_frames({"t": fname}, lambda: {"t": compute()}, force=force, verbose=verbose)["t"]

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
                                 bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
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
        fig.subplots_adjust(bottom=1.6 / (panel_size * nrows + 1.6) + 0.05, wspace=0.02, hspace=0.12)
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=False)
        self._savefig(fig, fname, dpi=self.params["dpi_maps"], force=force, verbose=verbose)
        return fig

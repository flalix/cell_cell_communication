#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single_cell_neuro_lib.py - SingleCellNeuro(SingleCell): mouse brain analyses of
Kleshchevnikov et al. "Comprehensive mapping of tissue cell architecture via integrated single cell and
spatial transcriptomics" (bioRxiv 2020; Nat Biotechnol 2022, cell2location), Figure 2.

Data (mouse_brain/data, see notebooks_mouse_brain/mb01_fig2_mouse_brain.ipynb)
    visium_mouse_brain_5sections.h5ad                5 Visium sections, E-MTAB-11114 (14,968 spots)
    snrna/all_cells_20200625.h5ad                     snRNA-seq raw counts, E-MTAB-11115 (40,572 nuclei)
    snrna/snRNA_annotation_..._refined59_*.csv        59 cell types (annotation_1) for 40,532 nuclei
    snrna/mouse_brain_snRNA_with_UMAP_cellxgene.h5ad  paper UMAP + broad_class (Fig 2B)
    snrna/regression_signatures_sc.h5ad               trained NB-regression signatures (59 types)
    c2l_paper_map/sp_with_clusters.h5ad               paper cell2location map, ST8059048 + ST8059051, region_cluster
    slideseq/slideseqv2.h5ad                          squidpy Slide-seqV2 hippocampus puck (Stickels 2021)
    allen_smartseq/                                   Allen mouse cortex + hippocampus Smart-seq (Yao 2020)

Methods (figure panels)
    snRNA     : add_broad_class, filter_snrna, add_paper_annotation, load_umap_obs, plot_umap_broad_subtypes (2B)
    Visium    : load_paper_map, calc_df_regions, plot_regions_he (2C), plot_density_panels (2D-F)
    Slide-seq : recover_counts, signatures_from_regression, plot_slideseq (2H)
    Allen     : allen_select_types, calc_df_allen_layer_prop, allen_signatures, assign_cortical_layers,
                calc_df_c2l_layer_prop, plot_layer_workflow (2I), plot_layer_heatmaps (2J)

Every DataFrame variable starts with 'df_'. All cutoffs / lists / colours are in NEURO_PARAMS.

@created on 2026/11/05
@author: Flavio Lichtenstein
"""

import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

import anndata as ad
import scanpy as sc

from libs.single_cell_lib import DEFAULT_PARAMS, KELLY_COLORS, SingleCell

__version__ = "1.0.0"

# cell2location.plt.plot_spatial default colour order (yellow, orange, blue, green, purple, grey)
DENSITY_COLORS = ("#FFE600", "#FF8000", "#2AA7FF", "#00C060", "#A040FF", "#E0E0E0")

NEURO_PARAMS: Dict[str, Any] = dict(
    # --- snRNA reference columns and QC (paper Methods: genes < 5000, mito fraction < 0.10, scrublet < 0.2)
    celltype_col="annotation_1",
    broad_col="broad_class",
    mt_frac_col="mt_frac",
    snrna_max_genes=5000,
    snrna_max_mt_frac=0.10,
    snrna_max_doublet=None,         # not applied: the published annotation is already post-scrublet (< 0.2);
                                    # Visium spots are multi-cell, so doublet scores do not apply to them
    # ordered prefix rules used when broad_class is missing (first match wins)
    broad_class_rules=(("Ext_Unk_1", "Other"), ("Ext_Unk_2", "Other"), ("Astro", "Astrocytes"), ("Ext", "Excitatory"),
                       ("Inh", "Inhibitory"), ("Oligo", "Oligo_OPC"), ("OPC", "Oligo_OPC"), ("Micro", "Micro"),
                       ("Endo", "Endo")),
    broad_default="Other",
    # display labels: the 59-type annotation has no 'Ext_L4'; Ext_L25 (L2-5 IT) is the L4-enriched type
    celltype_alias={"Ext_L25": "Ext_L25 (L4)"},
    # --- Fig 2D-F, 2H cell types (colours = DENSITY_COLORS in this order)
    fig2D_types=("Oligo_2", "Inh_Meis2_3", "Inh_4", "Ext_Thal_1", "Ext_L23", "Ext_L56"),
    fig2E_types=("Inh_Sst", "Inh_Lamp5", "Inh_Vip"),
    fig2F_types=("Ext_L56", "Ext_L6B", "Ext_L25", "Ext_L5_2", "Ext_L23"),
    fig2H_left=("Ext_Hpc_CA1", "Ext_Hpc_CA2", "Ext_Hpc_CA3", "Ext_Hpc_DG1", "Inh_6"),
    fig2H_right=("Ext_L23", "Ext_L25", "Ext_L5_2", "Ext_L56", "Ext_L6B"),
    density_colors=DENSITY_COLORS,
    density_spot_size=7.0,
    slideseq_spot_size=1.2,
    # --- paper cell2location map (old v0.05 layout)
    map_density_prefix="q05_spot_factors",       # obs columns q05_spot_factors<cell type> = cells per spot
    map_library_regex=r"(ST\d+)",               # uns['spatial'] keys -> sample ids
    # --- anatomical naming of region clusters: cell type -> region label (types not listed are ignored)
    region_celltype_map={
        "Ext_L23": "CTX_L2/3", "Ext_L25": "CTX_L4", "Ext_L5_1": "CTX_L5", "Ext_L5_2": "CTX_L5", "Ext_L5_3": "CTX_L5",
        "Ext_L56": "CTX_L5/6", "Ext_L6": "CTX_L6", "Ext_L6B": "CTX_L6b", "Astro_CTX": "CTX",
        "Ext_ClauPyr": "CLA", "Ext_Pir": "PIR", "Ext_Amy_1": "AMY", "Ext_Amy_2": "AMY", "Astro_AMY": "AMY",
        "Ext_Hpc_CA1": "HPC_CA1", "Ext_Hpc_CA2": "HPC_CA2", "Ext_Hpc_CA3": "HPC_CA3", "Ext_Hpc_DG1": "HPC_DG",
        "Ext_Hpc_DG2": "HPC_DG", "Astro_HPC": "HPC",
        "Ext_Thal_1": "THAL", "Ext_Thal_2": "THAL", "Astro_THAL_lat": "THAL", "Astro_THAL_med": "THAL",
        "Ext_Med": "HAB", "Astro_THAL_hab": "HAB", "Inh_6": "HAB",
        "Inh_Meis2_1": "STR", "Inh_Meis2_2": "STR", "Inh_Meis2_3": "STR", "Inh_Meis2_4": "STR", "Astro_STR": "STR",
        "Astro_HYPO": "HYPO", "Oligo_1": "WM", "Oligo_2": "WM", "Astro_WM": "WM",
    },
    region_label_min_enrich=1.5,    # the labelling type must be at least this enriched in the cluster
    region_label_top_k=3,           # labels voted by the k most enriched mapped types
    region_colors={                 # label -> colour (same palette as mb01_brain_regions.png); unknown labels: Kelly / tab20
        "CTX_L1": "#c6dbef", "CTX_L2/3": "#9ecae1", "CTX_L4": "#4292c6", "CTX_L5": "#2171b5", "CTX_L5/6": "#08519c",
        "CTX_L6b": "#08306b", "PIR": "#9e9ac8", "AMY": "#6a51a3", "HPC_CA1": "#74c476", "HPC_CA3": "#238b45",
        "HPC_DG": "#00441b", "STR": "#e7298a", "THAL": "#fd8d3c", "HAB": "#d94801", "HYPO": "#fdd0a2",
        "WM": "#969696", "edge_LowQ": "#f0f0f0"},
    region_label_override={},       # {region_cluster: label}, set after inspecting calc_df_edge_fraction
    edge_radius_factor=1.2,         # neighbours searched within factor x spot spacing
    edge_min_neighbors=6,           # fewer neighbours -> border spot
    # --- cortical layers for Fig 2I/2J (region label -> layer)
    layer_from_region={"CTX_L1": "L1", "CTX_L2/3": "L2/3", "CTX_L4": "L4", "CTX_L5": "L5", "CTX_L5/6": "L6",
                       "CTX_L6": "L6", "CTX_L6b": "L6"},
    layers=("L1", "L2/3", "L4", "L5", "L6"),
    layer_colors={"L1": "#1f77b4", "L2/3": "#9467bd", "L4": "#ff7f0e", "L5": "#2ca02c", "L6": "#d62728"},
    # --- Allen Smart-seq reference (Yao et al. 2020)
    allen_type_col="cluster_label",
    allen_region_col="region_label",
    allen_layer_col="cortical_layer_label",
    allen_ctx_region="SSp",
    allen_hpc_region="PAR-POST-PRE",
    allen_min_cells=3,              # discard cell types with fewer cells (paper)
    allen_name_replace={"/": "-"},  # h5ad keys cannot contain '/': '166_L2/3 IT CTX' -> '166_L2-3 IT CTX'
    allen_layer_merge={"L6b": "L6"},
    allen_top_min_ssp=15,           # 23 most prevalent SSp-exclusive types: count in SSp > 15, count in HPC = 0
    allen_n_top=23,
    c2l_allen_N=8,
    c2l_allen_A=7,                  # cell types per location (paper)
    c2l_allen_alpha=20,
    # --- Slide-seq mapping (paper: 1 cell / location)
    slideseq_N=1,
    slideseq_A=1,
    slideseq_B=1,
    slideseq_alpha=20,
    slideseq_count_tol=0.1,         # max |recovered - round| accepted when recovering counts
    # --- Fig 3: regional astrocyte subtypes
    astro_col="annotation_1",                 # 10 merged subtypes (paper)
    astro_col_premerge="annotation_1_old",    # 13 clusters before the location-based merge
    astro_order=("Astro_THAL_lat", "Astro_THAL_med", "Astro_HYPO", "Astro_THAL_hab", "Astro_WM", "Astro_AMY",
                 "Astro_HPC", "Astro_STR", "Astro_AMY_CTX", "Astro_CTX"),          # row order of Fig 3B/3D
    # Fig 3B marker groups, read from the published figure (Unc13c appears in THAL_med and CTX, as in the paper)
    fig3B_markers=(("Astro", ("Slc1a3", "Aldoc", "Hepacam")), ("WM", ("Gfap", "Igfbp5", "Cd9")),
                   ("diencephalic", ("Slc6a11", "Erbb4", "Lrig1", "Agt")), ("HYPO", ("Itih3", "Ttll3", "Sparc")),
                   ("THAL_hab", ("Slc38a1", "Angpt1", "Pkp4")), ("THAL_lat", ("Aldh1a1", "Esrrg", "Car10")),
                   ("THAL_med", ("Lgr6", "Unc13c", "Gria1")), ("telencephalic", ("Gldc", "Hapln1", "Slco1c1", "Foxg1")),
                   ("AMY_CTX", ("Mfge8", "Hes5", "St6galnac5")), ("AMY", ("Tenm4", "Gm29683", "Map3k19")),
                   ("HPC", ("Thrsp", "Id3", "Pdgfd")), ("STR", ("Efemp1", "Gfra1", "Thbs4")),
                   ("CTX", ("Zfhx3", "Unc13c", "Chrdl1", "Car8"))),
    astro_colors={"Astro_AMY": "#006FA6", "Astro_AMY_CTX": "#1E6E00", "Astro_CTX": "#D55E00", "Astro_HPC": "#56B4E9",
                  "Astro_HYPO": "#F0E442", "Astro_STR": "#6F0062", "Astro_THAL_hab": "#5A14A5",
                  "Astro_THAL_lat": "#009E73", "Astro_THAL_med": "#A77500", "Astro_WM": "#FF34FF"},  # paper object
    fig3C_types=("Astro_HYPO", "Astro_CTX", "Astro_HPC", "Astro_THAL_lat", "Astro_THAL_hab"),
    fig3E_types=("Astro_HYPO", "Astro_THAL_hab", "Astro_THAL_med", "Astro_THAL_lat"),
    fig3F_markers={"Astro_HYPO": "Ttll3", "Astro_THAL_hab": "Angpt1", "Astro_THAL_med": "Lgr6", "Astro_THAL_lat": "Aldh1a1"},
    fig3F_smooth=True,                         # Fig 3F display: mean over spot + 6 neighbours (rho uses raw spots)
    diencephalon_regions=("THAL", "HAB", "HYPO"),
    fig3E_outlines={"thalamus": ("THAL", "HAB"), "hypothalamus": ("HYPO",)},
    outline_min_spots=15,                      # region outlines: drop connected groups smaller than this
    fig3D_areas=(("hypothalamus", ("HYPO",)), ("thalamus", ("THAL", "HAB")), ("white matter", ("WM",)),
                 ("hippocampus", ("HPC_CA1", "HPC_CA3", "HPC_DG")), ("striatum", ("STR",)), ("amygdala", ("AMY",)),
                 ("piriform cortex", ("PIR",)),
                 ("cortex", ("CTX_L1", "CTX_L2/3", "CTX_L4", "CTX_L5", "CTX_L5/6", "CTX_L6b"))),
    area_colors={"hypothalamus": "#C8A97E", "thalamus": "#A78BCB", "white matter": "#B5651D", "hippocampus": "#7CC46C",
                 "striatum": "#F39C3C", "amygdala": "#A9D2E8", "piriform cortex": "#F7C46C", "cortex": "#3A8FB7"},
    density_cmap_he=("#FFFFFF", "#9E9AC8", "#54278F"),   # white -> purple over H&E (Fig 3E)
    marker_cmap_dark=("#000000", "#1B4F72", "#5DADE2", "#D6EAF8"),   # black -> light blue (Fig 3F style)
    # --- subregions (mccc03): clustering of the spots of each region on gene expression
    subregion_key="sub_region_id",            # obs column, e.g. 'THAL_2'
    subregion_skip=("edge_LowQ",),            # regions not subclustered (artefact)
    # gene sets: region signature (region vs all other regions), subregion signature (subregion vs rest of the region,
    # minus region signature), not present (detected in the rest of the region, absent in the subregion), other (rest)
    region_sig_lfc=1.0, region_sig_padj=0.05, region_sig_min_pct=0.25,
    sub_sig_lfc=0.75, sub_sig_padj=0.05, sub_sig_min_pct=0.20,
    notpres_max_pct=0.05,                     # 'absent': detected in <= 5% of the subregion's spots
    notpres_min_pct_rest=0.30,                # ... and in >= 30% of the rest of the region
    notpres_padj=0.05,
    gene_universe="detected",                 # 'detected' (in >= universe_min_pct of the region's spots) or 'all'
    universe_min_pct=0.01,
    n_genes_show=12,                          # genes listed per set in the summary tables
    gene_display_exclude=r"^Gm\d+$|Rik$",     # predicted genes / RIKEN clones not listed (still counted)
    coherence_min_ratio=1.5,                  # 'supported' subregion: spatial coherence ratio >= this
    sub_min_sig_genes=3,                      # ... and >= this many subregion signature genes
    section_min_frac=0.10,                    # below this fraction in one section -> flagged 'single-section'
    low_depth_ratio=0.6,                      # median UMIs of subregion / region below this -> flagged 'low-depth'
    high_depth_ratio=2.0,                     # ... above this -> 'high-depth' (signature counts inflated)
    border_min_frac=0.25,                     # >= this fraction of neighbours in another region -> 'border with X'
    composition_min_mean=0.05,                # cell types with mean density >= this (cells per spot) are ranked
    composition_top=3,
    plotly_include_js="cdn",                  # write_html: 'cdn' (small files, needs internet to view) or True
)


class SingleCellNeuro(SingleCell):
    """Mouse brain (neuro) analyses of the cell2location paper (Fig 2)."""

    default_params = {**DEFAULT_PARAMS, **NEURO_PARAMS}

    def __init__(self, ROOT0, project: str = "mouse_brain", symbol_col: str = "SYMBOL", sample_col: str = "sample",
                 mt_prefix: str = "mt-", **kw):
        super().__init__(ROOT0, project, symbol_col=symbol_col, sample_col=sample_col, mt_prefix=mt_prefix, **kw)

    def alias(self, ct: str) -> str:
        return self.params["celltype_alias"].get(ct, ct)

    # =====================================================================================================
    # snRNA-seq reference
    # =====================================================================================================
    def broad_class_of(self, ct: str) -> str:
        for prefix, cls in self.params["broad_class_rules"]:
            if str(ct).startswith(prefix):
                return cls
        return self.params["broad_default"]

    def add_broad_class(self, adata=None, overwrite: bool = False) -> pd.Series:
        """obs[broad_col] from obs[celltype_col] with broad_class_rules (kept if present unless overwrite)."""
        a = self._check_adata(adata)
        bc, ct = self.params["broad_col"], self.params["celltype_col"]
        if bc not in a.obs or overwrite:
            a.obs[bc] = pd.Categorical(a.obs[ct].astype(str).map(self.broad_class_of))
        return a.obs[bc]

    def filter_snrna(self, adata=None, max_genes: Optional[int] = None, max_mt_frac: Optional[float] = None,
                     max_doublet: Optional[float] = None, inplace: bool = True) -> Tuple[ad.AnnData, pd.DataFrame]:
        """
        Paper QC: n_genes_by_counts < 5000, mito fraction < 0.10. A doublet cutoff is applied only when
        max_doublet / params['snrna_max_doublet'] is set (default None: the paper's annotated nuclei are
        already post-scrublet). mt_frac (0-1) is converted to pct_counts_mt (0-100).
        """

        a = self._check_adata(adata)
        mt_col = self.params["mt_frac_col"]

        if "pct_counts_mt" not in a.obs and mt_col in a.obs:
            a.obs["pct_counts_mt"] = 100 * a.obs[mt_col].astype(float)

        if "n_genes_by_counts" not in a.obs or "total_counts" not in a.obs:
            self.calc_qc_metrics(a)

        '''
        max_doublet = self.get_param("snrna_max_doublet", max_doublet)
        kw = {} if max_doublet is None else {"max_doublet": max_doublet}
        out, df_report = self.filter_cells(a, max_genes=self.get_param("snrna_max_genes", max_genes),
                                           max_pct_mt=100 * self.get_param("snrna_max_mt_frac", max_mt_frac),
                                           inplace=False, **kw)
        '''
        
        if 'snrna_max_doublet' in self.params.keys():
            out, df_report = self.filter_cells(a, max_genes=self.get_param("snrna_max_genes", max_genes),
                                            max_pct_mt=100 * self.get_param("snrna_max_mt_frac", max_mt_frac),
                                            max_doublet=self.get_param("snrna_max_doublet", max_doublet), inplace=False)
        else:
            out, df_report = self.filter_cells(a, max_genes=self.get_param("snrna_max_genes", max_genes),
                                            max_pct_mt=100 * self.get_param("snrna_max_mt_frac", max_mt_frac),
                                            inplace=False)
            
        if inplace and adata is None:
            self.adata = out
        return out, df_report

    def add_paper_annotation(self, fname: str, adata=None, root: Optional[Path] = None,
                             keep_annotated: bool = True) -> Tuple[ad.AnnData, pd.DataFrame]:
        """Join the paper's 59-type annotation (csv: index = cell, column celltype_col); optionally keep only
        annotated cells. Returns (AnnData, df_annot)."""
        a = self._check_adata(adata)
        ct = self.params["celltype_col"]
        df_annot = pd.read_csv(Path(root or self.root_data) / fname, index_col=0)
        a.obs[ct] = df_annot[ct].reindex(a.obs_names).values
        if keep_annotated:
            a = a[a.obs[ct].notna()].copy()
        a.obs[ct] = a.obs[ct].astype("category")
        self.add_broad_class(a)
        if adata is None:
            self.adata = a
        return a, df_annot

    def load_umap_obs(self, fname: str, root: Optional[Path] = None) -> ad.AnnData:
        """
        Paper UMAP object without X (obs, obsm['X_umap'], *_colors): small AnnData for Fig 2B.
        """

        a = ad.read_h5ad(Path(root or self.root_data) / fname, backed="r")

        out = ad.AnnData(obs=a.obs.copy(), obsm={"X_umap": np.asarray(a.obsm["X_umap"])})

        for k in a.uns.keys():
            if k.endswith("_colors"):
                out.uns[k] = list(a.uns[k])
        a.file.close()

        return out

    def plot_umap_broad_subtypes(self, adata_umap: ad.AnnData, fname: Optional[str] = "fig2B_umap.png",
                                 subtype_col: Optional[str] = None, point_size: float = 0.6, figsize=(13, 6),
                                 label_subtypes: bool = False):
        """
        Fig 2B: UMAP coloured by broad class (labels on data) and by the 59 subtypes.
        """

        bc = self.params["broad_col"]
        ct = subtype_col or self.params["celltype_col"]
        U = adata_umap.obsm["X_umap"]
        obs = adata_umap.obs
        rng = np.random.default_rng(self.params["seed"])
        o = rng.permutation(len(U))

        def palette(col):
            cats = list(obs[col].astype("category").cat.categories)
            key = f"{col}_colors"
            cols = adata_umap.uns.get(key)
            if cols is None or len(cols) < len(cats):
                cols = (KELLY_COLORS + [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors] +
                        [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20b").colors])[:len(cats)]
            return dict(zip(cats, cols))

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, col, title in ((axes[0], bc, "broad class"), (axes[1], ct, f"{obs[ct].nunique()} subtypes")):
            pal = palette(col)
            lab = obs[col].astype(str).values
            ax.scatter(U[o, 0], U[o, 1], s=point_size, c=[pal[x] for x in lab[o]], lw=0, rasterized=True)
            ax.set_title(title, loc="center", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if col == bc or label_subtypes:
                for k in pal:
                    m = lab == k
                    if m.sum() == 0:
                        continue
                    x, y = np.median(U[m], 0)
                    ax.text(x, y, self.alias(k) if col == ct else k, fontsize=8 if col == bc else 5, ha="center",
                            va="center", fontweight="bold" if col == bc else "normal",
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        axes[0].text(0.0, -0.03, f"{adata_umap.n_obs:,} nuclei", transform=axes[0].transAxes, fontsize=7, va="top")
        self._savefig(fig, fname)
        return fig

    # =====================================================================================================
    # Paper cell2location map on Visium (Fig 2C-F)
    # =====================================================================================================
    def load_paper_map(self, fname: str, root: Optional[Path] = None, prefix: Optional[str] = None) -> ad.AnnData:
        """
        Paper cell2location map (v0.05 layout): obs '<prefix><cell type>' -> obsm[c2l_obsm_key] DataFrame;
        uns['spatial'] keys renamed to sample ids (map_library_regex); uns['mod']['factor_names'] set.
        Only libraries with spots are kept in uns['spatial'].
        """

        a = ad.read_h5ad(Path(root or self.root_data) / fname)
        prefix = self.get_param("map_density_prefix", prefix)
        cols = [c for c in a.obs.columns if c.startswith(prefix)]
        types = [c[len(prefix):] for c in cols]
        df_abund = a.obs[cols].astype(float).copy()
        df_abund.columns = types
        a.obsm[self.params["c2l_obsm_key"]] = df_abund
        a.obs = a.obs[[c for c in a.obs.columns if not re.match(r"^(mean|sd|q05|q95)_(spot_factors|nUMI_factors)", c)]]
        a.uns["mod"] = {"factor_names": np.array(types)}
        rx = re.compile(self.params["map_library_regex"])
        present = set(a.obs[self.sample_col].astype(str))
        spatial = {}
        for k, v in a.uns["spatial"].items():
            m = rx.search(k)
            sid = m.group(1) if m else k
            if sid in present:
                spatial[sid] = v
        a.uns["spatial"] = spatial
        return a

    def calc_df_regions(self, adata=None, key: Optional[str] = None, df_abund: Optional[pd.DataFrame] = None,
                        min_enrich: Optional[float] = None, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Anatomical label per region cluster from region_celltype_map.

        enrichment = cluster mean density / overall mean density. Among mapped cell types with enrichment >=
        min_enrich, the top_k most enriched vote for their region label (weighted by enrichment); unmapped types
        are ignored. Columns: region_cluster, n_spots, label, top_types, top_enrich, label_votes.
        """

        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        min_enrich = self.get_param("region_label_min_enrich", min_enrich)
        top_k = self.get_param("region_label_top_k", top_k)

        cmap_ = self.params["region_celltype_map"]

        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(a)
        df_abund = df_abund.loc[a.obs_names]
        
        grp = a.obs[key].astype(str).values
        df_mean = df_abund.groupby(grp).mean()
        df_enrich = df_mean / df_abund.mean(axis=0)
        rows = []
        for cl in df_enrich.index:
            e = df_enrich.loc[cl].sort_values(ascending=False)
            top_all = list(e.index[:3])
            em = e[[c for c in e.index if c in cmap_ and e[c] >= min_enrich]][:top_k]
            votes: Dict[str, float] = {}
            for c, v in em.items():
                votes[cmap_[c]] = votes.get(cmap_[c], 0) + float(v)
            label = max(votes, key=votes.get) if votes else "unassigned"
            rows.append(dict(region_cluster=cl, n_spots=int((grp == cl).sum()), label=label,
                             top_types=" / ".join(top_all), top_enrich=" / ".join(f"{e[c]:.1f}" for c in top_all),
                             label_votes="; ".join(f"{k}:{v:.1f}" for k, v in sorted(votes.items(), key=lambda x: -x[1]))))
        df_regions = pd.DataFrame(rows)
        df_regions["_k"] = pd.to_numeric(df_regions.region_cluster, errors="coerce")
        df_regions = df_regions.sort_values("_k").drop(columns="_k").reset_index(drop=True)
        return df_regions

    def calc_df_edge_fraction(self, adata=None, key: Optional[str] = None, radius_factor: Optional[float] = None,
                              min_neighbors: Optional[int] = None) -> pd.DataFrame:
        """Fraction of each region cluster's spots on the tissue border, per library (a Visium spot has 6
        neighbours; a border spot has fewer than min_neighbors within radius_factor x spot spacing).
        Clusters on the pial surface (layer 1) or the tissue rim show high fractions."""
        from scipy.spatial import cKDTree
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        rf = self.get_param("edge_radius_factor", radius_factor)
        mn = self.get_param("edge_min_neighbors", min_neighbors)
        rows = []
        for lib in self.library_ids(a):
            mk = (a.obs[self.sample_col].astype(str) == lib).values
            xy = np.asarray(a.obsm["spatial"], float)[mk]
            tree = cKDTree(xy)
            step = np.median(tree.query(xy, k=2)[0][:, 1])
            nb = np.array([len(x) - 1 for x in tree.query_ball_point(xy, rf * step)])
            cl = a.obs[key].astype(str).values[mk]
            for c in np.unique(cl):
                rows.append(dict(region_cluster=c, library=lib, n_spots=int((cl == c).sum()),
                                 frac_edge=float((nb[cl == c] < mn).mean())))
        df_edge = pd.DataFrame(rows).pivot(index="region_cluster", columns="library", values="frac_edge")
        df_edge["mean_frac_edge"] = df_edge.mean(1)
        df_edge = df_edge.loc[sorted(df_edge.index, key=lambda x: (len(x), x))]
        return df_edge.round(3)

    def add_region_labels(self, df_regions: pd.DataFrame, adata=None, key: Optional[str] = None,
                          label_col: str = "region_label", override: Optional[Dict[str, str]] = None) -> pd.Series:
        """obs[label_col] from df_regions.label; override (default params['region_label_override']) replaces
        the label of given clusters (e.g. {'11': 'CTX_L1'} after checking calc_df_edge_fraction)."""
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        override = self.get_param("region_label_override", override)
        lab = dict(zip(df_regions.region_cluster.astype(str), df_regions.label))
        lab.update({str(k): v for k, v in override.items()})
        a.obs[label_col] = pd.Categorical(a.obs[key].astype(str).map(lab))
        return a.obs[label_col]

    def _region_palette(self, labels: Sequence[str]) -> Dict[str, str]:
        given = self.params["region_colors"] or {}
        pal = KELLY_COLORS + [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]
        out, i = {}, 0
        for l in sorted(labels):
            if l in given:
                out[l] = given[l]
            else:
                out[l] = pal[i % len(pal)]
                i += 1
        return out

    def _crop(self, ax, xy, pad: float = 40):
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)

    def plot_regions_he(self, adata=None, library_id: Optional[str] = None, label_col: str = "region_label",
                        fname: Optional[str] = "fig2C_regions.png", figsize=(13, 6.5), spot_size: float = 11,
                        spot_alpha: float = 0.9):
        """Fig 2C: H&E of one section (left) and the same section with spots coloured by anatomical region label
        (right), region names at the label medians."""
        a = self._check_adata(adata)
        lib = library_id or self.library_id(a)
        img, _ = self.hires_image(lib, a)
        m, xy = self.spot_xy(lib, a)
        lab = a.obs[label_col].astype(str).values[m]
        pal = self._region_palette(np.unique(lab))
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(img)
        axes[1].imshow(img)
        axes[1].scatter(xy[:, 0], xy[:, 1], c=[pal[l] for l in lab], s=spot_size, lw=0, alpha=spot_alpha)
        for l in np.unique(lab):
            mm = lab == l
            x, y = np.median(xy[mm], 0)
            axes[1].text(x, y, l, fontsize=7, ha="center", va="center", fontweight="bold",
                         path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        for ax, t in zip(axes, (f"H&E - {lib}", "anatomical regions (cell2location region clusters)")):
            self._crop(ax, xy)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=9)
            for s_ in ax.spines.values():
                s_.set_visible(False)
        self._savefig(fig, fname)
        return fig

    def _density_legend(self, fig, ax, names, colors, caps, ncol: int = 3, labels: Optional[Dict[str, str]] = None):
        """Paper-style legend under a density panel: name in its colour + black->colour bar with 0, cap/2, cap."""
        pos = ax.get_position()
        n = len(names)
        nrow = int(np.ceil(n / ncol))
        w = pos.width / ncol
        for i, nm in enumerate(names):
            r, c = divmod(i, ncol)
            x0 = pos.x0 + c * w + 0.1 * w
            y0 = pos.y0 - 0.045 - r * 0.07
            cax = fig.add_axes([x0, y0, 0.75 * w, 0.012])
            cm_ = LinearSegmentedColormap.from_list(nm, ["#000000", colors[nm]])
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, caps[nm]), cmap=cm_), cax=cax,
                              orientation="horizontal")
            cb.set_ticks([0, caps[nm] / 2, caps[nm]])
            cb.ax.set_xticklabels([f"{t:.2f}" for t in (0, caps[nm] / 2, caps[nm])], fontsize=5, color="white")
            cb.ax.tick_params(length=1.5, colors="white")
            cax.set_title((labels or {}).get(nm, self.alias(nm)), fontsize=7, color=colors[nm], fontweight="bold", pad=2)
        return nrow

    def plot_density_panels(self, panels: Dict[str, Sequence[str]], adata=None, library_id: Optional[str] = None,
                            df_abund: Optional[pd.DataFrame] = None, fname: Optional[str] = "fig2DEF_density.png",
                            q: Optional[float] = None, spot_size: Optional[float] = None, panel_size: float = 5.0,
                            show_image: bool = False):
        """Fig 2D-F style: one black panel per entry of `panels` (title -> cell types); every cell type in its own
        colour (density_colors), opacity = density / q-quantile (cells per spot). Legends with 0-cap scales."""
        a = self._check_adata(adata)
        q = self.get_param("density_quantile", q)
        spot_size = self.get_param("density_spot_size", spot_size)
        lib = library_id or self.library_id(a)
        m, xy = self.spot_xy(lib, a)
        img, _ = self.hires_image(lib, a)
        df_abund = (df_abund if df_abund is not None else self.calc_df_abundance(a)).loc[a.obs_names[m]]
        cols = list(self.params["density_colors"])
        n = len(panels)
        fig = plt.figure(figsize=(panel_size * n, panel_size * 1.35), facecolor="black")
        df_caps = []
        for j, (title, types) in enumerate(panels.items()):
            ax = fig.add_axes([j / n + 0.01, 0.3, 1 / n - 0.02, 0.62])
            types = [t for t in types if t in df_abund.columns]
            colors = {t: cols[i % len(cols)] for i, t in enumerate(types)}
            caps = self.plot_density_overlay(ax, xy, {t: df_abund[t].values for t in types}, colors,
                                             img=img if show_image else None, q=q, s=spot_size,
                                             img_alpha=0.3 if show_image else 0)
            self._crop(ax, xy)
            ax.set_title(title, color="white", fontsize=11)
            self._density_legend(fig, ax, types, colors, caps)
            df_caps += [dict(panel=title, cell_type=t, cap_cells_per_spot=round(float(caps[t]), 3)) for t in types]
        self._savefig(fig, fname)
        fig.df_caps = pd.DataFrame(df_caps)
        return fig

    # =====================================================================================================
    # Slide-seq (Fig 2H)
    # =====================================================================================================
    def recover_counts(self, adata, total_col: str = "total_counts", tol: Optional[float] = None
                       ) -> Tuple[ad.AnnData, dict]:
        """Recover integer counts from X = log1p(normalize_total(target)) when obs[total_col] holds the raw
        library size. target is estimated per bead from its smallest non-zero value (count 1) and the median
        is used. Returns (AnnData with X = rounded counts, layers['lognorm'] = original X; diagnostics)."""
        tol = self.get_param("slideseq_count_tol", tol)
        X = sp.csr_matrix(adata.X)
        tot = adata.obs[total_col].values.astype(float)
        mins = np.array([np.expm1(X.data[X.indptr[i]:X.indptr[i + 1]].min()) * tot[i] if X.indptr[i + 1] > X.indptr[i]
                         else np.nan for i in range(X.shape[0])])
        target = float(np.nanmedian(mins))
        C = X.copy().astype(np.float64)
        C.data = np.expm1(C.data) * np.repeat(tot / target, np.diff(C.indptr))
        err = np.abs(C.data - np.round(C.data))
        C.data = np.round(C.data)
        C.eliminate_zeros()
        diag = dict(target_sum=target, max_err=float(err.max()), frac_values_err_gt_tol=float((err > tol).mean()),
                    median_err=float(np.median(err)))
        out = adata.copy()
        out.layers["lognorm"] = adata.X.copy()
        out.X = C.astype(np.float32)
        if diag["frac_values_err_gt_tol"] > 0.01:
            warnings.warn(f"count recovery is not exact: {diag}")
        return out, diag

    def signatures_from_regression(self, fname: str, root: Optional[Path] = None, use_symbols: bool = True,
                                   cell_types: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """59 reference signatures (genes x cell types) from the paper's v0.05 RegressionModel
        (uns['regression_mod']['post_sample_means']['gene_factors'], factors x genes). With use_symbols,
        index = var[symbol_col] (duplicated symbols: first kept)."""
        import h5py
        path = Path(root or self.root_data) / fname
        a = ad.read_h5ad(path, backed="r")
        with h5py.File(path) as h:
            g = h["uns/regression_mod"]
            fact = [x.decode() if isinstance(x, bytes) else str(x) for x in g["fact_names"][:]]
            var = [x.decode() if isinstance(x, bytes) else str(x) for x in g["var_names"][:]]
            G = g["post_sample_means/gene_factors"][:]
        df_sig = pd.DataFrame(G.T, index=var, columns=fact)
        ct = self.params["celltype_col"]
        keep = list(cell_types) if cell_types is not None else [c for c in fact if c.startswith(f"{ct}_") or
                                                                  c in set(a.obs[ct].astype(str))]
        keep_cols = []
        for c in keep:
            keep_cols.append(c if c in df_sig.columns else f"{ct}_{c}")
        df_sig = df_sig[keep_cols]
        df_sig.columns = [c.replace(f"{ct}_", "", 1) if c.startswith(f"{ct}_") else c for c in df_sig.columns]
        if use_symbols:
            sym = a.var[self.symbol_col].reindex(df_sig.index).astype(str).values
            df_sig.index = sym
            df_sig = df_sig[~df_sig.index.duplicated()]
        a.file.close()
        return df_sig

    def prepare_slideseq_mapping(self, fname_slideseq: str, fname_signatures: str
                                 ) -> Tuple[ad.AnnData, pd.DataFrame, dict]:
        """Slide-seq AnnData with recovered counts + 59 snRNA signatures (symbols). Returns (adata_ss, df_sig, diag)."""
        adata_ss = ad.read_h5ad(self.root_data / fname_slideseq)
        adata_ss, diag = self.recover_counts(adata_ss)
        df_sig = self.signatures_from_regression(fname_signatures)
        diag["n_genes_shared"] = int(len(np.intersect1d(adata_ss.var_names, df_sig.index)))
        return adata_ss, df_sig, diag

    def plot_slideseq(self, adata_ss, df_abund: Optional[pd.DataFrame] = None, panels: Optional[Dict[str, Sequence[str]]] = None,
                      fname: Optional[str] = "fig2H_slideseq.png", q: Optional[float] = None,
                      spot_size: Optional[float] = None, panel_size: float = 6.0):
        """Fig 2H: two black panels of Slide-seq bead densities (hippocampal + habenular types | cortical layers)."""
        q = self.get_param("density_quantile", q)
        spot_size = self.get_param("slideseq_spot_size", spot_size)
        panels = panels or {"hippocampus / habenula": self.params["fig2H_left"],
                            "cortical excitatory neurons": self.params["fig2H_right"]}
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(adata_ss)
        xy = np.asarray(adata_ss.obsm["spatial"], float).copy()
        xy[:, 1] = xy[:, 1].max() - xy[:, 1]        # puck y axis up -> image convention
        cols = list(self.params["density_colors"])
        n = len(panels)
        fig = plt.figure(figsize=(panel_size * n, panel_size * 1.3), facecolor="black")
        for j, (title, types) in enumerate(panels.items()):
            ax = fig.add_axes([j / n + 0.01, 0.28, 1 / n - 0.02, 0.66])
            types = [t for t in types if t in df_abund.columns]
            colors = {t: cols[i % len(cols)] for i, t in enumerate(types)}
            caps = self.plot_density_overlay(ax, xy, {t: df_abund[t].values for t in types}, colors, q=q, s=spot_size)
            self._crop(ax, xy, pad=50)
            ax.set_title(title, color="white", fontsize=11)
            self._density_legend(fig, ax, types, colors, caps)
        self._savefig(fig, fname)
        return fig

    # =====================================================================================================
    # Allen layer validation (Fig 2I, 2J)
    # =====================================================================================================
    def sanitize_name(self, name: str) -> str:
        for a_, b_ in self.params["allen_name_replace"].items():
            name = str(name).replace(a_, b_)
        return name

    def sanitize_allen_types(self, adata_allen: ad.AnnData) -> pd.Series:
        """obs[allen_type_col] with allen_name_replace applied (in place, idempotent). Needed before writing a
        mapped AnnData: obsm column names become HDF5 keys."""
        tc = self.params["allen_type_col"]
        adata_allen.obs[tc] = adata_allen.obs[tc].astype(str).map(self.sanitize_name)
        return adata_allen.obs[tc]

    def allen_select_types(self, df_meta: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Paper selection: SSp cell types (>= allen_min_cells cells in SSp) plus types exclusive to the
        hippocampal PAR-POST-PRE region (>= allen_min_cells there, 0 in SSp).
        Returns (df_type_counts per region, ssp_types, hpc_types)."""
        P = self.params
        tc, rc = P["allen_type_col"], P["allen_region_col"]
        df_cnt = pd.crosstab(df_meta[tc], df_meta[rc])
        for r in (P["allen_ctx_region"], P["allen_hpc_region"]):
            if r not in df_cnt:
                df_cnt[r] = 0
        ssp = df_cnt.index[df_cnt[P["allen_ctx_region"]] >= P["allen_min_cells"]].tolist()
        hpc = df_cnt.index[(df_cnt[P["allen_hpc_region"]] >= P["allen_min_cells"]) &
                           (df_cnt[P["allen_ctx_region"]] == 0)].tolist()
        df_type_counts = df_cnt[[P["allen_ctx_region"], P["allen_hpc_region"]]]
        return df_type_counts, ssp, hpc

    def calc_df_allen_layer_prop(self, df_meta: pd.DataFrame, types: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cell counts and proportions of each type within each SSp dissection layer (single-layer dissections only;
        allen_layer_merge applied, e.g. L6b -> L6). Proportion = count / all SSp cells of the layer (all types).
        Returns (df_counts types x layers, df_prop types x layers)."""
        P = self.params
        df_ssp = df_meta[df_meta[P["allen_region_col"]] == P["allen_ctx_region"]].copy()
        df_ssp["_layer"] = df_ssp[P["allen_layer_col"]].replace(P["allen_layer_merge"])
        df_ssp = df_ssp[df_ssp["_layer"].isin(P["layers"])]
        df_counts = pd.crosstab(df_ssp[P["allen_type_col"]], df_ssp["_layer"]).reindex(columns=list(P["layers"]), fill_value=0)
        df_prop = df_counts / df_counts.sum(0).replace(0, np.nan)
        return df_counts.reindex(types).fillna(0), df_prop.reindex(types).fillna(0)

    def allen_top_types(self, df_type_counts: pd.DataFrame, n_top: Optional[int] = None,
                        min_ssp: Optional[int] = None) -> List[str]:
        """Most prevalent SSp-exclusive types (SSp count > min_ssp, hippocampus count 0), largest first."""
        P = self.params
        ctx, hpc = P["allen_ctx_region"], P["allen_hpc_region"]
        df_sel = df_type_counts[(df_type_counts[ctx] > self.get_param("allen_top_min_ssp", min_ssp)) & (df_type_counts[hpc] == 0)]
        return df_sel.sort_values(ctx, ascending=False).index[:self.get_param("allen_n_top", n_top)].tolist()

    def allen_signatures(self, adata_allen: ad.AnnData, types: Sequence[str]) -> pd.DataFrame:
        """Per-cluster average expression (genes x types) of the Smart-seq counts (paper: NB regression is not
        suitable for Smart-seq, so per-cluster averages are used)."""
        tc = self.params["allen_type_col"]
        lab = adata_allen.obs[tc].astype(str).values
        X = sp.csr_matrix(adata_allen.X)
        cols = {}
        for t in types:
            m = lab == t
            cols[t] = np.asarray(X[m].mean(0)).ravel() if m.any() else np.zeros(X.shape[1])
        df_sig = pd.DataFrame(cols, index=adata_allen.var_names)
        return df_sig

    def prepare_allen_mapping(self, adata_allen: ad.AnnData, adata_map: ad.AnnData, label_col: str = "region_label",
                              layer_col: str = "cortical_layer") -> Tuple[ad.AnnData, pd.DataFrame, dict]:
        """Inputs of the Fig 2J mapping: Allen per-cluster signatures (SSp types + hippocampus-exclusive types,
        cell2location permissive gene filter) and the cortical-layer Visium spots of adata_map (needs
        obs[label_col]); var_names switched to gene symbols. Returns (adata_ctx, df_sig, info)."""
        tc = self.params["allen_type_col"]
        self.sanitize_allen_types(adata_allen)
        df_type_counts, ssp, hpc = self.allen_select_types(adata_allen.obs)
        types = ssp + hpc
        sub_ref = adata_allen[adata_allen.obs[tc].isin(types)].copy()
        _, genes_sel = self.filter_genes_c2l(sub_ref, inplace=False)
        df_sig = self.allen_signatures(adata_allen[:, genes_sel], types)
        self.assign_cortical_layers(adata_map, label_col=label_col, layer_col=layer_col)
        adata_ctx = adata_map[adata_map.obs[layer_col].notna()].copy()
        adata_ctx.var["ENSEMBL"] = adata_ctx.var_names
        adata_ctx.var_names = adata_ctx.var[self.symbol_col].astype(str).values
        adata_ctx = adata_ctx[:, ~adata_ctx.var_names.duplicated()].copy()
        info = dict(n_types_ssp=len(ssp), n_types_hpc=len(hpc), n_genes_filtered=len(genes_sel),
                    n_spots=int(adata_ctx.n_obs), n_genes_shared=int(len(np.intersect1d(adata_ctx.var_names, df_sig.index))),
                    types=types, df_type_counts=df_type_counts)
        return adata_ctx, df_sig, info

    def assign_cortical_layers(self, adata=None, label_col: str = "region_label", layer_col: str = "cortical_layer") -> pd.Series:
        """obs[layer_col] from region labels via layer_from_region (non-cortical spots -> NaN)."""
        a = self._check_adata(adata)
        a.obs[layer_col] = pd.Categorical(a.obs[label_col].astype(str).map(self.params["layer_from_region"]),
                                          categories=[l for l in self.params["layers"]])
        return a.obs[layer_col]

    def calc_df_c2l_layer_prop(self, adata_map, types: Sequence[str], layer_col: str = "cortical_layer",
                               df_abund: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """cell2location density averaged per cortical layer and normalised within the layer over all mapped
        types -> proportion (types x layers)."""
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(adata_map)
        lay = adata_map.obs[layer_col]
        df_mean = df_abund.groupby(lay.values, observed=False).mean().T
        df_mean = df_mean.reindex(columns=list(self.params["layers"]))
        df_prop = df_mean / df_mean.sum(0).replace(0, np.nan)
        return df_prop.reindex(types)

    def plot_layer_workflow(self, adata=None, library_id: Optional[str] = None, df_allen_counts: Optional[pd.DataFrame] = None,
                            layer_col: str = "cortical_layer", fname: Optional[str] = "fig2I_layers.png",
                            figsize=(12, 5), spot_size: float = 10, n_types_shown: int = 12):
        """Fig 2I (data version of the schematic): left = Allen SSp cells per dissected layer coloured by the
        most frequent cell types; right = Visium cortical-layer spots on the H&E."""
        a = self._check_adata(adata)
        P = self.params
        lib = library_id or self.library_id(a)
        img, _ = self.hires_image(lib, a)
        m, xy = self.spot_xy(lib, a)
        lay = a.obs[layer_col].astype(str).values[m]
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1, 1.3], wspace=0.25))
        ax = axes[0]
        if df_allen_counts is not None:
            df_c = df_allen_counts.reindex(columns=list(P["layers"])).fillna(0)
            top = df_c.sum(1).sort_values(ascending=False).index[:n_types_shown]
            df_plot = pd.concat([df_c.loc[top], df_c.drop(index=top).sum().to_frame("other").T])
            bottom = np.zeros(df_plot.shape[1])
            pal = KELLY_COLORS + [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]
            for i, t in enumerate(df_plot.index):
                ax.barh(range(df_plot.shape[1]), df_plot.loc[t].values, left=bottom, color=pal[i] if t != "other" else "#cccccc",
                        label=t, height=0.75)
                bottom += df_plot.loc[t].values
            ax.set_yticks(range(df_plot.shape[1])); ax.set_yticklabels(df_plot.columns)
            ax.invert_yaxis()
            ax.set_xlabel("cells (Smart-seq, SSp)")
            ax.set_title("Allen scRNA-seq reference: SSp layer dissections", fontsize=9)
            ax.legend(fontsize=5.5, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left", ncol=1)
        ax = axes[1]
        ax.imshow(img)
        ok = lay != "nan"
        ax.scatter(xy[ok, 0], xy[ok, 1], c=[P["layer_colors"].get(l, "#888") for l in lay[ok]], s=spot_size, lw=0, alpha=0.85)
        for l in P["layers"]:
            ax.scatter([], [], c=P["layer_colors"][l], s=20, label=f"{l} (n={int((lay == l).sum())})")
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  title="layer (region clusters)", title_fontsize=7)
        self._crop(ax, xy[ok] if ok.any() else xy, pad=80)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Visium cortical-layer spots - {lib}", fontsize=9)
        self._savefig(fig, fname)
        return fig

    def plot_layer_heatmaps(self, df_allen_prop: pd.DataFrame, df_c2l_prop: pd.DataFrame, types: Sequence[str],
                            fname: Optional[str] = "fig2J_layer_heatmap.png", vmin: float = 1e-3, vmax: float = 1.0,
                            figsize=(11, 5.2)) -> Tuple[plt.Figure, dict]:
        """Fig 2J: proportion of each cell type within each layer (log10 colour), Allen (top) vs cell2location
        (bottom); Pearson R2 over type x layer cells (linear and log10 scale)."""
        types = list(types)
        layers = list(self.params["layers"])
        A = df_allen_prop.reindex(index=types, columns=layers).astype(float)
        C = df_c2l_prop.reindex(index=types, columns=layers).astype(float)
        ok = (~A.isna() & ~C.isna()).values
        a_, c_ = A.values[ok], C.values[ok]
        r_lin = np.corrcoef(a_, c_)[0, 1]
        eps = vmin
        r_log = np.corrcoef(np.log10(a_ + eps), np.log10(c_ + eps))[0, 1]
        stats = dict(R2_linear=r_lin ** 2, R2_log10=r_log ** 2, n_cells_compared=int(ok.sum()))
        cmap = LinearSegmentedColormap.from_list("rp", ["#ffffff", "#f7c6dc", "#d6338a", "#7a0177", "#2d0042"])
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw=dict(hspace=0.25))
        for ax, M, t in ((axes[0], A, "scRNA-seq reference: known layer (Allen SSp dissections)"),
                         (axes[1], C, "cell2location: inferred layer (Visium)")):
            im = ax.imshow(np.clip(M.T.values, vmin, vmax), aspect="auto", cmap=cmap, norm=norm)
            ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
            ax.set_title(t, fontsize=9)
        axes[1].set_xticks(range(len(types))); axes[1].set_xticklabels(types, rotation=90, fontsize=6)
        cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
        cb.set_label("cell proportion in each layer (log10)", fontsize=7)
        fig.text(0.92, 0.95, f"R² = {stats['R2_linear']:.3f} (linear)\nR² = {stats['R2_log10']:.3f} (log10)",
                 fontsize=8, ha="left", va="top")
        self._savefig(fig, fname)
        return fig, stats

    # =====================================================================================================
    # Regional astrocyte subtypes (Fig 3)
    # =====================================================================================================
    def astro_short(self, ct: str) -> str:
        """'Astro_THAL_lat' -> 'THAL_lat' (paper labels)."""
        return str(ct).replace("Astro_", "", 1)

    def load_astro_reference(self, fname_astro: str, fname_snrna: str, root: Optional[Path] = None) -> ad.AnnData:
        """Astrocyte snRNA-seq object for Fig 3: raw counts of the astrocyte nuclei taken from the full snRNA file
        (all genes), obs / UMAP / colours from the paper's astrocyte cellxgene object (10 merged subtypes in
        astro_col, 13 pre-merge clusters in astro_col_premerge)."""
        root = Path(root or self.root_data)
        a_umap = ad.read_h5ad(root / fname_astro, backed="r")
        names = a_umap.obs_names
        a_raw = ad.read_h5ad(root / fname_snrna, backed="r")
        keep = np.asarray(a_raw.obs_names.isin(names))
        adata_astro = a_raw[keep].to_memory()
        adata_astro = adata_astro[names.intersection(adata_astro.obs_names, sort=False)].copy()
        for c in (self.params["astro_col"], self.params["astro_col_premerge"], self.sample_col):
            adata_astro.obs[c] = a_umap.obs.loc[adata_astro.obs_names, c].astype(str).astype("category").values
        adata_astro.obsm["X_umap"] = np.asarray(a_umap[adata_astro.obs_names].obsm["X_umap"])
        cats = list(a_umap.obs[self.params["astro_col"]].cat.categories)
        adata_astro.uns["astro_colors"] = dict(zip(cats, list(a_umap.uns[f"{self.params['astro_col']}_colors"])))
        a_umap.file.close(); a_raw.file.close()
        return adata_astro

    def astro_colors(self, adata_astro: Optional[ad.AnnData] = None) -> Dict[str, str]:
        """Subtype -> colour: from the astrocyte object when given, else params['astro_colors'] (paper palette)."""
        if adata_astro is not None and "astro_colors" in adata_astro.uns:
            return dict(adata_astro.uns["astro_colors"])
        return dict(self.params["astro_colors"])

    def plot_astro_umap(self, adata_astro: ad.AnnData, fname: Optional[str] = "fig3A_astro_umap.png",
                        figsize=(11, 5), point_size: float = 3):
        """Fig 3A: UMAP of the 10 astrocyte subtypes (left) and of the 13 clusters before merging (right, Fig S11/S12)."""
        U = adata_astro.obsm["X_umap"]
        col_m, col_p = self.params["astro_col"], self.params["astro_col_premerge"]
        cols_m = {self.astro_short(k): v for k, v in self.astro_colors(adata_astro).items()}
        lab_m = [self.astro_short(x) for x in adata_astro.obs[col_m].astype(str)]
        cats_p = sorted(adata_astro.obs[col_p].astype(str).unique())
        pal = KELLY_COLORS + [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]
        cols_p = {self.astro_short(k): pal[i % len(pal)] for i, k in enumerate(cats_p)}
        lab_p = [self.astro_short(x) for x in adata_astro.obs[col_p].astype(str)]
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self.plot_umap_labeled(axes[0], U, lab_m, cols_m, title=f"{len(cols_m)} regional astrocyte subtypes",
                               point_size=point_size, seed=self.params["seed"])
        self.plot_umap_labeled(axes[1], U, lab_p, cols_p, title=f"{len(cols_p)} clusters before merging",
                               point_size=point_size, fontsize=6.5, seed=self.params["seed"])
        axes[0].text(0.0, -0.08, f"{adata_astro.n_obs:,} astrocyte nuclei", transform=axes[0].transAxes, fontsize=7)
        self._savefig(fig, fname)
        return fig

    def plot_astro_markers(self, adata_astro: ad.AnnData, fname: Optional[str] = "fig3B_astro_markers.png",
                           markers: Optional[Sequence] = None):
        """Fig 3B: marker dot plot (size = fraction of nuclei with count > 0, colour = mean log1p CP10k scaled to the
        maximum subtype per gene). Returns (fig, df_frac, df_rel)."""
        markers = markers or self.params["fig3B_markers"]
        genes = [g for _, gl in markers for g in gl]
        order = list(self.params["astro_order"])
        df_frac, df_mean, df_rel = self.calc_df_dotplot(genes, self.params["astro_col"], adata=adata_astro, order=order)
        fig = self.plot_dotplot(df_frac, df_rel, gene_groups=markers, fname=fname,
                                row_labels={t: self.astro_short(t) for t in order})
        return fig, df_frac, df_rel

    def _density_he(self, ax, img, xy, v, cap, cmap, s: float = 9, min_alpha: float = 0.04):
        """Densities over the H&E: white -> colour, opacity proportional to density / cap."""
        r = np.clip(np.asarray(v, float) / max(cap, 1e-9), 0, 1)
        rgba = cmap(r)
        rgba[:, 3] = np.clip(r, min_alpha, 1)
        o = np.argsort(r)
        ax.imshow(img)
        ax.scatter(xy[o, 0], xy[o, 1], c=rgba[o], s=s, lw=0)

    def plot_astro_density_regions(self, adata_map, library_id: Optional[str] = None, types: Optional[Sequence[str]] = None,
                                   colors: Optional[Dict[str, str]] = None, df_abund: Optional[pd.DataFrame] = None,
                                   label_col: str = "region_label", fname: Optional[str] = "fig3C_astro_regions.png",
                                   q: Optional[float] = None, panel_size: float = 5.5):
        """Fig 3C: selected astrocyte subtype densities (left, black panel, one colour per subtype) and the brain
        regions from location clustering (right)."""
        lib = library_id or self.library_id(adata_map)
        types = list(types or self.params["fig3C_types"])
        q = self.get_param("density_quantile", q)
        colors = colors or self.astro_colors()
        m, xy = self.spot_xy(lib, adata_map)
        df_abund = (df_abund if df_abund is not None else self.calc_df_abundance(adata_map)).loc[adata_map.obs_names[m]]
        lab = adata_map.obs[label_col].astype(str).values[m]
        pal = self._region_palette(np.unique(lab))
        fig = plt.figure(figsize=(panel_size * 2, panel_size * 1.3), facecolor="black")
        ax = fig.add_axes([0.01, 0.28, 0.48, 0.64])
        caps = self.plot_density_overlay(ax, xy, {t: df_abund[t].values for t in types},
                                         {t: colors[t] for t in types}, q=q, s=self.params["density_spot_size"])
        self._crop(ax, xy); ax.set_title("astrocyte subtypes", color="white", fontsize=11)
        names = {t: self.astro_short(t) for t in types}
        self._density_legend(fig, ax, types, {t: colors[t] for t in types}, caps, labels=names)
        ax2 = fig.add_axes([0.51, 0.28, 0.48, 0.64])
        ax2.set_facecolor("black")
        ax2.scatter(xy[:, 0], xy[:, 1], c=[pal[l] for l in lab], s=self.params["density_spot_size"], lw=0)
        self._crop(ax2, xy); ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_aspect("equal")
        for s_ in ax2.spines.values():
            s_.set_visible(False)
        ax2.set_title("brain regions", color="white", fontsize=11)
        for l in np.unique(lab):
            if l == "edge_LowQ":
                continue
            pts = xy[lab == l]; c = np.median(pts, 0); p = pts[np.argmin(((pts - c) ** 2).sum(1))]
            ax2.text(p[0], p[1], l, fontsize=6, ha="center", va="center", color="white", fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
        self._savefig(fig, fname)
        return fig

    def calc_df_astro_region_density(self, adata_map, types: Optional[Sequence[str]] = None, label_col: str = "region_label",
                                     df_abund: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Mean density (cells per spot) of each astrocyte subtype in each region (fig3D_areas order, both sections);
        df_rel = df_mean / max over regions per subtype (Fig 3D scaling). Returns (df_mean, df_rel), rows = subtypes."""
        types = list(types or self.params["astro_order"])
        regions = [r for _, rl in self.params["fig3D_areas"] for r in rl]
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(adata_map)
        lab = adata_map.obs.loc[df_abund.index, label_col].astype(str).values
        df_mean = df_abund[types].groupby(lab).mean().T.reindex(columns=regions)
        df_rel = df_mean.div(df_mean.max(1), axis=0)
        return df_mean, df_rel

    def plot_astro_region_dotplot(self, df_rel: pd.DataFrame, fname: Optional[str] = "fig3D_astro_region_dotplot.png",
                                  size_max: float = 110, cmap: str = "RdPu", figsize=(10, 4.6)):
        """Fig 3D: dot size and colour = relative density of each astrocyte subtype (rows) across regions (columns),
        scaled 0-1 per subtype; colour bars = broad areas (top) and regions (bottom)."""
        areas = self.params["fig3D_areas"]
        rows, cols = list(df_rel.index), list(df_rel.columns)
        nr, nc = len(rows), len(cols)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.14, 0.3, 0.72, 0.52])
        yy, xx = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        R = np.nan_to_num(df_rel.values)
        sca = ax.scatter(xx.ravel(), yy.ravel(), s=R.ravel() * size_max, c=R.ravel(), cmap=cmap, vmin=0, vmax=1, lw=0)
        ax.set_xlim(-0.6, nc - 0.4); ax.set_ylim(nr - 0.4, -0.6)
        ax.set_yticks(range(nr)); ax.set_yticklabels([self.astro_short(r) for r in rows])
        ax.set_xticks([])
        for s_ in ax.spines.values():
            s_.set_visible(True)
        pal = self._region_palette(cols)
        axb = fig.add_axes([0.14, 0.25, 0.72, 0.04]); axt = fig.add_axes([0.14, 0.83, 0.72, 0.04])
        axb.bar(range(nc), 1, width=1, color=[pal[c] for c in cols]); axb.set_xlim(-0.5, nc - 0.5)
        axb.set_xticks(range(nc)); axb.set_xticklabels(cols, rotation=90, fontsize=7); axb.set_yticks([])
        x0 = 0
        for label, rl in areas:
            n = len([r for r in rl if r in cols])
            axt.bar(np.arange(x0, x0 + n), 1, width=1, color=self.params["area_colors"].get(label, "#cccccc"))
            axt.text(x0 + (n - 1) / 2, 1.3, label, rotation=30, ha="left", va="bottom", fontsize=7)
            x0 += n
        axt.set_xlim(-0.5, nc - 0.5); axt.set_ylim(0, 1); axt.axis("off")
        for a_ in (axb,):
            for s_ in a_.spines.values():
                s_.set_visible(False)
        cax = fig.add_axes([0.88, 0.3, 0.012, 0.52])
        fig.colorbar(sca, cax=cax).set_label("relative cell density in region", fontsize=7)
        self._savefig(fig, fname)
        return fig

    def plot_diencephalic_astro(self, adata_map, types: Optional[Sequence[str]] = None, df_abund: Optional[pd.DataFrame] = None,
                                label_col: str = "region_label", fname: Optional[str] = "fig3E_diencephalic_astro.png",
                                q: Optional[float] = None, panel_size: float = 3.6):
        """Fig 3E: densities of the diencephalic astrocyte subtypes (columns) over the H&E of each section (rows);
        yellow outlines = thalamus (THAL + HAB) and hypothalamus (fig3E_outlines). Colour scale per subtype."""
        types = list(types or self.params["fig3E_types"])
        q = self.get_param("density_quantile", q)
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(adata_map)
        libs = self.library_ids(adata_map)
        cmap = LinearSegmentedColormap.from_list("he_purple", list(self.params["density_cmap_he"]))
        caps = {t: float(np.quantile(df_abund[t].values, q)) for t in types}
        fig, axes = plt.subplots(len(libs), len(types), figsize=(panel_size * len(types) + 0.8, panel_size * len(libs)),
                                 squeeze=False, gridspec_kw=dict(wspace=0.05, hspace=0.12))
        for i, lib in enumerate(libs):
            img, _ = self.hires_image(lib, adata_map)
            m, xy = self.spot_xy(lib, adata_map)
            lab = adata_map.obs[label_col].astype(str).values[m]
            for j, t in enumerate(types):
                ax = axes[i, j]
                self._density_he(ax, img, xy, df_abund.loc[adata_map.obs_names[m], t].values, caps[t], cmap)
                for name, regs in self.params["fig3E_outlines"].items():
                    self.plot_region_outline(ax, xy, np.isin(lab, regs), min_spots=self.params["outline_min_spots"])
                self._crop(ax, xy); ax.set_xticks([]); ax.set_yticks([])
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if i == 0:
                    ax.set_title(self.astro_short(t), fontsize=11, color="#54278F", fontweight="bold")
                if j == 0:
                    ax.set_ylabel(lib, fontsize=8)
        for j, t in enumerate(types):
            pos = axes[-1, j].get_position()
            cax = fig.add_axes([pos.x0 + pos.width * 0.15, pos.y0 - 0.035, pos.width * 0.7, 0.012])
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, caps[t]), cmap=cmap), cax=cax,
                              orientation="horizontal")
            cb.set_label("cells per spot", fontsize=6); cb.ax.tick_params(labelsize=6)
        self._savefig(fig, fname)
        return fig

    def spot_log1p_cp10k(self, adata_map, genes: Sequence[str]) -> pd.DataFrame:
        """log1p(CP10k) per spot for genes (symbols) from the raw counts in adata_map.X (spots x genes)."""
        found, ids = self.resolve_gene_ids(genes, adata_map)
        X = adata_map[:, ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        tot = np.asarray(adata_map.X.sum(1)).ravel()
        df_expr = pd.DataFrame(np.log1p(X / np.maximum(tot, 1)[:, None] * self.params["target_sum"]),
                               index=adata_map.obs_names, columns=found)
        return df_expr

    def plot_diencephalic_markers(self, adata_map, markers: Optional[Dict[str, str]] = None, label_col: str = "region_label",
                                  fname: Optional[str] = "fig3F_diencephalic_markers.png", q: float = 99.0,
                                  panel_size: float = 3.6, spot_size: float = 14, smooth: Optional[bool] = None):
        """Fig 3F substitute (the paper's smFISH images are not public): Visium expression (log1p CP10k) of the
        diencephalic subtype markers, diencephalon spots only (black panels). Spot expression includes every cell
        type in the spot, so it is not astrocyte-specific. smooth=True (params['fig3F_smooth']) shows the mean over
        each spot and its 6 neighbours, because single spots detect these genes rarely."""
        smooth = self.get_param("fig3F_smooth", smooth)
        markers = markers or self.params["fig3F_markers"]
        genes = list(markers.values())
        df_expr = self.spot_log1p_cp10k(adata_map, genes)
        libs = self.library_ids(adata_map)
        cmap = LinearSegmentedColormap.from_list("dark_blue", list(self.params["marker_cmap_dark"]))
        regs = self.params["diencephalon_regions"]
        fig, axes = plt.subplots(len(libs), len(genes), figsize=(panel_size * len(genes), panel_size * len(libs)),
                                 squeeze=False, facecolor="black", gridspec_kw=dict(wspace=0.04, hspace=0.1))
        for i, lib in enumerate(libs):
            m, xy = self.spot_xy(lib, adata_map)
            lab = adata_map.obs[label_col].astype(str).values[m]
            d = np.isin(lab, regs)
            for j, (t, g) in enumerate(markers.items()):
                ax = axes[i, j]; ax.set_facecolor("black")
                v = df_expr.loc[adata_map.obs_names[m], g].values
                if smooth:
                    A, _ = self.spot_neighbor_graph(xy)
                    v = self.smooth_on_spots(v, A)
                cap = max(np.percentile(v[d], q), 1e-9)
                o = np.argsort(v[d])
                ax.scatter(xy[d][o, 0], xy[d][o, 1], c=np.clip(v[d][o], 0, cap), cmap=cmap, vmin=0, vmax=cap,
                           s=spot_size, lw=0)
                for name, rr in self.params["fig3E_outlines"].items():
                    self.plot_region_outline(ax, xy, np.isin(lab, rr), color="#FFD400", lw=1.0, min_spots=self.params["outline_min_spots"])
                self._crop(ax, xy[d], pad=60); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                for s_ in ax.spines.values():
                    s_.set_visible(False)
                if i == 0:
                    ax.set_title(f"{g}  ({self.astro_short(t)})", color="#5DADE2", fontsize=10, style="italic")
                    if j == 0:
                        ax.text(0, 1.12, "Visium log1p(CP10k)" + (", mean of spot + 6 neighbours" if smooth else ""),
                                transform=ax.transAxes, color="white", fontsize=7)
                if j == 0:
                    ax.text(-0.02, 0.5, lib, transform=ax.transAxes, rotation=90, color="white", ha="right",
                            va="center", fontsize=8)
        self._savefig(fig, fname)
        return fig

    def calc_df_marker_density_rho(self, adata_map, markers: Optional[Dict[str, str]] = None,
                                   types: Optional[Sequence[str]] = None, label_col: str = "region_label",
                                   df_abund: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Spearman rho, over diencephalon spots of both sections, between each marker's Visium expression and each
        diencephalic astrocyte subtype's density (rows = markers, columns = subtypes). A marker that tracks its own
        subtype should have its highest rho on the diagonal."""
        from scipy.stats import spearmanr
        markers = markers or self.params["fig3F_markers"]
        types = list(types or self.params["fig3E_types"])
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(adata_map)
        d = adata_map.obs[label_col].astype(str).isin(self.params["diencephalon_regions"]).values
        df_expr = self.spot_log1p_cp10k(adata_map, list(markers.values()))
        rows = []
        for t_m, g in markers.items():
            rows.append({t: spearmanr(df_expr[g].values[d], df_abund[t].values[d])[0] for t in types})
        df_rho = pd.DataFrame(rows, index=[f"{g} ({self.astro_short(t)})" for t, g in markers.items()])
        df_rho.columns = [self.astro_short(t) for t in types]
        df_rho.attrs["n_spots"] = int(d.sum())
        return df_rho.round(3)

    # =====================================================================================================
    # Subregions (mccc03): spots of each region clustered on gene expression; Plotly figures
    # =====================================================================================================
    def prepare_expression(self, adata_map, label_col: str = "region_label") -> ad.AnnData:
        """Spots x genes for subregion analysis: var_names = unique gene symbols (Ensembl in var['ENSEMBL']),
        layers['counts'] = raw counts, X = log1p(CP10k). Needs obs[label_col] (add_region_labels)."""
        a = adata_map.copy()
        a.raw = None                       # the paper map carries .raw (Ensembl names); scanpy would test on it
        a.var["ENSEMBL"] = a.var_names
        a.var_names = a.var[self.symbol_col].astype(str).values
        a.var_names_make_unique()
        a.layers[self.params["counts_layer"]] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=self.params["target_sum"])
        sc.pp.log1p(a)
        a.obs[label_col] = a.obs[label_col].astype(str).astype("category")
        return a

    def run_subregions(self, adata_expr, label_col: str = "region_label", regions: Optional[Sequence[str]] = None
                       ) -> Tuple[Dict[str, ad.AnnData], pd.DataFrame]:
        """subcluster_adaptive on the spots of each region (batch = section). Writes obs[subregion_key] = '<region>_<k>'
        (k = 1 is the largest; regions in subregion_skip stay NaN). Returns ({region: adata_sub}, df_grid)."""
        key = self.params["subregion_key"]
        regions = regions or [r for r in sorted(adata_expr.obs[label_col].unique()) if r not in self.params["subregion_skip"]]
        adata_expr.obs[key] = pd.Series(np.nan, index=adata_expr.obs_names, dtype=object)
        subs, grids = {}, []
        for reg in regions:
            a_reg = adata_expr[adata_expr.obs[label_col] == reg]
            a_sub, df_grid = self.subcluster_adaptive(a_reg, batch_key=self.sample_col)
            a_sub.obs[key] = pd.Categorical([f"{reg}_{k}" for k in a_sub.obs["subcluster"]],
                                            categories=[f"{reg}_{k}" for k in a_sub.obs["subcluster"].cat.categories])
            adata_expr.obs.loc[a_sub.obs_names, key] = a_sub.obs[key].astype(str).values
            df_grid.insert(0, "region_id", reg)
            subs[reg] = a_sub
            grids.append(df_grid)
        return subs, pd.concat(grids, ignore_index=True)

    def _display_genes(self, genes: Sequence[str], n: Optional[int] = None) -> List[str]:
        rx = re.compile(self.params["gene_display_exclude"]) if self.params["gene_display_exclude"] else None
        out = [g for g in genes if not (rx and rx.search(g))]
        return out[: (n or self.params["n_genes_show"])]

    def calc_df_region_signature(self, adata_expr, label_col: str = "region_label") -> Tuple[pd.DataFrame, Dict[str, set]]:
        """Each region vs all other spots (Wilcoxon). Returns (df_rank of all genes, {region: signature gene set})
        with signature = logFC >= region_sig_lfc, padj <= region_sig_padj, pct_in >= region_sig_min_pct."""
        P = self.params
        df_rank = self.calc_df_rank_genes(adata_expr, groupby=label_col)
        df_rank = df_rank.rename(columns={"group": "region_id", "names": "gene"})
        m = (df_rank.logfoldchanges >= P["region_sig_lfc"]) & (df_rank.pvals_adj <= P["region_sig_padj"]) & \
            (df_rank.pct_in >= P["region_sig_min_pct"])
        df_rank["region_signature"] = m
        sig = {r: set(g.gene) for r, g in df_rank[m].groupby("region_id", observed=True)}
        return df_rank, sig

    def calc_df_subregion_gene_sets(self, adata_expr, region_sig: Dict[str, set], label_col: str = "region_label"
                                    ) -> pd.DataFrame:
        """Partition of each subregion's gene universe into 4 disjoint sets (counts add up to the universe):
          region_signature    - in the region signature (region vs all other regions)
          subregion_signature - higher in the subregion than in the rest of the region (logFC >= sub_sig_lfc,
                                padj <= sub_sig_padj, pct >= sub_sig_min_pct), not in the region signature
          not_present         - detected in >= notpres_min_pct_rest of the rest of the region, in <= notpres_max_pct
                                of the subregion, significantly lower (padj <= notpres_padj)
          other               - the remaining genes of the universe
        Universe: genes detected in >= universe_min_pct of the region's spots ('detected') or all genes ('all').
        Returns long df_gene_sets: region_id, sub_region_id, gene, gene_set, logfoldchanges, pvals_adj, pct_sub,
        pct_rest, scores, mean_log1p_sub."""
        P = self.params
        key = P["subregion_key"]
        rows = []
        for reg in [r for r in adata_expr.obs[label_col].cat.categories if r not in P["subregion_skip"]]:
            a = adata_expr[adata_expr.obs[label_col] == reg].copy()
            a.obs[key] = a.obs[key].astype(str).astype("category")
            subs = list(a.obs[key].cat.categories)
            X = a.X
            pct_reg = np.asarray((X > 0).mean(0)).ravel()
            universe = a.var_names[pct_reg >= P["universe_min_pct"]] if P["gene_universe"] == "detected" else a.var_names
            rsig = region_sig.get(reg, set())
            if len(subs) == 1:
                df_r = pd.DataFrame({"group": subs[0], "names": a.var_names, "scores": 0.0, "logfoldchanges": 0.0,
                                     "pvals": 1.0, "pvals_adj": 1.0, "pct_in": pct_reg, "pct_out": np.nan})
            else:
                df_r = self.calc_df_rank_genes(a, groupby=key)
            for s in subs:
                d = df_r[df_r.group == s].set_index("names").reindex(universe)
                ms = (a.obs[key] == s).values
                d["mean_log1p_sub"] = np.asarray(X[ms][:, a.var_names.get_indexer(universe)].mean(0)).ravel()
                gs = pd.Series("other", index=universe)
                gs[gs.index.isin(rsig)] = "region_signature"
                up = (d.logfoldchanges >= P["sub_sig_lfc"]) & (d.pvals_adj <= P["sub_sig_padj"]) & (d.pct_in >= P["sub_sig_min_pct"])
                gs[(gs == "other") & up.values] = "subregion_signature"
                np_ = (d.pct_in <= P["notpres_max_pct"]) & (d.pct_out >= P["notpres_min_pct_rest"]) & \
                      (d.pvals_adj <= P["notpres_padj"]) & (d.logfoldchanges < 0)
                gs[(gs == "other") & np_.values] = "not_present"
                rows.append(pd.DataFrame({"region_id": reg, "sub_region_id": s, "gene": universe, "gene_set": gs.values,
                                          "logfoldchanges": d.logfoldchanges.values, "pvals_adj": d.pvals_adj.values,
                                          "pct_sub": d.pct_in.values, "pct_rest": d.pct_out.values,
                                          "scores": d.scores.values, "mean_log1p_sub": d.mean_log1p_sub.values}))
        df_gene_sets = pd.concat(rows, ignore_index=True)
        return df_gene_sets

    def calc_df_composition_enrichment(self, df_abund: pd.DataFrame, labels: pd.Series, reference: Optional[pd.Series] = None,
                                       min_mean: Optional[float] = None) -> pd.DataFrame:
        """Mean cell-type density per label vs its reference (default: all other spots; with `reference`, the other spots
        of the same reference group). Returns long table label, cell_type, mean_in, mean_ref, ratio (cell types with
        mean_in >= min_mean)."""
        min_mean = self.get_param("composition_min_mean", min_mean)
        labels = labels.astype(str)
        rows = []
        for lab in sorted(labels.dropna().unique()):
            if lab == "nan":
                continue
            m = (labels == lab).values
            ref = (~m) if reference is None else ((reference == reference[m].iloc[0]).values & ~m)
            if ref.sum() == 0:
                continue
            mi, mr = df_abund[m].mean(), df_abund[ref].mean()
            ratio = (mi + 0.01) / (mr + 0.01)
            for ct in df_abund.columns:
                if mi[ct] >= min_mean:
                    rows.append(dict(label=lab, cell_type=ct, mean_in=round(float(mi[ct]), 3),
                                     mean_ref=round(float(mr[ct]), 3), ratio=round(float(ratio[ct]), 2)))
        df_comp = pd.DataFrame(rows).sort_values(["label", "ratio"], ascending=[True, False])
        return df_comp

    @staticmethod
    def _top_ratio_text(df_comp: pd.DataFrame, label: str, n: int) -> str:
        d = df_comp[df_comp.label == label].head(n)
        return ", ".join(f"{r.cell_type} (x{r.ratio:.1f})" for r in d.itertuples())

    def calc_df_region_table(self, adata_expr, df_rank_region: pd.DataFrame, region_sig: Dict[str, set],
                             df_abund: pd.DataFrame, df_brain_regions: Optional[pd.DataFrame] = None,
                             label_col: str = "region_label") -> pd.DataFrame:
        """Table region: region_id, region_name, n_spots, n_subregions, n_signature_genes, signature_genes (top by score),
        enriched_cell_types (vs all other spots), functional_description (mouse_brain_regions.tsv)."""
        key = self.params["subregion_key"]
        lab = adata_expr.obs[label_col].astype(str)
        df_comp = self.calc_df_composition_enrichment(df_abund.loc[adata_expr.obs_names], lab)
        desc = dict(zip(df_brain_regions.region_id, df_brain_regions.description)) if df_brain_regions is not None else {}
        names = dict(zip(df_brain_regions.region_id, df_brain_regions.region_name)) if df_brain_regions is not None else {}
        rows = []
        for reg in adata_expr.obs[label_col].cat.categories:
            d = df_rank_region[(df_rank_region.region_id == reg) & df_rank_region.region_signature].sort_values("scores", ascending=False)
            n_sub = adata_expr.obs.loc[lab == reg, key].dropna().nunique() if key in adata_expr.obs else np.nan
            rows.append(dict(region_id=reg, region_name=names.get(reg, ""), n_spots=int((lab == reg).sum()), n_subregions=(int(n_sub) if pd.notna(n_sub) else None),
                             n_signature_genes=len(region_sig.get(reg, ())),
                             signature_genes=", ".join(self._display_genes(d.gene)),
                             enriched_cell_types=self._top_ratio_text(df_comp, reg, self.params["composition_top"]),
                             functional_description=desc.get(reg, "")))
        df_region_table = pd.DataFrame(rows)
        return df_region_table

    def calc_df_subregion_borders(self, adata_expr, label_col: str = "region_label") -> pd.DataFrame:
        """For each subregion: fraction of its spots' tissue neighbours (same section) that belong to another region
        (border_frac) and the most frequent of those regions (border_region). Index = sub_region_id."""
        from scipy.spatial import cKDTree
        key = self.params["subregion_key"]
        rf = self.params["coherence_radius_factor"]
        obs = adata_expr.obs
        lab = obs[label_col].astype(str).values
        sub = obs[key].astype(str).values
        libs = obs[self.sample_col].astype(str).values
        xy = np.asarray(adata_expr.obsm["spatial"], float)
        counts: Dict[str, Dict[str, int]] = {}
        totals: Dict[str, int] = {}
        for lb in np.unique(libs):
            idx = np.where(libs == lb)[0]
            tree = cKDTree(xy[idx])
            step = np.median(tree.query(xy[idx], k=2)[0][:, 1])
            for i_loc, nb in enumerate(tree.query_ball_point(xy[idx], rf * step)):
                i = idx[i_loc]
                if sub[i] == "nan":
                    continue
                for j_loc in nb:
                    j = idx[j_loc]
                    if j == i:
                        continue
                    totals[sub[i]] = totals.get(sub[i], 0) + 1
                    if lab[j] != lab[i]:
                        counts.setdefault(sub[i], {})
                        counts[sub[i]][lab[j]] = counts[sub[i]].get(lab[j], 0) + 1
        rows = []
        for s, tot in totals.items():
            c = counts.get(s, {})
            n_other = sum(c.values())
            top = max(c, key=c.get) if c else ""
            rows.append(dict(sub_region_id=s, border_frac=n_other / tot if tot else 0.0, border_region=top))
        df_border = pd.DataFrame(rows).set_index("sub_region_id")
        return df_border

    def calc_df_subregion_table(self, adata_expr, df_gene_sets: pd.DataFrame, df_abund: pd.DataFrame,
                                df_descriptions: Optional[pd.DataFrame] = None, label_col: str = "region_label",
                                n_other_top: int = 10) -> pd.DataFrame:
        """Table subregion: region_id, sub_region_id, n_spots, section fractions, spatial coherence, counts of the 4 gene
        sets (+ universe), subregion_signature_genes, not_present_genes, other_genes_top_expressed, enriched cell types
        (vs the rest of the region), status, functional_description (curated table when given, else automatic)."""
        P = self.params
        key = P["subregion_key"]
        obs = adata_expr.obs[adata_expr.obs[key].notna()]
        df_coh = self.calc_df_spatial_coherence(adata_expr[obs.index], key=key, within_key=label_col)
        coh = dict(zip(df_coh.category, df_coh.coherence_ratio))
        df_comp = self.calc_df_composition_enrichment(df_abund.loc[obs.index], obs[key], reference=obs[label_col])
        libs = sorted(obs[self.sample_col].astype(str).unique())
        df_border = self.calc_df_subregion_borders(adata_expr, label_col=label_col)
        umis = np.asarray(adata_expr.layers[P["counts_layer"]].sum(1)).ravel()
        umis = pd.Series(umis, index=adata_expr.obs_names)
        curated = {}
        if df_descriptions is not None:
            curated = {r.sub_region_id: r for r in df_descriptions.itertuples()}
        rows = []
        for (reg, s), d in df_gene_sets.groupby(["region_id", "sub_region_id"], sort=False):
            n_set = d.gene_set.value_counts()
            o = obs[obs[key] == s]
            frac = o[self.sample_col].astype(str).value_counts(normalize=True).reindex(libs).fillna(0)
            reg_libs = obs.loc[obs[label_col] == reg, self.sample_col].astype(str).unique()
            sig = d[d.gene_set == "subregion_signature"].sort_values("scores", ascending=False)
            npr = d[d.gene_set == "not_present"].sort_values("scores")
            oth = d[d.gene_set == "other"].sort_values("mean_log1p_sub", ascending=False)
            n_sig = int(n_set.get("subregion_signature", 0))
            single = len(reg_libs) > 1 and frac[list(reg_libs)].min() < P["section_min_frac"]
            n_subs = obs.loc[obs[label_col] == reg, key].nunique()
            depth_ratio = float(umis[o.index].median() / umis[obs.index[obs[label_col] == reg]].median())
            low_depth = depth_ratio < P["low_depth_ratio"]
            b = df_border.loc[s] if s in df_border.index else None
            border = b is not None and b.border_frac >= P["border_min_frac"]
            if n_subs == 1:
                status = "no subregions (one cluster)"
            elif coh.get(s, 0) >= P["coherence_min_ratio"] and n_sig >= P["sub_min_sig_genes"]:
                status = "supported"
            else:
                status = "weak"
            high_depth = depth_ratio > P["high_depth_ratio"]
            status += ("; single-section" if single else "") + ("; low-depth" if low_depth else "") + \
                      ("; high-depth" if high_depth else "") + (f"; border with {b.border_region}" if border else "")
            sig_genes = self._display_genes(sig.gene)
            np_genes = self._display_genes(npr.gene)
            cell_types = self._top_ratio_text(df_comp, s, P["composition_top"])
            cur = curated.get(s)
            if cur is not None:
                description, desc_source = cur.functional_description, "curated"
                check = [g.strip() for g in str(getattr(cur, "signature_check", "")).split(",") if g.strip()]
                if check and not set(check) & set(self._display_genes(sig.gene, 20)) and n_sig > 0:
                    desc_source = "curated - CHECK: signature changed since curation"
            else:
                description = (f"Part of {reg}" + (f"; enriched for {cell_types}" if cell_types else "")
                               + (f"; markers {', '.join(sig_genes[:5])}" if sig_genes else "")
                               + (f"; lacks {', '.join(np_genes[:3])}" if np_genes else "") + ".")
                desc_source = "automatic"
            row = dict(region_id=reg, sub_region_id=s, n_spots=len(o))
            row.update({f"frac_{lb}": round(float(frac[lb]), 2) for lb in libs})
            row.update(dict(coherence_ratio=coh.get(s, np.nan), depth_ratio=round(depth_ratio, 2),
                            border_region=(b.border_region if b is not None else ""),
                            border_frac=(round(float(b.border_frac), 2) if b is not None else np.nan), status=status,
                            n_universe=int(len(d)), n_region_signature=int(n_set.get("region_signature", 0)),
                            n_subregion_signature=n_sig, n_not_present=int(n_set.get("not_present", 0)),
                            n_other=int(n_set.get("other", 0)),
                            subregion_signature_genes=", ".join(sig_genes), not_present_genes=", ".join(np_genes),
                            other_genes_top_expressed=", ".join(self._display_genes(oth.gene, n_other_top)),
                            enriched_cell_types=cell_types, functional_description=description,
                            description_source=desc_source))
            rows.append(row)
        df_sub_table = pd.DataFrame(rows)
        assert (df_sub_table[["n_region_signature", "n_subregion_signature", "n_not_present", "n_other"]].sum(1)
                == df_sub_table.n_universe).all(), "gene sets do not partition the universe"
        return df_sub_table

    # ---------------------------------------------------------------------------------------- Plotly figures
    def save_plotly(self, fig, fname: Optional[str], root_figure: Optional[Path] = None):
        """fig.write_html to (root_figure or self.root_figure) / fname (plotly.js from params['plotly_include_js'])."""
        if fname is None:
            return None
        path = Path(root_figure or self.root_figure) / fname
        fig.write_html(path, include_plotlyjs=self.params["plotly_include_js"])
        return path

    def subregion_colors(self, sub_ids: Sequence[str], region: str) -> Dict[str, str]:
        """Shades of the region colour (params['region_colors']), dark -> light by subregion number."""
        import colorsys
        base = self._region_palette([region])[region]
        r, g, b = mpl.colors.to_rgb(base)
        h, l, s_ = colorsys.rgb_to_hls(r, g, b)
        n = len(sub_ids)
        light = np.linspace(0.28, 0.78, n) if n > 1 else [min(max(l, 0.35), 0.6)]
        sat = max(s_, 0.45)
        return {sid: mpl.colors.to_hex(colorsys.hls_to_rgb(h, li, sat)) for sid, li in zip(sub_ids, light)}

    def subregion_palette(self, sub_ids: Sequence[str]) -> Dict[str, str]:
        """Distinct colours for the subregions of one region (Plotly qualitative palette)."""
        import plotly.express as px
        pal = px.colors.qualitative.Bold + px.colors.qualitative.Safe
        return {s: pal[i % len(pal)] for i, s in enumerate(sub_ids)}

    def _jpeg_uri(self, img: np.ndarray) -> str:
        import base64, io
        from PIL import Image
        a_ = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a_[..., :3]).save(buf, format="JPEG", quality=self.params["plotly_jpeg_quality"])
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def plotly_region_subregions(self, region: str, adata_sub: ad.AnnData, adata_map, df_sub_table: Optional[pd.DataFrame] = None,
                                 pad: float = 80, height: int = 470, spot_size: float = 7):
        """One region: UMAP by subregion | UMAP by section | subregions on the H&E of each section (cropped to the region).
        Legend entries toggle a subregion in every panel. Hover: subregion, section, status, signature genes."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        key = self.params["subregion_key"]
        subs = list(adata_sub.obs[key].cat.categories)
        cols = self.subregion_palette(subs)
        libs = [lb for lb in self.library_ids(adata_map) if (adata_sub.obs[self.sample_col].astype(str) == lb).any()]
        info = {}
        if df_sub_table is not None:
            for r in df_sub_table[df_sub_table.region_id == region].itertuples():
                info[r.sub_region_id] = f"{r.status}; coherence {r.coherence_ratio}<br>up: {r.subregion_signature_genes[:80]}"
        titles = ["UMAP - subregions", "UMAP - sections"] + [f"{lb}" for lb in libs]
        fig = make_subplots(rows=1, cols=2 + len(libs), subplot_titles=titles, horizontal_spacing=0.03)
        U = adata_sub.obsm["X_umap"]
        lab = adata_sub.obs[key].astype(str).values
        sec = adata_sub.obs[self.sample_col].astype(str).values
        for s in subs:
            m = lab == s
            fig.add_trace(go.Scattergl(x=U[m, 0], y=U[m, 1], mode="markers", name=f"{s} (n={m.sum()})", legendgroup=s,
                                       marker=dict(size=5, color=cols[s]), text=adata_sub.obs_names[m],
                                       hovertemplate=f"<b>{s}</b><br>{info.get(s, '')}<br>%{{text}}<extra></extra>"),
                          row=1, col=1)
        sec_cols = dict(zip(self.library_ids(adata_map), ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]))
        for lb in np.unique(sec):
            m = sec == lb
            fig.add_trace(go.Scattergl(x=U[m, 0], y=U[m, 1], mode="markers", name=lb, legendgroup=lb,
                                       marker=dict(size=5, color=sec_cols.get(lb, "#777")), hoverinfo="skip"), row=1, col=2)
        for j, lb in enumerate(libs):
            img, _ = self.hires_image(lb, adata_map)
            mm, xy_all = self.spot_xy(lb, adata_map)
            names_lb = adata_map.obs_names[mm]
            in_reg = names_lb.isin(adata_sub.obs_names)
            xy = xy_all[in_reg]
            x0, x1 = int(max(xy[:, 0].min() - pad, 0)), int(min(xy[:, 0].max() + pad, img.shape[1]))
            y0, y1 = int(max(xy[:, 1].min() - pad, 0)), int(min(xy[:, 1].max() + pad, img.shape[0]))
            col = 3 + j
            fig.add_trace(go.Image(source=self._jpeg_uri(img[y0:y1, x0:x1]), x0=x0, y0=y0, dx=1, dy=1, hoverinfo="skip"),
                          row=1, col=col)
            lab_lb = adata_sub.obs.loc[names_lb[in_reg], key].astype(str).values
            for s in subs:
                m = lab_lb == s
                if not m.any():
                    continue
                fig.add_trace(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=s, legendgroup=s, showlegend=False,
                                           marker=dict(size=spot_size, color=cols[s], line=dict(width=0.4, color="black")),
                                           text=names_lb[in_reg][m],
                                           hovertemplate=f"<b>{s}</b> - {lb}<br>{info.get(s, '')}<br>%{{text}}<extra></extra>"),
                              row=1, col=col)
            fig.update_xaxes(range=[x0, x1], visible=False, row=1, col=col)
            fig.update_yaxes(range=[y1, y0], visible=False, scaleanchor=f"x{col}", row=1, col=col)
        for c in (1, 2):
            fig.update_xaxes(showticklabels=False, title_text="UMAP1", row=1, col=c)
            fig.update_yaxes(showticklabels=False, title_text="UMAP2" if c == 1 else None, row=1, col=c)
        res = adata_sub.uns.get("subcluster_resolution", "")
        fig.update_layout(title=f"{region}: {len(subs)} subregion(s), {adata_sub.n_obs} spots (Leiden resolution {res})",
                          height=height, width=420 * (2 + len(libs)), template="plotly_white",
                          legend=dict(itemsizing="constant"), margin=dict(l=20, r=20, t=70, b=20))
        return fig

    def plotly_subregion_map(self, adata_expr, adata_map, library_id: str, df_sub_table: Optional[pd.DataFrame] = None,
                             label_col: str = "region_label", spot_size: float = 6, width: int = 1300, height: int = 950):
        """All subregions of one section on the H&E. Colours = shades of each region's colour; legend grouped by region
        (click a region title to toggle its subregions). Hover: region, subregion, status, signature genes, cell types."""
        import plotly.graph_objects as go
        key = self.params["subregion_key"]
        img, _ = self.hires_image(library_id, adata_map)
        mm, xy = self.spot_xy(library_id, adata_map)
        names_lb = adata_map.obs_names[mm]
        obs = adata_expr.obs.loc[names_lb]
        info = {}
        if df_sub_table is not None:
            for r in df_sub_table.itertuples():
                info[r.sub_region_id] = (f"{r.status}<br>up: {r.subregion_signature_genes[:70]}"
                                         f"<br>absent: {r.not_present_genes[:60]}<br>cells: {r.enriched_cell_types[:80]}")
        fig = go.Figure()
        fig.add_trace(go.Image(source=self._jpeg_uri(img), hoverinfo="skip"))
        regions = [r for r in adata_expr.obs[label_col].cat.categories]
        for reg in regions:
            mreg = (obs[label_col].astype(str) == reg).values
            if not mreg.any():
                continue
            if reg in self.params["subregion_skip"]:
                fig.add_trace(go.Scattergl(x=xy[mreg, 0], y=xy[mreg, 1], mode="markers", name=reg, legendgroup=reg,
                                           legendgrouptitle_text=reg, marker=dict(size=spot_size, color="#dddddd"),
                                           hovertemplate=f"<b>{reg}</b> (not subclustered)<extra></extra>"))
                continue
            subs = sorted(obs.loc[mreg, key].dropna().unique(), key=lambda x: int(x.rsplit("_", 1)[1]))
            cols = self.subregion_colors(subs, reg)
            for s in subs:
                m = (obs[key] == s).values
                fig.add_trace(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=s, legendgroup=reg,
                                           legendgrouptitle_text=reg,
                                           marker=dict(size=spot_size, color=cols[s], line=dict(width=0.3, color="black")),
                                           text=names_lb[m],
                                           hovertemplate=f"<b>{s}</b><br>{info.get(s, '')}<br>%{{text}}<extra></extra>"))
        fig.update_layout(title=f"{library_id}: subregions of each brain region", width=width, height=height,
                          template="plotly_white", legend=dict(groupclick="toggleitem", itemsizing="constant", font=dict(size=10)),
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(visible=False, range=[xy[:, 0].min() - 40, xy[:, 0].max() + 40])
        fig.update_yaxes(visible=False, range=[xy[:, 1].max() + 40, xy[:, 1].min() - 40], scaleanchor="x")
        return fig

    def plotly_subregion_diagnostics(self, df_sub_table: pd.DataFrame, height: int = 520):
        """Subregion size vs spatial coherence ratio (marker size = number of subregion signature genes, colour = region);
        dashed line = coherence_min_ratio."""
        import plotly.express as px
        d = df_sub_table.copy()
        d["n_sig_plot"] = d.n_subregion_signature.clip(lower=1)
        cmap = {r: self._region_palette([r])[r] for r in d.region_id.unique()}
        fig = px.scatter(d, x="n_spots", y="coherence_ratio", color="region_id", size="n_sig_plot", size_max=28,
                         hover_name="sub_region_id", color_discrete_map=cmap, log_x=True,
                         hover_data=dict(status=True, n_subregion_signature=True, n_not_present=True, n_sig_plot=False,
                                         subregion_signature_genes=True))
        fig.add_hline(y=self.params["coherence_min_ratio"], line_dash="dash", line_color="#888")
        fig.update_layout(title="Subregion diagnostics: size vs spatial coherence (marker size = signature genes)",
                          template="plotly_white", height=height,
                          xaxis_title="spots in subregion (log)", yaxis_title="spatial coherence ratio (1 = random)")
        return fig

    def plotly_region_signature_heatmap(self, adata_expr, df_rank_region: pd.DataFrame, n_per_region: int = 3,
                                        label_col: str = "region_label", height: int = 700):
        """Mean log1p(CP10k) of the top region signature genes (rows) in every region (columns), z-scored per gene."""
        import plotly.graph_objects as go
        regions = list(adata_expr.obs[label_col].cat.categories)
        genes = []
        for reg in regions:
            d = df_rank_region[(df_rank_region.region_id == reg) & df_rank_region.region_signature].sort_values("scores", ascending=False)
            genes += [g for g in self._display_genes(d.gene, n_per_region) if g not in genes]
        X = adata_expr[:, genes].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        df_mean = pd.DataFrame(X, columns=genes).groupby(adata_expr.obs[label_col].astype(str).values).mean().loc[regions].T
        Z = df_mean.sub(df_mean.mean(1), axis=0).div(df_mean.std(1).replace(0, 1), axis=0)
        fig = go.Figure(go.Heatmap(z=Z.values, x=regions, y=genes, colorscale="RdBu_r", zmid=0,
                                   customdata=df_mean.values.round(2),
                                   hovertemplate="%{y} in %{x}<br>z = %{z:.2f}<br>mean log1p = %{customdata}<extra></extra>",
                                   colorbar=dict(title="z (per gene)")))
        fig.update_layout(title=f"Top {n_per_region} signature genes per region", height=height, template="plotly_white",
                          yaxis=dict(autorange="reversed", tickfont=dict(size=9)))
        return fig

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
    region_colors=None,             # dict label -> colour; None = Kelly / tab20
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

    def _density_legend(self, fig, ax, names, colors, caps, ncol: int = 3):
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
            cax.set_title(self.alias(nm), fontsize=7, color=colors[nm], fontweight="bold", pad=2)
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

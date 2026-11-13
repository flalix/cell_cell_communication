#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
single_cell_lib.py - SingleCell: generic Scanpy / Squidpy / cell2location / LIANA methods.

Tissue-specific analyses live in subclasses (one file each):
    libs/single_cell_neuro_lib.py     SingleCellNeuro(SingleCell)      mouse brain (Kleshchevnikov et al., Fig 2)
    libs/single_cell_lymphoid_lib.py  SingleCellLymphoid(SingleCell)   human lymph node / lymphoid-organ reference

Parameters
----------
Everything tunable is a parameter:
- project layout : ROOT0, project, data_dir, figure_dir, result_dir  -> self.root_project, root_data, root_figure, root_result
- cutoffs        : DEFAULT_PARAMS (copied to self.params; override with SingleCell(..., params={...}) or self.set_params())
- filenames      : every method that writes a file has a `fname=` argument
Method arguments left as None are read from self.params (self.get_param(key, value)).

Naming: every pandas DataFrame variable starts with 'df_'.

Sections of SingleCell
----------------------
1. I/O and components          read_h5ad, read_h5ad_slots, read_visium_samples, write_h5ad, summary, calc_df_components,
                               x_info, is_raw_counts
2. Gene lookup and clipping    symbols, values, resolve_gene_ids, percentile_cap, clip_for_plot
3. Filtering                   flag_mt_genes, remove_mt_genes, filter_cells, filter_genes, filter_genes_c2l
4. Quality control             calc_qc_metrics, calc_df_qc, plot_qc
5. Scanpy preprocessing        normalize_log, highly_variable, embed (pca/neighbors/umap), leiden, rank_genes
6. Squidpy spatial             library_ids, hires_image, spot_xy, select_slide, spatial_scatter_genes,
                               spatial_neighbors, spatial_autocorr, nhood_enrichment
7. cell2location               train_reference, get_reference_signatures, train_mapping, ensure_posterior_samples,
                               calc_df_abundance, cluster_regions, umap_regions, region_names
7b. cell2location models       fit_or_load_reference, fit_or_load_mapping (return the model), add_expected_layers,
                               abundance_to_obs, plot_c2l_history, plot_c2l_qc, posterior_samples_cached,
                               plot_posterior_median_vs_mean, run_colocation_cached, calc_df_coloc_loadings
8. Plots                       legend_figure_bottom, umap_spatial_3plot, plot_spatial_visium, plotly_spatial_visium,
                               plot_genes_per_cell_type, plot_slide_loc_cell_types, plot_spatial_features,
                               plot_density_overlay
8b. Subclustering              subcluster_adaptive, calc_df_rank_genes, calc_df_spatial_coherence
8c. Subregions                 prepare_expression, run_subregions, calc_df_region_signature, calc_df_subregion_gene_sets,
                               calc_df_region_table, calc_df_subregion_table (+ borders, composition), Plotly figures
9. LIANA                       spatial_weights (per section), liana_bivariate (cached), lr_significant,
                               calc_df_lr_by_group, calc_celltype_expr (c2l count split), calc_df_celltype_lr,
                               calc_df_group_celltype_links, liana_rank_aggregate_reference, calc_df_celltype_coloc,
                               calc_df_ref_lr_support, calc_df_group_ccc; figures plot_lr_spatial,
                               plotly_lr_group_heatmap, plotly_group_celltype_links, plot_sender_receiver_heatmaps,
                               plotly_coloc_heatmap
10. Static helpers             wtab

Usage (notebook)
----------------
    ROOT0 = Path("/home/flavio/uv/cell_cell_communication")
    sys.path.append(str(ROOT0 / "src"))
    from libs.single_cell_lib import SingleCell
    scl = SingleCell(ROOT0, project="mouse_brain", params={"qc_max_genes": 5000})

libs/cell2location_lib.py (function version, used by ccc01/ccc02) is kept unchanged for backward compatibility.

@created on 2026/11/05
@author: Flavio Lichtenstein
"""

import copy
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import anndata as ad
import scanpy as sc

from libs.Basic import create_dir, pdwritecsv, pdreadcsv

__version__ = "1.4.0"

colors_byr = ("#57a4e2", "#ebd246", "#f30707")
cmap_byr = LinearSegmentedColormap.from_list("blue_yellow_red", list(colors_byr))
cmap_lbwr = LinearSegmentedColormap.from_list("lightblue_white_lightred", ["#6baed6", "#ffffff", "#fb6a4a"])  # blue -> white -> red (mid tones)

# Kelly's colours of maximum contrast, without near-white and near-black
KELLY_COLORS = ['#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280',
                '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97',
                '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522',
                '#E25822', '#2B3D26']

# Region colours: bright and distinct, visible on H&E. Excluded: black, gray, violet/purple (HSV hue 230-335),
# brown/tan, dark colours (all CIELAB L* >= 55). Ordered so that consecutive regions get contrasting hues.
REGION_COLORS = ['#F3C300',  # yellow
                 '#3FB9DB',  # sky blue
                 '#F38400',  # orange
                 '#FA5D98',  # pink
                 '#8DB600',  # lime
                 '#6296ED',  # cornflower blue
                 '#F96560',  # coral red
                 '#3BF0D5',  # turquoise
                 '#FFBB78',  # peach
                 '#4DAF4A',  # green
                 '#E68FAC',  # rose
                 '#A1CAF1',  # light blue
                 '#E25822',  # red-orange
                 '#00C49A',  # teal
                 '#BCE667',  # light lime
                 '#FF9DA7',  # light pink
                 '#17BECF',  # cyan
                 '#98DF8A',  # pale green
                 '#F44336',  # red
                 '#71E9ED',  # pale cyan
                 '#36F587']  # spring green

# Light sequential scale (low -> high; all CIELAB L* >= 60): continuous values such as log10 UMIs
LIGHT_SEQUENTIAL = ['#FFEDA0', '#FEB24C', '#FD8D3C', '#F96560']

# -----------------------------------------------------------------------------------------------------------
# Default parameters (cutoffs, model settings, plotting). Copied to self.params; override per project.
# -----------------------------------------------------------------------------------------------------------
DEFAULT_PARAMS: Dict[str, Any] = dict(
    seed=0,
    # --- QC / filtering of cells or spots
    qc_min_genes=None,             # n_genes_by_counts >= qc_min_genes
    qc_max_genes=None,             # n_genes_by_counts <  qc_max_genes
    qc_min_counts=None,            # total_counts >= qc_min_counts
    qc_max_pct_mt=None,            # pct_counts_mt < qc_max_pct_mt (percent, 0-100)
    qc_max_doublet=None,           # doublet_score < qc_max_doublet
    qc_doublet_col="doublet_score",
    qc_max_points=20000,           # points in the QC scatter
    # --- gene filtering
    gene_min_cells=3,
    c2l_cell_count_cutoff=5,       # cell2location.utils.filtering.filter_genes
    c2l_cell_percentage_cutoff2=0.03,
    c2l_nonz_mean_cutoff=1.12,
    # --- scanpy
    target_sum=1e4,
    counts_layer="counts",
    n_top_genes=3000,
    hvg_flavor="seurat_v3",
    n_pcs=50,
    n_neighbors=15,
    umap_min_dist=0.5,
    leiden_resolution=1.0,
    rank_method="wilcoxon",
    rank_n_genes=50,
    # --- squidpy
    sq_coord_type="grid",
    sq_n_rings=1,
    sq_n_perms=100,
    sq_n_jobs=4,
    # --- cell2location reference (RegressionModel)
    ref_max_epochs=250,
    ref_batch_size=2500,
    # --- cell2location mapping
    c2l_N_cells_per_location=8,
    c2l_detection_alpha=20,
    c2l_max_epochs=30000,
    c2l_num_samples=1000,
    c2l_quantiles=("q05", "q50", "q95", "q0001"),
    c2l_obsm_key="q05_cell_abundance_w_sf",
    accelerator="auto",
    torch_threads=None,            # CPU threads for torch (None = os.cpu_count()); ignored on GPU
    # --- regions on cell-abundance space (paper: kNN 38, Leiden 1.3)
    region_key="region_cluster",
    region_n_neighbors=38,
    region_resolution=1.3,
    region_top_n=2,
    region_min_enrich=1.2,
    region_method="abundance",
    region_leiden_flavor="igraph",  # 'igraph' (fast) or 'leidenalg' (scanpy default; gives different clusters)
    region_umap_min_dist=0.3,      # UMAP of spots on the abundance kNN graph
    region_umap_spread=1.0,
    # --- cell2location model objects (fit_or_load_reference / fit_or_load_mapping)
    c2l_signature_summary="q50",   # signatures used for mapping: 'means', 'q50', 'q05'
    c2l_expected_quantile="q05",   # posterior summary for compute_expected_per_cell_type (layers)
    c2l_posterior_sites=("w_sf", "m_g", "z_sr_groups_factors"),
    # --- NMF co-location (cell2location.run_colocation)
    coloc_n_facts=(11, 12),
    coloc_n_restarts=3,
    coloc_alpha=0.01,
    # --- LIANA
    liana_resource="consensus",    # 'mouseconsensus' for mouse
    liana_expr_prop=0.1,
    liana_n_perms=100,
    liana_bandwidth_spots=1.1,     # Gaussian kernel bandwidth in spot spacings (lymph node: 150 px / 137 px); one
                                   # kernel per section; ~18 neighbours incl. self (rings 1-2) with liana_cutoff 0.1
    liana_cutoff=0.1,              # kernel weights below this are removed
    liana_fdr=0.05,                # LR pairs with global score > 0 at this FDR are 'significant'
    liana_n_jobs=8,
    liana_local="cosine",
    liana_global="morans",
    liana_nz_prop=0.05,            # ligand and receptor (all subunits) expressed in >= this fraction of spots
    # --- LR per group (calc_df_lr_by_group), cell-type-resolved LR (calc_df_celltype_lr), reference LIANA
    lr_group_min_auroc=0.65,       # 'enriched' in a group: AUROC of the local score >= this ...
    lr_local_p=0.05,               # ... and more high-high spots (local p < this) than elsewhere
    lr_n_show=5,                   # LR pairs listed per group
    lr_display_exclude=r"(?i)^(HLA-|H2-)|^(B2M|LCK|GNAS|GNAI2|GNB1|ARF1|CALM[123]|RIMS1|SPTBN2|SHANK1|PTPN6)\^",
                                   # MHC / B2M and intracellular signalling proteins as 'ligand': computed, not listed
                                   # nor counted as cell-type links (both species; case-insensitive)
    lr_top_n=100,                  # significant LR pairs (by global score) scored per sender / receiver cell type
    c2l_lr_n_perms=200,            # receptor-spot permutations, whole tissue
    c2l_lr_group_n_perms=50,       # ... within each group
    c2l_lr_group_top_k=10,         # sender -> receiver links kept per (group, LR pair)
    c2l_lr_min_z=3.0,              # group link counted when z >= this and score >= the whole-tissue score
    ref_lr_cells_per_type=400,     # reference cells per type for rank_aggregate
    ref_lr_spec_rank=0.05, ref_lr_mag_rank=0.2,    # specific and strong reference interactions (~5% of all,
                                                   # lymph node and brain; canonical chemokine pairs ~0.13-0.16)
    coloc_min=0.2,                 # spatially supported: tissue co-localisation r >= this ...
    coloc_group_min=1.0,           # ... or group co-localisation share >= this (~ top 5% of group values)
    # --- subclustering (subcluster_adaptive)
    sub_n_hvg=2000,
    sub_n_pcs=20,
    sub_n_neighbors=15,
    sub_batch="combat",            # 'combat' (scanpy, per batch) or None
    sub_min_gene_frac=0.02,        # genes detected in >= this fraction of the subset's spots / cells
    sub_resolutions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
    sub_min_size=20,               # every subcluster must have >= this many spots / cells
    sub_max_k=6,                   # at most this many subclusters
    # --- subregions (section 8c; mp_mouse_brain03 / mp_lymphoid03): gene sets, flags, tables
    subregion_key="sub_region_id",            # obs column, e.g. 'THAL_2'
    subregion_skip=(),                        # regions not subclustered (artefacts, e.g. low-quality spots)
    region_colors={},                         # region -> colour (others: REGION_COLORS)
    region_table_first=(),                    # calc_df_region_table: these columns first (the others keep their order)
    region_table_depth=False,                 # calc_df_region_table: add median_UMIs, median_cells_per_spot
    subregion_table_first=(),                 # calc_df_subregion_table: these columns first
    region_sig_lfc=1.0, region_sig_padj=0.05, region_sig_min_pct=0.25,
    sub_sig_lfc=0.75, sub_sig_padj=0.05, sub_sig_min_pct=0.20,
    notpres_max_pct=0.05,                     # 'absent': detected in <= 5% of the subregion's spots
    notpres_min_pct_rest=0.30,                # ... and in >= 30% of the rest of the region
    notpres_padj=0.05,
    gene_universe="detected",                 # 'detected' (in >= universe_min_pct of the region's spots) or 'all'
    universe_min_pct=0.01,
    n_genes_show=12,                          # genes listed per set in the summary tables
    gene_display_exclude=None,                # regex of genes not listed (still counted), e.g. r"^Gm\d+$|Rik$"
    coherence_min_ratio=1.5,                  # 'supported' subregion: spatial coherence ratio >= this
    sub_min_sig_genes=3,                      # ... and >= this many subregion signature genes
    section_min_frac=0.10,                    # below this fraction in one section -> flagged 'single-section'
    low_depth_ratio=0.6,                      # median UMIs of subregion / region below this -> flagged 'low-depth'
    high_depth_ratio=2.0,                     # ... above this -> 'high-depth' (signature counts inflated)
    border_min_frac=0.25,                     # >= this fraction of neighbours in another region -> 'border with X'
    border_min_rel=0.0,                       # ... and border_frac / region_border_frac >= this (0 = not used)
    composition_min_mean=0.05,                # cell types with mean density >= this (cells per spot) are ranked
    composition_top=3,
    subregion_map_title="{lib}: subregions of each region",
    plotly_include_js="cdn",                  # write_html: 'cdn' (small files, needs internet to view) or True
    # --- marker tests (calc_df_rank_genes)
    rank_corr_method="benjamini-hochberg",
    # --- spatial coherence
    coherence_radius_factor=1.2,   # spot neighbours within factor x spot spacing (Visium: the 6 neighbours)
    # --- plotting
    plotly_image_format="jpeg",    # 'jpeg' (small HTML) or 'raw'
    plotly_jpeg_quality=85,
    clip_quantile=99.2,            # percent, for colour scales
    density_quantile=0.992,        # fraction, for density overlays
    spot_size=1.3,
    img_key="hires",
    dpi=300,
    # --- figure style: defaults of the plot methods' font_color / bg_color / fontsize_scale (argument None = these)
    font_color="black",
    bg_color="white",
    fontsize_scale=1.0,            # x the font sizes set inside the matplotlib plot methods (style_figure)
)


class SingleCell(object):
    """Generic single-cell / spatial analysis object wrapping one AnnData (self.adata).

    Parameters
    ----------
    ROOT0       : project root (e.g. /home/flavio/uv/cell_cell_communication)
    project     : project sub-folder (e.g. 'mouse_brain'); root_project = ROOT0 / project
    adata       : AnnData or None (set later with read_h5ad / read_visium_samples)
    data_dir, figure_dir, result_dir : sub-folders of root_project (created if missing)
    symbol_col  : var column with gene symbols when var_names are Ensembl IDs
    sample_col  : obs column with the sample / section id
    mt_prefix   : mitochondrial gene prefix ('mt-' mouse, 'MT-' human; matched case-insensitively)
    params      : overrides of DEFAULT_PARAMS
    """

    default_params: Dict[str, Any] = DEFAULT_PARAMS
    default_files: Dict[str, Tuple[str, str]] = {}      # key -> (root, name); see file_path
    analysis_files: Dict[str, Dict[str, Tuple[str, str]]] = {}   # {'inputs': {...}, '<analysis>': {...}} (subclasses)
    analysis_params: Dict[str, Dict[str, Any]] = {}              # {'<analysis>': {param: value}} (subclasses)

    def __init__(self, ROOT0: Path, project: str, adata: Optional[ad.AnnData] = None,
                 data_dir: str = "data", figure_dir: str = "figures", result_dir: str = "results",
                 symbol_col: str = "SYMBOL", sample_col: str = "sample", mt_prefix: str = "mt-",
                 params: Optional[Dict[str, Any]] = None, name: str = "",
                 files: Optional[Dict[str, Tuple[str, str]]] = None, analysis: Optional[str] = None):
        
        self.ROOT0 = Path(ROOT0)
        self.project = project

        self.root_project = create_dir(self.ROOT0, project)
        self.root_data = create_dir(self.root_project, data_dir)
        self.root_figure = create_dir(self.root_project, figure_dir)
        self.root_result = create_dir(self.root_project, result_dir)

        self.adata = adata
        self.symbol_col = symbol_col
        self.sample_col = sample_col
        self.mt_prefix = mt_prefix
        self.name = name or project
        # files: analysis_files['inputs'] always; an analysis (e.g. 'lymph03', 'mb03') adds its outputs, which turns on
        # the caching of its methods (force=False: read the file when it exists); files= adds or overrides entries
        if analysis is not None and analysis not in self.analysis_files:
            raise KeyError(f"analysis {analysis!r} not in {sorted(k for k in self.analysis_files if k != 'inputs')}")
        self.analysis = analysis
        # params: default_params (DEFAULT_PARAMS + project) < analysis_params[analysis] < params=
        self.params = copy.deepcopy(self.default_params)
        if analysis in self.analysis_params:
            self.set_params(**copy.deepcopy(self.analysis_params[analysis]))
        if params:
            self.set_params(**params)
        self.files: Dict[str, Tuple[str, str]] = {**copy.deepcopy(self.default_files),
                                                  **self.analysis_files.get("inputs", {}),
                                                  **(self.analysis_files[analysis] if analysis else {}), **(files or {})}
        self.create_file_dirs()

    def __repr__(self):
        shape = None if self.adata is None else self.adata.shape
        return (f"{type(self).__name__}(project={self.project!r}, adata={shape}, "
                f"symbol_col={self.symbol_col!r}, sample_col={self.sample_col!r})")

    # -----------------------------------------------------------------------------------------------------
    # parameters
    # -----------------------------------------------------------------------------------------------------
    def set_params(self, **kw) -> Dict[str, Any]:
        """Override parameters; unknown keys raise KeyError (catches typos)."""
        unknown = [k for k in kw if k not in self.params]
        if unknown:
            raise KeyError(f"unknown parameter(s): {unknown}; known: {sorted(self.params)}")
        self.params.update(kw)
        return self.params

    def get_param(self, key: str, value=None):
        """value if given, else self.params[key]."""
        return self.params[key] if value is None else value

    def calc_df_params(self) -> pd.DataFrame:
        """Current parameters as a table (parameter, value, default)."""
        return pd.DataFrame([dict(parameter=k, value=v, default=self.default_params.get(k))
                             for k, v in self.params.items()])

    # -----------------------------------------------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------------------------------------------
    def _check_adata(self, adata=None) -> ad.AnnData:
        a = self.adata if adata is None else adata
        if a is None:
            raise ValueError("no AnnData: pass adata= or set self.adata")
        return a

    # -----------------------------------------------------------------------------------------------------
    # files: registry self.files (key -> (root, name)) and cached tables
    # -----------------------------------------------------------------------------------------------------
    def file_path(self, key: str, **fmt) -> Path:
        """Path of a registered file. self.files[key] = (root, name): root 'data' | 'result' | 'figure' (the project
        folders) or 'root0' (ROOT0, inputs outside the project); name may contain sub-folders and {placeholders}
        filled from fmt (e.g. lib=..., region=...)."""
        if key not in self.files:
            raise KeyError(f"file key {key!r} is not registered; known: {sorted(self.files)}")
        root, name = self.files[key]
        roots = {"data": self.root_data, "result": self.root_result, "figure": self.root_figure, "root0": self.ROOT0}
        fmt = {k: str(v).replace("/", "-") for k, v in fmt.items()}      # 'CTX_L2/3' -> 'CTX_L2-3' (no sub-folder)
        return Path(roots[root]) / name.format(**fmt)

    def fname(self, key: str, **fmt) -> str:
        """Name of a registered file relative to its root folder, for the methods that take a file name in root_data /
        root_figure / root_result (fname=, fname_tsv=, ...). Placeholders in fmt are filled ('/' -> '-'); the others
        are kept for the method that writes the file (e.g. '{k}', the figure number of spatial_scatter_genes)."""
        if key not in self.files:
            raise KeyError(f"file key {key!r} is not registered; known: {sorted(self.files)}")

        class _Keep(dict):
            def __missing__(self, k):
                return "{" + k + "}"

        return self.files[key][1].format_map(_Keep({k: str(v).replace("/", "-") for k, v in fmt.items()}))

    def create_file_dirs(self) -> List[Path]:
        """create_dir for the sub-folders of the registered files (e.g. results/lymph03_subregions); called by __init__."""
        roots = {"data": self.root_data, "result": self.root_result, "figure": self.root_figure, "root0": self.ROOT0}
        subs = sorted({(root, str(Path(name).parent)) for root, name in self.files.values() if str(Path(name).parent) != "."})
        return [create_dir(roots[root], sub) for root, sub in subs]

    def _resolve_path(self, fname: str, root: Optional[Path], default_root: Path, **fmt) -> Path:
        """Registered key (and no explicit root) -> file_path(key); otherwise (root or default_root) / fname."""
        if root is None and fname in self.files:
            return self.file_path(fname, **fmt)
        return Path(root or default_root) / (fname.format(**fmt) if fmt else fname)

    def calc_df_files(self) -> pd.DataFrame:
        """Registered files: key, root, name, path, exists."""
        rows = []
        for k, (root, name) in self.files.items():
            p = self.file_path(k) if "{" not in name else None
            rows.append(dict(key=k, root=root, name=name, path=str(p) if p else "", exists=bool(p and p.exists())))
        return pd.DataFrame(rows)

    def read_tsv(self, key: str, str_cols: Optional[Sequence[str]] = None, verbose: bool = False, **fmt) -> pd.DataFrame:
        """pdreadcsv of a registered table (FileNotFoundError if missing). Empty strings are written as empty fields
        and read back as NaN: NaN -> '' in object columns and in str_cols (text columns that may be all empty)."""
        path = self.file_path(key, **fmt)
        if not path.exists():
            raise FileNotFoundError(path)
        df = pdreadcsv(path.name, path.parent)
        str_cols = set(str_cols or ())
        for c in df.columns:
            if c in str_cols:
                df[c] = df[c].fillna("").astype(str)
            elif df[c].dtype == object:
                df[c] = df[c].fillna("")
        if verbose:
            print(f"Reading {path} ({len(df):,} rows)")
        return df

    def write_tsv(self, df: pd.DataFrame, key: str, index: bool = False, verbose: bool = False, **fmt) -> Path:
        """pdwritecsv of df to the registered file (its folder is created by __init__: create_file_dirs)."""
        path = self.file_path(key, **fmt)
        if not pdwritecsv(df, path.name, path.parent, index=index, verbose=verbose):
            raise IOError(f"could not write {path}")
        return path

    def _save_figure(self, fig, fname: Optional[str],
    force: bool = False, verbose: bool = False, **fmt) -> Optional[Path]:
        """Save a figure inside its plot method. fname: key of self.files ({placeholders} from fmt, e.g. lib=..., region=...)
        or a file name with an extension in root_figure; None, or a key that is not registered (e.g. another analysis),
        -> not saved. An existing file is kept unless force (as _savefig). Plotly -> write_html (params['plotly_include_js']),
        matplotlib -> savefig (params['dpi'])."""
        if fname is None or (fname not in self.files and not Path(fname).suffix):
            return None
        path = self._resolve_path(fname, None, self.root_figure, **fmt)
        if path.exists() and not force:
            if verbose:
                print(f"File already exists: {path}")
            return path
        if hasattr(fig, "write_html"):
            fig.write_html(path, include_plotlyjs=self.params["plotly_include_js"])
        else:
            fig.savefig(path, dpi=self.params["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
        if verbose:
            print(f"Saving figure to {path}")
        return path

    def _cached_table(self, key: str, compute, force: bool = False, verbose: bool = False,
                      str_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
        """Cached table: file registered, present and not force -> read it; else df = compute() and write it.
        Key not registered (no analysis files): compute only, nothing written. After writing, the table is read back,
        so a computed and a cached run return the same values (text float parsing can differ by 1 ulp from the
        in-memory float). The file holds the result of the parameters of the run that wrote it: use force=True after
        changing parameters or inputs."""
        if key is None or key not in self.files:
            return compute()
        if not (self.file_path(key).exists() and not force):
            self.write_tsv(compute(), key, verbose=verbose)
        return self.read_tsv(key, str_cols=str_cols, verbose=verbose)

    def _savefig(self, fig, fname: Optional[str], root_figure: Optional[Path] = None, dpi: Optional[int] = None, 
                 force: bool = False, verbose: bool = False):
        """Save fig to (root_figure or self.root_figure) / fname; fname=None -> not saved."""
        if fname is None:
            return None
        filename = self._resolve_path(fname, root_figure, self.root_figure)

        if filename.exists() and not force:
            if verbose:
                print(f"File already exists: {filename}")
        else:
            fig.savefig(filename, dpi=self.get_param("dpi", dpi), bbox_inches="tight", facecolor=fig.get_facecolor())
            if verbose:
                print(f"Saving figure to {filename}")

        return filename

    def _write_table(self, df: pd.DataFrame, fname: Optional[str], force: bool = False, verbose: bool = False,
                     sep: str = "\t") -> Optional[Path]:
        """Write a table a method returns (key of self.files or a name in root_result; None = not written). As the
        figures (_savefig), an existing file is kept unless force; pdwritecsv prints with verbose."""
        if fname is None:
            return None
        path = self._resolve_path(fname, None, self.root_result)
        if path.exists() and not force:
            if verbose:
                print(f"File already exists: {path}")
            return path
        pdwritecsv(df, path.name, path.parent, sep=sep, verbose=verbose)
        return path

    # ---- colours and font scale of a finished matplotlib figure -------------------------------------------------
    def _muted(self, bg_color: str = "white") -> str:
        """Muted grey readable on bg_color: dark grey on a light background, light grey on a dark one."""
        return "#555555" if self._luminance(bg_color) > 0.5 else "#AAAAAA"

    @staticmethod
    def _luminance(color) -> float:
        """Relative luminance (0 = black, 1 = white) of a matplotlib colour."""
        r, g, b = mpl.colors.to_rgb(mpl.colors.to_rgba(color))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def style_figure(self, fig, font_color: str = "black", bg_color: str = "white", fontsize_scale: float = 1.0,
                     min_contrast: float = 0.3, relayout: bool = False):
        """Recolour and rescale a finished matplotlib figure, including figures drawn by scanpy, squidpy and
        cell2location, so that one call gives e.g. white text on black (font_color='white', bg_color='black').

        - figure background, legend frames and every axes whose background was the figure background -> bg_color
          (axes with their own background, e.g. the black panels of the spatial plots, keep it);
        - texts (titles, labels, tick labels, legends, colour bars, figure texts), tick marks and spines whose
          colour is too close to bg_color (|luminance difference| < min_contrast) -> font_color; coloured labels
          (lineage colours) keep their colour; annotations inside image axes (heatmap values, labels on the H&E)
          are left as they are; white halos (path effects) around recoloured text -> bg_color;
        - every font size x fontsize_scale (tick labels through the axis, so new ticks get it too).
        relayout: re-run tight_layout (figures laid out with tight_layout before styling).
        Defaults (black on white, scale 1) leave the figure unchanged. Returns fig."""
        from matplotlib.text import Text
        import matplotlib.patheffects as mpe

        if fig is None:
            return fig
        lum_bg = self._luminance(bg_color)

        def dark_ok(c):
            """Keep colour c: transparent, saturated (lineage colours, even dark ones) or visible on bg_color."""
            rgba = mpl.colors.to_rgba(c)
            mx, mn = max(rgba[:3]), min(rgba[:3])
            saturated = mx > 0 and (mx - mn) / mx >= 0.35
            return rgba[3] == 0 or saturated or abs(self._luminance(rgba) - lum_bg) >= min_contrast
        if font_color == "black" and bg_color == "white" and fontsize_scale == 1:
            return fig

        fig_bg0 = mpl.colors.to_hex(fig.get_facecolor(), keep_alpha=False)
        fig.patch.set_facecolor(bg_color)
        tick_texts, skip_texts = set(), set()

        for ax in fig.axes:
            if mpl.colors.to_hex(ax.get_facecolor(), keep_alpha=False) == fig_bg0:
                ax.set_facecolor(bg_color)
            for sp_ in ax.spines.values():
                if not dark_ok(sp_.get_edgecolor()):
                    sp_.set_edgecolor(font_color)
            for axis in (ax.xaxis, ax.yaxis):
                for which in ("major", "minor"):
                    ticks = axis.get_major_ticks() if which == "major" else axis.get_minor_ticks()
                    if not ticks:
                        continue
                    labs = [t.label1 for t in ticks] + [t.label2 for t in ticks]
                    tick_texts.update(map(id, labs))
                    kw = {}
                    if not dark_ok(ticks[0].tick1line.get_color()):
                        kw["color"] = font_color
                    shown = [l for l in labs if l.get_text()]       # colour-bar labels are filled at draw time
                    if (not shown and not dark_ok(labs[0].get_color())) or \
                            (shown and all(not dark_ok(l.get_color()) for l in shown)):
                        kw["labelcolor"] = font_color                       # persists for ticks created later
                    else:
                        for l in shown:
                            if not dark_ok(l.get_color()):
                                l.set_color(font_color)
                    if fontsize_scale != 1:
                        kw["labelsize"] = ticks[0].label1.get_fontsize() * fontsize_scale
                    if kw:
                        sizes = [l.get_fontsize() for l in labs]
                        axis.set_tick_params(which=which, **kw)
                        if fontsize_scale != 1:                            # keep per-label sizes, if they differ
                            for l, s in zip(labs, sizes):
                                l.set_fontsize(s * fontsize_scale)
            if ax.images:                                                   # heatmap values, labels on the H&E
                skip_texts.update(id(t) for t in ax.texts)

        for leg in list(fig.legends) + [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]:
            fr = leg.get_frame()
            fr.set_facecolor(bg_color)
            if not dark_ok(fr.get_edgecolor()):
                fr.set_edgecolor(font_color)

        seen = set()
        for t in fig.findobj(Text):
            if id(t) in seen:
                continue
            seen.add(id(t))
            if fontsize_scale != 1 and id(t) not in tick_texts:
                t.set_fontsize(t.get_fontsize() * fontsize_scale)
            if id(t) in skip_texts or id(t) in tick_texts:
                continue
            if not dark_ok(t.get_color()):
                t.set_color(font_color)
            for p in (t.get_path_effects() or []):                          # halo of the text colour -> bg
                gc = getattr(p, "_gc", None)
                if isinstance(gc, dict) and "foreground" in gc and not \
                        abs(self._luminance(gc["foreground"]) - self._luminance(t.get_color())) >= min_contrast:
                    gc["foreground"] = bg_color
        if relayout:
            try:
                fig.tight_layout()
            except Exception:
                pass
        return fig


    # =====================================================================================================
    # 1. I/O and components
    # =====================================================================================================
    def read_h5ad(self, fname: str, root: Optional[Path] = None, backed: Optional[str] = None,
                  set_self: bool = True, verbose: bool = False, **fmt) -> ad.AnnData:
        """
        Read (root or self.root_data) / fname, or the registered file when fname is a key of self.files
        (backed='r' keeps X on disk). Sets self.adata unless set_self=False.
        """
        filename = self._resolve_path(fname, root, self.root_data, **fmt)

        if not filename.exists():
            raise FileNotFoundError(filename)
        
        a = ad.read_h5ad(filename, backed=backed)
        if set_self:
            self.adata = a

        if verbose:
            print(f"Reading {filename}")

        return a

    def write_h5ad(self, fname: str, adata=None, root: Optional[Path] = None,
                   compression: Optional[str] = "gzip", verbose: bool = False, **fmt) -> Path:
        """Write to (root or self.root_data) / fname, or to the registered file when fname is a key of self.files
        ({placeholders} from fmt, e.g. region=...)."""
        filename = self._resolve_path(fname, root, self.root_data, **fmt)

        self._check_adata(adata).write_h5ad(filename, compression=compression)

        if verbose:
            print(f"Writing {filename}")

        return filename

    def read_h5ad_slots(self, fname: str, root: Optional[Path] = None, skip: Sequence[str] = ("layers",),
                        set_self: bool = False, verbose: bool = False) -> ad.AnnData:
        """
        Read an h5ad without some slots (default: layers). Useful for cell2location maps whose
        per-cell-type expected-expression layers are dense (e.g. 34 layers x spots x genes, several GB).
        fname: file name in (root or root_data), or a key of self.files.
        """
        import h5py
        from anndata.io import read_elem
        filename = self._resolve_path(fname, root, self.root_data)
        if not filename.exists():
            raise FileNotFoundError(filename)
        slots = ("X", "obs", "var", "obsm", "varm", "obsp", "varp", "layers", "uns")
        kw = {}
        with h5py.File(filename, "r") as f:
            for s in slots:
                if s in skip or s not in f:
                    continue
                kw[s] = read_elem(f[s])
        a = ad.AnnData(**kw)
        if set_self:
            self.adata = a
        if verbose:
            print(f"Reading {filename} (skipped: {list(skip)})")
        return a

    def read_visium_samples(self, root_raw: Path, samples: Sequence[str],
                            count_file: str = "filtered_feature_bc_matrix.h5",
                            ensembl_index: bool = True, set_self: bool = True, verbose: bool = False) -> ad.AnnData:
        """
        Read several Space Ranger outputs (root_raw/<sample>/) into one AnnData.

        obs[sample_col] = sample id; obs_names = '<sample>_<barcode>'; uns['spatial'] keeps every image.
        ensembl_index=True -> var_names = Ensembl IDs, symbols in var[symbol_col] (cell2location convention).
        """
        ads = []
        for s in samples:
            a = sc.read_visium(Path(root_raw) / s, count_file=count_file, library_id=s)

            if verbose:
                print(f"Reading {s}")

            a.var_names_make_unique()
            a.obs[self.sample_col] = s

            if ensembl_index and "gene_ids" in a.var:
                a.var[self.symbol_col] = a.var_names
                a.var_names = a.var.pop("gene_ids")

            a.obs_names = s + "_" + a.obs_names
                
            ads.append(a)

        out = ad.concat(ads, join="outer", uns_merge="unique")
        if ensembl_index:
            out.var[self.symbol_col] = ads[0].var[self.symbol_col].reindex(out.var_names)
        if set_self:
            self.adata = out

        return out

    def calc_df_components(self, adata=None) -> pd.DataFrame:
        """
        One row per AnnData slot entry (X, obs, var, obsm, varm, obsp, layers, uns) with type and shape.
        """

        a = self._check_adata(adata)
        rows = [dict(slot="X", key="", type=type(a.X).__name__, shape=str(a.shape))]
        rows += [dict(slot="obs", key=c, type=str(a.obs[c].dtype), shape=str(a.obs[c].shape)) for c in a.obs.columns]
        rows += [dict(slot="var", key=c, type=str(a.var[c].dtype), shape=str(a.var[c].shape)) for c in a.var.columns]

        for slot in ("obsm", "varm", "obsp", "varp", "layers"):
            m = getattr(a, slot)
            for k in m.keys():
                rows.append(dict(slot=slot, key=k, type=type(m[k]).__name__, shape=str(getattr(m[k], "shape", ""))))

        for k, v in a.uns.items():
            rows.append(dict(slot="uns", key=k, type=type(v).__name__,
                             shape=str(list(v.keys())[:6]) if isinstance(v, dict) else str(getattr(v, "shape", ""))))
            
        df_comp = pd.DataFrame(rows)

        return df_comp

    def x_info(self, adata=None, n_rows: int = 2000) -> dict:
        """
        Type, density and value range of X on the first n_rows: is it raw counts?
        """

        a = self._check_adata(adata)
        sub = a.X[:min(n_rows, a.n_obs)]

        if sp.issparse(sub):
            d, nnz = sub.data, sub.nnz
        else:
            sub = np.asarray(sub)
            d = sub[sub != 0]
            nnz = d.size

        tot = sub.shape[0] * sub.shape[1]

        return dict(type=type(a.X).__name__, dtype=str(a.X.dtype), shape=a.shape,
                    density=round(nnz / tot, 4) if tot else 0.0,
                    min=float(d.min()) if d.size else 0.0, max=float(d.max()) if d.size else 0.0,
                    integer=bool(np.allclose(d, np.round(d))) if d.size else True,
                    nnz_per_row=round(nnz / max(sub.shape[0], 1), 1))

    def is_raw_counts(self, adata=None) -> bool:
        info = self.x_info(adata)
        return info["integer"] and info["min"] >= 0

    def summary(self, adata=None, max_cols: int = 30) -> None:
        """Print shape, obs / var columns, obsm / layers / uns keys, Visium images and an X check."""
        a = self._check_adata(adata)
        print(a)
        print("\nX:", self.x_info(a))
        print(f"\nobs columns ({a.obs.shape[1]}):", list(a.obs.columns)[:max_cols])
        print(f"var columns ({a.var.shape[1]}):", list(a.var.columns)[:max_cols])
        print("obsm:", {k: getattr(v, "shape", None) for k, v in a.obsm.items()})
        print("layers:", list(a.layers.keys()))
        print("uns:", list(a.uns.keys()))
        for lib, d in a.uns.get("spatial", {}).items():
            print(f"  library {lib}: images {[(k, v.shape) for k, v in d.get('images', {}).items()]}")

    def value_counts(self, cols: Sequence[str], adata=None, top: int = 40) -> Dict[str, pd.Series]:
        a = self._check_adata(adata)
        return {c: a.obs[c].value_counts(dropna=False).head(top) for c in cols if c in a.obs}

    # =====================================================================================================
    # 2. Gene lookup and clipping
    # =====================================================================================================
    def symbols(self, adata=None) -> pd.Series:
        """var_names -> gene symbol (var[symbol_col] when present, otherwise var_names)."""
        a = self._check_adata(adata)
        if self.symbol_col in a.var:
            return a.var[self.symbol_col].astype(str)
        return pd.Series(a.var_names, index=a.var_names)

    def values(self, feature: str, adata=None, layer: Optional[str] = None) -> np.ndarray:
        """1-D array for an obs column, a var_name, or a gene symbol."""
        a = self._check_adata(adata)
        if feature in a.obs:
            return a.obs[feature].values
        if feature in a.var_names:
            gid = feature
        else:
            hits = a.var_names[self.symbols(a).values == feature]
            if len(hits) == 0:
                raise KeyError(f"{feature} not found in obs, var_names or var['{self.symbol_col}']")
            gid = hits[0]
        x = a[:, gid].layers[layer] if layer else a[:, gid].X
        return np.asarray(x.toarray() if sp.issparse(x) else x).ravel()

    def resolve_gene_ids(self, genes: Iterable[str], adata=None) -> Tuple[List[str], List[str]]:
        """Gene symbols -> var_names. Missing genes are warned about and skipped. Returns (symbols, var_ids)."""
        sym = self.symbols(adata)
        genes = list(genes)
        found = [g for g in genes if (sym == g).any()]
        missing = [g for g in genes if g not in found]
        if missing:
            warnings.warn(f"not in var['{self.symbol_col}']: {missing}")
        ids = [sym.index[sym == g][0] for g in found]
        return found, ids

    @staticmethod
    def percentile_cap(v: np.ndarray, q: float = 99.2) -> float:
        """q-th percentile of v; if 0 (very sparse feature), q-th percentile of the non-zero values."""
        v = np.asarray(v, dtype=float)
        cap = float(np.percentile(v, q))
        if cap == 0:
            nz = v[v > 0]
            cap = float(np.percentile(nz, q)) if nz.size else 1.0
        return cap

    def clip_for_plot(self, features: Sequence[str], adata=None, q: Optional[float] = None, vmin: float = 0,
                      sparse_fallback: bool = True) -> Tuple[List[str], List[str]]:
        """Write percentile-clipped copies to obs['_plot_<feature>']. Returns (features_kept, obs_columns)."""
        a = self._check_adata(adata)
        q = self.get_param("clip_quantile", q)
        sym = set(self.symbols(a))
        kept = [g for g in features if g in a.obs or g in a.var_names or g in sym]
        dropped = [g for g in features if g not in kept]
        if dropped:
            warnings.warn(f"not found, dropped: {dropped}")
        cols = []
        for g in kept:
            col = f"_plot_{g}"
            if g in a.obs and isinstance(a.obs[g].dtype, pd.CategoricalDtype):
                a.obs[col] = a.obs[g]
            else:
                v = np.asarray(self.values(g, a), dtype=float)
                vmax = self.percentile_cap(v, q) if sparse_fallback else float(np.percentile(v, q))
                a.obs[col] = np.clip(v, vmin, vmax)
            cols.append(col)
        return kept, cols

    # =====================================================================================================
    # 3. Filtering
    # =====================================================================================================
    def flag_mt_genes(self, adata=None, prefix: Optional[str] = None) -> pd.Series:
        """var['MT_gene'] = symbol starts with the mitochondrial prefix (case-insensitive)."""
        a = self._check_adata(adata)
        prefix = (prefix or self.mt_prefix).lower()
        a.var["MT_gene"] = self.symbols(a).str.lower().str.startswith(prefix).values
        return a.var["MT_gene"]

    def remove_mt_genes(self, adata=None, keep_in_obsm: bool = True) -> ad.AnnData:
        """
        Drop mitochondrial genes (cell2location recommendation); 
        their counts go to obsm['MT'].
        Returns the filtered AnnData (also stored in self.adata when adata is None).
        """

        a = self._check_adata(adata)
        mt = self.flag_mt_genes(a).values
        if keep_in_obsm:
            X = a[:, mt].X
            a.obsm["MT"] = X.toarray() if sp.issparse(X) else np.asarray(X)
        out = a[:, ~mt].copy()
        if adata is None:
            self.adata = out
        return out

    def filter_cells(self, adata=None, min_genes: Optional[int] = None, max_genes: Optional[int] = None,
                     min_counts: Optional[int] = None, max_pct_mt: Optional[float] = None,
                     max_doublet: Optional[float] = None, doublet_col: Optional[str] = None,
                     inplace: bool = True) -> Tuple[ad.AnnData, pd.DataFrame]:
        """
        Cell / spot filter on QC metrics (run calc_qc_metrics first). Arguments left None come from
        self.params (qc_min_genes, qc_max_genes, qc_min_counts, qc_max_pct_mt, qc_max_doublet); a None
        parameter disables that criterion. Returns (filtered AnnData, df_report of cells removed per criterion).
        """

        a = self._check_adata(adata)
        obs = a.obs

        doublet_col = self.get_param("qc_doublet_col", doublet_col)

        crit = [("n_genes_by_counts >= min_genes", self.get_param("qc_min_genes", min_genes), "n_genes_by_counts", np.greater_equal),
                ("n_genes_by_counts < max_genes", self.get_param("qc_max_genes", max_genes), "n_genes_by_counts", np.less),
                ("total_counts >= min_counts", self.get_param("qc_min_counts", min_counts), "total_counts", np.greater_equal),
                ("pct_counts_mt < max_pct_mt", self.get_param("qc_max_pct_mt", max_pct_mt), "pct_counts_mt", np.less),
                (f"{doublet_col} < max_doublet", self.get_param("qc_max_doublet", max_doublet), doublet_col, np.less)]
        
        keep = np.ones(a.n_obs, bool)
        rows = []
        for label, thr, col, op in crit:
            if thr is None:
                continue
            if col not in obs:
                warnings.warn(f"obs['{col}'] missing: criterion '{label}' skipped")
                continue
            m = op(obs[col].values, thr)
            rows.append(dict(criterion=label, threshold=thr, failing=int((~m).sum()), removed_here=int((~m & keep).sum())))
            keep &= m

        df_report = pd.DataFrame(rows, columns=["criterion", "threshold", "failing", "removed_here"])
        df_report.attrs["kept"] = int(keep.sum())
        df_report.attrs["total"] = int(a.n_obs)

        out = a[keep].copy()
        if inplace and adata is None:
            self.adata = out

        return out, df_report

    def filter_genes(self, adata=None, min_cells: Optional[int] = None, inplace: bool = True) -> ad.AnnData:
        a = self._check_adata(adata)
        keep = sc.pp.filter_genes(a, min_cells=self.get_param("gene_min_cells", min_cells), inplace=False)[0]
        out = a[:, keep].copy()
        if inplace and adata is None:
            self.adata = out
        return out

    def filter_genes_c2l(self, adata=None, cell_count_cutoff: Optional[int] = None,
                         cell_percentage_cutoff2: Optional[float] = None, nonz_mean_cutoff: Optional[float] = None,
                         figsize=(3, 4), inplace: bool = True, font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None):
        """
        cell2location permissive gene selection (keeps rare-cell-type markers). Returns (AnnData, genes).
        Style of the filter plot: font_color, bg_color, fontsize_scale (style_figure).
        """
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        from cell2location.utils.filtering import filter_genes

        a = self._check_adata(adata)
        selected = filter_genes(a, cell_count_cutoff=self.get_param("c2l_cell_count_cutoff", cell_count_cutoff),
                                cell_percentage_cutoff2=self.get_param("c2l_cell_percentage_cutoff2", cell_percentage_cutoff2),
                                nonz_mean_cutoff=self.get_param("c2l_nonz_mean_cutoff", nonz_mean_cutoff), figsize=figsize)
        self.style_figure(plt.gcf(), font_color, bg_color, fontsize_scale)
        out = a[:, selected].copy()
        if inplace and adata is None:
            self.adata = out
        return out, list(selected)

    # =====================================================================================================
    # 4. Quality control
    # =====================================================================================================
    def calc_qc_metrics(self, adata=None, layer: Optional[str] = None) -> pd.DataFrame:
        """scanpy calculate_qc_metrics with an 'mt' variable (n_genes_by_counts, total_counts, pct_counts_mt)."""
        a = self._check_adata(adata)
        a.var["mt"] = self.flag_mt_genes(a).values
        sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], layer=layer, inplace=True, percent_top=None, log1p=True)
        df_qc = a.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]]
        return df_qc

    def calc_df_qc(self, groupby: Optional[str] = None, adata=None) -> pd.DataFrame:
        """n, median and IQR of n_genes_by_counts, total_counts, pct_counts_mt per group (default sample_col)."""
        a = self._check_adata(adata)
        g = groupby or self.sample_col
        cols = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        grp = a.obs.groupby(g, observed=True)
        df_qc = pd.concat({"n": grp.size(), **{f"median_{c}": grp[c].median() for c in cols},
                           **{f"q25_{c}": grp[c].quantile(.25) for c in cols},
                           **{f"q75_{c}": grp[c].quantile(.75) for c in cols}}, axis=1)
        return df_qc.round(2)

    def plot_qc(self, groupby: Optional[str] = None, adata=None, fname: Optional[str] = "qc_violin.png",
                title: str = "", thresholds: Optional[Dict[str, float]] = None, max_points: Optional[int] = None,
                force: bool = False, verbose: bool = False):
        """Violins of n_genes_by_counts, total_counts, pct_counts_mt per group + counts-vs-genes scatter.
        thresholds: {'n_genes_by_counts': 5000, 'pct_counts_mt': 10} draws dashed lines."""
        a = self._check_adata(adata)
        g = groupby or self.sample_col
        thresholds = thresholds or {}
        cols = ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
        obs = a.obs
        cats = list(obs[g].astype("category").cat.categories) if g in obs else ["all"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), gridspec_kw=dict(wspace=0.35))
        for ax, c in zip(axes[:3], cols):
            data = [obs.loc[obs[g] == k, c].values for k in cats] if g in obs else [obs[c].values]
            parts = ax.violinplot(data, showextrema=False, showmedians=True)
            for b in parts["bodies"]:
                b.set_alpha(0.6)
            ax.set_xticks(range(1, len(cats) + 1))
            ax.set_xticklabels(cats, rotation=60, ha="right")
            ax.set_title(c)
            if c in thresholds:
                ax.axhline(thresholds[c], color="#b2182b", ls="--", lw=0.8)
        rng = np.random.default_rng(self.params["seed"])
        idx = rng.choice(a.n_obs, min(self.get_param("qc_max_points", max_points), a.n_obs), replace=False)
        sca = axes[3].scatter(obs["total_counts"].values[idx], obs["n_genes_by_counts"].values[idx],
                              c=obs["pct_counts_mt"].values[idx], s=1, cmap="viridis", rasterized=True)
        axes[3].set_xscale("log")
        axes[3].set_xlabel("total_counts")
        axes[3].set_ylabel("n_genes_by_counts")
        fig.colorbar(sca, ax=axes[3], fraction=0.05).set_label("pct_counts_mt")
        if title:
            fig.suptitle(title, x=0.02, ha="left", y=1.04)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    # =====================================================================================================
    # 5. Scanpy preprocessing / embedding
    # =====================================================================================================
    def normalize_log(self, adata=None, target_sum: Optional[float] = None, counts_layer: Optional[str] = None) -> ad.AnnData:
        """Keep raw counts in layers[counts_layer]; X <- log1p(CP<target_sum>)."""
        a = self._check_adata(adata)
        counts_layer = self.get_param("counts_layer", counts_layer)
        if counts_layer not in a.layers:
            a.layers[counts_layer] = a.X.copy()
        sc.pp.normalize_total(a, target_sum=self.get_param("target_sum", target_sum))
        sc.pp.log1p(a)
        return a

    def highly_variable(self, adata=None, n_top_genes: Optional[int] = None, flavor: Optional[str] = None,
                        counts_layer: Optional[str] = None, batch_key: Optional[str] = None):
        a = self._check_adata(adata)
        flavor = self.get_param("hvg_flavor", flavor)
        counts_layer = self.get_param("counts_layer", counts_layer)
        layer = counts_layer if flavor == "seurat_v3" and counts_layer in a.layers else None
        sc.pp.highly_variable_genes(a, n_top_genes=self.get_param("n_top_genes", n_top_genes), flavor=flavor,
                                    layer=layer, batch_key=batch_key)
        return a.var["highly_variable"]

    def embed(self, adata=None, n_pcs: Optional[int] = None, n_neighbors: Optional[int] = None, use_hvg: bool = True,
              use_rep: Optional[str] = None, min_dist: Optional[float] = None) -> ad.AnnData:
        """PCA (on HVGs if present) -> kNN graph -> UMAP."""
        a = self._check_adata(adata)
        seed = self.params["seed"]
        if use_rep is None:
            mask = a.var["highly_variable"].values if (use_hvg and "highly_variable" in a.var) else None
            sc.pp.pca(a, n_comps=self.get_param("n_pcs", n_pcs), mask_var=mask, random_state=seed)
            use_rep = "X_pca"
        sc.pp.neighbors(a, n_neighbors=self.get_param("n_neighbors", n_neighbors), use_rep=use_rep, random_state=seed)
        sc.tl.umap(a, min_dist=self.get_param("umap_min_dist", min_dist), random_state=seed)
        return a

    def leiden(self, adata=None, resolution: Optional[float] = None, key_added: str = "leiden"):
        a = self._check_adata(adata)
        sc.tl.leiden(a, resolution=self.get_param("leiden_resolution", resolution), key_added=key_added,
                     random_state=self.params["seed"], flavor="igraph", n_iterations=2, directed=False)
        return a.obs[key_added]

    def rank_genes(self, groupby: str, adata=None, method: Optional[str] = None, n_genes: Optional[int] = None,
                   key_added: str = "rank_genes_groups") -> pd.DataFrame:
        a = self._check_adata(adata)
        sc.tl.rank_genes_groups(a, groupby=groupby, method=self.get_param("rank_method", method),
                                n_genes=self.get_param("rank_n_genes", n_genes), key_added=key_added)
        df_rank = sc.get.rank_genes_groups_df(a, group=None, key=key_added)
        return df_rank

    # =====================================================================================================
    # 6. Squidpy / spatial helpers
    # =====================================================================================================
    def library_ids(self, adata=None) -> List[str]:
        return list(self._check_adata(adata).uns.get("spatial", {}).keys())

    def library_id(self, adata=None) -> str:
        return self.library_ids(adata)[0]

    def hires_image(self, library_id: Optional[str] = None, adata=None, img_key: Optional[str] = None):
        """(image, scale factor) of one Visium library."""
        a = self._check_adata(adata)
        img_key = self.get_param("img_key", img_key)
        d = a.uns["spatial"][library_id or self.library_id(a)]
        return d["images"][img_key], d["scalefactors"][f"tissue_{img_key}_scalef"]

    def hires_image_bg(self, library_id: Optional[str] = None, adata=None, bg_color: str = "#3a3a3a",
                       white_min: float = 0.90, min_area: int = 5000, edge_px: int = 3, img_key: Optional[str] = None,
                       force: bool = False, verbose: bool = False) -> np.ndarray:
        """Copy of the hires image (float RGB in [0, 1]) with the glass background outside the tissue set to bg_color;
        the image in adata is not changed. Tissue = connected areas of pixels with min(R, G, B) < white_min larger
        than min_area pixels, holes filled (ventricles stay as in the image); smaller blobs (fiducial frame, dust)
        become background. The last edge_px pixels inside the tissue edge are blended by whiteness (no white rim).
        Cached as files['hires_bg'] ({lib}, {color}) when registered (PNG, force=False: read it); otherwise kept in
        memory for the session."""
        from scipy import ndimage as ndi
        lib = library_id or self.library_id(self._check_adata(adata))
        color = mpl.colors.to_hex(bg_color).lstrip("#").lower()
        key_mem = (lib, color, white_min, min_area, edge_px)
        cache = self.__dict__.setdefault("_hires_bg_cache", {})
        if key_mem in cache and not force:
            return cache[key_mem]
        reg = "hires_bg" in self.files
        path = self.file_path("hires_bg", lib=lib, color=color) if reg else None
        if reg and path.exists() and not force:
            from PIL import Image
            out = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
            if verbose:
                print(f"Reading {path}")
        else:
            img, _ = self.hires_image(lib, adata, img_key)
            x = np.asarray(img[..., :3], dtype=np.float32)
            x = x / 255 if x.max() > 1.5 else x
            mn = x.min(2)
            lab_, n = ndi.label(ndi.binary_closing(mn < white_min, iterations=2))
            sizes = ndi.sum(np.ones_like(mn), lab_, index=np.arange(1, n + 1))
            tissue = ndi.binary_fill_holes(np.isin(lab_, 1 + np.flatnonzero(sizes >= min_area)))
            band = tissue & ~ndi.binary_erosion(tissue, iterations=edge_px)
            alpha = np.where(tissue, 0.0, 1.0)
            alpha[band] = np.clip((mn[band] - 0.75) / (white_min + 0.05 - 0.75), 0, 1)
            bg = np.array(mpl.colors.to_rgb("#" + color), dtype=np.float32)
            out = (x * (1 - alpha[..., None]) + bg * alpha[..., None]).astype(np.float32)
            if reg:
                from PIL import Image
                Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8)).save(path)
                if verbose:
                    print(f"Saving {path} (tissue {tissue.mean():.0%} of the image)")
        cache[key_mem] = out
        return out

    def _resolve_table(self, fname: Optional[str], root: Optional[Path] = None) -> Optional[Path]:
        """Path of a table given as a key of self.files or as a file name with an extension (in root or root_result);
        None when fname is None or an unregistered key."""
        if fname is None or (fname not in self.files and not Path(fname).suffix):
            return None
        return self._resolve_path(fname, root, self.root_result)

    def _plotly_style(self, fig, fontsize_title: int, fontsize_xticks: int, fontsize_yticks: int, fontsize_hover: int,
                      font_color: str, bg_color: str):
        """Common Plotly styling of the heatmaps: title / tick fonts, global font colour, backgrounds, hover font."""
        fig.update_layout(title_font=dict(size=fontsize_title, color=font_color), font=dict(color=font_color),
                          paper_bgcolor=bg_color, plot_bgcolor=bg_color, hoverlabel=dict(font=dict(size=fontsize_hover)))
        fig.update_xaxes(tickfont=dict(size=fontsize_xticks, color=font_color))
        fig.update_yaxes(tickfont=dict(size=fontsize_yticks, color=font_color))
        return fig

    def _map_image(self, library_id: str, adata_map, image_bg_color: Optional[str] = None, force: bool = False,
                   verbose: bool = False) -> np.ndarray:
        """hires image of a map: the original, or the copy with the background set to image_bg_color."""
        if image_bg_color is None:
            return self.hires_image(library_id, adata_map)[0]
        return self.hires_image_bg(library_id, adata_map, bg_color=image_bg_color, force=force, verbose=verbose)

    def spot_xy(self, library_id: Optional[str] = None, adata=None, img_key: Optional[str] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """(mask of the spots of one library, their centres in image pixels)."""
        a = self._check_adata(adata)
        lib = library_id or self.library_id(a)
        m = (a.obs[self.sample_col].astype(str) == lib).values if self.sample_col in a.obs else np.ones(a.n_obs, bool)
        _, sf = self.hires_image(lib, a, img_key)
        return m, np.asarray(a.obsm["spatial"])[m] * sf

    def select_slide(self, adata=None, library_id: Optional[str] = None, batch_key: Optional[str] = None) -> ad.AnnData:
        """
        One Visium section (cell2location.utils.select_slide); default: the first library.
        slide is a smaller AnnData that contains only one section
        like,
        adata_vis.uns['spatial']['V1_Human_Lymph_Node']
        """
        from cell2location.utils import select_slide
        a = self._check_adata(adata)
        lib = library_id or self.library_id(a)
        return select_slide(a, lib, batch_key=batch_key or self.sample_col)

    def spatial_scatter_genes(self, genes: Sequence[str], adata=None, library_id: Optional[str] = None,
                              ncols: int = 4, nframe: Optional[int] = None, q: Optional[float] = None,
                              img_alpha: float = 0.6, size: Optional[float] = None, cmap=None,
                              font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                              fname: Optional[str] = None,
                              force: bool = False, verbose: bool = False) -> Tuple[List[str], List[Any]]:
        """squidpy spatial_scatter of genes (symbols or obs columns), each clipped at its q-th percentile
        (clip_for_plot), nframe panels per figure. fname may contain '{k}' (figure number, 1-based).
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status.
        Returns (features plotted, figures)."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        import squidpy as sq
        a = self._check_adata(adata)
        kept, cols = self.clip_for_plot(list(genes), a, q=q)
        nframe = nframe or len(kept)
        lib = library_id or self.library_id(a)
        figs = []
        for k, i in enumerate(range(0, len(kept), nframe)):
            sq.pl.spatial_scatter(a, color=cols[i:i + nframe], title=kept[i:i + nframe],
                                  img_res_key=self.params["img_key"], img_alpha=img_alpha,
                                  size=self.get_param("spot_size", size), cmap=cmap or cmap_byr, ncols=ncols,
                                  library_id=lib)
            fig = self.style_figure(plt.gcf(), font_color, bg_color, fontsize_scale)
            figs.append(fig)
            if fname is not None:
                self._savefig(fig, fname.format(k=k + 1) if "{k}" in fname else fname, force=force, verbose=verbose)

        return kept, figs

    def spatial_neighbors(self, adata=None, coord_type: Optional[str] = None, n_rings: Optional[int] = None,
                          library_key: Optional[str] = None, **kw):
        import squidpy as sq
        a = self._check_adata(adata)
        lk = library_key or (self.sample_col if self.sample_col in a.obs and a.obs[self.sample_col].nunique() > 1 else None)
        if lk is not None:
            a.obs[lk] = a.obs[lk].astype("category")
        sq.gr.spatial_neighbors(a, coord_type=self.get_param("sq_coord_type", coord_type),
                                n_rings=self.get_param("sq_n_rings", n_rings), library_key=lk, **kw)
        return a

    def spatial_autocorr(self, genes: Optional[Sequence[str]] = None, adata=None, mode: str = "moran",
                         n_perms: Optional[int] = None, n_jobs: Optional[int] = None) -> pd.DataFrame:
        import squidpy as sq
        a = self._check_adata(adata)
        if genes is None and "highly_variable" in a.var:
            genes = a.var_names[a.var["highly_variable"]].tolist()
        sq.gr.spatial_autocorr(a, mode=mode, genes=genes, n_perms=self.get_param("sq_n_perms", n_perms),
                               n_jobs=self.get_param("sq_n_jobs", n_jobs))
        df_autocorr = a.uns["moranI" if mode == "moran" else "gearyC"]
        return df_autocorr

    def nhood_enrichment(self, cluster_key: str, adata=None, n_perms: Optional[int] = None):
        import squidpy as sq
        a = self._check_adata(adata)
        sq.gr.nhood_enrichment(a, cluster_key=cluster_key, n_perms=self.get_param("sq_n_perms", n_perms),
                               seed=self.params["seed"])
        return a.uns[f"{cluster_key}_nhood_enrichment"]

    # =====================================================================================================
    # 7. cell2location
    # =====================================================================================================
    def train_reference(self, adata_ref: ad.AnnData, run_dir: Path, labels_key: str, batch_key: Optional[str] = None,
                        categorical_covariate_keys: Optional[List[str]] = None, max_epochs: Optional[int] = None,
                        accelerator: Optional[str] = None, fname: str = "sc.h5ad",
                        force: bool = False) -> Tuple[ad.AnnData, pd.DataFrame]:
        """cell2location RegressionModel (NB regression) -> reference signatures; cache-or-train in run_dir.
        Returns (adata_ref with posterior in varm, df_inf_aver genes x cell types)."""
        from cell2location.models import RegressionModel
        run_dir = create_dir(run_dir)
        filename = run_dir / fname
        accelerator = self.get_param("accelerator", accelerator)
        if filename.exists() and not force:
            adata_ref = ad.read_h5ad(filename)
        else:
            RegressionModel.setup_anndata(adata=adata_ref, batch_key=batch_key, labels_key=labels_key,
                                          categorical_covariate_keys=categorical_covariate_keys)
            mod = RegressionModel(adata_ref)
            mod.train(max_epochs=self.get_param("ref_max_epochs", max_epochs), accelerator=accelerator)
            adata_ref = mod.export_posterior(adata_ref, use_quantiles=True, add_to_varm=list(self.params["c2l_quantiles"]),
                                             sample_kwargs={"batch_size": self.params["ref_batch_size"],
                                                            "accelerator": accelerator})
            mod.save(str(run_dir), overwrite=True)
            adata_ref.write_h5ad(filename)
        df_inf_aver, _ = self.get_reference_signatures(adata_ref)
        return adata_ref, df_inf_aver

    @staticmethod
    def get_reference_signatures(adata_ref, prefer: Sequence[str] = ("means", "q50", "q05")) -> Tuple[pd.DataFrame, str]:
        """
        Per-cell-type signatures (genes x cell types) from RegressionModel.export_posterior (varm or var).
        Also reads the cell2location v0.05 layout (var['means_per_cluster_mu_fg_<type>']).
        """

        factor_names = list(adata_ref.uns["mod"]["factor_names"])

        for summary in prefer:
            key = f"{summary}_per_cluster_mu_fg"
            cols = [f"{key}_{i}" for i in factor_names]
            
            if key in adata_ref.varm:
                df_inf_aver = adata_ref.varm[key][cols].copy()
            elif cols[0] in adata_ref.var:
                df_inf_aver = adata_ref.var[cols].copy()
            else:
                continue
            
            df_inf_aver.columns = factor_names

            return df_inf_aver, summary
        
        raise KeyError(f"No per-cluster signatures for {list(prefer)}; rerun mod.export_posterior(adata_ref, ...)")

    def train_mapping(self, adata_vis: ad.AnnData, df_inf_aver: pd.DataFrame, run_dir: Path,
                      N_cells_per_location: Optional[float] = None, detection_alpha: Optional[float] = None,
                      batch_key: Optional[str] = None, max_epochs: Optional[int] = None,
                      accelerator: Optional[str] = None, fname: str = "sp.h5ad", force: bool = False,
                      **model_kw) -> ad.AnnData:
        """cell2location spatial mapping; cache-or-train (run_dir/fname + model).

        Genes are intersected between adata_vis and df_inf_aver. Posterior quantiles go to
        obsm['<q>_cell_abundance_w_sf']. Returns the mapped AnnData.
        """
        import os
        import torch
        import cell2location
        run_dir = create_dir(run_dir)
        filename = run_dir / fname
        if filename.exists() and not force:
            return ad.read_h5ad(filename)
        accelerator = self.get_param("accelerator", accelerator)
        torch.set_num_threads(self.params["torch_threads"] or len(os.sched_getaffinity(0)))
        max_epochs = self.get_param("c2l_max_epochs", max_epochs)
        N = self.get_param("c2l_N_cells_per_location", N_cells_per_location)
        alpha = self.get_param("c2l_detection_alpha", detection_alpha)
        genes = np.intersect1d(adata_vis.var_names, df_inf_aver.index)
        a = adata_vis[:, genes].copy()
        df_inf = df_inf_aver.loc[genes, :].copy()
        cell2location.models.Cell2location.setup_anndata(adata=a, batch_key=batch_key)
        mod = cell2location.models.Cell2location(a, cell_state_df=df_inf, N_cells_per_location=N,
                                                 detection_alpha=alpha, **model_kw)
        mod.train(max_epochs=max_epochs, batch_size=None, train_size=1, accelerator=accelerator)
        a = mod.export_posterior(a, use_quantiles=True, add_to_obsm=list(self.params["c2l_quantiles"]),
                                 sample_kwargs={"batch_size": a.n_obs, "accelerator": accelerator})
        a.uns["c2l_training"] = dict(max_epochs=max_epochs, N_cells_per_location=N, detection_alpha=alpha,
                                     n_genes=len(genes), accelerator=str(accelerator),
                                     elbo_history=np.asarray(mod.history["elbo_train"]).ravel().astype(float))
        mod.save(str(run_dir), overwrite=True)
        a.write_h5ad(filename)
        return a

    @staticmethod
    def ensure_posterior_samples(mod, adata, quantiles: Sequence[str] = ("q05", "q50", "q95", "q0001"),
                                 accelerator: str = "gpu", batch_size: Optional[int] = None, spatial: bool = True):
        """Re-create mod.samples after re-loading a model (mod.samples is not saved with the model)."""
        if getattr(mod, "samples", None):
            return adata
        if batch_size is None:
            batch_size = mod.adata.n_obs if spatial else 2500
        slot = {"add_to_obsm": list(quantiles)} if spatial else {"add_to_varm": list(quantiles)}
        return mod.export_posterior(adata, use_quantiles=True,
                                    sample_kwargs={"batch_size": batch_size, "accelerator": accelerator}, **slot)

    def calc_df_abundance(self, adata=None, obsm_key: Optional[str] = None) -> pd.DataFrame:
        """Spots x cell types abundance table from obsm (column prefixes removed)."""
        a = self._check_adata(adata)
        obsm_key = self.get_param("c2l_obsm_key", obsm_key)
        X = a.obsm[obsm_key]
        if isinstance(X, pd.DataFrame):
            df_abund = X.copy()
        else:
            df_abund = pd.DataFrame(np.asarray(X), index=a.obs_names, columns=list(a.uns["mod"]["factor_names"]))
        prefix = obsm_key.replace("_cell_abundance_w_sf", "cell_abundance_w_sf_")
        df_abund.columns = [str(c).replace(prefix, "") for c in df_abund.columns]
        return df_abund

    def cluster_regions(self, adata=None, df_abund: Optional[pd.DataFrame] = None, n_neighbors: Optional[int] = None,
                        resolution: Optional[float] = None, key_added: Optional[str] = None,
                        flavor: Optional[str] = None) -> pd.Series:
        """Leiden clusters of spots on cell-abundance space (paper: kNN 38, resolution 1.3).

        The kNN graph is stored under neighbors_key='abund' (umap_regions uses it).
        flavor (params['region_leiden_flavor']): 'igraph' (n_iterations=2, undirected) or 'leidenalg'
        (scanpy defaults, as in the cell2location tutorial). The two give different clusters."""
        a = self._check_adata(adata)
        key_added = self.get_param("region_key", key_added)
        flavor = self.get_param("region_leiden_flavor", flavor)
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(a)
        a.obsm["_abund"] = df_abund.loc[a.obs_names].values
        seed = self.params["seed"]
        sc.pp.neighbors(a, use_rep="_abund", n_neighbors=self.get_param("region_n_neighbors", n_neighbors),
                        key_added="abund", random_state=seed)
        resolution = self.get_param("region_resolution", resolution)
        if flavor == "igraph":
            sc.tl.leiden(a, resolution=resolution, neighbors_key="abund", key_added=key_added, random_state=seed,
                         flavor="igraph", n_iterations=2, directed=False)
        else:
            sc.tl.leiden(a, resolution=resolution, neighbors_key="abund", key_added=key_added, random_state=seed,
                         flavor="leidenalg")
        a.obs[key_added] = a.obs[key_added].astype("category")
        return a.obs[key_added]

    def umap_regions(self, adata=None, min_dist: Optional[float] = None, spread: Optional[float] = None,
                     neighbors_key: str = "abund") -> np.ndarray:
        """UMAP of the spots on the abundance kNN graph built by cluster_regions (obsm['X_umap'])."""
        a = self._check_adata(adata)
        sc.tl.umap(a, min_dist=self.get_param("region_umap_min_dist", min_dist),
                   spread=self.get_param("region_umap_spread", spread), neighbors_key=neighbors_key,
                   random_state=self.params["seed"])
        return a.obsm["X_umap"]

    def region_names(self, adata=None, key: Optional[str] = None, df_abund: Optional[pd.DataFrame] = None,
                     top_n: Optional[int] = None, method: Optional[str] = None,
                     min_enrich: Optional[float] = None) -> Dict[str, str]:
        """Name each region by top_n cell types (enrichment = region mean / overall mean).
        method='abundance': among types with enrichment > min_enrich, the highest mean abundance; 'ratio': by enrichment."""
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        top_n = self.get_param("region_top_n", top_n)
        method = self.get_param("region_method", method)
        min_enrich = self.get_param("region_min_enrich", min_enrich)
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(a)
        df_abund = df_abund.loc[a.obs_names]
        df_mean = df_abund.groupby(a.obs[key].values, observed=False).mean()
        df_enrich = df_mean / df_abund.mean(axis=0)
        names = {}
        for cl in df_enrich.index:
            by_ratio = list(df_enrich.loc[cl].sort_values(ascending=False).index)
            if method == "ratio":
                top = by_ratio[:top_n]
            else:
                cand = df_mean.loc[cl][df_enrich.loc[cl] > min_enrich].sort_values(ascending=False)
                top = list(cand.index[:top_n])
                top += [c for c in by_ratio if c not in top][:top_n - len(top)]
            names[str(cl)] = " / ".join(top)
        return names

    # =====================================================================================================
    # 7b. cell2location model objects: cache-or-train returning the model (plot_history, plot_QC, samples)
    # =====================================================================================================
    def fit_or_load_reference(self, adata_ref: ad.AnnData, run_dir: Path, labels_key: str,
                              batch_key: Optional[str] = None, categorical_covariate_keys: Optional[List[str]] = None,
                              max_epochs: Optional[int] = None, accelerator: Optional[str] = None,
                              fname: str = "sc.h5ad", quantiles: Optional[Sequence[str]] = None,
                              force: bool = False, verbose: bool = True):
        """
        cell2location RegressionModel (NB regression of reference signatures), tutorial workflow.

        setup_anndata -> if run_dir/fname is missing (or force): train, export the posterior (num_samples draws:
        means/stds/q05/q95 in varm), save model + h5ad; else load the saved model bound to adata_ref.
        Then the posterior quantiles (params['c2l_quantiles'], use_quantiles=True) are exported to varm if missing,
        which also sets mod.samples (needed by mod.plot_QC).

        Returns (mod_reg, adata_ref, trained).
        """
        from cell2location.models import RegressionModel
        
        run_dir = create_dir(run_dir)
        filename = run_dir / fname
        accelerator = self.get_param("accelerator", accelerator)
        quantiles = list(self.get_param("c2l_quantiles", quantiles))
        
        RegressionModel.setup_anndata(adata=adata_ref, batch_key=batch_key, labels_key=labels_key,
                                      categorical_covariate_keys=categorical_covariate_keys)
        
        trained = False
        if force or not filename.exists():
            mod = RegressionModel(adata_ref)
            mod.train(max_epochs=self.get_param("ref_max_epochs", max_epochs), accelerator=accelerator)
            adata_ref = mod.export_posterior(adata_ref, sample_kwargs={"num_samples": self.params["c2l_num_samples"],
                                                                       "batch_size": self.params["ref_batch_size"],
                                                                       "accelerator": accelerator})
            mod.save(str(run_dir), overwrite=True, save_anndata=True)
            adata_ref.write_h5ad(filename)
            
            trained = True
            
            if verbose:
                print(f"RegressionModel trained and saved to {run_dir}")
        else:
            mod = RegressionModel.load(str(run_dir), adata_ref, accelerator=accelerator)
            
            adata_ref = mod.adata
            
            if verbose:
                print(f"RegressionModel loaded from {run_dir}")
        
        # a loaded model has no mod.samples (not saved with the model): re-export also when varm already has the
        # quantiles (e.g. the cell re-run with adata_ref = mod.adata of the previous run), else plot_QC fails
        if not all(f"{q}_per_cluster_mu_fg" in adata_ref.varm for q in quantiles) or not getattr(mod, "samples", None):
            adata_ref = mod.export_posterior(adata_ref, use_quantiles=True, add_to_varm=quantiles,
                                             sample_kwargs={"batch_size": self.params["ref_batch_size"],
                                                            "accelerator": accelerator})
            if verbose:
                print(f"posterior quantiles exported to varm: {quantiles}")
        return mod, adata_ref, trained

    def fit_or_load_mapping(self, adata_vis: ad.AnnData, df_inf_aver: pd.DataFrame, run_dir: Path,
                            N_cells_per_location: Optional[float] = None, detection_alpha: Optional[float] = None,
                            batch_key: Optional[str] = None, max_epochs: Optional[int] = None,
                            accelerator: Optional[str] = None, fname: str = "sp.h5ad", expected_layers: bool = True,
                            read_layers: bool = True, force: bool = False, verbose: bool = True):
        """
        cell2location spatial mapping, tutorial workflow.

        Cache missing (or force): genes shared with df_inf_aver, setup_anndata(batch_key), Cell2location(N, alpha),
        train (full data, batch_size=None, train_size=1), export posterior quantiles to obsm, optional
        expected expression per cell type in layers (compute_expected_per_cell_type), save model + h5ad.
        Cache present: read run_dir/fname (read_layers=False skips the dense per-cell-type layers), load the
        saved model and re-create mod.samples (ensure_posterior_samples).
        In both cases the q05 abundances are copied to obs (one column per cell type).

        Returns (mod_c2l, adata_vis, trained).
        """
        import cell2location

        run_dir = create_dir(run_dir)
        filename = run_dir / fname
        accelerator = self.get_param("accelerator", accelerator)
        batch_key = batch_key or self.sample_col
        quantiles = list(self.params["c2l_quantiles"])
        
        trained = False
        if filename.exists() and not force:
            adata_vis = ad.read_h5ad(filename) if read_layers else self.read_h5ad_slots(fname, root=run_dir)
            mod = cell2location.models.Cell2location.load(str(run_dir), adata_vis, accelerator=accelerator)
            adata_vis = self.ensure_posterior_samples(mod, adata_vis, quantiles=quantiles, accelerator=accelerator)
            if verbose:
                print(f"Cell2location map read from {filename}; model loaded; posterior samples: {sorted(mod.samples)}")
        else:
            genes = np.intersect1d(adata_vis.var_names, df_inf_aver.index)
            adata_vis = adata_vis[:, genes].copy()
            df_inf = df_inf_aver.loc[genes, :].copy()
            cell2location.models.Cell2location.setup_anndata(adata=adata_vis, batch_key=batch_key)
            mod = cell2location.models.Cell2location(
                adata_vis, cell_state_df=df_inf,
                N_cells_per_location=self.get_param("c2l_N_cells_per_location", N_cells_per_location),
                detection_alpha=self.get_param("c2l_detection_alpha", detection_alpha))
            mod.train(max_epochs=self.get_param("c2l_max_epochs", max_epochs), batch_size=None, train_size=1,
                      accelerator=accelerator)
            adata_vis = mod.export_posterior(adata_vis, use_quantiles=True, add_to_obsm=quantiles,
                                             sample_kwargs={"batch_size": mod.adata.n_obs, "accelerator": accelerator})
            if expected_layers:
                self.add_expected_layers(mod, adata_vis)
            mod.save(str(run_dir), overwrite=True)
            self.abundance_to_obs(adata_vis)
            adata_vis.write_h5ad(filename)
            trained = True
            if verbose:
                print(f"Cell2location trained ({len(genes):,} genes); model and map saved to {run_dir}")
        self.abundance_to_obs(adata_vis)
        return mod, adata_vis, trained

    def add_expected_layers(self, mod, adata, quantile: Optional[str] = None) -> List[str]:
        """Expected expression of every gene per cell type (RCTD-like) -> adata.layers[<cell type>]."""
        quantile = self.get_param("c2l_expected_quantile", quantile)
        expected = mod.module.model.compute_expected_per_cell_type(mod.samples[f"post_sample_{quantile}"],
                                                                   mod.adata_manager)
        for i, n in enumerate(mod.factor_names_):
            adata.layers[n] = expected["mu"][i]
        return list(mod.factor_names_)

    def abundance_to_obs(self, adata=None, obsm_key: Optional[str] = None) -> List[str]:
        """Copy the abundance table (default q05, 'at least this many cells') to obs, one column per cell type."""
        a = self._check_adata(adata)
        df_abund = self.calc_df_abundance(a, obsm_key)
        for c in df_abund.columns:
            a.obs[c] = df_abund[c].values
        return list(df_abund.columns)

    def plot_c2l_history(self, mod, iter_start: int = 0, label: Optional[str] = None, figsize=(5, 3.5),
                         font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                         fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """ELBO loss history (mod.plot_history) on a new figure, first iter_start epochs omitted.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        fig, ax = plt.subplots(figsize=figsize)
        mod.plot_history(iter_start, ax=ax)
        if label:
            ax.legend(labels=[label])
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=True)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def plot_c2l_qc(self, mod, summary_name: str = "q50", figsize=(3, 3), use_n_obs: int = 1000,
                    font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                    fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> list:
        """mod.plot_QC(summary_name) with every figure kept (cell2location calls plt.show() internally; each
        figure is styled before it is shown). fname='x.png' saves x_1.png, x_2.png, ... Returns the figures.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        if not getattr(mod, "samples", None):
            raise RuntimeError("mod.samples is missing (model loaded from disk without its posterior): use the model "
                               "returned by fit_or_load_reference / fit_or_load_mapping, which re-export it")
        figs = []
        show = plt.show

        def _show(*a, **k):
            figs.append(self.style_figure(plt.gcf(), font_color, bg_color, fontsize_scale, relayout=True))
            show(*a, **k)

        plt.show = _show
        try:
            with plt.rc_context({"figure.figsize": figsize}):
                mod.plot_QC(summary_name=summary_name, use_n_obs=use_n_obs)
        finally:
            plt.show = show
        if not figs:
            figs = [self.style_figure(plt.gcf(), font_color, bg_color, fontsize_scale, relayout=True)]
        if fname:
            for i, f in enumerate(figs):
                self._savefig(f, fname.replace(".png", f"_{i + 1}.png"), force=force, verbose=verbose)
        return figs

    def posterior_samples_cached(self, mod, adata, run_dir: Path, fname: str = "posterior_samples_w_sf.npz",
                                 sites: Optional[Sequence[str]] = None, num_samples: Optional[int] = None,
                                 batch_size: Optional[int] = None, accelerator: Optional[str] = None,
                                 verbose: bool = True) -> Dict[str, np.ndarray]:
        """Posterior draws of selected sites (default w_sf, m_g, z_sr_groups_factors), cached in an npz with
        obs/factor/var names (the cache is checked against adata.obs_names). Returns {site: array}."""
        import scvi
        filename = Path(run_dir) / fname
        sites = list(self.get_param("c2l_posterior_sites", sites))
        meta = {"obs_names", "factor_names", "var_names"}
        if filename.exists():
            with np.load(filename, allow_pickle=False) as z:
                if len(z["obs_names"]) != adata.n_obs or not (z["obs_names"] == np.asarray(adata.obs_names, dtype=str)).all():
                    raise ValueError(f"{filename}: obs_names differ from adata")
                ps = {k: z[k] for k in z.files if k not in meta}
            if verbose:
                print(f"posterior samples loaded from {filename}: {sorted(ps)}")
            return ps
        scvi.settings.seed = self.params["seed"]
        out = mod.sample_posterior(num_samples=self.get_param("c2l_num_samples", num_samples),
                                   accelerator=self.get_param("accelerator", accelerator), return_samples=True,
                                   batch_size=batch_size or adata.n_obs, return_sites=sites)
        ps = {k: np.asarray(out["posterior_samples"][k], dtype=np.float32) for k in sites}
        np.savez(filename, **ps, obs_names=np.asarray(adata.obs_names, dtype=str),
                 factor_names=np.asarray(mod.factor_names_, dtype=str), var_names=np.asarray(adata.var_names, dtype=str))
        if verbose:
            print(f"posterior samples saved to {filename}")
        return ps

    def plot_posterior_median_vs_mean(self, mod, ps: Optional[Dict[str, np.ndarray]] = None, site: str = "w_sf",
                                      accelerator: Optional[str] = None, figsize=(4, 4), font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                      fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """Posterior median (mod.posterior_quantile q=0.5) vs posterior mean of one site (from mod.samples
        'post_sample_means' when present, otherwise the mean of the draws in ps).
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        medians = mod.posterior_quantile(q=0.5, batch_size=mod.adata.n_obs,
                                         accelerator=self.get_param("accelerator", accelerator))
        if "post_sample_means" in getattr(mod, "samples", {}):
            means = np.asarray(mod.samples["post_sample_means"][site])
        elif ps is not None:
            means = ps[site].mean(0)
        else:
            raise ValueError("no posterior means: pass ps from posterior_samples_cached")
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(np.asarray(medians[site]).ravel(), np.asarray(means).ravel(), s=12, color="steelblue",
                   edgecolors="black", linewidths=0.3, alpha=0.8, rasterized=True)
        ax.set_xlabel("posterior median"); ax.set_ylabel("posterior mean")
        ax.set_title(f"{site}: median vs mean", loc="left")
        self.style_figure(fig, font_color, bg_color, fontsize_scale, relayout=True)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def run_colocation_cached(self, adata_vis: ad.AnnData, run_dir: Path, n_facts: Optional[Sequence[int]] = None,
                              n_restarts: Optional[int] = None, alpha: Optional[float] = None,
                              force: bool = False, verbose: bool = True) -> Tuple[ad.AnnData, Optional[dict], bool]:
        """
        NMF of the cell abundances (cell2location.run_colocation, CoLocatedGroupsSklearnNMF) for each number of
        factors; skipped when adata.uns already holds every 'mod_coloc_n_fact<n>'.
        Returns (adata_vis, res_dict or None when cached, edited).
        """
        n_facts = list(self.get_param("coloc_n_facts", n_facts))

        keys = [f"mod_coloc_n_fact{n}" for n in n_facts]

        if all(k in adata_vis.uns for k in keys) and not force:
            if verbose:
                print(f"NMF co-location already in adata.uns: {keys}")
            return adata_vis, None, False
        
        from cell2location import run_colocation
        
        res_dict, adata_vis = run_colocation(
            adata_vis, model_name="CoLocatedGroupsSklearnNMF",
            train_args={"n_fact": np.asarray(n_facts), "sample_name_col": self.sample_col,
                        "n_restarts": self.get_param("coloc_n_restarts", n_restarts)},
            model_kwargs={"alpha": self.get_param("coloc_alpha", alpha), "init": "random",
                          "nmf_kwd_args": {"tol": 1e-6}},
            export_args={"path": f"{run_dir}/CoLocatedComb/"})
        return adata_vis, res_dict, True

    @staticmethod
    def calc_df_coloc_loadings(adata, n_fact: int, normalize: bool = True) -> pd.DataFrame:
        """
        NMF cell-type loadings (cell types x factors) from adata.uns['mod_coloc_n_fact<n>'];
        normalize=True: each cell type's weights divided by their sum (relative weight across factors,
        as in plot_cell_type_loadings).
        """

        d = adata.uns[f"mod_coloc_n_fact{n_fact}"]
        W = np.asarray(d["post_sample_means"]["cell_type_factors"], dtype=float)
        df_load = pd.DataFrame(W, index=[str(x) for x in d["var_names"]], columns=[str(x) for x in d["fact_names"]])
        df_load.index = [i.replace("mean_cell_type_factors", "") for i in df_load.index]
        
        if normalize:
            df_load = df_load.div(df_load.sum(1).replace(0, np.nan), axis=0).fillna(0)

        return df_load

    # =====================================================================================================
    # 8. Plots
    # =====================================================================================================
    @staticmethod
    def legend_figure_bottom(fig, adata, key, names=None, ncol=4, fontsize=9):
        """
        Legend 'cluster: name' along the bottom of a layout='constrained' figure.
        """
        cats = adata.obs[key].cat.categories
        cols = adata.uns[f"{key}_colors"]
        names = names or {}
        handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=c, markeredgecolor="none",
                          label=f"{k}: {names[k]}" if k in names else str(k)) for k, c in zip(cats, cols)]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(ncol, len(handles)), frameon=False,
                   fontsize=fontsize, handletextpad=0.3, columnspacing=1.5)

    def umap_spatial_3plot(self, adata=None, names: Optional[Dict[str, str]] = None, 
                           figsize: Tuple[float, float] = (15, 6.5), key: Optional[str] = None,
                           fname: Optional[str] = "umap_region_sample.png", library_id: Optional[str] = None,
                           font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                           force: bool = False, verbose: bool = False):
        """UMAP by region, UMAP by sample, spatial map by region (one library) with a shared legend.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
        sc.pl.umap(a, color=key, size=30, legend_loc="on data", legend_fontsize=12, ax=axes[0], show=False, title=key)
        sc.pl.umap(a, color=self.sample_col, size=30, legend_loc="none", ax=axes[1], show=False, title=self.sample_col)
        lib = library_id or self.library_id(a)
        sub = a[a.obs[self.sample_col] == lib] if self.sample_col in a.obs else a
        sc.pl.spatial(sub, color=key, library_id=lib, ax=axes[2], size=self.params["spot_size"],
                      img_key=self.params["img_key"], alpha=0.5, title=lib, legend_loc="none", show=False)
        names = names if names is not None else self.region_names(a, key=key)
        self.legend_figure_bottom(fig, a, key, names=names, ncol=4)
        self.style_figure(fig, font_color, bg_color, fontsize_scale)          # constrained layout: re-laid out on draw
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def plot_spatial_visium(self, names: Dict[str, str], adata=None, key: Optional[str] = None,
                            library_id: Optional[str] = None, figsize=(8, 4), fname: Optional[str] = None,
                            force: bool = False, verbose: bool = False):
        """Spatial map of a categorical key on the hires image with a 'cluster – name' legend."""
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        s = library_id or self.library_id(a)
        sub = a[a.obs[self.sample_col] == s] if self.sample_col in a.obs else a
        with mpl.rc_context({"axes.facecolor": "black"}):
            fig, ax = plt.subplots(figsize=figsize)
            sc.pl.spatial(sub, color=key, library_id=s, size=self.params["spot_size"], img_key=self.params["img_key"],
                          alpha=0.5, legend_loc=None, ax=ax, show=False, title=s)
            cats = a.obs[key].cat.categories
            cols = a.uns[f"{key}_colors"]
            handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=c, markeredgecolor="none",
                              label=f"{k} – {names.get(k, '')}") for k, c in zip(cats, cols)]
            fig.legend(handles=handles, loc="center left", ncol=1, fontsize=6, frameon=False,
                       bbox_to_anchor=(0.92, 0.5), handletextpad=0.3)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def plotly_spatial_visium(self, names: Dict[str, str], adata=None, key: Optional[str] = None,
                              library_id: Optional[str] = None, set_colors: bool = False,
                              width: int = 1400, height: int = 900, image_format: Optional[str] = None,
                              marker_size: float = 6, hover_extra: Optional[Dict[str, str]] = None,
                              fontsize_title: int = 17, fontsize_legend: int = 12, fontsize_hover: int = 12,
                              font_color: Optional[str] = None, bg_color: Optional[str] = None, marker_line_color: str = "black",
                              fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """Interactive (plotly) spatial map of a categorical key on the hires image.

        image_format: 'raw' embeds the pixel array (about 30 MB per HTML file), 'jpeg' an encoded image (about 6 MB);
        default params['plotly_image_format']. hover_extra: category -> extra hover text (e.g. signature genes).
        Font sizes: fontsize_title, fontsize_legend, fontsize_hover. Colours: font_color, bg_color (paper and plot),
        marker_line_color (spot outline). fname: key of self.files or an .html name in root_figure (None = not
        saved); an existing file is kept unless force."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import base64
        import io
        import plotly.graph_objects as go
        image_format = self.get_param("plotly_image_format", image_format)
        hover_extra = hover_extra or {}
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        cats = a.obs[key].cat.categories
        cols = list(a.uns.get(f"{key}_colors", []))
        if set_colors or len(cols) < len(cats):
            palette = KELLY_COLORS + [mpl.colors.to_hex(c) for c in plt.get_cmap("tab20").colors]
            if len(cats) > len(palette):
                raise ValueError(f"{len(cats)} categories but only {len(palette)} colours")
            cols = palette[:len(cats)]
            a.uns[f"{key}_colors"] = cols
        s = library_id or self.library_id(a)
        img, _ = self.hires_image(s, a)
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        m, xy = self.spot_xy(s, a)
        clust = a.obs[key].astype(str).values[m]
        spot_names = a.obs_names[m]
        fig = go.Figure()
        if image_format == "jpeg":
            buf = io.BytesIO()
            from PIL import Image
            Image.fromarray(img[..., :3]).save(buf, format="JPEG", quality=self.params["plotly_jpeg_quality"])
            fig.add_trace(go.Image(source="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                                   hoverinfo="skip"))
        else:
            fig.add_trace(go.Image(z=img, hoverinfo="skip"))
        for k, c in zip(cats, cols):
            mk = clust == k
            if not mk.any():
                continue
            label = f"{k} – {names.get(k, '')}" if names.get(k) else str(k)
            extra = f"<br>{hover_extra[k]}" if k in hover_extra else ""
            fig.add_trace(go.Scattergl(x=xy[mk, 0], y=xy[mk, 1], mode="markers", name=label, text=spot_names[mk],
                                       marker=dict(size=marker_size, color=c, opacity=0.9,
                                                   line=dict(width=0.5, color=marker_line_color)),
                                       hovertemplate=f"<b>{label}</b>{extra}<br>spot: %{{text}}<extra></extra>"))
        fig.update_layout(title=dict(text=s, font=dict(size=fontsize_title, color=font_color)), width=width, height=height,
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          margin=dict(l=10, r=10, t=40 + fontsize_title, b=10),
                          hoverlabel=dict(font=dict(size=fontsize_hover)),
                          legend=dict(font=dict(size=fontsize_legend, color=font_color), itemsizing="constant",
                                      x=1.01, y=0.5, yanchor="middle"))
        fig.update_xaxes(visible=False, range=[0, img.shape[1]])
        fig.update_yaxes(visible=False, range=[img.shape[0], 0], scaleanchor="x")
        self._save_figure(fig, fname, force, verbose)
        return fig

    def plot_genes_per_cell_type(self, slide, genes, ctypes, q: Optional[float] = None, font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None):
        """Total expression (column 0) and cell-type-specific expected expression (layers[<cell type>]) per gene.
        Requires slide.layers[<cell type>] from mod.module.model.compute_expected_per_cell_type.
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). Not saved: returns (fig, axs)."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        qf = self.get_param("density_quantile", None if q is None else q)
        fig, axs = plt.subplots(nrows=len(genes), ncols=len(ctypes) + 1,
                                figsize=(4.5 * (len(ctypes) + 1) + 2, 5 * len(genes) + 1), squeeze=False)
        for j, g in enumerate(genes):
            gmask = (slide.var[self.symbol_col] == g).values
            caps = np.array([np.quantile(slide.layers[n][:, gmask].toarray(), qf) for n in slide.uns["mod"]["factor_names"]])
            cap = np.partition(caps.flatten(), -2)[-2]
            sc.pl.spatial(slide, cmap=cmap_byr, color=g, gene_symbols=self.symbol_col, size=self.params["spot_size"],
                          img_key=self.params["img_key"], vmin=0, vmax=f"p{100 * qf:g}", ax=axs[j, 0], show=False)
            for i, ct in enumerate(ctypes):
                sc.pl.spatial(slide, cmap=cmap_byr, color=g, layer=ct, gene_symbols=self.symbol_col,
                              size=self.params["spot_size"], img_key=self.params["img_key"], vmin=0, vmax=cap,
                              ax=axs[j, i + 1], show=False)
                axs[j, i + 1].set_title(f"{g} {ct}")
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        return fig, axs

    def plot_slide_loc_cell_types(self, slide, clust_labels, c2l_colors, figsize=(5, 5), font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                                  fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """cell2location.plt.plot_spatial for up to 6 cell types in one panel (legend outside, right).
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        from cell2location.plt import plot_spatial
        from matplotlib.patches import Patch
        with mpl.rc_context({"figure.figsize": figsize}):
            fig = plot_spatial(adata=slide, color=[str(i) for i in clust_labels], labels=clust_labels,
                               show_img=True, style="fast", max_color_quantile=self.params["density_quantile"],
                               circle_diameter=6, colorbar_position=None)
        handles = [Patch(facecolor=c2l_colors[i], edgecolor="black", label=lab) for i, lab in enumerate(clust_labels)]
        fig.axes[0].legend(handles=handles, title="cell type", loc="center left", bbox_to_anchor=(1.02, 0.5),
                           fontsize=10, title_fontsize=11, frameon=False, borderaxespad=0)
        if fname is None:
            fname = "spatial_" + "_".join(str(c).replace("+", "pos").replace("/", "-") for c in clust_labels) + ".png"
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    def plot_spatial_features(self, slide, features: Sequence[str], ncols: int = 4, figsize=(4.5, 5),
                              q: Optional[float] = None, cmap=None, font_color: Optional[str] = None, bg_color: Optional[str] = None, fontsize_scale: Optional[float] = None,
                              fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """sc.pl.spatial of several obs features (e.g. cell-type abundances) on a black background, colour scale
        clipped at the q-th percentile of each feature (cell2location tutorial style).
        Style (None = params): font_color, bg_color, fontsize_scale (style_figure). force: rewrite an existing file; verbose: print the file status."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        fontsize_scale = self.get_param("fontsize_scale", fontsize_scale)
        q = self.get_param("clip_quantile", q)
        with mpl.rc_context({"axes.facecolor": "black", "figure.figsize": list(figsize)}):
            sc.pl.spatial(slide, cmap=cmap or cmap_byr, color=list(features), ncols=ncols,
                          size=self.params["spot_size"], img_key=self.params["img_key"], vmin=0, vmax=f"p{q:g}",
                          show=False)
            fig = plt.gcf()
        self.style_figure(fig, font_color, bg_color, fontsize_scale)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    @staticmethod
    def plot_density_overlay(ax, xy: np.ndarray, values: Dict[str, np.ndarray], colors: Dict[str, str],
                             img: Optional[np.ndarray] = None, q: float = 0.992, s: float = 6, img_alpha: float = 0.0,
                             background: str = "black", min_alpha: float = 0.05):
        """Several cell-type densities in one panel (paper Fig 2D-F style). Each density is divided by its
        q quantile and clipped to [0, 1]; a spot takes the colour of the cell type with the highest scaled
        density and that value as opacity. Returns the per-type caps (cells per spot at the q quantile)."""
        ax.set_facecolor(background)
        if img is not None and img_alpha > 0:
            ax.imshow(img, alpha=img_alpha)
        names = list(values)
        V = np.column_stack([np.asarray(values[n], float) for n in names])
        cap = np.array([max(np.quantile(V[:, i], q), 1e-9) for i in range(V.shape[1])])
        S = np.clip(V / cap, 0, 1)
        best = S.argmax(1)
        alpha = S.max(1)
        rgba = np.array([mpl.colors.to_rgba(colors[names[b]]) for b in best])
        rgba[:, 3] = np.clip(alpha, min_alpha, 1)
        o = np.argsort(alpha)
        ax.scatter(xy[o, 0], xy[o, 1], c=rgba[o], s=s, lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        return dict(zip(names, cap))

    def calc_df_dotplot(self, genes: Sequence[str], groupby: str, adata=None, order: Optional[Sequence[str]] = None,
                        target_sum: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Marker table per group from raw counts in X (paper Fig 3B style).

        Genes are symbols (or var_names); repeated genes are allowed and kept in order.
        Returns (df_frac: fraction of cells with count > 0, df_mean: mean log1p(CP<target_sum>),
                 df_rel: df_mean / max over groups per gene, 0-1); rows = groups, columns = genes."""
        a = self._check_adata(adata)
        target_sum = self.get_param("target_sum", target_sum)
        uniq = list(dict.fromkeys(genes))
        found, ids = self.resolve_gene_ids(uniq, a)
        X = a[:, ids].X
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        tot = np.asarray(a.X.sum(1)).ravel()
        E = np.log1p(X / np.maximum(tot, 1)[:, None] * target_sum)
        grp = a.obs[groupby].astype(str).values
        df_frac = pd.DataFrame(X > 0, columns=found).groupby(grp).mean()
        df_mean = pd.DataFrame(E, columns=found).groupby(grp).mean()
        cols = [g for g in genes if g in found]
        rows = [r for r in (order or df_mean.index) if r in df_mean.index]
        df_frac, df_mean = df_frac.loc[rows, cols], df_mean.loc[rows, cols]
        df_rel = df_mean / df_mean.max(0).replace(0, np.nan)
        return df_frac, df_mean, df_rel.fillna(0)

    def plot_dotplot(self, df_frac: pd.DataFrame, df_rel: pd.DataFrame,
                     gene_groups: Optional[Sequence[Tuple[str, Sequence[str]]]] = None, fname: Optional[str] = None,
                     row_labels: Optional[Dict[str, str]] = None, cmap: str = "RdPu", size_max: float = 90,
                     title: str = "", figsize: Optional[Tuple[float, float]] = None,
                     size_label: str = "fraction of cells\nin group (%)", color_label: str = "relative expression\nin group",
                     force: bool = False, verbose: bool = False):
        """Dot plot: dot size = df_frac (0-1), dot colour = df_rel (0-1); columns may repeat a gene.
        gene_groups = [(label, genes), ...] draws brackets above the columns (paper Fig 3B)."""
        rows, cols = list(df_frac.index), list(df_frac.columns)
        row_labels = row_labels or {}
        nr, nc = len(rows), len(cols)
        figsize = figsize or (0.28 * nc + 3.2, 0.32 * nr + 2.2)
        fig, ax = plt.subplots(figsize=figsize)
        yy, xx = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
        F, R = df_frac.values, df_rel.values
        sca = ax.scatter(xx.ravel(), yy.ravel(), s=F.ravel() * size_max, c=R.ravel(), cmap=cmap, vmin=0, vmax=1,
                         lw=0.3, edgecolors="#999999")
        ax.set_xticks(range(nc)); ax.set_xticklabels(cols, rotation=90, style="italic")
        ax.set_yticks(range(nr)); ax.set_yticklabels([row_labels.get(r, r) for r in rows])
        ax.set_xlim(-0.6, nc - 0.4); ax.set_ylim(nr - 0.4, -0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(True)
        if gene_groups:
            x0 = 0
            for label, gl in gene_groups:
                n = len([g for g in gl if g in cols])
                if n == 0:
                    continue
                ax.annotate("", xy=(x0 - 0.3, -0.9), xytext=(x0 + n - 0.7, -0.9), annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color="#333333", lw=0.8))
                ax.text(x0 + (n - 1) / 2, -1.2, label, rotation=90, ha="center", va="bottom", fontsize=7)
                x0 += n
        cb = fig.colorbar(sca, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label(color_label, fontsize=7)
        for p in (0.2, 0.4, 0.6, 0.8, 1.0):
            ax.scatter([], [], s=p * size_max, c="#777777", label=f"{int(p * 100)}")
        ax.legend(title=size_label, loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=6,
                  title_fontsize=7, labelspacing=0.9)
        if title:
            ax.set_title(title, loc="left", pad=70 if gene_groups else 6)
        self._savefig(fig, fname, force=force, verbose=verbose)
        return fig

    @staticmethod
    def plot_umap_labeled(ax, U: np.ndarray, labels: Sequence[str], colors: Dict[str, str], title: str = "",
                          point_size: float = 2.0, fontsize: float = 8, seed: int = 0, label_on_data: bool = True):
        """Scatter of a 2-D embedding coloured by category, with category names at their medians."""
        labels = np.asarray(labels).astype(str)
        o = np.random.default_rng(seed).permutation(len(U))
        ax.scatter(U[o, 0], U[o, 1], s=point_size, c=[colors.get(l, "#cccccc") for l in labels[o]], lw=0,
                   rasterized=True)
        if label_on_data:
            import matplotlib.patheffects as pe
            for k in colors:
                m = labels == k
                if m.any():
                    x, y = np.median(U[m], 0)
                    ax.text(x, y, k, fontsize=fontsize, ha="center", va="center", color=colors[k], fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        if title:
            ax.set_title(title)
        return ax

    @staticmethod
    def spot_neighbor_graph(xy: np.ndarray, radius_factor: float = 1.2):
        """Sparse adjacency of spots closer than radius_factor x median spot spacing (Visium: the 6 neighbours)."""
        from scipy.spatial import cKDTree
        tree = cKDTree(xy)
        step = np.median(tree.query(xy, k=2)[0][:, 1])
        pairs = np.array(list(tree.query_pairs(r=radius_factor * step)), dtype=int).reshape(-1, 2)
        n = len(xy)
        A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        return (A + A.T).tocsr(), step

    @staticmethod
    def smooth_on_spots(v: np.ndarray, A) -> np.ndarray:
        """Mean of v over each spot and its neighbours (A from spot_neighbor_graph)."""
        v = np.asarray(v, float)
        deg = np.asarray(A.sum(1)).ravel()
        return (A @ v + v) / (deg + 1)

    @staticmethod
    def plot_region_outline(ax, xy: np.ndarray, mask: np.ndarray, color: str = "#FFD400", lw: float = 1.4,
                            max_edge_factor: float = 1.6, min_spots: int = 15, smooth: bool = True):
        """Outline of a set of spots (mask): connected groups smaller than min_spots are dropped (scattered spots),
        the indicator is optionally averaged over each spot's neighbours, and the 0.5 contour is drawn on the Delaunay
        triangulation of all spots (triangles with an edge > max_edge_factor x spot spacing are masked)."""
        import matplotlib.tri as mtri
        from scipy.sparse.csgraph import connected_components
        mask = np.asarray(mask, bool).copy()
        A, step = SingleCell.spot_neighbor_graph(xy)
        if mask.sum() and min_spots > 1:
            idx = np.where(mask)[0]
            _, lab = connected_components(A[idx][:, idx], directed=False)
            big = np.isin(lab, np.where(np.bincount(lab) >= min_spots)[0])
            mask[:] = False
            mask[idx[big]] = True
        if not mask.any():
            return ax
        z = SingleCell.smooth_on_spots(mask.astype(float), A) if smooth else mask.astype(float)
        tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
        t = tri.triangles
        e = np.max([np.linalg.norm(xy[t[:, i]] - xy[t[:, (i + 1) % 3]], axis=1) for i in range(3)], axis=0)
        tri.set_mask(e > max_edge_factor * step)
        ax.tricontour(tri, z, levels=[0.5], colors=[color], linewidths=lw)
        return ax

    # =====================================================================================================
    # 8b. Subclustering, marker tests, spatial coherence
    # =====================================================================================================
    def subcluster_adaptive(self, adata_sub: ad.AnnData, batch_key: Optional[str] = None, counts_layer: Optional[str] = None,
                            resolutions: Optional[Sequence[float]] = None, min_size: Optional[int] = None,
                            max_k: Optional[int] = None, key_added: str = "subcluster") -> Tuple[ad.AnnData, pd.DataFrame]:
        """Cluster a subset (X = log-normalised, counts in layers[counts_layer]): genes detected in >= sub_min_gene_frac,
        HVGs (seurat_v3 on counts, per batch when possible), optional ComBat on the HVGs, scale, PCA, kNN, UMAP,
        Leiden over a resolution grid. The chosen resolution is the highest one at which every cluster has
        >= min_size members and there are <= max_k clusters (else the lowest resolution).

        Returns (adata_sub with obs[key_added] = '1', '2', ... by decreasing size and obsm['X_umap'],
                 df_grid: resolution, k, min_size, chosen)."""
        P = self.params
        counts_layer = self.get_param("counts_layer", counts_layer)
        resolutions = list(self.get_param("sub_resolutions", resolutions))
        min_size = self.get_param("sub_min_size", min_size)
        max_k = self.get_param("sub_max_k", max_k)
        seed = P["seed"]
        a = adata_sub.copy()
        sc.pp.filter_genes(a, min_cells=max(3, int(P["sub_min_gene_frac"] * a.n_obs)))
        n_batches = a.obs[batch_key].nunique() if batch_key and batch_key in a.obs else 1
        layer = counts_layer if counts_layer in a.layers else None
        n_hvg = min(P["sub_n_hvg"], a.n_vars - 1)
        try:
            sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", layer=layer,
                                        batch_key=batch_key if n_batches > 1 else None)
        except Exception:
            try:
                sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat_v3", layer=layer)
            except ValueError:      # loess singularity on very small subsets (e.g. 41 low-depth spots): dispersion on X
                sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat")
        b = a[:, a.var["highly_variable"].values].copy()
        if P["sub_batch"] == "combat" and n_batches > 1:
            sc.pp.combat(b, key=batch_key)
            b.X = np.array(b.X, dtype=np.float64, copy=True)   # combat can leave X read-only; sc.pp.scale works in place
        sc.pp.scale(b, max_value=10)
        n_pcs = int(min(P["sub_n_pcs"], b.n_obs - 2, b.n_vars - 1))
        sc.pp.pca(b, n_comps=n_pcs, random_state=seed)
        sc.pp.neighbors(b, n_neighbors=min(P["sub_n_neighbors"], b.n_obs - 1), random_state=seed)
        sc.tl.umap(b, random_state=seed)
        rows, labels = [], {}
        for r in resolutions:
            sc.tl.leiden(b, resolution=r, key_added="_l", random_state=seed, flavor="igraph", n_iterations=2,
                         directed=False)
            vc = b.obs["_l"].value_counts()
            labels[r] = b.obs["_l"].astype(str).values.copy()
            rows.append(dict(resolution=r, k=len(vc), min_size=int(vc.min()), ok=bool(vc.min() >= min_size and len(vc) <= max_k)))
        df_grid = pd.DataFrame(rows)
        ok = df_grid[df_grid.ok]
        chosen = float(ok.resolution.max()) if len(ok) else float(df_grid.resolution.min())
        df_grid["chosen"] = df_grid.resolution == chosen
        lab = pd.Series(labels[chosen], index=b.obs_names)
        rank = {old: str(i + 1) for i, old in enumerate(lab.value_counts().index)}       # '1' = largest
        adata_sub = adata_sub.copy()
        adata_sub.obs[key_added] = pd.Categorical(lab.map(rank).loc[adata_sub.obs_names].values,
                                                  categories=[str(i + 1) for i in range(len(rank))])
        adata_sub.obsm["X_umap"] = b.obsm["X_umap"]
        adata_sub.obsm["X_pca_sub"] = b.obsm["X_pca"]
        adata_sub.uns[f"{key_added}_resolution"] = chosen
        return adata_sub, df_grid

    def calc_df_rank_genes(self, adata, groupby: str, groups: Any = "all", reference: str = "rest",
                           method: Optional[str] = None) -> pd.DataFrame:
        """scanpy rank_genes_groups on log-normalised X (never .raw) with pts=True, as one tidy table:
        group, names, scores, logfoldchanges, pvals, pvals_adj, pct_in (fraction of the group with expression > 0),
        pct_out (fraction of the reference)."""
        a = self._check_adata(adata)
        sc.tl.rank_genes_groups(a, groupby=groupby, groups=groups, reference=reference, use_raw=False,
                                method=self.get_param("rank_method", method), pts=True, n_genes=a.n_vars,
                                corr_method=self.params["rank_corr_method"], key_added="_rg")
        df_rank = sc.get.rank_genes_groups_df(a, group=None, key="_rg")
        if "group" not in df_rank.columns:
            df_rank.insert(0, "group", groups[0] if isinstance(groups, (list, tuple)) else groups)
        df_rank = df_rank.rename(columns={"pct_nz_group": "pct_in", "pct_nz_reference": "pct_out"})
        del a.uns["_rg"]
        return df_rank

    def calc_df_spatial_coherence(self, adata, key: str, within_key: Optional[str] = None,
                                  library_key: Optional[str] = None, radius_factor: Optional[float] = None) -> pd.DataFrame:
        """Spatial coherence of each category of obs[key]: for its spots, the fraction of tissue neighbours (same
        library; optionally only neighbours with the same obs[within_key]) that share the category, divided by the
        category's share of those neighbourhoods' universe (its fraction of spots in the same library / within group).
        ratio ~ 1: scattered like random labels; ratio >> 1: a contiguous patch."""
        from scipy.spatial import cKDTree
        a = self._check_adata(adata)
        library_key = library_key or self.sample_col
        rf = self.get_param("coherence_radius_factor", radius_factor)
        lab = a.obs[key].astype(str).values
        grp = a.obs[within_key].astype(str).values if within_key else np.zeros(a.n_obs, dtype=str)
        libs = a.obs[library_key].astype(str).values if library_key in a.obs else np.zeros(a.n_obs, dtype=str)
        xy = np.asarray(a.obsm["spatial"], float)
        same = np.full(a.n_obs, np.nan)
        n_nb = np.zeros(a.n_obs, int)
        for lb in np.unique(libs):
            idx = np.where(libs == lb)[0]
            tree = cKDTree(xy[idx])
            step = np.median(tree.query(xy[idx], k=2)[0][:, 1])
            for i_loc, nb in enumerate(tree.query_ball_point(xy[idx], rf * step)):
                i = idx[i_loc]
                nb = [idx[j] for j in nb if idx[j] != i and grp[idx[j]] == grp[i]]
                n_nb[i] = len(nb)
                if nb:
                    same[i] = np.mean(lab[nb] == lab[i])
        rows = []
        for (g, c), sub in pd.DataFrame({"g": grp, "c": lab, "lib": libs, "same": same, "n": n_nb}).groupby(["g", "c"]):
            exp = []
            for lb, s2 in sub.groupby("lib"):
                pool = (grp == g) & (libs == lb)
                exp.append((len(s2), (lab[pool] == c).mean()))
            expected = np.average([e for _, e in exp], weights=[w for w, _ in exp])
            obs_ = np.nanmean(sub["same"].values) if np.isfinite(sub["same"].values).any() else np.nan
            rows.append(dict(group=g, category=c, n=len(sub), frac_same_neighbours=round(float(obs_), 3),
                             expected=round(float(expected), 3),
                             coherence_ratio=round(float(obs_ / expected), 2) if expected > 0 else np.nan))
        df_coh = pd.DataFrame(rows)
        if not within_key:
            df_coh = df_coh.drop(columns="group")
        return df_coh

    # =====================================================================================================
    # 8c. Subregions: the spots of each region clustered on gene expression (mp_mouse_brain03, mp_lymphoid03)
    #     prepare_expression, run_subregions, calc_df_region_signature, calc_df_subregion_gene_sets,
    #     calc_df_composition_enrichment, calc_df_region_table, calc_df_subregion_borders, calc_df_subregion_table;
    #     Plotly: save_plotly, plotly_region_subregions, plotly_subregion_map, plotly_subregion_diagnostics,
    #     plotly_region_signature_heatmap
    # =====================================================================================================
    def _region_palette(self, labels: Sequence[str]) -> Dict[str, str]:
        """Region -> colour: params['region_colors'] first, then REGION_COLORS (no black/gray/violet/brown/dark)
        in sorted order; cycles if there are more than len(REGION_COLORS) regions."""
        given = self.params.get("region_colors") or {}
        pal = REGION_COLORS
        out, i = {}, 0
        for l in sorted(labels):
            if l in given:
                out[l] = given[l]
            else:
                out[l] = pal[i % len(pal)]
                i += 1
        return out


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

    def run_subregions(self, adata_expr, label_col: str = "region_label", regions: Optional[Sequence[str]] = None,
                       force: bool = False, verbose: bool = False) -> Tuple[Dict[str, ad.AnnData], pd.DataFrame]:
        """subcluster_adaptive on the spots of each region (batch = section). Writes obs[subregion_key] = '<region>_<k>'
        (k = 1 is the largest; regions in subregion_skip stay NaN). Returns ({region: adata_sub}, df_grid).
        Saved: df_grid -> files['subregion_grid'], each adata_sub -> files['subregion_adata'] (one h5ad per region).
        When all of them exist and not force they are read back as they are (no recomputation)."""
        key = self.params["subregion_key"]
        regions = regions or [r for r in sorted(adata_expr.obs[label_col].unique()) if r not in self.params["subregion_skip"]]
        adata_expr.obs[key] = pd.Series(np.nan, index=adata_expr.obs_names, dtype=object)
        cached = all(k in self.files for k in ("subregion_grid", "subregion_adata"))

        if cached and not force and self.file_path("subregion_grid").exists() and \
                all(self.file_path("subregion_adata", region=reg).exists() for reg in regions):
            df_grid = self.read_tsv("subregion_grid", verbose=verbose)
            subs = {reg: self.read_h5ad("subregion_adata", set_self=False, verbose=verbose, region=reg) for reg in regions}
            for a_sub in subs.values():                      # labels back into adata_expr (used by the next methods)
                adata_expr.obs.loc[a_sub.obs_names, key] = a_sub.obs[key].astype(str).values
            return subs, df_grid

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
        df_grid = pd.concat(grids, ignore_index=True)

        if cached:
            self.write_tsv(df_grid, "subregion_grid", verbose=verbose)
            for reg, a_sub in subs.items():
                self.write_h5ad("subregion_adata", a_sub, verbose=verbose, region=reg)
            df_grid = self.read_tsv("subregion_grid")        # same dtypes as a cached run
        return subs, df_grid

    def _display_genes(self, genes: Sequence[str], n: Optional[int] = None) -> List[str]:
        rx = re.compile(self.params["gene_display_exclude"]) if self.params["gene_display_exclude"] else None
        out = [g for g in genes if not (rx and rx.search(g))]
        return out[: (n or self.params["n_genes_show"])]

    def calc_df_region_signature(self, adata_expr, label_col: str = "region_label", force: bool = False,
                                 verbose: bool = False) -> Tuple[pd.DataFrame, Dict[str, set]]:
        """Each region vs all other spots (Wilcoxon). Returns (df_rank of all genes, {region: signature gene set})
        with signature = logFC >= region_sig_lfc, padj <= region_sig_padj, pct_in >= region_sig_min_pct.
        Cached in files 'region_rank_genes' (the signature sets are rebuilt from its region_signature column)."""
        P = self.params

        def _compute():
            df = self.calc_df_rank_genes(adata_expr, groupby=label_col)
            df = df.rename(columns={"group": "region_id", "names": "gene"})
            df["region_id"] = df["region_id"].astype(str)
            df["region_signature"] = (df.logfoldchanges >= P["region_sig_lfc"]) & (df.pvals_adj <= P["region_sig_padj"]) & \
                                     (df.pct_in >= P["region_sig_min_pct"])
            return df

        df_rank = self._cached_table("region_rank_genes", _compute, force, verbose, str_cols=("region_id", "gene"))
        df_rank["region_signature"] = df_rank["region_signature"].astype(bool)
        sig = {r: set(g.gene) for r, g in df_rank[df_rank.region_signature].groupby("region_id", observed=True)}
        return df_rank, sig

    def calc_df_subregion_gene_sets(self, adata_expr, region_sig: Dict[str, set], label_col: str = "region_label",
                                    force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached in files 'subregion_gene_sets' (see _calc_df_subregion_gene_sets)."""
        return self._cached_table("subregion_gene_sets",
                                  lambda: self._calc_df_subregion_gene_sets(adata_expr, region_sig, label_col),
                                  force, verbose, str_cols=("region_id", "sub_region_id", "gene", "gene_set"))

    def _calc_df_subregion_gene_sets(self, adata_expr, region_sig: Dict[str, set], label_col: str = "region_label"
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

    def calc_df_region_depth(self, adata_expr, df_abund: pd.DataFrame, label_col: str = "region_label") -> pd.DataFrame:
        """Per region: median UMIs per spot (layers[counts_layer]) and median cells per spot (sum of df_abund).
        Index = region id (sorted)."""
        umis = pd.Series(np.asarray(adata_expr.layers[self.params["counts_layer"]].sum(1)).ravel(), index=adata_expr.obs_names)
        df = pd.DataFrame({"median_UMIs": umis, "median_cells_per_spot": df_abund.loc[adata_expr.obs_names].sum(1)})
        return df.groupby(adata_expr.obs[label_col].astype(str).values).median().round(0).astype(int)

    REGION_TABLE_STR_COLS = ("region_id", "region_name", "compartment", "signature_genes", "enriched_cell_types",
                             "functional_description")

    def calc_df_region_table(self, adata_expr, df_rank_region: pd.DataFrame, region_sig: Dict[str, set],
                             df_abund: pd.DataFrame, df_region_info: Optional[pd.DataFrame] = None,
                             label_col: str = "region_label",
                             force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached in files 'region_table' (see _calc_df_region_table). n_subregions is empty until the subregions exist:
        update it with update_region_table_subregions."""
        return self._cached_table("region_table",
                                  lambda: self._calc_df_region_table(adata_expr, df_rank_region, region_sig, df_abund,
                                                                     df_region_info, label_col),
                                  force, verbose, str_cols=self.REGION_TABLE_STR_COLS)

    def update_region_table_subregions(self, df_region_table: pd.DataFrame, adata_expr, label_col: str = "region_label",
                                       force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """n_subregions from obs[subregion_key] (after run_subregions). The table file ('region_table') is rewritten
        when the column changed, or always with force."""
        key = self.params["subregion_key"]
        df = df_region_table.copy()
        lab = adata_expr.obs[label_col].astype(str)
        n_sub = adata_expr.obs[key].groupby(lab.values).nunique() if key in adata_expr.obs else pd.Series(dtype=float)
        new = df.region_id.map(n_sub).astype(float)          # 0 = not subclustered (subregion_skip)
        old = pd.to_numeric(df["n_subregions"], errors="coerce").astype(float) if "n_subregions" in df else new * np.nan
        changed = not np.array_equal(new.fillna(-1).to_numpy(), old.fillna(-1).to_numpy())
        df["n_subregions"] = new.astype("Int64")
        if "region_table" in self.files and (changed or force):
            self.write_tsv(df, "region_table", verbose=verbose)
        elif verbose:
            print("n_subregions unchanged: region table not rewritten")
        return df

    def _calc_df_region_table(self, adata_expr, df_rank_region: pd.DataFrame, region_sig: Dict[str, set],
                              df_abund: pd.DataFrame, df_region_info: Optional[pd.DataFrame] = None,
                              label_col: str = "region_label") -> pd.DataFrame:
        """Table region: region_id, region_name, n_spots, n_subregions, n_signature_genes, signature_genes (top by score),
        enriched_cell_types (vs all other spots), functional_description (curated region table df_region_info:
        region_id, region_name, description; e.g. mouse_brain_regions.tsv, lymph_node_regions.tsv)."""
        key = self.params["subregion_key"]
        lab = adata_expr.obs[label_col].astype(str)
        df_comp = self.calc_df_composition_enrichment(df_abund.loc[adata_expr.obs_names], lab)
        desc = dict(zip(df_region_info.region_id, df_region_info.description)) if df_region_info is not None else {}
        names = dict(zip(df_region_info.region_id, df_region_info.region_name)) if df_region_info is not None else {}
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
        if df_region_info is not None and "compartment" in df_region_info.columns:
            comp = dict(zip(df_region_info.region_id, df_region_info.compartment))
            df_region_table.insert(2, "compartment", df_region_table.region_id.map(comp).fillna(""))
        if self.params["region_table_depth"]:
            df_depth = self.calc_df_region_depth(adata_expr, df_abund, label_col=label_col)
            i = df_region_table.columns.get_loc("n_spots") + 1
            df_region_table.insert(i, "median_UMIs", df_region_table.region_id.map(df_depth.median_UMIs).values)
            df_region_table.insert(i + 1, "median_cells_per_spot",
                                   df_region_table.region_id.map(df_depth.median_cells_per_spot).values)
        df_region_table["n_subregions"] = df_region_table["n_subregions"].astype("Int64")
        return self._first_cols(df_region_table, self.params["region_table_first"])

    @staticmethod
    def _first_cols(df: pd.DataFrame, first: Sequence[str]) -> pd.DataFrame:
        """Columns in `first` (those present) first, the others in their order."""
        first = [c for c in first if c in df.columns]
        return df[first + [c for c in df.columns if c not in first]]

    def calc_df_subregion_borders(self, adata_expr, label_col: str = "region_label") -> pd.DataFrame:
        """For each subregion: fraction of its spots' tissue neighbours (same section) that belong to another region
        (border_frac), the most frequent of those regions (border_region), the same fraction for its whole region
        (region_border_frac) and border_rel = border_frac / region_border_frac (> 1: the subregion sits on the region's
        edge more than the region as a whole; useful when regions are interleaved patches). Index = sub_region_id."""
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
        reg_tot: Dict[str, int] = {}
        reg_out: Dict[str, int] = {}
        sub_reg: Dict[str, str] = {}
        for lb in np.unique(libs):
            idx = np.where(libs == lb)[0]
            tree = cKDTree(xy[idx])
            step = np.median(tree.query(xy[idx], k=2)[0][:, 1])
            for i_loc, nb in enumerate(tree.query_ball_point(xy[idx], rf * step)):
                i = idx[i_loc]
                for j_loc in nb:
                    j = idx[j_loc]
                    if j == i:
                        continue
                    reg_tot[lab[i]] = reg_tot.get(lab[i], 0) + 1
                    reg_out[lab[i]] = reg_out.get(lab[i], 0) + int(lab[j] != lab[i])
                    if sub[i] == "nan":
                        continue
                    sub_reg[sub[i]] = lab[i]
                    totals[sub[i]] = totals.get(sub[i], 0) + 1
                    if lab[j] != lab[i]:
                        counts.setdefault(sub[i], {})
                        counts[sub[i]][lab[j]] = counts[sub[i]].get(lab[j], 0) + 1
        rows = []
        for s, tot in totals.items():
            c = counts.get(s, {})
            n_other = sum(c.values())
            top = max(c, key=c.get) if c else ""
            bf = n_other / tot if tot else 0.0
            r = sub_reg[s]
            rbf = reg_out.get(r, 0) / reg_tot[r] if reg_tot.get(r) else 0.0
            rows.append(dict(sub_region_id=s, border_frac=bf, border_region=top, region_border_frac=rbf,
                             border_rel=bf / rbf if rbf > 0 else np.nan))
        df_border = pd.DataFrame(rows).set_index("sub_region_id")
        return df_border

    SUBREGION_TABLE_STR_COLS = ("region_id", "sub_region_id", "subregion_name", "subregion_signature_genes", "other_genes_top_expressed",
                                "not_present_genes", "functional_description", "border_region", "status",
                                "enriched_cell_types", "description_source", "ref_cell_types")

    def calc_df_subregion_table(self, adata_expr, df_gene_sets: pd.DataFrame, df_abund: pd.DataFrame,
                                df_descriptions: Optional[pd.DataFrame] = None, label_col: str = "region_label",
                                n_other_top: int = 10, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached in files 'subregion_table' (see _calc_df_subregion_table); columns params['subregion_table_first'] first."""
        return self._cached_table("subregion_table",
                                  lambda: self._first_cols(self._calc_df_subregion_table(
                                      adata_expr, df_gene_sets, df_abund, df_descriptions, label_col, n_other_top),
                                      self.params["subregion_table_first"]),
                                  force, verbose, str_cols=self.SUBREGION_TABLE_STR_COLS)

    @staticmethod
    def _short_label(text: str, n: int = 40) -> str:
        """Short name from a description: the text before the first ':' or ';', cut to n characters."""
        t = re.split(r"[:;]", str(text or ""))[0].strip().rstrip(".")
        return t if len(t) <= n else t[:n - 3].rstrip() + "..."

    def _calc_df_subregion_table(self, adata_expr, df_gene_sets: pd.DataFrame, df_abund: pd.DataFrame,
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
            border = b is not None and b.border_frac >= P["border_min_frac"] and \
                (not P["border_min_rel"] or (pd.notna(b.border_rel) and b.border_rel >= P["border_min_rel"]))
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
            name = str(getattr(cur, "subregion_name", "") or "") if cur is not None else ""
            row = dict(region_id=reg, sub_region_id=s, subregion_name=name or self._short_label(description), n_spots=len(o))
            row.update({f"frac_{lb}": round(float(frac[lb]), 2) for lb in libs})
            row.update(dict(coherence_ratio=coh.get(s, np.nan), depth_ratio=round(depth_ratio, 2),
                            border_region=(b.border_region if b is not None else ""),
                            border_frac=(round(float(b.border_frac), 2) if b is not None else np.nan),
                            border_rel=(round(float(b.border_rel), 2) if b is not None and pd.notna(b.border_rel) else np.nan),
                            status=status,
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
    def save_plotly(self, fig, fname: Optional[str], root_figure: Optional[Path] = None, verbose: bool = False, **fmt):
        """fig.write_html to (root_figure or self.root_figure) / fname, or to the registered file when fname is a key of
        self.files ({placeholders} from fmt, e.g. lib=..., region=...). plotly.js from params['plotly_include_js']."""
        if fname is None:
            return None
        path = self._resolve_path(fname, root_figure, self.root_figure, **fmt)
        fig.write_html(path, include_plotlyjs=self.params["plotly_include_js"])
        if verbose:
            print(f"Saving figure to {path}")
        return path

    def subregion_colors(self, sub_ids: Sequence[str], region: str, regions: Optional[Sequence[str]] = None,
                         region_colors: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Shades of the region colour, dark -> light by subregion number. The region colour comes from
        _region_palette(regions) (all regions, so that regions without params['region_colors'] get distinct colours),
        overridden by region_colors (region -> colour)."""
        import colorsys
        pal = {**self._region_palette(list(regions) if regions is not None else [region]), **(region_colors or {})}
        base = pal[region]
        r, g, b = mpl.colors.to_rgb(base)
        h, l, s_ = colorsys.rgb_to_hls(r, g, b)
        n = len(sub_ids)
        light = np.linspace(0.28, 0.78, n) if n > 1 else [min(max(l, 0.35), 0.6)]
        sat = max(s_, 0.45)
        return {sid: mpl.colors.to_hex(colorsys.hls_to_rgb(h, li, sat)) for sid, li in zip(sub_ids, light)}

    def subregion_palette(self, sub_ids: Sequence[str]) -> Dict[str, str]:
        """Distinct light colours for the subregions of one region: REGION_COLORS in order (no dark, gray, violet,
        brown); consecutive subregions get contrasting hues."""
        return {s: REGION_COLORS[i % len(REGION_COLORS)] for i, s in enumerate(sub_ids)}

    def _jpeg_uri(self, img: np.ndarray) -> str:
        import base64, io
        from PIL import Image
        a_ = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a_[..., :3]).save(buf, format="JPEG", quality=self.params["plotly_jpeg_quality"])
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def plotly_region_subregions(self, region: str, adata_sub: ad.AnnData, adata_map, df_sub_table: Optional[pd.DataFrame] = None,
                                 df_region_table: Optional[pd.DataFrame] = None, pad: float = 80, height: int = 470, width: Optional[int] = None, spot_size: float = 7,
                                 umap_size: float = 5, fontsize_title: int = 17, fontsize_subplot_title: int = 16,
                                 fontsize_axis: int = 14, fontsize_legend: int = 12, fontsize_colorbar: int = 12,
                                 fontsize_hover: int = 13, subregion_colors: Optional[Dict[str, str]] = None,
                                 section_colors: Optional[Dict[str, str]] = None, umi_colorscale=LIGHT_SEQUENTIAL,
                                 marker_line_color: str = "black", marker_line_width: float = 0.4,
                                 font_color: Optional[str] = None, bg_color: Optional[str] = None, webgl: bool = False,
                                 image_bg_color: Optional[str] = None,
                                 fname: Optional[str] = "fig_region_sub", force: bool = False, verbose: bool = False):
        """One region: UMAP by subregion | UMAP by section (one section: by log10 UMIs, the depth confounder) |
        subregions on the H&E of each section (cropped to the region).
        Legend entries toggle a subregion in every panel. Hover: subregion, section, status, signature genes.
        webgl=False draws the points as SVG (go.Scatter): many of these figures shown in one notebook would otherwise
        exceed the browser's WebGL context limit and the first ones turn blank. Spots per region are few (< 500).
        Title: 'R06 - <region_name>' (region_name from df_region_table). Legend: 'R06_1 - <subregion_name> (n=92)'
        (df_sub_table['subregion_name'], else the start of functional_description).

        Sizes: spot_size (H&E spots), umap_size (UMAP points), width (default 420 per panel). Font sizes: fontsize_title,
        fontsize_subplot_title (panel titles), fontsize_axis (UMAP1/UMAP2), fontsize_legend, fontsize_colorbar (log10 UMIs),
        fontsize_hover. Colours (light by default): subregion_colors (subregion -> colour; others from subregion_palette =
        REGION_COLORS), section_colors (section -> colour; others REGION_COLORS), umi_colorscale (Plotly scale name or
        list of colours; default LIGHT_SEQUENTIAL), marker_line_color / marker_line_width (H&E spot outline), font_color, bg_color."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        Scatter = go.Scattergl if webgl else go.Scatter      # SVG by default: browsers keep only ~16 WebGL contexts per page
        key = self.params["subregion_key"]
        subs = list(adata_sub.obs[key].cat.categories)
        cols = {**self.subregion_palette(subs), **(subregion_colors or {})}
        libs = [lb for lb in self.library_ids(adata_map) if (adata_sub.obs[self.sample_col].astype(str) == lb).any()]
        info, sub_names = {}, {}
        if df_sub_table is not None:
            for r in df_sub_table[df_sub_table.region_id == region].itertuples():
                info[r.sub_region_id] = f"{r.status}; coherence {r.coherence_ratio}<br>up: {r.subregion_signature_genes[:80]}"
                nm = getattr(r, "subregion_name", "") or self._short_label(getattr(r, "functional_description", ""))
                if nm:
                    sub_names[r.sub_region_id] = nm
        region_name = ""
        if df_region_table is not None and "region_name" in df_region_table.columns:
            region_name = dict(zip(df_region_table.region_id, df_region_table.region_name)).get(region, "") or ""
        one_section = adata_sub.obs[self.sample_col].astype(str).nunique() == 1
        titles = ["UMAP - subregions", "UMAP - log10 UMIs" if one_section else "UMAP - sections"] + [f"{lb}" for lb in libs]
        fig = make_subplots(rows=1, cols=2 + len(libs), subplot_titles=titles, horizontal_spacing=0.03)
        U = adata_sub.obsm["X_umap"]
        lab = adata_sub.obs[key].astype(str).values
        sec = adata_sub.obs[self.sample_col].astype(str).values
        for s in subs:
            m = lab == s
            legend_name = f"{s} - {sub_names[s]} (n={m.sum()})" if s in sub_names else f"{s} (n={m.sum()})"
            fig.add_trace(Scatter(x=U[m, 0], y=U[m, 1], mode="markers", name=legend_name, legendgroup=s,
                                       marker=dict(size=umap_size, color=cols[s]), text=adata_sub.obs_names[m],
                                       hovertemplate=f"<b>{s}</b><br>{info.get(s, '')}<br>%{{text}}<extra></extra>"),
                          row=1, col=1)
        sec_cols = {**dict(zip(self.library_ids(adata_map), REGION_COLORS[1:])), **(section_colors or {})}
        if one_section and self.params["counts_layer"] in adata_sub.layers:
            umi = np.log10(np.asarray(adata_sub.layers[self.params["counts_layer"]].sum(1)).ravel() + 1)
            fig.add_trace(Scatter(x=U[:, 0], y=U[:, 1], mode="markers", name="log10 UMIs", showlegend=False,
                                       marker=dict(size=umap_size, color=umi, colorscale=umi_colorscale,
                                                   colorbar=dict(title=dict(text="log10<br>UMIs", font=dict(size=fontsize_colorbar)),
                                                                 tickfont=dict(size=fontsize_colorbar),
                                                                 len=0.6, x=0.33, thickness=10)),
                                       text=[f"{v:.2f}" for v in umi],
                                       hovertemplate="log10 UMIs %{text}<extra></extra>"), row=1, col=2)
        else:
            for lb in np.unique(sec):
                m = sec == lb
                fig.add_trace(Scatter(x=U[m, 0], y=U[m, 1], mode="markers", name=lb, legendgroup=lb,
                                           marker=dict(size=umap_size, color=sec_cols.get(lb, REGION_COLORS[0])), hoverinfo="skip"),
                              row=1, col=2)
        for j, lb in enumerate(libs):
            img = self._map_image(lb, adata_map, image_bg_color, force, verbose)
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
                fig.add_trace(Scatter(x=xy[m, 0], y=xy[m, 1], mode="markers", name=s, legendgroup=s, showlegend=False,
                                           marker=dict(size=spot_size, color=cols[s],
                                                       line=dict(width=marker_line_width, color=marker_line_color)),
                                           text=names_lb[in_reg][m],
                                           hovertemplate=f"<b>{s}</b> - {lb}<br>{info.get(s, '')}<br>%{{text}}<extra></extra>"),
                              row=1, col=col)
            fig.update_xaxes(range=[x0, x1], visible=False, row=1, col=col)
            fig.update_yaxes(range=[y1, y0], visible=False, scaleanchor=f"x{col}", row=1, col=col)
        axis_font = dict(size=fontsize_axis, color=font_color)
        for c in (1, 2):
            fig.update_xaxes(showticklabels=False, title_text="UMAP1", title_font=axis_font, row=1, col=c)
            fig.update_yaxes(showticklabels=False, title_text="UMAP2" if c == 1 else None, title_font=axis_font, row=1, col=c)
        fig.update_annotations(font=dict(size=fontsize_subplot_title, color=font_color))      # subplot titles
        res = adata_sub.uns.get("subcluster_resolution", "")
        head = f"{region} - {region_name}" if region_name else region
        fig.update_layout(title=dict(text=f"{head}: {len(subs)} subregion(s), {adata_sub.n_obs} spots (Leiden resolution {res})",
                                     font=dict(size=fontsize_title, color=font_color)),
                          height=height, width=width or 420 * (2 + len(libs)) + (300 if sub_names else 0), template="plotly_white",
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          hoverlabel=dict(font=dict(size=fontsize_hover)),
                          legend=dict(itemsizing="constant", font=dict(size=fontsize_legend, color=font_color)),
                          margin=dict(l=20, r=20, t=70, b=20))
        self._save_figure(fig, fname, force, verbose, region=region)
        return fig

    def plotly_subregion_map(self, adata_expr, adata_map, library_id: str, df_sub_table: Optional[pd.DataFrame] = None,
                             label_col: str = "region_label", spot_size: float = 6, width: int = 1300, height: int = 950,
                             fontsize_title: int = 17, fontsize_legend: int = 10, fontsize_legend_title: int = 10,
                             fontsize_hover: int = 13, region_colors: Optional[Dict[str, str]] = None,
                             skip_color: str = "#dddddd", marker_line_color: str = "black", marker_line_width: float = 0.3,
                             font_color: Optional[str] = None, bg_color: Optional[str] = None, legend_ncols: int = 1,
                             image_bg_color: Optional[str] = None,
                             legend_col_width: int = 110,
                             fname: Optional[str] = "fig_subregion_map", force: bool = False, verbose: bool = False):
        """All subregions of one section on the H&E. Colours = shades of each region's colour; legend grouped by region
        (click a region title to toggle its subregions). Hover: region, subregion, status, signature genes, cell types.

        Font sizes: fontsize_title, fontsize_legend (subregions), fontsize_legend_title (region group titles), fontsize_hover.
        Colours: region_colors (region -> base colour of its shades; others from _region_palette), skip_color (regions in
        subregion_skip), marker_line_color / marker_line_width (spot outline), font_color, bg_color.
        Legend: legend_ncols columns side by side on the right (regions kept whole, split in order into columns of about
        equal length; each column is a separate Plotly legend), legend_col_width pixels per column."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        key = self.params["subregion_key"]
        img = self._map_image(library_id, adata_map, image_bg_color, force, verbose)
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
        reg_traces: Dict[str, List[int]] = {}             # region -> indices of its traces (legend columns)
        for reg in regions:
            mreg = (obs[label_col].astype(str) == reg).values
            if not mreg.any():
                continue
            n_before = len(fig.data)
            if reg in self.params["subregion_skip"]:
                fig.add_trace(go.Scattergl(x=xy[mreg, 0], y=xy[mreg, 1], mode="markers", name=reg, legendgroup=reg,
                                           legendgrouptitle_text=reg, marker=dict(size=spot_size, color=skip_color),
                                           hovertemplate=f"<b>{reg}</b> (not subclustered)<extra></extra>"))
                reg_traces[reg] = list(range(n_before, len(fig.data)))
                continue
            subs = sorted(obs.loc[mreg, key].dropna().unique(), key=lambda x: int(x.rsplit("_", 1)[1]))
            cols = self.subregion_colors(subs, reg, regions, region_colors=region_colors)
            for s in subs:
                m = (obs[key] == s).values
                fig.add_trace(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=s, legendgroup=reg,
                                           legendgrouptitle_text=reg,
                                           marker=dict(size=spot_size, color=cols[s],
                                                       line=dict(width=marker_line_width, color=marker_line_color)),
                                           text=names_lb[m],
                                           hovertemplate=f"<b>{s}</b><br>{info.get(s, '')}<br>%{{text}}<extra></extra>"))
            reg_traces[reg] = list(range(n_before, len(fig.data)))
        fig.update_traces(legendgrouptitle_font=dict(size=fontsize_legend_title, color=font_color), selector=dict(type="scattergl"))
        legend_style = dict(groupclick="toggleitem", itemsizing="constant", font=dict(size=fontsize_legend, color=font_color))
        margin = dict(l=10, r=10, t=50, b=10)
        ncols = max(1, int(legend_ncols))
        if ncols > 1:
            lines = {reg: len(ix) + 1 for reg, ix in reg_traces.items()}          # entries + group title
            target = sum(lines.values()) / ncols
            col, acc = 0, 0
            for reg, ix in reg_traces.items():
                if acc >= target * (col + 1) and col < ncols - 1:
                    col += 1
                acc += lines[reg]
                for i in ix:
                    fig.data[i].legend = "legend" if col == 0 else f"legend{col + 1}"
            margin["r"] = ncols * legend_col_width + 10
            plot_w = width - margin["l"] - margin["r"]
            for k in range(ncols):
                fig.update_layout({("legend" if k == 0 else f"legend{k + 1}"):
                                   dict(legend_style, x=1 + (10 + k * legend_col_width) / plot_w, xanchor="left",
                                        y=1, yanchor="top")})
        fig.update_layout(title=dict(text=self.params["subregion_map_title"].format(lib=library_id),
                                     font=dict(size=fontsize_title, color=font_color)),
                          width=width, height=height, template="plotly_white",
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          hoverlabel=dict(font=dict(size=fontsize_hover)), margin=margin)
        if ncols == 1:
            fig.update_layout(legend=legend_style)
        fig.update_xaxes(visible=False, range=[xy[:, 0].min() - 40, xy[:, 0].max() + 40])
        fig.update_yaxes(visible=False, range=[xy[:, 1].max() + 40, xy[:, 1].min() - 40], scaleanchor="x")
        self._save_figure(fig, fname, force, verbose, lib=library_id)
        return fig

    def plotly_region_map(self, adata_expr, adata_map, library_id: str, df_region_table: Optional[pd.DataFrame] = None,
                          label_col: str = "region_label", spot_size: float = 6, show_image: bool = True,
                          width: int = 1300, height: int = 950, fontsize_legend: int = 11, fontsize_title: int = 18,
                          fontsize_hover: int = 13, n_genes_hover: int = 8, region_colors: Optional[Dict[str, str]] = None,
                          marker_line_color: str = "black", marker_line_width: float = 0.3,
                          marker_opacity: Optional[float] = None, font_color: Optional[str] = None, bg_color: Optional[str] = None,
                          image_bg_color: Optional[str] = None,
                          fname: Optional[str] = "fig_region_map", force: bool = False, verbose: bool = False):
        """All regions of one section on the H&E (one trace per region, colours = _region_palette, as in the subregion map).
        Legend: 'R10 - Interferon-active zone (n=173)'; click to toggle, double-click to isolate.
        Hover (from df_region_table, any of these columns that exist): region_name, compartment, n_spots, median_UMIs,
        median_cells_per_spot, n_signature_genes, signature_genes (first n_genes_hover), enriched_cell_types.

        Font sizes: fontsize_title, fontsize_legend, fontsize_hover. Colours: region_colors (region -> colour; others from
        _region_palette), marker_line_color / marker_line_width (spot outline), marker_opacity, font_color, bg_color."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        mm, xy = self.spot_xy(library_id, adata_map)
        names_lb = adata_map.obs_names[mm]
        keep = names_lb.isin(adata_expr.obs_names)
        names_lb, xy = names_lb[keep], xy[keep]
        lab = adata_expr.obs.loc[names_lb, label_col].astype(str).values
        regions = list(adata_expr.obs[label_col].cat.categories)
        cols = {**self._region_palette(regions), **(region_colors or {})}
        tab = df_region_table.set_index("region_id") if df_region_table is not None else pd.DataFrame(index=regions)
        fields = [("compartment", "compartment"), ("n_spots", "spots"), ("median_UMIs", "median UMIs"),
                  ("median_cells_per_spot", "median cells/spot"), ("n_signature_genes", "signature genes (n)")]

        def _hover(reg: str) -> str:
            if reg not in tab.index:
                return f"<b>{reg}</b>"
            r = tab.loc[reg]
            name = r.get("region_name", "")
            lines = [f"<b>{reg}</b>" + (f" - {name}" if isinstance(name, str) and name else "")]
            lines += [f"{lbl}: {r[c]}" for c, lbl in fields if c in tab.columns and pd.notna(r[c]) and r[c] != ""]
            if "signature_genes" in tab.columns and isinstance(r["signature_genes"], str) and r["signature_genes"]:
                g = [x.strip() for x in r["signature_genes"].split(",")]
                lines.append("up: " + ", ".join(g[:n_genes_hover]) + (" ..." if len(g) > n_genes_hover else ""))
            if "enriched_cell_types" in tab.columns and isinstance(r["enriched_cell_types"], str):
                lines.append("cells: " + r["enriched_cell_types"])
            return "<br>".join(lines).replace("%{", "% {")      # no accidental Plotly %{var} in the text

        fig = go.Figure()
        if show_image:
            img = self._map_image(library_id, adata_map, image_bg_color, force, verbose)
            fig.add_trace(go.Image(source=self._jpeg_uri(img), hoverinfo="skip"))
        for reg in regions:
            m = lab == reg
            if not m.any():
                continue
            name = tab.loc[reg, "region_name"] if (reg in tab.index and "region_name" in tab.columns) else ""
            legend = f"{reg} - {name} (n={m.sum()})" if isinstance(name, str) and name else f"{reg} (n={m.sum()})"
            fig.add_trace(go.Scattergl(x=xy[m, 0], y=xy[m, 1], mode="markers", name=legend, legendgroup=reg,
                                       marker=dict(size=spot_size, color=cols[reg], opacity=marker_opacity,
                                                   line=dict(width=marker_line_width, color=marker_line_color)),
                                       text=names_lb[m],
                                       hovertemplate=_hover(reg) + "<br>%{text}<extra></extra>"))
        fig.update_layout(title=dict(text=f"Regions - {library_id} ({len(names_lb)} spots, {len(regions)} regions)",
                                     font=dict(size=fontsize_title, color=font_color)),
                          width=width, height=height, template="plotly_white",
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          hoverlabel=dict(font=dict(size=fontsize_hover)),
                          legend=dict(itemsizing="constant", font=dict(size=fontsize_legend, color=font_color)),
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_xaxes(visible=False, range=[xy[:, 0].min() - 40, xy[:, 0].max() + 40])
        fig.update_yaxes(visible=False, range=[xy[:, 1].max() + 40, xy[:, 1].min() - 40], scaleanchor="x")
        self._save_figure(fig, fname, force, verbose, lib=library_id)
        return fig

    def plotly_subregion_diagnostics(self, df_sub_table: pd.DataFrame, height: int = 520, width: Optional[int] = None,
                                     fontsize_title: int = 17, fontsize_axis: int = 14, fontsize_ticks: int = 12,
                                     fontsize_legend: int = 12, fontsize_legend_title: int = 12, fontsize_hover: int = 13,
                                     region_colors: Optional[Dict[str, str]] = None, font_color: Optional[str] = None,
                                     line_color: str = "#888888", line_width: float = 1.5,
                                     marker_line_color: str = "#333333", marker_line_width: float = 0.0,
                                     marker_opacity: Optional[float] = None, bg_color: Optional[str] = None,
                                     grid_color: str = "#EBF0F8", size_max: int = 28,
                                     fname: Optional[str] = "fig_diagnostics",
                                     force: bool = False, verbose: bool = False):
        """Subregion size vs spatial coherence ratio (marker size = number of subregion signature genes, colour = region);
        dashed line = coherence_min_ratio.

        Font sizes: fontsize_title, fontsize_axis (axis titles), fontsize_ticks (tick labels), fontsize_legend,
        fontsize_legend_title, fontsize_hover. Colours: region_colors (region -> colour; regions not given keep
        _region_palette), font_color (all text), line_color / line_width (threshold line), marker_line_color /
        marker_line_width (marker outline; 0 = none), marker_opacity, bg_color (plot + paper), grid_color.
        Defaults reproduce the plotly_white look."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.express as px
        d = df_sub_table.copy()
        d["n_sig_plot"] = d.n_subregion_signature.clip(lower=1)
        cmap = {**self._region_palette(d.region_id.unique()), **(region_colors or {})}
        fig = px.scatter(d, x="n_spots", y="coherence_ratio", color="region_id", size="n_sig_plot", size_max=size_max,
                         hover_name="sub_region_id", color_discrete_map=cmap, log_x=True,
                         hover_data=dict(status=True, n_subregion_signature=True, n_not_present=True, n_sig_plot=False,
                                         subregion_signature_genes=True))
        marker = dict(line=dict(color=marker_line_color, width=marker_line_width))
        if marker_opacity is not None:
            marker["opacity"] = marker_opacity
        fig.update_traces(marker=marker)
        fig.add_hline(y=self.params["coherence_min_ratio"], line_dash="dash", line_color=line_color, line_width=line_width)
        axis_title = dict(font=dict(size=fontsize_axis, color=font_color))
        fig.update_layout(title=dict(text="Subregion diagnostics: size vs spatial coherence (marker size = signature genes)",
                                     font=dict(size=fontsize_title, color=font_color)),
                          template="plotly_white", height=height, width=width,
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          legend=dict(font=dict(size=fontsize_legend, color=font_color),
                                      title=dict(font=dict(size=fontsize_legend_title, color=font_color))),
                          hoverlabel=dict(font=dict(size=fontsize_hover)))
        fig.update_xaxes(title=dict(text="spots in subregion (log)", **axis_title),
                         tickfont=dict(size=fontsize_ticks, color=font_color), gridcolor=grid_color)
        fig.update_yaxes(title=dict(text="spatial coherence ratio (1 = random)", **axis_title),
                         tickfont=dict(size=fontsize_ticks, color=font_color), gridcolor=grid_color)
        self._save_figure(fig, fname, force, verbose)
        return fig

    def plotly_region_signature_heatmap(self, adata_expr, df_rank_region: pd.DataFrame, n_per_region: int = 3,
                                        label_col: str = "region_label", height: int = 700, width: Optional[int] = None,
                                        fontsize_title: int = 17, fontsize_xticks: int = 12, fontsize_yticks: int = 9,
                                        fontsize_colorbar: int = 12, fontsize_hover: int = 13, colorscale: str = "RdBu_r",
                                        font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                        fname: Optional[str] = "fig_region_heatmap",
                                        force: bool = False, verbose: bool = False):
        """Mean log1p(CP10k) of the top region signature genes (rows) in every region (columns), z-scored per gene.

        Font sizes: fontsize_title, fontsize_xticks (regions), fontsize_yticks (genes), fontsize_colorbar, fontsize_hover.
        Colours: colorscale (diverging, centred at 0), font_color, bg_color."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
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
        fig = go.Figure(go.Heatmap(z=Z.values, x=regions, y=genes, colorscale=colorscale, zmid=0,
                                   customdata=df_mean.values.round(2),
                                   hovertemplate="%{y} in %{x}<br>z = %{z:.2f}<br>mean log1p = %{customdata}<extra></extra>",
                                   colorbar=dict(title=dict(text="z (per gene)", font=dict(size=fontsize_colorbar)),
                                                 tickfont=dict(size=fontsize_colorbar))))
        fig.update_layout(title=dict(text=f"Top {n_per_region} signature genes per region",
                                     font=dict(size=fontsize_title, color=font_color)),
                          height=height, width=width, template="plotly_white",
                          font=dict(color=font_color), paper_bgcolor=bg_color, plot_bgcolor=bg_color,
                          hoverlabel=dict(font=dict(size=fontsize_hover)),
                          xaxis=dict(tickfont=dict(size=fontsize_xticks, color=font_color)),
                          yaxis=dict(autorange="reversed", tickfont=dict(size=fontsize_yticks, color=font_color)))
        self._save_figure(fig, fname, force, verbose)
        return fig

    # =====================================================================================================
    # 9. LIANA: ligand-receptor (LR) communication in regions / subregions and between cell types
    #    (mp_lymphoid04, mp_mouse_brain04). Dataset-specific values are parameters (liana_*, lr_*, c2l_lr_*,
    #    ref_lr_*, coloc_*): resource ('consensus' human, 'mouseconsensus' mouse), kernel bandwidth in spot
    #    spacings (one kernel per section), signatures and abundances from cell2location.
    # =====================================================================================================
    @staticmethod
    def _liana_compat():
        """liana <= 1.10 with pandas >= 3: single-score methods (logfc, sca) store their missing score in a column
        labelled None and then fail on lr_res.drop([None]). Recompiles liana's _run_method with a tolerant drop
        (only when that exact line is present; no files are changed)."""
        import inspect
        import liana.method.sc._liana_pipe as lp
        if getattr(lp, "_compat_patched", False):
            return
        src = inspect.getsource(lp._run_method)
        old = "lr_res = lr_res.drop([None], axis=1)"
        if old in src:
            new = ("lr_res = lr_res.drop([c for c in lr_res.columns if c is None or c == 'None' or "
                   "(isinstance(c, float) and c != c)], axis=1)")
            exec(compile(src.replace(old, new), lp.__file__, "exec"), lp.__dict__)
        lp._compat_patched = True

    @staticmethod
    def _bh(p) -> np.ndarray:
        """Benjamini-Hochberg adjusted p-values."""
        p = np.asarray(p, float)
        n = len(p)
        o = np.argsort(p)
        q = p[o] * n / np.arange(1, n + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(n)
        out[o] = np.clip(q, 0, 1)
        return out

    def _groups(self, adata, group_key: str, skip: Optional[Sequence[str]] = None) -> pd.Series:
        """obs[group_key] as str (NaN / skipped groups -> None)."""
        skip = set(self.params["subregion_skip"] if skip is None else skip)
        g = adata.obs[group_key].astype(object).where(adata.obs[group_key].notna(), None)
        g = g.map(lambda x: None if x is None or str(x) in skip or str(x) == "nan" else str(x))
        return g

    def spatial_weights(self, adata=None, bandwidth_spots: Optional[float] = None, cutoff: Optional[float] = None,
                        max_neighbours: int = 100, key_added: str = "spatial", fname: Optional[str] = "spatial_weights",
                        force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Gaussian spot kernel w = exp(-d^2 / (2 bw^2)) built per section (obs[sample_col]; block-diagonal, sections
        never connected), bw = bandwidth_spots x the section's spot spacing, self weight 1 (several cells share a
        spot), weights < cutoff removed (liana keeps them). Writes obsp[f'{key_added}_connectivities'].
        Returns df_w: library, spacing_px, bandwidth_px, ring (distance / spacing), n neighbours, weight.
        The weights are always computed (cheap; obsp is needed); df_w is written to files[fname] when registered and
        missing (or force)."""
        import liana as li
        from scipy.spatial import cKDTree
        a = self._check_adata(adata)
        bw = self.get_param("liana_bandwidth_spots", bandwidth_spots)
        cutoff = self.get_param("liana_cutoff", cutoff)
        libs = a.obs[self.sample_col].astype(str).values if self.sample_col in a.obs else np.zeros(a.n_obs, str)
        xy_all = np.asarray(a.obsm["spatial"], float)
        R, C, V, rows = [], [], [], []
        for lb in pd.unique(libs):
            idx = np.where(libs == lb)[0]
            xy = xy_all[idx]
            step = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
            tmp = ad.AnnData(obs=pd.DataFrame(index=a.obs_names[idx]))
            tmp.obsm["spatial"] = xy
            li.pp.spatial_neighbors(tmp, bandwidth=bw * step, cutoff=cutoff, kernel="gaussian", set_diag=True,
                                    max_neighbours=max_neighbours)
            Wl = sp.coo_matrix(tmp.obsp["spatial_connectivities"])
            k = Wl.data >= cutoff
            R.append(idx[Wl.row[k]]); C.append(idx[Wl.col[k]]); V.append(Wl.data[k])
            Wc = sp.csr_matrix((Wl.data[k], (Wl.row[k], Wl.col[k])), shape=(len(idx), len(idx)))
            i = int(np.argmin(np.linalg.norm(xy - np.median(xy, 0), axis=1)))            # a central spot
            d = np.linalg.norm(xy[Wc[i].indices] - xy[i], axis=1) / step
            df_r = pd.DataFrame({"ring": np.round(d, 1), "weight": Wc[i].data}).groupby("ring") \
                     .agg(n=("weight", "size"), weight=("weight", "mean")).reset_index()
            df_r.insert(0, "library", lb); df_r.insert(1, "spacing_px", round(step, 1))
            df_r.insert(2, "bandwidth_px", round(bw * step, 1))
            df_r["mean_neighbours_incl_self"] = round(float(np.diff(Wc.indptr).mean()), 1)
            rows.append(df_r)
        W = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(a.n_obs, a.n_obs))
        a.obsp[f"{key_added}_connectivities"] = W
        df_w = pd.concat(rows, ignore_index=True)
        df_w["weight"] = df_w.weight.round(3)
        if fname is not None and fname in self.files and (force or not self.file_path(fname).exists()):
            self.write_tsv(df_w, fname, verbose=verbose)
        return df_w

    def liana_bivariate(self, adata=None, resource_name: Optional[str] = None, local_name: Optional[str] = None,
                        global_name: Optional[str] = None, n_perms: Optional[int] = None,
                        nz_prop: Optional[float] = None, fname: Optional[str] = "liana_bivariate",
                        fname_global: Optional[str] = "liana_global", force: bool = False,
                        verbose: bool = False, **kw) -> Tuple[pd.DataFrame, ad.AnnData]:
        """
        LIANA bivariate on log-normalised X with obsp['spatial_connectivities'] (spatial_weights is called when
        missing): global Moran's R (permutation p) + local score (cosine) with categories (1 = high-high) and local
        p-values. Cached as h5ad: fname = key of self.files or a file name in root_result (force=False: read it).
        The global table (df_inter) is written to files[fname_global] when registered. Returns (df_inter sorted by
        Moran's R, with morans_fdr (BH over all tested pairs), adata_lr: spots x LR pairs, X = local score, layers
        'cats', 'pvals').
        """
        import liana as li
        
        a = self._check_adata(adata)
        path = self._resolve_table(fname)
        computed = False
        if path is not None and path.exists() and not force:
            adata_lr = ad.read_h5ad(path)
            if verbose:
                print(f"Reading {path}")
        else:
            if "spatial_connectivities" not in a.obsp:
                self.spatial_weights(a)
            adata_lr = li.mt.bivariate(a, resource_name=self.get_param("liana_resource", resource_name),
                                       local_name=self.get_param("liana_local", local_name),
                                       global_name=self.get_param("liana_global", global_name),
                                       n_perms=self.get_param("liana_n_perms", n_perms), mask_negatives=False,
                                       add_categories=True, nz_prop=self.get_param("liana_nz_prop", nz_prop),
                                       use_raw=False, verbose=False, **kw)
            for k_ in ("cats", "pvals"):
                if k_ in adata_lr.layers and not sp.issparse(adata_lr.layers[k_]):
                    adata_lr.layers[k_] = sp.csr_matrix(adata_lr.layers[k_])
            computed = True
            if path is not None:
                adata_lr.write_h5ad(path)
                if verbose:
                    print(f"Saving {path}")
        g = self.params["liana_global"]
        df_inter = adata_lr.var.sort_values(g, ascending=False).copy()
        df_inter[f"{g}_fdr"] = self._bh(df_inter[f"{g}_pvals"].values)
        df_inter.index.name = "interaction"
        if fname_global is not None and fname_global in self.files and \
                (computed or force or not self.file_path(fname_global).exists()):
            self.write_tsv(df_inter.reset_index(), fname_global, verbose=verbose)
        return df_inter, adata_lr

    def lr_significant(self, df_inter: pd.DataFrame, fdr: Optional[float] = None) -> pd.DataFrame:
        """LR pairs with global score > 0 at FDR <= liana_fdr (sorted by the global score)."""
        g = self.params["liana_global"]
        fdr = self.get_param("liana_fdr", fdr)
        return df_inter[(df_inter[f"{g}_fdr"] <= fdr) & (df_inter[g] > 0)]

    def _lr_display(self, interactions: Sequence[str]) -> List[str]:
        rx = re.compile(self.params["lr_display_exclude"]) if self.params["lr_display_exclude"] else None
        return [i for i in interactions if not (rx and rx.search(i))]

    LR_STR_COLS = ("group", "interaction", "ligand", "receptor", "level", "sender", "receiver", "link", "lr", "source",
                   "target", "ligand_complex", "receptor_complex", "best_group", "ref_links")

    def calc_df_lr_by_group(self, adata_lr, groups: pd.Series, interactions: Optional[Sequence[str]] = None,
                            min_auroc: Optional[float] = None, local_p: Optional[float] = None,
                            fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached in files[fname] (e.g. 'lr_by_region', 'lr_by_subregion'; see _calc_df_lr_by_group)."""
        return self._cached_table(fname, lambda: self._calc_df_lr_by_group(adata_lr, groups, interactions, min_auroc, local_p),
                                  force, verbose, str_cols=("group", "interaction", "ligand", "receptor"))

    def _calc_df_lr_by_group(self, adata_lr, groups: pd.Series, interactions: Optional[Sequence[str]] = None,
                            min_auroc: Optional[float] = None, local_p: Optional[float] = None) -> pd.DataFrame:
        """LR activity of every group (region / subregion) vs the rest of the tissue, from the LIANA local scores:
        mean_in / mean_out (local score), AUROC (P(local score of a group spot > a spot elsewhere); 0.5 = no
        difference), hh_frac_in / hh_frac_out (fraction of spots in the high-high category with local p < local_p),
        Mann-Whitney p and BH fdr per group. enriched = AUROC >= min_auroc and hh_frac_in > hh_frac_out.
        Spots are spatially autocorrelated, so p-values are optimistic: rank by AUROC (effect size).
        groups: Series aligned with adata_lr.obs_names (None = not assigned)."""
        from scipy.stats import norm
        min_auroc = self.get_param("lr_group_min_auroc", min_auroc)
        local_p = self.get_param("lr_local_p", local_p)
        ints = list(interactions) if interactions is not None else list(adata_lr.var_names)
        A = adata_lr[:, ints]
        X = A.X.toarray() if sp.issparse(A.X) else np.asarray(A.X)
        C = A.layers["cats"]; C = C.toarray() if sp.issparse(C) else np.asarray(C)
        P = A.layers["pvals"]; P = P.toarray() if sp.issparse(P) else np.asarray(P)
        HH = (C == 1) & (P < local_p)
        g = pd.Series(groups).reindex(adata_lr.obs_names)
        ok = g.notna().values
        X, HH, gv = X[ok], HH[ok], g[ok].astype(str).values
        from scipy.stats import rankdata
        Rk = np.apply_along_axis(rankdata, 0, X)
        n = len(gv)
        rows = []
        for grp in pd.unique(gv):
            m = gv == grp
            n1, n0 = int(m.sum()), int(n - m.sum())
            if n1 == 0 or n0 == 0:
                continue
            U = Rk[m].sum(0) - n1 * (n1 + 1) / 2
            auc = U / (n1 * n0)
            zU = (U - n1 * n0 / 2) / np.sqrt(n1 * n0 * (n1 + n0 + 1) / 12)
            pv = 2 * norm.sf(np.abs(zU))
            rows.append(pd.DataFrame({"group": grp, "interaction": ints, "n_spots": n1,
                                      "mean_in": X[m].mean(0), "mean_out": X[~m].mean(0), "auroc": auc,
                                      "hh_frac_in": HH[m].mean(0), "hh_frac_out": HH[~m].mean(0), "p": pv,
                                      "fdr": self._bh(pv)}))
        df_lr_group = pd.concat(rows, ignore_index=True)
        df_lr_group["enriched"] = (df_lr_group.auroc >= min_auroc) & (df_lr_group.hh_frac_in > df_lr_group.hh_frac_out)
        lr_cols = adata_lr.var[["ligand", "receptor"]]
        df_lr_group = df_lr_group.join(lr_cols, on="interaction")
        return df_lr_group.sort_values(["group", "auroc"], ascending=[True, False]).reset_index(drop=True)

    def calc_df_lr_group_top(self, df_lr_group: pd.DataFrame, n: Optional[int] = None, enriched_only: bool = True) -> pd.Series:
        """Top-n LR pairs per group (display filter lr_display_exclude): 'CCL19^CCR7 (0.96), ...'."""
        n = self.get_param("lr_n_show", n)
        d = df_lr_group[df_lr_group.enriched] if enriched_only else df_lr_group
        d = d[d.interaction.isin(set(self._lr_display(d.interaction.unique())))]
        return d.groupby("group", sort=False).apply(
            lambda x: ", ".join(f"{i} ({v:.2f})" for i, v in zip(x.interaction.head(n), x.auroc.head(n))),
            include_groups=False).rename("top_lr")

    def calc_celltype_expr(self, adata, df_abund: pd.DataFrame, df_sig: pd.DataFrame, genes: Sequence[str],
                           counts_layer: Optional[str] = None, target_sum: Optional[float] = None
                           ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Cell-type-attributed expression without the c2l per-cell-type layers: the observed counts of every spot are
        split across cell types in proportion to abundance x reference signature,
            counts_A(s, g) = counts(s, g) * w(s, A) g(A, g) / sum_B w(s, B) g(B, g),
        which is how cell2location's per-cell-type layers are made (checked: identical to the layers of the lymph node
        map with q05 abundances and 'means' signatures). Returns (E = log1p(CP10k of the split counts; spot library
        size), spots x cell types x genes, float32; genes found; cell types).
        df_sig: genes (same names as adata.var_names) x cell types; df_abund: spots x cell types."""
        counts_layer = self.get_param("counts_layer", counts_layer)
        target_sum = self.get_param("target_sum", target_sum)
        celltypes = [c for c in df_abund.columns if c in df_sig.columns]
        genes = [x for x in dict.fromkeys(genes) if x in adata.var_names and x in df_sig.index]
        Xc = adata[:, genes].layers[counts_layer]
        Xc = (Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)).astype(np.float32)
        tot = np.asarray(adata.layers[counts_layer].sum(1)).ravel().astype(np.float32)
        Wab = df_abund.loc[adata.obs_names, celltypes].values.astype(np.float32)
        G = df_sig.loc[genes, celltypes].values.astype(np.float32)
        E = np.empty((adata.n_obs, len(celltypes), len(genes)), np.float32)
        for j in range(len(genes)):                                # gene by gene: memory n x k
            mu = Wab * G[j][None, :]
            E[:, :, j] = Xc[:, [j]] * mu / np.maximum(mu.sum(1, keepdims=True), 1e-12)
        E = np.log1p(E / np.maximum(tot, 1)[:, None, None] * target_sum)
        return E, genes, celltypes

    @staticmethod
    def _lr_entity(E, gpos, entity, sep="_"):
        subs = entity.split(sep)
        if not all(s_ in gpos for s_ in subs):
            return None
        return np.min(np.stack([E[:, :, gpos[s_]] for s_ in subs], -1), -1)     # complex = min over subunits

    def calc_df_celltype_lr(self, E: np.ndarray, genes: Sequence[str], celltypes: Sequence[str], df_lr: pd.DataFrame,
                            W, groups=None, n_perms: Optional[int] = None, group_n_perms: Optional[int] = None,
                            group_top_k: Optional[int] = None, seed: Optional[int] = None,
                            fname: Optional[str] = "celltype_lr_tissue", fname_group: Optional[str] = "celltype_lr_by_group",
                            force: bool = False, verbose: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Cached in files[fname] (whole tissue) and files[fname_group] (per group) when registered: both present and
        not force -> read (the permutations are the slow part). See _calc_df_celltype_lr."""
        keys = [fname] + ([fname_group] if groups is not None else [])
        reg = all(k is not None and k in self.files for k in keys)
        str_cols = ("interaction", "ligand", "receptor", "sender", "receiver", "level", "group")
        if reg and not force and all(self.file_path(k).exists() for k in keys):
            df_ct = self.read_tsv(fname, str_cols=str_cols, verbose=verbose)
            df_group_ct = self.read_tsv(fname_group, str_cols=str_cols, verbose=verbose) if groups is not None else None
            return df_ct, df_group_ct
        df_ct, df_group_ct = self._calc_df_celltype_lr(E, genes, celltypes, df_lr, W, groups, n_perms, group_n_perms,
                                                       group_top_k, seed)
        if reg:
            self.write_tsv(df_ct, fname, verbose=verbose)
            df_ct = self.read_tsv(fname, str_cols=str_cols)
            if df_group_ct is not None:
                self.write_tsv(df_group_ct, fname_group, verbose=verbose)
                df_group_ct = self.read_tsv(fname_group, str_cols=str_cols)
        return df_ct, df_group_ct

    def _calc_df_celltype_lr(self, E: np.ndarray, genes: Sequence[str], celltypes: Sequence[str], df_lr: pd.DataFrame,
                            W, groups=None, n_perms: Optional[int] = None,
                            group_n_perms: Optional[int] = None, group_top_k: Optional[int] = None,
                            seed: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Sender cell type A -> receiver cell type B for every LR pair (df_lr: ligand, receptor; complexes '_'):
            score[A, B] = mean_s ( (W_norm L_A)(s) * R_B(s) ),  L_A / R_B = ligand of A / receptor of B (E).
        Whole tissue: receptor spots permuted n_perms times -> z, p (Gaussian tail of z when < 5 permutations reach
        the score), BH fdr; score_share = share of the LR total; rank_in_lr.
        Per group (groups: Series aligned with the spots, None = skip; or {level: Series} for several levels, e.g.
        regions and subregions -> column 'level'): the same score over the group's spots,
        receptor spots permuted within the group (group_n_perms), ratio = group score / whole-tissue score; the
        group_top_k links per (group, LR) are kept. The spot-permutation null ignores spatial autocorrelation:
        rank by score / z rather than by fdr."""
        from scipy import stats
        n_perms = self.get_param("c2l_lr_n_perms", n_perms)
        group_n_perms = self.get_param("c2l_lr_group_n_perms", group_n_perms)
        group_top_k = self.get_param("c2l_lr_group_top_k", group_top_k)
        rng = np.random.default_rng(self.get_param("seed", seed))
        gpos = {x: i for i, x in enumerate(genes)}
        W = sp.csr_matrix(W, dtype=np.float64)
        s_ = np.asarray(W.sum(1)).ravel(); s_[s_ == 0] = 1
        Wn = sp.csr_matrix(sp.diags(1 / s_) @ W)
        n = E.shape[0]
        perms = [rng.permutation(n) for _ in range(n_perms)]
        S, Rr = np.meshgrid(celltypes, celltypes, indexing="ij")
        levels = {} if groups is None else (dict(groups) if isinstance(groups, dict) else {"group": groups})
        grp_idx, grp_perms = {}, {}
        for lev, gs in levels.items():
            gv = pd.Series(gs).values
            for grp in pd.unique(gv[pd.notna(gv)]):
                if grp is None:
                    continue
                ix = np.where(gv == grp)[0]
                grp_idx[(lev, grp)] = ix
                grp_perms[(lev, grp)] = [rng.permutation(len(ix)) for _ in range(group_n_perms)]
        rows, grows = [], []

        def _p(z, n_ge, n_p):
            pe = (1 + n_ge) / (1 + n_p)
            return pe, np.where(n_ge < 5, np.minimum(pe, np.maximum(stats.norm.sf(z), 1e-12)), pe)

        for _, r in df_lr.iterrows():
            Lk, Rk = self._lr_entity(E, gpos, r.ligand), self._lr_entity(E, gpos, r.receptor)
            if Lk is None or Rk is None:
                continue
            WL = Wn @ Lk
            obs = WL.T @ Rk / n
            null = np.stack([WL.T @ Rk[p] / n for p in perms])
            z = (obs - null.mean(0)) / (null.std(0) + 1e-12)
            pe, p = _p(z, (null >= obs).sum(0), n_perms)
            inter = f"{r.ligand}^{r.receptor}"
            rows.append(pd.DataFrame({"interaction": inter, "ligand": r.ligand, "receptor": r.receptor,
                                      "sender": S.ravel(), "receiver": Rr.ravel(), "score": obs.ravel(),
                                      "z": z.ravel(), "p_emp": pe.ravel(), "p": p.ravel()}))
            for (lev, grp), ix in grp_idx.items():
                WLg, Rg = WL[ix], Rk[ix]
                og = WLg.T @ Rg / len(ix)
                ng = np.stack([WLg.T @ Rg[p] / len(ix) for p in grp_perms[(lev, grp)]])
                zg = (og - ng.mean(0)) / (ng.std(0) + 1e-12)
                _, pg = _p(zg, (ng >= og).sum(0), group_n_perms)
                top = np.argsort(og.ravel())[::-1][:group_top_k]
                grows.append(pd.DataFrame({"level": lev, "group": grp, "interaction": inter, "ligand": r.ligand,
                                           "receptor": r.receptor, "sender": S.ravel()[top], "receiver": Rr.ravel()[top],
                                           "score": og.ravel()[top], "score_tissue": obs.ravel()[top],
                                           "ratio": og.ravel()[top] / np.maximum(obs.ravel()[top], 1e-12),
                                           "z": zg.ravel()[top], "p": pg.ravel()[top],
                                           "rank_in_group_lr": np.arange(1, len(top) + 1)}))
        df_ct = pd.concat(rows, ignore_index=True)
        df_ct["fdr"] = self._bh(df_ct.p.values)
        df_ct["score_share"] = df_ct.score / df_ct.groupby("interaction").score.transform("sum")
        df_ct["rank_in_lr"] = df_ct.groupby("interaction").score.rank(ascending=False, method="first").astype(int)
        df_ct = df_ct.sort_values(["interaction", "rank_in_lr"]).reset_index(drop=True)
        df_group_ct = None
        if grows:
            df_group_ct = pd.concat(grows, ignore_index=True)
            df_group_ct["fdr"] = df_group_ct.groupby("level").p.transform(lambda x: self._bh(x.values))
        return df_ct, df_group_ct

    def calc_df_group_celltype_links(self, df_group_ct: pd.DataFrame, min_z: Optional[float] = None,
                                     min_ratio: float = 1.0, display_filter: bool = True,
                                     fname: Optional[str] = "celltype_links_by_group", force: bool = False,
                                     verbose: bool = False) -> pd.DataFrame:
        """Cached in files[fname] (see _calc_df_group_celltype_links)."""
        return self._cached_table(fname, lambda: self._calc_df_group_celltype_links(df_group_ct, min_z, min_ratio, display_filter),
                                  force, verbose, str_cols=("group", "link", "sender", "receiver", "lr"))

    def _calc_df_group_celltype_links(self, df_group_ct: pd.DataFrame, min_z: Optional[float] = None,
                                     min_ratio: float = 1.0, display_filter: bool = True) -> pd.DataFrame:
        """Per group and sender -> receiver: number of LR pairs for which the link is among the group's top links,
        with z >= min_z (within-group permutation) and score >= the whole-tissue score (ratio >= min_ratio);
        lr = those LR pairs (by score); LR pairs matching lr_display_exclude are not counted (display_filter);
        self_link = sender and receiver are the same type (L and R attributed to one type in the same spots)."""
        min_z = self.get_param("c2l_lr_min_z", min_z)
        d = df_group_ct[(df_group_ct.z >= min_z) & (df_group_ct.ratio >= min_ratio)].sort_values("score", ascending=False)
        if display_filter:
            d = d[d.interaction.isin(set(self._lr_display(d.interaction.unique())))]
        d = d.assign(link=d.sender + " -> " + d.receiver)
        df_links = d.groupby(["group", "link", "sender", "receiver"], sort=False).agg(
            n_lr=("interaction", "size"), sum_score=("score", "sum"),
            lr=("interaction", lambda x: ", ".join(list(x)[:6]))).reset_index()
        df_links["self_link"] = df_links.sender == df_links.receiver
        return df_links.sort_values(["group", "n_lr", "sum_score"], ascending=[True, False, False]).reset_index(drop=True)

    def liana_rank_aggregate_reference(self, adata_ref, groupby: str, symbol_col: Optional[str] = None,
                                       n_per_type: Optional[int] = None, resource_name: Optional[str] = None,
                                       expr_prop: Optional[float] = None, n_perms: Optional[int] = None,
                                       fname: Optional[str] = "liana_reference", force: bool = False,
                                       verbose: bool = False) -> pd.DataFrame:
        """Cell-type-resolved, non-spatial LIANA rank_aggregate (CellPhoneDB, Connectome, log2FC, NATMI, SCA,
        CellChat-like) on the scRNA reference: up to n_per_type cells per type (seed), raw counts -> log1p(CP10k),
        gene symbols (var[symbol_col] or var_names). Cached as TSV: fname = key of self.files or a file name in
        root_result (force=False: read it)."""
        import liana as li
        path = self._resolve_table(fname)
        if path is not None and path.exists() and not force:
            if verbose:
                print(f"Reading {path}")
            return pdreadcsv(path.name, path.parent)
        self._liana_compat()
        n_per_type = self.get_param("ref_lr_cells_per_type", n_per_type)
        rng = np.random.default_rng(self.params["seed"])
        lab = adata_ref.obs[groupby].astype(str)
        keep = np.sort(np.concatenate([rng.choice(ix, size=min(n_per_type, len(ix)), replace=False)
                                       for ix in lab.groupby(lab.values).indices.values()]))
        sub = adata_ref[keep]
        X = sub.X[:] if not hasattr(sub.X, "to_memory") else sub.X.to_memory()
        X = sp.csr_matrix(X, dtype=np.float32)
        names = sub.var[symbol_col].astype(str).values if symbol_col else np.asarray(sub.var_names).astype(str)
        a = ad.AnnData(X=X, obs=pd.DataFrame({groupby: lab.values[keep]}, index=sub.obs_names),
                       var=pd.DataFrame(index=pd.Index(names)))
        a.var_names_make_unique()
        a.obs[groupby] = a.obs[groupby].astype("category")
        sc.pp.normalize_total(a, target_sum=self.params["target_sum"])
        sc.pp.log1p(a)
        li.mt.rank_aggregate(a, groupby=groupby, resource_name=self.get_param("liana_resource", resource_name),
                             expr_prop=self.get_param("liana_expr_prop", expr_prop), use_raw=False,
                             n_perms=self.get_param("liana_n_perms", n_perms), n_jobs=self.params["liana_n_jobs"],
                             seed=self.params["seed"], verbose=False)
        df_liref = a.uns["liana_res"].copy()
        df_liref.attrs["n_cells"] = int(a.n_obs)
        if path is not None:
            pdwritecsv(df_liref, path.name, path.parent)
            if verbose:
                print(f"Saving {path} ({a.n_obs} cells, {len(df_liref)} interactions)")
            df_liref = pdreadcsv(path.name, path.parent)
        return df_liref

    def calc_df_celltype_coloc(self, df_abund: pd.DataFrame, W, groups: Optional[pd.Series] = None
                               ) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """Cell-type co-localisation: Pearson r between the abundance of A at a spot and the W-weighted (niche)
        abundance of B, symmetrised (whole tissue). Per group: the group's share of that correlation,
        mean over the group's spots of z(A) z(W B) (z over the whole tissue), symmetrised; > 0 when A and the
        neighbourhood of B are high together in the group."""
        A = df_abund.values.astype(float)
        W = sp.csr_matrix(W, dtype=np.float64)
        s_ = np.asarray(W.sum(1)).ravel(); s_[s_ == 0] = 1
        WA = sp.diags(1 / s_) @ W @ A
        Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
        WAz = (WA - WA.mean(0)) / (WA.std(0) + 1e-12)
        Cm = Az.T @ WAz / len(A)
        df_coloc = pd.DataFrame((Cm + Cm.T) / 2, index=df_abund.columns, columns=df_abund.columns)
        out = None
        if groups is not None:
            gv = pd.Series(groups).values
            out = {}
            for grp in pd.unique(gv[pd.notna(gv)]):
                if grp is None:
                    continue
                m = gv == grp
                Cg = Az[m].T @ WAz[m] / m.sum()
                out[grp] = pd.DataFrame((Cg + Cg.T) / 2, index=df_abund.columns, columns=df_abund.columns)
        return df_coloc, out

    def calc_df_ref_lr_support(self, df_liref: pd.DataFrame, df_coloc: pd.DataFrame,
                               coloc_group: Optional[Dict[str, pd.DataFrame]] = None,
                               spec_rank: Optional[float] = None, mag_rank: Optional[float] = None,
                               min_coloc: Optional[float] = None, min_coloc_group: Optional[float] = None,
                               fname: Optional[str] = "liana_reference_supported", force: bool = False,
                               verbose: bool = False) -> pd.DataFrame:
        """Cached in files[fname] (see _calc_df_ref_lr_support)."""
        return self._cached_table(fname, lambda: self._calc_df_ref_lr_support(df_liref, df_coloc, coloc_group, spec_rank, mag_rank,
                                                                              min_coloc, min_coloc_group),
                                  force, verbose, str_cols=("source", "target", "interaction", "best_group"))

    def _calc_df_ref_lr_support(self, df_liref: pd.DataFrame, df_coloc: pd.DataFrame,
                               coloc_group: Optional[Dict[str, pd.DataFrame]] = None,
                               spec_rank: Optional[float] = None, mag_rank: Optional[float] = None,
                               min_coloc: Optional[float] = None, min_coloc_group: Optional[float] = None) -> pd.DataFrame:
        """Reference interactions (rank_aggregate) that are specific and strong (specificity_rank, magnitude_rank <=
        cutoffs) annotated with the co-localisation of source and target (coloc), the group where they co-localise
        most (best_group, coloc_group) and spatially_supported = coloc >= min_coloc or coloc_group >= min_coloc_group."""
        spec_rank = self.get_param("ref_lr_spec_rank", spec_rank)
        mag_rank = self.get_param("ref_lr_mag_rank", mag_rank)
        min_coloc = self.get_param("coloc_min", min_coloc)
        min_coloc_group = self.get_param("coloc_group_min", min_coloc_group)
        d = df_liref[(df_liref.specificity_rank <= spec_rank) & (df_liref.magnitude_rank <= mag_rank)].copy()
        d = d[d.source.isin(df_coloc.index) & d.target.isin(df_coloc.index)]
        d["interaction"] = d.ligand_complex + "^" + d.receptor_complex
        d["coloc"] = [df_coloc.at[s_, t] for s_, t in zip(d.source, d.target)]
        if coloc_group:
            pos = {c: i for i, c in enumerate(df_coloc.index)}
            si, ti = d.source.map(pos).values, d.target.map(pos).values
            G = np.stack([coloc_group[g].reindex(index=df_coloc.index, columns=df_coloc.index).values[si, ti]
                          for g in coloc_group], 1)
            gnames = np.array(list(coloc_group))
            d["best_group"] = gnames[G.argmax(1)]
            d["coloc_group"] = G.max(1)
        else:
            d["best_group"], d["coloc_group"] = "", np.nan
        d["spatially_supported"] = (d.coloc >= min_coloc) | (d.coloc_group >= min_coloc_group)
        d["listed"] = d.interaction.isin(set(self._lr_display(d.interaction.unique())))   # passes lr_display_exclude
        return d.sort_values(["spatially_supported", "magnitude_rank"], ascending=[False, True]).reset_index(drop=True)

    def calc_df_group_ccc(self, df_lr_group: pd.DataFrame, df_group_ct: Optional[pd.DataFrame] = None,
                          df_ref_sup: Optional[pd.DataFrame] = None, n: Optional[int] = None,
                          group_names: Optional[Dict[str, str]] = None, name_col: str = "region_name",
                          fname: Optional[str] = None, force: bool = False, verbose: bool = False) -> pd.DataFrame:
        """Cached in files[fname] (e.g. 'ccc_by_region', 'ccc_by_subregion'); group_names: adds name_col after 'group'.
        See _calc_df_group_ccc. Missing cell-type links read back as '' (count them with (col != '')."""
        def _compute():
            out = self._calc_df_group_ccc(df_lr_group, df_group_ct, df_ref_sup, n)
            if group_names:
                out.insert(1, name_col, out.group.map(group_names))
            return out
        return self._cached_table(fname, _compute, force, verbose, str_cols=("group", "interaction", "ref_links"))

    def _calc_df_group_ccc(self, df_lr_group: pd.DataFrame, df_group_ct: Optional[pd.DataFrame] = None,
                          df_ref_sup: Optional[pd.DataFrame] = None, n: Optional[int] = None) -> pd.DataFrame:
        """Consensus per group: its top-n enriched LR pairs (AUROC; display filter) with the best cell2location
        sender -> receiver of the pair in the group (z, ratio vs tissue) and the reference support: whether
        rank_aggregate finds the same LR between that sender and receiver (ref_same_link) and the best spatially
        supported reference source -> target of the pair (ref_links)."""
        n = self.get_param("lr_n_show", n)
        d = df_lr_group[df_lr_group.enriched]
        d = d[d.interaction.isin(set(self._lr_display(d.interaction.unique())))]
        d = d.groupby("group", sort=False).head(n)
        cols = ["group", "interaction", "auroc", "hh_frac_in", "hh_frac_out", "mean_in", "mean_out"]
        out = d[cols].copy()
        if df_group_ct is not None:
            best = df_group_ct[df_group_ct.rank_in_group_lr == 1].set_index(["group", "interaction"])
            key = list(zip(out.group, out.interaction))
            for c_src, c_dst in [("sender", "c2l_sender"), ("receiver", "c2l_receiver"), ("z", "c2l_z"),
                                 ("ratio", "c2l_ratio")]:
                out[c_dst] = [best[c_src].get(k, np.nan) for k in key]
        if df_ref_sup is not None:
            sup = df_ref_sup[df_ref_sup.spatially_supported]
            links = sup.groupby("interaction").apply(
                lambda x: ", ".join(f"{s_} -> {t}" for s_, t in zip(x.source.head(3), x.target.head(3))),
                include_groups=False)
            out["ref_links"] = out.interaction.map(links).fillna("")
            trip = set(zip(df_ref_sup.interaction, df_ref_sup.source, df_ref_sup.target))
            if "c2l_sender" in out:
                out["ref_same_link"] = [(i, s_, t) in trip for i, s_, t in zip(out.interaction, out.c2l_sender, out.c2l_receiver)]
        return out.reset_index(drop=True)

    # ---------------------------------------------------------------------------------------- LIANA figures
    def plot_lr_spatial(self, adata_lr, adata, interactions: Sequence[str], library_id: Optional[str] = None,
                        ncols: int = 4, figsize:tuple=(12,8), titles: Optional[Sequence[str]] = None,
                        outlines: Optional[Sequence[np.ndarray]] = None, cmap=None, point_size: float = 4,
                        image_alpha: float = 0.5, image_bg_color: Optional[str] = None, outline_color: str = "#00E5FF",
                        outline_width: float = 1.0, fontsize_title: float = 8, fontsize_suptitle: float = 9,
                        fontsize_colorbar: float = 6, font_color: Optional[str] = None, bg_color: Optional[str] = None,
                        fname: Optional[str] = "lr_local_scores.png", force: bool = False, verbose: bool = False):
        """LIANA local scores of LR pairs on the H&E of one section (colour 2nd-99.5th percentile). One panel per
        entry of interactions (repeats allowed); titles / outlines (boolean masks over adata.obs_names, e.g. the
        spots of a region, drawn with plot_region_outline) are given per panel.

        Image: image_bg_color (H&E copy with the glass background in this colour, hires_image_bg), image_alpha.
        Sizes: point_size, outline_width. Font sizes: fontsize_title (panels), fontsize_suptitle,
        fontsize_colorbar. Colours: cmap, outline_color, font_color, bg_color (figure and panels).
        fname: key of self.files ({lib}) or a file name in root_figure; an existing file is kept unless force."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        
        lib = library_id or self.library_ids(adata)[0]
        img = self._map_image(lib, adata, image_bg_color, verbose=verbose)
        m, xy = self.spot_xy(lib, adata)
        names = adata.obs_names[m]
        keep = [k for k, i in enumerate(interactions) if i in adata_lr.var_names]
        titles = [list(titles)[k] for k in keep] if titles is not None else [interactions[k] for k in keep]
        outlines = [list(outlines)[k] for k in keep] if outlines is not None else [None] * len(keep)
        interactions = [interactions[k] for k in keep]
        nrows = int(np.ceil(len(interactions) / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, facecolor=bg_color)
        cmap = cmap or cmap_byr

        for ax, it, tt, ol in zip(axes.ravel(), interactions, titles, outlines):
            ax.set_facecolor(bg_color)
            v = adata_lr[names, it].X
            v = np.asarray(v.toarray() if sp.issparse(v) else v).ravel()
            ax.imshow(img, alpha=image_alpha)
            o = np.argsort(v)
            sca = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=point_size, cmap=cmap, vmin=float(np.quantile(v, 0.02)),
                             vmax=float(np.quantile(v, 0.995)), lw=0)
            ax.set_xlim(xy[:, 0].min() - 30, xy[:, 0].max() + 30); ax.set_ylim(xy[:, 1].max() + 30, xy[:, 1].min() - 30)
            ax.set_axis_off()
            if ol is not None:
                self.plot_region_outline(ax, xy, np.asarray(ol)[m], color=outline_color, lw=outline_width, min_spots=5)
            ax.set_title(tt, fontsize=fontsize_title, loc="left", color=font_color)
            cb = fig.colorbar(sca, ax=ax, fraction=0.04, pad=0.01)
            cb.ax.tick_params(labelsize=fontsize_colorbar, colors=font_color)
            cb.outline.set_edgecolor(font_color)
        for ax in axes.ravel()[len(interactions):]:
            ax.set_visible(False)
        
        fig.tight_layout()
        
        fig.suptitle(f"{lib}: LIANA local {self.params['liana_local']} score", fontsize=fontsize_suptitle, color=font_color,
                     x=0.01, y=1.0, ha="left", va="bottom")
        
        self._save_figure(fig, fname, force, verbose, lib=lib)
        return fig

    def plotly_lr_group_heatmap(self, df_lr_group: pd.DataFrame, n_per_group: int = 3, groups: Optional[Sequence[str]] = None,
                                group_names: Optional[Dict[str, str]] = None, height: int = 900, width: int = 1200,
                                fontsize_title: int = 17, fontsize_xticks: int = 8, fontsize_yticks: int = 9,
                                fontsize_colorbar: int = 12, fontsize_hover: int = 13, colorscale: str = "RdBu_r",
                                font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                fname: Optional[str] = None, force: bool = False, verbose: bool = False):
        """Groups (rows) x LR pairs (columns: union of each group's top-n enriched pairs, display filter): AUROC of
        the local score (0.5 = as elsewhere). Hover: high-high fraction in / out.
        Font sizes: fontsize_title, fontsize_xticks (LR pairs), fontsize_yticks (groups), fontsize_colorbar,
        fontsize_hover. Colours: colorscale (diverging at 0.5), font_color, bg_color. fname: key ('fig_lr_region_hm')."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        groups = list(groups) if groups is not None else list(pd.unique(df_lr_group.group))
        d = df_lr_group[df_lr_group.enriched & df_lr_group.interaction.isin(set(self._lr_display(df_lr_group.interaction.unique())))]
        cols = []
        for grp in groups:
            for i in d[d.group == grp].interaction.head(n_per_group):
                if i not in cols:
                    cols.append(i)
        M = df_lr_group.pivot(index="group", columns="interaction", values="auroc").reindex(index=groups, columns=cols)
        H = df_lr_group.assign(t=lambda x: [f"high-high {a:.2f} vs {b:.2f}" for a, b in zip(x.hh_frac_in, x.hh_frac_out)]) \
                       .pivot(index="group", columns="interaction", values="t").reindex(index=groups, columns=cols)
        ylab = [f"{grp} {group_names.get(grp, '')}"[:45] if group_names else grp for grp in groups]
        fig = go.Figure(go.Heatmap(z=M.values, x=cols, y=ylab, colorscale=colorscale, zmid=0.5, zmin=0.2, zmax=0.8,
                                   customdata=H.values,
                                   colorbar=dict(title=dict(text="AUROC", font=dict(size=fontsize_colorbar)),
                                                 tickfont=dict(size=fontsize_colorbar)),
                                   hovertemplate="%{x} in %{y}<br>AUROC %{z:.2f}<br>%{customdata}<extra></extra>"))
        fig.update_layout(title=f"LR pairs enriched in each group (top {n_per_group}; AUROC of the local score vs the rest)",
                          height=height, width=width, template="plotly_white",
                          yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-60))
        self._plotly_style(fig, fontsize_title, fontsize_xticks, fontsize_yticks, fontsize_hover, font_color, bg_color)
        self._save_figure(fig, fname, force, verbose)
        return fig

    def plotly_group_celltype_links(self, df_links: pd.DataFrame, n_links: int = 30, groups: Optional[Sequence[str]] = None,
                                    group_names: Optional[Dict[str, str]] = None, exclude_self: bool = True,
                                    height: Optional[int] = None, width: Optional[int] = None, fontsize_title: int = 17,
                                    fontsize_xticks: int = 8, fontsize_yticks: int = 9, fontsize_colorbar: int = 12,
                                    fontsize_hover: int = 13, colorscale: str = "Reds", font_color: Optional[str] = None,
                                    bg_color: Optional[str] = None, fname: Optional[str] = None, force: bool = False,
                                    verbose: bool = False):
        """Groups (rows) x the n_links sender -> receiver links with most LR pairs over all groups (columns):
        number of LR pairs (calc_df_group_celltype_links). Hover: the LR pairs. exclude_self: links between
        different cell types only.
        Font sizes: fontsize_title, fontsize_xticks (links), fontsize_yticks (groups), fontsize_colorbar, fontsize_hover.
        Colours: colorscale, font_color, bg_color. fname: key ('fig_links_region', 'fig_links_sub')."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        groups = list(groups) if groups is not None else list(pd.unique(df_links.group))
        if exclude_self:
            df_links = df_links[~df_links.self_link]
        top = df_links.groupby("link").n_lr.sum().sort_values(ascending=False).index[:n_links]
        M = df_links.pivot_table(index="group", columns="link", values="n_lr", aggfunc="sum").reindex(index=groups, columns=top)
        T = df_links.pivot_table(index="group", columns="link", values="lr", aggfunc="first").reindex(index=groups, columns=top)
        ylab = [f"{grp} {group_names.get(grp, '')}"[:45] if group_names else grp for grp in groups]
        fig = go.Figure(go.Heatmap(z=M.fillna(0).values, x=list(top), y=ylab, colorscale=colorscale, customdata=T.fillna("").values,
                                   colorbar=dict(title=dict(text="LR pairs", font=dict(size=fontsize_colorbar)),
                                                 tickfont=dict(size=fontsize_colorbar)),
                                   hovertemplate="%{x} in %{y}<br>%{z} LR pairs<br>%{customdata}<extra></extra>"))
        fig.update_layout(title="Cell type -> cell type communication per group (cell2location-resolved; LR pairs with z >= "
                                f"{self.params['c2l_lr_min_z']} and score >= tissue" + ("; different types only)" if exclude_self else ")"),
                          height=height or max(450, 22 * len(groups) + 300), width=width or max(900, 18 * len(top) + 380),
                          template="plotly_white", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-60))
        self._plotly_style(fig, fontsize_title, fontsize_xticks, fontsize_yticks, fontsize_hover, font_color, bg_color)
        self._save_figure(fig, fname, force, verbose)
        return fig

    def plot_sender_receiver_heatmaps(self, df_ct: pd.DataFrame, interactions: Sequence[str], n_top: int = 10,
                                      ncols: int = 3, panel_size: Tuple[float, float] = (4.6, 4.0), cmap: str = "Reds",
                                      fontsize_title: float = 8, fontsize_ticks: float = 6, fontsize_label: float = 7,
                                      fontsize_colorbar: float = 6, font_color: Optional[str] = None, bg_color: Optional[str] = None,
                                      fname: Optional[str] = "celltype_lr_sender_receiver.png", force: bool = False,
                                      verbose: bool = False):
        """One heatmap per LR pair: the n_top senders (rows) x n_top receivers (columns) by total score.
        Font sizes: fontsize_title, fontsize_ticks (cell types), fontsize_label (axis labels), fontsize_colorbar.
        Colours: cmap, font_color, bg_color. fname: key of self.files or a file name; kept unless force."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        interactions = [i for i in interactions if i in set(df_ct.interaction)]
        nrows = int(np.ceil(len(interactions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False,
                                 facecolor=bg_color)
        for ax, it in zip(axes.ravel(), interactions):
            ax.set_facecolor(bg_color)
            M = df_ct[df_ct.interaction == it].pivot(index="sender", columns="receiver", values="score")
            M = M.loc[M.sum(1).nlargest(n_top).index, M.sum(0).nlargest(n_top).index]
            im = ax.imshow(M.values, cmap=cmap, aspect="auto")
            ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=90, fontsize=fontsize_ticks, color=font_color)
            ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=fontsize_ticks, color=font_color)
            ax.tick_params(colors=font_color)
            for sp_ in ax.spines.values():
                sp_.set_edgecolor(font_color)
            ax.set_title(it, fontsize=fontsize_title, loc="left", color=font_color)
            ax.set_xlabel("receiver (receptor)", fontsize=fontsize_label, color=font_color)
            ax.set_ylabel("sender (ligand)", fontsize=fontsize_label, color=font_color)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.ax.tick_params(labelsize=fontsize_colorbar, colors=font_color)
            cb.outline.set_edgecolor(font_color)
        for ax in axes.ravel()[len(interactions):]:
            ax.set_visible(False)
        fig.tight_layout()
        self._save_figure(fig, fname, force, verbose)
        return fig

    def plotly_coloc_heatmap(self, df_coloc: pd.DataFrame, height: int = 800, width: Optional[int] = None,
                             fontsize_title: int = 17, fontsize_ticks: int = 8, fontsize_colorbar: int = 12,
                             fontsize_hover: int = 13, colorscale: str = "RdBu_r", font_color: Optional[str] = None,
                             bg_color: Optional[str] = None, fname: Optional[str] = "fig_coloc", force: bool = False,
                             verbose: bool = False):
        """Cell-type co-localisation (symmetric), rows / columns ordered by hierarchical clustering.
        Font sizes: fontsize_title, fontsize_ticks (cell types), fontsize_colorbar, fontsize_hover.
        Colours: colorscale (diverging at 0), font_color, bg_color. fname: key of self.files."""
        font_color = self.get_param("font_color", font_color)   # style: None -> params
        bg_color = self.get_param("bg_color", bg_color)
        import plotly.graph_objects as go
        from scipy.cluster.hierarchy import linkage, leaves_list
        o = leaves_list(linkage(df_coloc.values, "average", metric="euclidean"))
        M = df_coloc.iloc[o, o]
        fig = go.Figure(go.Heatmap(z=M.values, x=list(M.columns), y=list(M.index), colorscale=colorscale, zmid=0, zmin=-1,
                                   zmax=1, colorbar=dict(title=dict(text="r", font=dict(size=fontsize_colorbar)),
                                                         tickfont=dict(size=fontsize_colorbar)),
                                   hovertemplate="%{y} with niche of %{x}: r = %{z:.2f}<extra></extra>"))
        fig.update_layout(title="Cell-type co-localisation (abundance at a spot vs kernel-weighted abundance around it)",
                          height=height, width=width or height + 80, template="plotly_white",
                          yaxis=dict(autorange="reversed"))
        self._plotly_style(fig, fontsize_title, fontsize_ticks, fontsize_ticks, fontsize_hover, font_color, bg_color)
        self._save_figure(fig, fname, force, verbose)
        return fig

    # =====================================================================================================
    # 10. Static helpers
    # =====================================================================================================
    @staticmethod
    def wtab(group, rgg) -> pd.DataFrame:
        """One group of a scanpy rank_genes_groups result as a DataFrame indexed by gene."""
        df_wtab = pd.DataFrame({"gene": rgg["names"][group], "lfc": rgg["logfoldchanges"][group],
                                "padj": rgg["pvals_adj"][group], "score": rgg["scores"][group]}).set_index("gene")
        return df_wtab

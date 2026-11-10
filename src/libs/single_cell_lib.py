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
9. LIANA                       liana_rank_aggregate, liana_bivariate
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

__version__ = "1.2.0"

colors_byr = ("#57a4e2", "#ebd246", "#f30707")
cmap_byr = LinearSegmentedColormap.from_list("blue_yellow_red", list(colors_byr))

# Kelly's colours of maximum contrast, without near-white and near-black
KELLY_COLORS = ['#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280',
                '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97',
                '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522',
                '#E25822', '#2B3D26']

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
    liana_bandwidth=200,
    liana_local="cosine",
    liana_global="morans",
    liana_nz_prop=0.2,
    # --- subclustering (subcluster_adaptive)
    sub_n_hvg=2000,
    sub_n_pcs=20,
    sub_n_neighbors=15,
    sub_batch="combat",            # 'combat' (scanpy, per batch) or None
    sub_min_gene_frac=0.02,        # genes detected in >= this fraction of the subset's spots / cells
    sub_resolutions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
    sub_min_size=20,               # every subcluster must have >= this many spots / cells
    sub_max_k=6,                   # at most this many subclusters
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

    def __init__(self, ROOT0: Path, project: str, adata: Optional[ad.AnnData] = None,
                 data_dir: str = "data", figure_dir: str = "figures", result_dir: str = "results",
                 symbol_col: str = "SYMBOL", sample_col: str = "sample", mt_prefix: str = "mt-",
                 params: Optional[Dict[str, Any]] = None, name: str = ""):
        
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
        self.params = copy.deepcopy(self.default_params)
        if params:
            self.set_params(**params)

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

    def _savefig(self, fig, fname: Optional[str], root_figure: Optional[Path] = None, dpi: Optional[int] = None, 
                 force: bool = False, verbose: bool = False):
        """Save fig to (root_figure or self.root_figure) / fname; fname=None -> not saved."""
        if fname is None:
            return None
        filename = Path(root_figure or self.root_figure) / fname

        if filename.exists() and not force:
            print(f"File already exists: {filename}")
        else:
            fig.savefig(filename, dpi=self.get_param("dpi", dpi), bbox_inches="tight")
            if verbose:
                print(f"Saving figure to {filename}")

        return filename


    # =====================================================================================================
    # 1. I/O and components
    # =====================================================================================================
    def read_h5ad(self, fname: str, root: Optional[Path] = None, backed: Optional[str] = None,
                  set_self: bool = True, verbose: bool = False) -> ad.AnnData:
        """
        Read (root or self.root_data) / fname (backed='r' keeps X on disk). 
        Sets self.adata unless set_self=False.
        """
        filename = Path(root or self.root_data) / fname

        if not filename.exists():
            raise FileNotFoundError(filename)
        
        a = ad.read_h5ad(filename, backed=backed)
        if set_self:
            self.adata = a

        if verbose:
            print(f"Reading {filename}")

        return a

    def write_h5ad(self, fname: str, adata=None, root: Optional[Path] = None,
                   compression: Optional[str] = "gzip", verbose: bool = False) -> Path:
        
        filename = Path(root or self.root_data) / fname

        self._check_adata(adata).write_h5ad(filename, compression=compression)

        if verbose:
            print(f"Writing {filename}")

        return filename

    def read_h5ad_slots(self, fname: str, root: Optional[Path] = None, skip: Sequence[str] = ("layers",),
                        set_self: bool = False, verbose: bool = False) -> ad.AnnData:
        """
        Read an h5ad without some slots (default: layers). Useful for cell2location maps whose
        per-cell-type expected-expression layers are dense (e.g. 34 layers x spots x genes, several GB).
        """
        import h5py
        from anndata.io import read_elem
        filename = Path(root or self.root_data) / fname
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
                         figsize=(3, 4), inplace: bool = True):
        """
        cell2location permissive gene selection (keeps rare-cell-type markers). Returns (AnnData, genes).
        """
        from cell2location.utils.filtering import filter_genes

        a = self._check_adata(adata)
        selected = filter_genes(a, cell_count_cutoff=self.get_param("c2l_cell_count_cutoff", cell_count_cutoff),
                                cell_percentage_cutoff2=self.get_param("c2l_cell_percentage_cutoff2", cell_percentage_cutoff2),
                                nonz_mean_cutoff=self.get_param("c2l_nonz_mean_cutoff", nonz_mean_cutoff), figsize=figsize)
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
                title: str = "", thresholds: Optional[Dict[str, float]] = None, max_points: Optional[int] = None):
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
        self._savefig(fig, fname)
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

    def spot_xy(self, library_id: Optional[str] = None, adata=None, img_key: Optional[str] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """(mask of the spots of one library, their centres in image pixels)."""
        a = self._check_adata(adata)
        lib = library_id or self.library_id(a)
        m = (a.obs[self.sample_col].astype(str) == lib).values if self.sample_col in a.obs else np.ones(a.n_obs, bool)
        _, sf = self.hires_image(lib, a, img_key)
        return m, np.asarray(a.obsm["spatial"])[m] * sf

    def select_slide(self, adata=None, library_id: Optional[str] = None, batch_key: Optional[str] = None) -> ad.AnnData:
        """One Visium section (cell2location.utils.select_slide); default: the first library."""
        from cell2location.utils import select_slide
        a = self._check_adata(adata)
        lib = library_id or self.library_id(a)
        return select_slide(a, lib, batch_key=batch_key or self.sample_col)

    def spatial_scatter_genes(self, genes: Sequence[str], adata=None, library_id: Optional[str] = None,
                              ncols: int = 4, nframe: Optional[int] = None, q: Optional[float] = None,
                              img_alpha: float = 0.6, size: Optional[float] = None, cmap=None,
                              fname: Optional[str] = None) -> Tuple[List[str], List[Any]]:
        """squidpy spatial_scatter of genes (symbols or obs columns), each clipped at its q-th percentile
        (clip_for_plot), nframe panels per figure. fname may contain '{k}' (figure number, 1-based).
        Returns (features plotted, figures)."""
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
            fig = plt.gcf()
            figs.append(fig)

            self._savefig(fig, fname.format(k=k + 1) if "{k}" in fname else fname)

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
        fn = run_dir / fname
        accelerator = self.get_param("accelerator", accelerator)
        if fn.exists() and not force:
            adata_ref = ad.read_h5ad(fn)
        else:
            RegressionModel.setup_anndata(adata=adata_ref, batch_key=batch_key, labels_key=labels_key,
                                          categorical_covariate_keys=categorical_covariate_keys)
            mod = RegressionModel(adata_ref)
            mod.train(max_epochs=self.get_param("ref_max_epochs", max_epochs), accelerator=accelerator)
            adata_ref = mod.export_posterior(adata_ref, use_quantiles=True, add_to_varm=list(self.params["c2l_quantiles"]),
                                             sample_kwargs={"batch_size": self.params["ref_batch_size"],
                                                            "accelerator": accelerator})
            mod.save(str(run_dir), overwrite=True)
            adata_ref.write_h5ad(fn)
        df_inf_aver, _ = self.get_reference_signatures(adata_ref)
        return adata_ref, df_inf_aver

    @staticmethod
    def get_reference_signatures(adata_ref, prefer: Sequence[str] = ("means", "q50", "q05")) -> Tuple[pd.DataFrame, str]:
        """Per-cell-type signatures (genes x cell types) from RegressionModel.export_posterior (varm or var).
        Also reads the cell2location v0.05 layout (var['means_per_cluster_mu_fg_<type>'])."""
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
        fn = run_dir / fname
        if fn.exists() and not force:
            return ad.read_h5ad(fn)
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
        a.write_h5ad(fn)
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
        fn = run_dir / fname
        accelerator = self.get_param("accelerator", accelerator)
        quantiles = list(self.get_param("c2l_quantiles", quantiles))
        RegressionModel.setup_anndata(adata=adata_ref, batch_key=batch_key, labels_key=labels_key,
                                      categorical_covariate_keys=categorical_covariate_keys)
        trained = False
        if force or not fn.exists():
            mod = RegressionModel(adata_ref)
            mod.train(max_epochs=self.get_param("ref_max_epochs", max_epochs), accelerator=accelerator)
            adata_ref = mod.export_posterior(adata_ref, sample_kwargs={"num_samples": self.params["c2l_num_samples"],
                                                                       "batch_size": self.params["ref_batch_size"],
                                                                       "accelerator": accelerator})
            mod.save(str(run_dir), overwrite=True, save_anndata=True)
            adata_ref.write_h5ad(fn)
            trained = True
            if verbose:
                print(f"RegressionModel trained and saved to {run_dir}")
        else:
            mod = RegressionModel.load(str(run_dir), adata_ref, accelerator=accelerator)
            adata_ref = mod.adata
            if verbose:
                print(f"RegressionModel loaded from {run_dir}")
        if not all(f"{q}_per_cluster_mu_fg" in adata_ref.varm for q in quantiles):
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
        fn = run_dir / fname
        accelerator = self.get_param("accelerator", accelerator)
        batch_key = batch_key or self.sample_col
        quantiles = list(self.params["c2l_quantiles"])
        trained = False
        if fn.exists() and not force:
            adata_vis = ad.read_h5ad(fn) if read_layers else self.read_h5ad_slots(fname, root=run_dir)
            mod = cell2location.models.Cell2location.load(str(run_dir), adata_vis, accelerator=accelerator)
            adata_vis = self.ensure_posterior_samples(mod, adata_vis, quantiles=quantiles, accelerator=accelerator)
            if verbose:
                print(f"Cell2location map read from {fn}; model loaded; posterior samples: {sorted(mod.samples)}")
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
            adata_vis.write_h5ad(fn)
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
                         fname: Optional[str] = None):
        """ELBO loss history (mod.plot_history) on a new figure, first iter_start epochs omitted."""
        fig, ax = plt.subplots(figsize=figsize)
        mod.plot_history(iter_start, ax=ax)
        if label:
            ax.legend(labels=[label])
        self._savefig(fig, fname)
        return fig

    def plot_c2l_qc(self, mod, summary_name: str = "q50", figsize=(3, 3), use_n_obs: int = 1000,
                    fname: Optional[str] = None) -> list:
        """mod.plot_QC(summary_name) with every figure kept (cell2location calls plt.show() internally).
        fname='x.png' saves x_1.png, x_2.png, ... Returns the figures."""
        figs = []
        show = plt.show

        def _show(*a, **k):
            figs.append(plt.gcf())
            show(*a, **k)

        plt.show = _show
        try:
            with plt.rc_context({"figure.figsize": figsize}):
                mod.plot_QC(summary_name=summary_name, use_n_obs=use_n_obs)
        finally:
            plt.show = show
        if not figs:
            figs = [plt.gcf()]
        if fname:
            for i, f in enumerate(figs):
                self._savefig(f, fname.replace(".png", f"_{i + 1}.png"))
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
                                      accelerator: Optional[str] = None, figsize=(4, 4), fname: Optional[str] = None):
        """Posterior median (mod.posterior_quantile q=0.5) vs posterior mean of one site (from mod.samples
        'post_sample_means' when present, otherwise the mean of the draws in ps)."""
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
        self._savefig(fig, fname)
        return fig

    def run_colocation_cached(self, adata_vis: ad.AnnData, run_dir: Path, n_facts: Optional[Sequence[int]] = None,
                              n_restarts: Optional[int] = None, alpha: Optional[float] = None,
                              force: bool = False, verbose: bool = True) -> Tuple[ad.AnnData, Optional[dict], bool]:
        """NMF of the cell abundances (cell2location.run_colocation, CoLocatedGroupsSklearnNMF) for each number of
        factors; skipped when adata.uns already holds every 'mod_coloc_n_fact<n>'.
        Returns (adata_vis, res_dict or None when cached, edited)."""
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
        """NMF cell-type loadings (cell types x factors) from adata.uns['mod_coloc_n_fact<n>'];
        normalize=True: each cell type's weights divided by their sum (relative weight across factors,
        as in plot_cell_type_loadings)."""
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
        """Legend 'cluster: name' along the bottom of a layout='constrained' figure."""
        cats = adata.obs[key].cat.categories
        cols = adata.uns[f"{key}_colors"]
        names = names or {}
        handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=c, markeredgecolor="none",
                          label=f"{k}: {names[k]}" if k in names else str(k)) for k, c in zip(cats, cols)]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(ncol, len(handles)), frameon=False,
                   fontsize=fontsize, handletextpad=0.3, columnspacing=1.5)

    def umap_spatial_3plot(self, adata=None, names: Optional[Dict[str, str]] = None, 
                           figsize: Tuple[float, float] = (15, 6.5), key: Optional[str] = None,
                           fname: Optional[str] = "umap_region_sample.png", library_id: Optional[str] = None):
        """UMAP by region, UMAP by sample, spatial map by region (one library) with a shared legend."""
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
        self._savefig(fig, fname)
        return fig

    def plot_spatial_visium(self, names: Dict[str, str], adata=None, key: Optional[str] = None,
                            library_id: Optional[str] = None, figsize=(8, 4), fname: Optional[str] = None):
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
        self._savefig(fig, fname)
        return fig

    def plotly_spatial_visium(self, names: Dict[str, str], adata=None, key: Optional[str] = None,
                              library_id: Optional[str] = None, set_colors: bool = False,
                              width: int = 1400, height: int = 900, image_format: Optional[str] = None,
                              marker_size: float = 6, hover_extra: Optional[Dict[str, str]] = None):
        """Interactive (plotly) spatial map of a categorical key on the hires image.

        image_format: 'raw' embeds the pixel array (about 30 MB per HTML file), 'jpeg' an encoded image (about 6 MB);
        default params['plotly_image_format']. hover_extra: category -> extra hover text (e.g. signature genes)."""
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
                                                   line=dict(width=0.5, color="black")),
                                       hovertemplate=f"<b>{label}</b>{extra}<br>spot: %{{text}}<extra></extra>"))
        fig.update_layout(title=s, width=width, height=height, plot_bgcolor="black", margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(font=dict(size=12), itemsizing="constant", x=1.01, y=0.5, yanchor="middle"))
        fig.update_xaxes(visible=False, range=[0, img.shape[1]])
        fig.update_yaxes(visible=False, range=[img.shape[0], 0], scaleanchor="x")
        return fig

    def plot_genes_per_cell_type(self, slide, genes, ctypes, q: Optional[float] = None):
        """Total expression (column 0) and cell-type-specific expected expression (layers[<cell type>]) per gene.
        Requires slide.layers[<cell type>] from mod.module.model.compute_expected_per_cell_type."""
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
        return fig, axs

    def plot_slide_loc_cell_types(self, slide, clust_labels, c2l_colors, figsize=(5, 5), fname: Optional[str] = None):
        """cell2location.plt.plot_spatial for up to 6 cell types in one panel (legend outside, right)."""
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
        self._savefig(fig, fname)
        return fig

    def plot_spatial_features(self, slide, features: Sequence[str], ncols: int = 4, figsize=(4.5, 5),
                              q: Optional[float] = None, cmap=None, fname: Optional[str] = None):
        """sc.pl.spatial of several obs features (e.g. cell-type abundances) on a black background, colour scale
        clipped at the q-th percentile of each feature (cell2location tutorial style)."""
        q = self.get_param("clip_quantile", q)
        with mpl.rc_context({"axes.facecolor": "black", "figure.figsize": list(figsize)}):
            sc.pl.spatial(slide, cmap=cmap or cmap_byr, color=list(features), ncols=ncols,
                          size=self.params["spot_size"], img_key=self.params["img_key"], vmin=0, vmax=f"p{q:g}",
                          show=False)
            fig = plt.gcf()
        self._savefig(fig, fname)
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
                     size_label: str = "fraction of cells\nin group (%)", color_label: str = "relative expression\nin group"):
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
        self._savefig(fig, fname)
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
        try:
            sc.pp.highly_variable_genes(a, n_top_genes=min(P["sub_n_hvg"], a.n_vars - 1), flavor="seurat_v3", layer=layer,
                                        batch_key=batch_key if n_batches > 1 else None)
        except Exception:
            sc.pp.highly_variable_genes(a, n_top_genes=min(P["sub_n_hvg"], a.n_vars - 1), flavor="seurat_v3", layer=layer)
        b = a[:, a.var["highly_variable"].values].copy()
        if P["sub_batch"] == "combat" and n_batches > 1:
            sc.pp.combat(b, key=batch_key)
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
    # 9. LIANA
    # =====================================================================================================
    def liana_rank_aggregate(self, groupby: str, adata=None, resource_name: Optional[str] = None,
                             expr_prop: Optional[float] = None, n_perms: Optional[int] = None,
                             use_raw: bool = False, key_added: str = "liana_res", **kw) -> pd.DataFrame:
        """LIANA consensus rank_aggregate on log-normalised X ('mouseconsensus' resource for mouse)."""
        import liana as li
        a = self._check_adata(adata)
        li.mt.rank_aggregate(a, groupby=groupby, resource_name=self.get_param("liana_resource", resource_name),
                             expr_prop=self.get_param("liana_expr_prop", expr_prop), use_raw=use_raw,
                             n_perms=self.get_param("liana_n_perms", n_perms), key_added=key_added, verbose=False, **kw)
        df_liana = a.uns[key_added]
        return df_liana

    def liana_bivariate(self, adata=None, resource_name: Optional[str] = None, local_name: Optional[str] = None,
                        global_name: Optional[str] = None, bandwidth: Optional[int] = None,
                        n_perms: Optional[int] = None, nz_prop: Optional[float] = None,
                        **kw) -> Tuple[pd.DataFrame, ad.AnnData]:
        """LIANA spatial bivariate LR scores (global + local metric). Returns (df_global, lr AnnData)."""
        import liana as li
        a = self._check_adata(adata)
        global_name = self.get_param("liana_global", global_name)
        if "spatial_connectivities" not in a.obsp:
            li.ut.spatial_neighbors(a, bandwidth=self.get_param("liana_bandwidth", bandwidth), cutoff=0.1,
                                    kernel="gaussian", set_diag=True)
        lrdata = li.mt.bivariate(a, resource_name=self.get_param("liana_resource", resource_name),
                                 local_name=self.get_param("liana_local", local_name), global_name=global_name,
                                 n_perms=self.get_param("liana_n_perms", n_perms), mask_negatives=False, add_categories=True,
                                 nz_prop=self.get_param("liana_nz_prop", nz_prop), use_raw=False, verbose=False, **kw)
        df_global = lrdata.var.sort_values(global_name, ascending=False)
        return df_global, lrdata

    # =====================================================================================================
    # 10. Static helpers
    # =====================================================================================================
    @staticmethod
    def wtab(group, rgg) -> pd.DataFrame:
        """One group of a scanpy rank_genes_groups result as a DataFrame indexed by gene."""
        df_wtab = pd.DataFrame({"gene": rgg["names"][group], "lfc": rgg["logfoldchanges"][group],
                                "padj": rgg["pvals_adj"][group], "score": rgg["scores"][group]}).set_index("gene")
        return df_wtab

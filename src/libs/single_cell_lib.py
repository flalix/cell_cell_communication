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
1. I/O and components          read_h5ad, read_visium_samples, write_h5ad, summary, calc_df_components, x_info, is_raw_counts
2. Gene lookup and clipping    symbols, values, resolve_gene_ids, percentile_cap, clip_for_plot
3. Filtering                   flag_mt_genes, remove_mt_genes, filter_cells, filter_genes, filter_genes_c2l
4. Quality control             calc_qc_metrics, calc_df_qc, plot_qc
5. Scanpy preprocessing        normalize_log, highly_variable, embed (pca/neighbors/umap), leiden, rank_genes
6. Squidpy spatial             library_ids, hires_image, spot_xy, spatial_neighbors, spatial_autocorr, nhood_enrichment
7. cell2location               train_reference, get_reference_signatures, train_mapping, ensure_posterior_samples,
                               calc_df_abundance, cluster_regions, region_names
8. Plots                       legend_figure_bottom, umap_spatial_3plot, plot_spatial_visium, plotly_spatial_visium,
                               plot_genes_per_cell_type, plot_slide_loc_cell_types, plot_density_overlay
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

__version__ = "1.1.0"

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
    # --- LIANA
    liana_resource="consensus",    # 'mouseconsensus' for mouse
    liana_expr_prop=0.1,
    liana_n_perms=100,
    liana_bandwidth=200,
    liana_local="cosine",
    liana_global="morans",
    liana_nz_prop=0.2,
    # --- plotting
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

    def _savefig(self, fig, fname: Optional[str], root_figure: Optional[Path] = None, dpi: Optional[int] = None, verbose: bool = False):
        """Save fig to (root_figure or self.root_figure) / fname; fname=None -> not saved."""
        if fname is None:
            return None
        filename = Path(root_figure or self.root_figure) / fname
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
                        resolution: Optional[float] = None, key_added: Optional[str] = None) -> pd.Series:
        """Leiden clusters of spots on cell-abundance space (paper: kNN 38, resolution 1.3)."""
        a = self._check_adata(adata)
        key_added = self.get_param("region_key", key_added)
        df_abund = df_abund if df_abund is not None else self.calc_df_abundance(a)
        a.obsm["_abund"] = df_abund.loc[a.obs_names].values
        seed = self.params["seed"]
        sc.pp.neighbors(a, use_rep="_abund", n_neighbors=self.get_param("region_n_neighbors", n_neighbors),
                        key_added="abund", random_state=seed)
        sc.tl.leiden(a, resolution=self.get_param("region_resolution", resolution), neighbors_key="abund",
                     key_added=key_added, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
        return a.obs[key_added]

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

    def umap_spatial_3plot(self, adata=None, names: Optional[Dict[str, str]] = None, key: Optional[str] = None,
                           fname: Optional[str] = "umap_region_sample.png", library_id: Optional[str] = None):
        """UMAP by region, UMAP by sample, spatial map by region (one library) with a shared legend."""
        a = self._check_adata(adata)
        key = self.get_param("region_key", key)
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), layout="constrained")
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
                              width: int = 1400, height: int = 900):
        """Interactive (plotly) spatial map of a categorical key on the hires image."""
        import plotly.graph_objects as go
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
        fig.add_trace(go.Image(z=img, hoverinfo="skip"))
        for k, c in zip(cats, cols):
            mk = clust == k
            label = f"{k} – {names.get(k, '')}"
            fig.add_trace(go.Scattergl(x=xy[mk, 0], y=xy[mk, 1], mode="markers", name=label, text=spot_names[mk],
                                       marker=dict(size=6, color=c, opacity=0.9, line=dict(width=0.5, color="black")),
                                       hovertemplate=f"<b>{label}</b><br>spot: %{{text}}<extra></extra>"))
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

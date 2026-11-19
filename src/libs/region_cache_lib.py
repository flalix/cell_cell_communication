#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
region_cache_lib.py: file cache for the region clustering (cluster_regions: kNN on the cell-type abundances + Leiden;
umap_regions: UMAP on that kNN graph) of the SingleCell projects.

Usage (MRO: the mixin first, so its methods wrap SingleCell's):

    from libs.region_cache_lib import RegionCacheMixin
    class SingleCellPDAC(RegionCacheMixin, SingleCell): ...

    scp.cluster_regions(adata_vis, df_abund=df_abund, force=force, verbose=verbose)
    scp.umap_regions(adata_vis, force=force, verbose=verbose)

Cache: files['regions_cache'] (e.g. ('result', 'pdac01_regions_cache.tsv.gz')) + a .json fingerprint of the inputs
(abundance source, cell types, abundance checksum, barcodes, n_neighbors, resolution, Leiden flavor, seed; UMAP
min_dist, spread). Same inputs -> read (seconds); other inputs (e.g. rctd -> c2l, new resolution) -> recomputed and
rewritten. force=True: recompute; cache=False: no file. The same code as SingleCellHeart (single_cell_heart_dvlp_lib).
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc


class RegionCacheMixin:
    """cluster_regions / umap_regions with a file cache (see the module docstring). Needs a SingleCell subclass."""

    def _regions_cache_paths(self):
        """(table, json) of the cache: files['regions_cache'] when registered, else next to files['region_labels']
        ('<prefix>_region_labels.tsv' -> '<prefix>_regions_cache.tsv.gz'), else root_result/regions_cache.tsv.gz."""
        if "regions_cache" in self.files:
            path = self.file_path("regions_cache")
        elif "region_labels" in self.files:
            p = self.file_path("region_labels")
            path = p.with_name(p.name.replace("_region_labels.tsv", "") + "_regions_cache.tsv.gz")
        else:
            path = Path(self.root_result) / "regions_cache.tsv.gz"
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

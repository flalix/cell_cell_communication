#!/usr/bin/env python
"""
pdac_reference_subclusters.py - re-derived subclusters of the PDAC reference (GEO has only the 14 major types).

For the compartments of PDAC_SUBCLUSTERS (ductal, fibroblast = CAF, myeloid, Schwann): tissue cells (Tumor +
Adjacent_normal; PBMC excluded) of sc + sn, scVI integration (batch = sample, covariate = technology; Schwann: PCA),
Leiden, marker-set scores and cluster labels with the paper's names (SingleCellPDAC.annotate_clusters).

Outputs (pdac/results/):
    pdac01_reference_subclusters.tsv           cell, compartment, leiden, subcluster, umap1, umap2
    pdac01_reference_subcluster_clusters.tsv   one row per Leiden cluster: label, scores, anchor %, sn / normal share
    pdac01_subclusters/<compartment>.h5ad      log-normalised cells of the compartment (UMAP, scores)

Usage (CPU, ~15-30 min):
    python src/programs/pdac_reference_subclusters.py [--compartments ductal fibroblast myeloid schwann] [--force]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "src"))

import warnings                                           # noqa: E402
warnings.filterwarnings("ignore")

import pandas as pd                                       # noqa: E402

from libs.Basic import pdwritecsv                         # noqa: E402
from libs.single_cell_pdac_lib import PDAC_SUBCLUSTERS, SingleCellPDAC   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--compartments", nargs="+", default=list(PDAC_SUBCLUSTERS))
p.add_argument("--accelerator", default="cpu")
p.add_argument("--force", action="store_true")
args = p.parse_args()

t0 = time.time()
scp = SingleCellPDAC(ROOT0)
out_dir = scp.root_result / "pdac01_subclusters"
out_dir.mkdir(exist_ok=True)
adata_ref = scp.read_h5ad("reference", set_self=False, backed="r")
labels, clusters = [], []
for comp in args.compartments:
    f = out_dir / f"{comp}.h5ad"
    if f.exists() and not args.force:
        import anndata as ad
        import scanpy as sc
        a = ad.read_h5ad(f)                  # integration and kNN graph cached; Leiden at the spec resolution
        sc.tl.leiden(a, resolution=PDAC_SUBCLUSTERS[comp]["resolution"], flavor="igraph", n_iterations=2,
                     directed=False, random_state=scp.params["seed"])
        df_cl = scp.annotate_clusters(a, comp)
        a.write_h5ad(f, compression="gzip")
        print(f"[{time.time() - t0:6.0f} s] {comp}: read {f}")
    else:
        a, df_cl = scp.subcluster_compartment(adata_ref, comp, accelerator=args.accelerator)
        del a.layers["counts"]
        a.write_h5ad(f, compression="gzip")
        print(f"[{time.time() - t0:6.0f} s] {comp}: {a.n_obs:,} cells -> {f}")
    labels.append(pd.DataFrame({"cell": a.obs_names, "compartment": comp, "leiden": a.obs.leiden.astype(str).values,
                                "subcluster": a.obs.subcluster.astype(str).values,
                                "umap1": a.obsm["X_umap"][:, 0].round(4), "umap2": a.obsm["X_umap"][:, 1].round(4)}))
    clusters.append(df_cl)
    print(df_cl[["leiden", "n_cells", "label", "score", "z", "second", "anchor_pct", "pct_sn", "best_raw"]].to_string())
pdwritecsv(pd.concat(labels, ignore_index=True), "pdac01_reference_subclusters.tsv", scp.root_result, verbose=True)
pdwritecsv(pd.concat(clusters, ignore_index=True), "pdac01_reference_subcluster_clusters.tsv", scp.root_result, verbose=True)
print(f"[{time.time() - t0:6.0f} s] done")

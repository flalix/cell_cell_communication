#!/usr/bin/env python
"""
run_c2l_heart_dvlp.py - cell2location for the developing human heart (mp_heart_dvlp01_prepare.ipynb), outside the
notebook because the runs are long (38 Visium sections, 69,115 spots; scRNA reference 75k cells x 31 types).
Results are cached where the notebooks read them (SingleCellHeart: results/c2l_<tag>/reference_signatures,
results/c2l_<tag>/cell2location_map; fit_or_load_reference / fit_or_load_mapping then only load them).

    mode reference : NB regression of the 31 high-level cell types (labels cell_type, batch = 10x library sample_id)
    mode mapping   : the reference signatures (q50) -> the 38 Visium sections (batch = section)
    mode all       : both

Usage (GPU, full epochs; tag 'gpu' = HEART_PARAMS['c2l_tag'], the folder the notebooks read):
    python src/programs/run_c2l_heart_dvlp.py --mode all --tag gpu --ref_epochs 250 --map_epochs 30000 --accelerator gpu
    (default batch: HEART_PARAMS c2l_batch_size = 10000 spots; full data and 20000 run out of memory on a 24 GB GPU)
CPU (reference only, ~160 s per epoch; mapping > 25 min per full-data epoch); notebooks then need params=dict(c2l_tag='cpu'):
    python src/programs/run_c2l_heart_dvlp.py --mode reference --tag cpu --ref_epochs 100 --num_samples 200 --accelerator cpu
"""
import argparse
import sys
import time
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "src"))

import matplotlib                                         # noqa: E402
matplotlib.use("Agg")
import numpy as np                                        # noqa: E402

from libs.single_cell_heart_dvlp_lib import SingleCellHeart   # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["reference", "mapping", "all"], default="all")
p.add_argument("--tag", default="gpu", help="results/c2l_<tag>/ (HEART_PARAMS['c2l_tag'])")
p.add_argument("--ref_epochs", type=int, default=250)
p.add_argument("--map_epochs", type=int, default=30000)
p.add_argument("--accelerator", default="auto")
p.add_argument("--N_cells_per_location", type=float, default=None)
p.add_argument("--detection_alpha", type=float, default=None)
p.add_argument("--batch_size", type=int, default=None, help="spots per step (None = all; e.g. 10000 on a 24 GB GPU)")
p.add_argument("--num_samples", type=int, default=None, help="posterior draws of the reference export (default 1000)")
p.add_argument("--force", action="store_true")
args = p.parse_args()

params = dict(c2l_tag=args.tag, accelerator=args.accelerator)
if args.N_cells_per_location:
    params["c2l_N_cells_per_location"] = args.N_cells_per_location
if args.batch_size:
    params["c2l_batch_size"] = args.batch_size
if args.num_samples:
    params["c2l_num_samples"] = args.num_samples
if args.detection_alpha:
    params["c2l_detection_alpha"] = args.detection_alpha
sch = SingleCellHeart(ROOT0, params=params)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


ref_dir = sch.file_path("c2l_ref_dir")
map_dir = sch.file_path("c2l_map_dir")

if args.mode in ("reference", "all"):
    adata_ref = sch.filter_reference(sch.read_h5ad("reference", set_self=False))
    adata_ref, selected = sch.filter_genes_c2l(adata_ref)
    log(f"reference: {adata_ref.n_obs:,} cells x {adata_ref.n_vars:,} genes (filter_genes_c2l), "
        f"{adata_ref.obs[sch.params['ref_labels_key']].nunique()} cell types")
    mod_reg, adata_ref, trained = sch.fit_or_load_reference(
        adata_ref, ref_dir, labels_key=sch.params["ref_labels_key"], batch_key=sch.params["ref_batch_key"],
        categorical_covariate_keys=list(sch.params["ref_covariate_keys"]) or None, max_epochs=args.ref_epochs,
        force=args.force)
    # fit_or_load_reference writes sc.h5ad right after training; the posterior quantiles (q05/q50/q95, varm) are
    # exported afterwards: save again so that the mapping (read from disk) finds params['c2l_signature_summary']
    key = f"{sch.params['c2l_signature_summary']}_per_cluster_mu_fg"
    if key in adata_ref.varm:
        adata_ref.write_h5ad(sch.file_path("signatures"))
        log(f"signatures ({key}) saved: {sch.file_path('signatures')}")
    log(f"reference model {'trained' if trained else 'loaded'}: {ref_dir}")

if args.mode in ("mapping", "all"):
    import anndata as ad
    adata_ref = ad.read_h5ad(sch.file_path("signatures"))
    df_inf_aver, used = sch.get_reference_signatures(adata_ref, prefer=(sch.params["c2l_signature_summary"],))
    del adata_ref
    adata_vis = sch.read_h5ad("visium", set_self=False)
    adata_vis = sch.remove_mt_genes(adata_vis)
    for k in ("stereoscope_hl", "stereoscope_dl"):          # not needed by the model; kept in the source h5ad
        adata_vis.obsm.pop(k, None)
    log(f"mapping: {adata_vis.n_obs:,} spots x {adata_vis.n_vars:,} genes; signatures {used} {df_inf_aver.shape}; "
        f"N={sch.params['c2l_N_cells_per_location']}, alpha={sch.params['c2l_detection_alpha']}, epochs={args.map_epochs}")
    mod, adata_map, trained = sch.fit_or_load_mapping(adata_vis, df_inf_aver, map_dir, max_epochs=args.map_epochs,
                                                      expected_layers=False, force=args.force)
    elbo = np.asarray(mod.history["elbo_train"]).ravel() if trained else None
    log(f"mapping {'trained' if trained else 'loaded'}: {map_dir}" +
        (f"; ELBO first {elbo[0]:.4g}, last {elbo[-1]:.4g}" if elbo is not None else ""))
log("done")

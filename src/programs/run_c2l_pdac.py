#!/usr/bin/env python
"""
run_c2l_pdac.py - cell2location for the PDAC project (mp_pdac01_prepare.ipynb), outside the notebook because the runs
are long (21 Visium sections, 81,846 spots; reference: tissue sc + sn cells, <= 3,000 per cell type). Results are
cached where the notebooks read them (SingleCellPDAC: results/c2l_<tag>/reference_signatures, .../cell2location_map).

    mode reference : NB regression (labels cell_type = 14 major types + re-derived subclusters; batch = sample_id;
                     covariate = technology sc5p / snFlex)
    mode mapping   : signatures (q50) -> the 21 sections (batch = section), genes of the three Visium panels only
    mode all       : both

Needs pdac/results/pdac01_reference_subclusters.tsv (src/programs/pdac_reference_subclusters.py) for the subclusters;
without it the 14 major types are used.

Usage (GPU; tag 'gpu' = PDAC_PARAMS['c2l_tag'], the folder the notebooks read):
    python src/programs/run_c2l_pdac.py --mode all --tag gpu --ref_epochs 250 --map_epochs 6000 --accelerator gpu
    (batch: PDAC_PARAMS c2l_batch_size = 10000 spots per step; ~1.1 GB GPU memory per 1000 spots)
CPU code test (a few epochs, tag 'cputest'; the notebooks ignore it):
    python src/programs/run_c2l_pdac.py --mode all --tag cputest --ref_epochs 2 --map_epochs 2 --num_samples 20 --accelerator cpu
"""
import argparse
import sys
import time
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "src"))

import warnings                                           # noqa: E402
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib                                         # noqa: E402
matplotlib.use("Agg")
import numpy as np                                        # noqa: E402

from libs.single_cell_pdac_lib import SingleCellPDAC     # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["reference", "mapping", "all"], default="all")
p.add_argument("--tag", default="gpu", help="results/c2l_<tag>/ (PDAC_PARAMS['c2l_tag'])")
p.add_argument("--ref_epochs", type=int, default=250)
p.add_argument("--map_epochs", type=int, default=6000)
p.add_argument("--accelerator", default="auto")
p.add_argument("--N_cells_per_location", type=float, default=None)
p.add_argument("--detection_alpha", type=float, default=None)
p.add_argument("--batch_size", type=int, default=None, help="spots per step (default PDAC_PARAMS c2l_batch_size)")
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
scp = SingleCellPDAC(ROOT0, params=params)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


ref_dir = scp.file_path("c2l_ref_dir")
map_dir = scp.file_path("c2l_map_dir")

if args.mode in ("reference", "all"):
    adata_ref = scp.reference_for_deconvolution(scp.read_h5ad("reference", set_self=False, backed="r"))
    adata_ref, selected = scp.filter_genes_c2l(adata_ref)
    log(f"reference: {adata_ref.n_obs:,} cells x {adata_ref.n_vars:,} genes (filter_genes_c2l), "
        f"{adata_ref.obs[scp.params['ref_labels_key']].nunique()} cell types")
    mod_reg, adata_ref, trained = scp.fit_or_load_reference(
        adata_ref, ref_dir, labels_key=scp.params["ref_labels_key"], batch_key=scp.params["ref_batch_key"],
        categorical_covariate_keys=list(scp.params["ref_covariate_keys"]) or None, max_epochs=args.ref_epochs,
        force=args.force)
    key = f"{scp.params['c2l_signature_summary']}_per_cluster_mu_fg"
    if key in adata_ref.varm:
        adata_ref.write_h5ad(scp.file_path("signatures"))
        log(f"signatures ({key}) saved: {scp.file_path('signatures')}")
    log(f"reference model {'trained' if trained else 'loaded'}: {ref_dir}")

if args.mode in ("mapping", "all"):
    import anndata as ad
    adata_ref = ad.read_h5ad(scp.file_path("signatures"))
    df_inf_aver, used = scp.get_reference_signatures(adata_ref, prefer=(scp.params["c2l_signature_summary"],))
    del adata_ref
    adata_vis = scp.read_h5ad("visium", set_self=False)
    adata_vis = scp.remove_mt_genes(adata_vis)
    adata_vis = adata_vis[:, adata_vis.var["in_all_panels"].values.astype(bool)].copy()   # zeros outside a panel
    log(f"mapping: {adata_vis.n_obs:,} spots x {adata_vis.n_vars:,} genes (three panels); signatures {used} "
        f"{df_inf_aver.shape}; N={scp.params['c2l_N_cells_per_location']}, alpha={scp.params['c2l_detection_alpha']}, "
        f"epochs={args.map_epochs}")
    mod, adata_map, trained = scp.fit_or_load_mapping(adata_vis, df_inf_aver, map_dir, max_epochs=args.map_epochs,
                                                      expected_layers=False, force=args.force)
    elbo = np.asarray(mod.history["elbo_train"]).ravel() if trained else None
    log(f"mapping {'trained' if trained else 'loaded'}: {map_dir}" +
        (f"; ELBO first {elbo[0]:.4g}, last {elbo[-1]:.4g}" if elbo is not None else ""))
log("done")

#!/usr/bin/env python
"""
Cell2location mappings for the mouse brain Fig 2 notebook (notebooks_mouse_brain/mb01_fig2_mouse_brain.ipynb),
run outside the notebook because they are long on CPU. Results are cached where the notebook reads them
(SingleCellNeuro.train_mapping: <run_dir>/sp.h5ad + model).

    mode slideseq : 59 snRNA signatures (paper regression model) -> Slide-seqV2 puck (recovered counts)  [Fig 2H]
    mode allen    : Allen Smart-seq per-cluster signatures (SSp + PAR-POST-PRE types) -> Visium cortex spots [Fig 2J]

Usage
    python src/programs/run_c2l_mouse_brain.py --mode slideseq --max_epochs 30000 --accelerator gpu
    python src/programs/run_c2l_mouse_brain.py --mode allen    --max_epochs 30000 --accelerator gpu
"""
import argparse
import sys
import time
from pathlib import Path

ROOT0 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT0 / "src"))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import anndata as ad                     # noqa: E402

from libs.single_cell_neuro_lib import SingleCellNeuro   # noqa: E402

'''
cd ~/uv/cell_cell_communication
.venv/bin/python src/programs/run_c2l_mouse_brain.py --mode slideseq --max_epochs 30000 --accelerator gpu   # Fig 2H
.venv/bin/python src/programs/run_c2l_mouse_brain.py --mode allen    --max_epochs 30000 --accelerator gpu   # Fig 2J
'''

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["slideseq", "allen"], required=True)
p.add_argument("--project", default="mouse_brain")
p.add_argument("--max_epochs", type=int, default=30000)
p.add_argument("--accelerator", default="auto")
p.add_argument("--run_tag", default="")
p.add_argument("--force", action="store_true")
p.add_argument("--region_override", default="11:CTX_L1,21:edge_LowQ",
               help="region_cluster:label pairs (see calc_df_edge_fraction in the notebook)")
# input file names (relative to <project>/data)
p.add_argument("--fname_signatures", default="snrna/regression_signatures_sc.h5ad")
p.add_argument("--fname_slideseq", default="slideseq/slideseqv2.h5ad")
p.add_argument("--fname_map", default="c2l_paper_map/sp_with_clusters.h5ad")
p.add_argument("--fname_allen", default="allen_smartseq/allen_smartseq_SSp_PARPOSTPRE.h5ad")
args = p.parse_args()

nr = SingleCellNeuro(ROOT0, args.project)
tag = args.run_tag or f"{args.max_epochs}ep"
t0 = time.time()

if args.mode == "slideseq":
    adata_ss, df_sig, diag = nr.prepare_slideseq_mapping(args.fname_slideseq, args.fname_signatures)
    print("slideseq inputs:", diag, flush=True)
    run_dir = nr.root_result / f"c2l_slideseq_{tag}"
    out = nr.train_mapping(adata_ss, df_sig, run_dir, N_cells_per_location=nr.params["slideseq_N"],
                           detection_alpha=nr.params["slideseq_alpha"], max_epochs=args.max_epochs,
                           accelerator=args.accelerator, force=args.force,
                           A_factors_per_location=nr.params["slideseq_A"], B_groups_per_location=nr.params["slideseq_B"])
else:
    import matplotlib; matplotlib.use("Agg")
    adata_allen = ad.read_h5ad(nr.root_data / args.fname_allen)
    m = nr.load_paper_map(args.fname_map)
    df_regions = nr.calc_df_regions(m)
    override = dict(kv.split(":", 1) for kv in args.region_override.split(",") if kv)
    nr.add_region_labels(df_regions, m, override=override)
    adata_ctx, df_sig, info = nr.prepare_allen_mapping(adata_allen, m)
    print("allen inputs:", {k: v for k, v in info.items() if k not in ("types", "df_type_counts")}, flush=True)
    run_dir = nr.root_result / f"c2l_allen_ssp_{tag}"
    out = nr.train_mapping(adata_ctx, df_sig, run_dir, N_cells_per_location=nr.params["c2l_allen_N"],
                           detection_alpha=nr.params["c2l_allen_alpha"], max_epochs=args.max_epochs,
                           accelerator=args.accelerator, force=args.force, batch_key=nr.sample_col,
                           A_factors_per_location=nr.params["c2l_allen_A"])

print(f"done: {out.shape} -> {run_dir} in {(time.time() - t0) / 60:.1f} min", flush=True)

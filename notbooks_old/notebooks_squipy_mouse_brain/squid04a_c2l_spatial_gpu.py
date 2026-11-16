"""cell2location spatial model - Visium H&E mouse brain x sc_mouse_cortex (Smart-seq) signatures.

Run on the GPU (RTX 3090): ~15-30 min.   python squid04a_c2l_spatial_gpu.py
Inputs : results/c2l_reference_signatures.tsv  (per-subclass averages, 16,304 genes x 22 types)
         data/visium_hne.h5ad                  (.raw = raw Visium counts)
Outputs: data/visium_hne_c2l.h5ad, data/c2l_spatial_model/, results/c2l_abundance_q05.tsv,
         figures/c2l_spatial_training.png

Priors follow Kleshchevnikov et al. 2020 (bioRxiv 10.1101/2020.11.15.378125), Methods:
  N_cells_per_location = 8   (mouse-brain Visium, from histology)
  detection_mean       = 1/20 (Smart-seq reference vs Visium sensitivity)
  detection_alpha      = 20, 30,000 iterations, all shared genes.
Requires: cell2location 0.1.5, pyro-ppl==1.9.1 (1.9.2 breaks the import), pandas<3.
Set C2L_TEST=1 for a 20-iteration smoke test written to ./c2l_test_out.
"""
import os, time
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cell2location.models import Cell2location

TEST = os.environ.get("C2L_TEST") == "1"
root0 = Path("/home/flavio/uv/squidpy")
root_data, root_result, root_figure = root0 / "data", root0 / "results", root0 / "figures"
if TEST:
    root_data = root_result = root_figure = Path("c2l_test_out")
root_data.mkdir(exist_ok=True); root_result.mkdir(exist_ok=True); root_figure.mkdir(exist_ok=True)

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | accelerator: {accelerator}"
      + (f" ({torch.cuda.get_device_name(0)})" if accelerator == "gpu" else ""), flush=True)

# ---- inputs
inf_aver = pd.read_csv(root0 / "results" / "c2l_reference_signatures.tsv", sep="\t", index_col=0)
vis = sc.read_h5ad(root0 / "data" / "visium_hne.h5ad")
ad = sc.AnnData(vis.raw.X.copy(), obs=vis.obs.copy(), var=pd.DataFrame(index=vis.raw.var_names))
ad.obsm["spatial"] = vis.obsm["spatial"]
ad.uns["spatial"] = vis.uns["spatial"]
if "cluster_colors" in vis.uns:
    ad.uns["cluster_colors"] = vis.uns["cluster_colors"]
genes = inf_aver.index.intersection(ad.var_names)
ad, inf_aver = ad[:, genes].copy(), inf_aver.loc[genes]
assert float(ad.X.max()) > 100 and np.allclose(ad.X[:50].toarray() % 1, 0), "expected raw integer counts"
print(f"spots x genes: {ad.shape} | cell types: {inf_aver.shape[1]}", flush=True)

# ---- model
Cell2location.setup_anndata(adata=ad)
mod = Cell2location(ad, cell_state_df=inf_aver,
                    N_cells_per_location=8, detection_alpha=20, detection_mean=1 / 20)
t = time.time()
mod.train(max_epochs=20 if TEST else 30000, batch_size=None, train_size=1, accelerator=accelerator)
print(f"training: {(time.time() - t) / 60:.1f} min", flush=True)

# ---- training curve
h = mod.history["elbo_train"]
skip = 0 if TEST else 1000
fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(h.index[skip:], h.values[skip:, 0], lw=0.6)
ax.set_xlabel("iteration"); ax.set_ylabel("-ELBO")
ax.set_title(f"cell2location training (from iteration {skip:,})")
fig.savefig(root_figure / "c2l_spatial_training.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---- posterior + save
ad = mod.export_posterior(ad, sample_kwargs={"num_samples": 20 if TEST else 1000,
                                             "batch_size": ad.n_obs, "accelerator": accelerator})
mod.save(str(root_data / "c2l_spatial_model"), overwrite=True)

# anndata >= 0.13 refuses '/' in keys ("L2/3 IT") -> rename to "L2-3 IT" everywhere
safe = lambda s: str(s).replace("/", "-")
for k in list(ad.obsm.keys()):
    if isinstance(ad.obsm[k], pd.DataFrame):
        ad.obsm[k].columns = [safe(c) for c in ad.obsm[k].columns]
ad.uns["mod"]["factor_names"] = np.array([safe(c) for c in ad.uns["mod"]["factor_names"]])

q05 = ad.obsm["q05_cell_abundance_w_sf"].copy()
q05.columns = ad.uns["mod"]["factor_names"]
ad.obs[q05.columns] = q05
q05.to_csv(root_result / "c2l_abundance_q05.tsv", sep="\t")
ad.write_h5ad(root_data / "visium_hne_c2l.h5ad")
print("saved:", root_data / "visium_hne_c2l.h5ad", flush=True)

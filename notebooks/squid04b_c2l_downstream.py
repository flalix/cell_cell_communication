"""Downstream of cell2location: per-spot cell-type fractions, maps, cluster summary, LIANA link.

Inputs : data/visium_hne_c2l.h5ad (from squid04a), data/visium_hne.h5ad, data/visium_hne_liana_bivariate.h5ad
Outputs: results/c2l_abundance_q05.tsv, c2l_fractions.tsv, c2l_cluster_by_celltype.tsv,
         lr_vs_celltype_spearman.tsv, liana_dominant_celltype_results.tsv
         figures/c2l_celltype_maps.png, c2l_cluster_by_celltype_heatmap.png, c2l_dominant_celltype_map.png,
                 lr_vs_celltype_spearman_heatmap.png
Set C2L_IN=<path to visium_hne_c2l.h5ad> and C2L_OUT=<dir> to test on another file.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, scanpy as sc, squidpy as sq, liana as li
from scipy.stats import rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.append("/home/flavio/uv/gwas/src")
from libs.Basic import create_dir, pdwritecsv
from libs.matplotlib_figure_style import apply_figure_style

root0 = Path("/home/flavio/uv/squidpy"); root_data = root0 / "data"
c2l_in = Path(os.environ.get("C2L_IN", root_data / "visium_hne_c2l.h5ad"))
out = os.environ.get("C2L_OUT")
root_result = create_dir(Path(out) if out else root0, "" if out else "results")
root_figure = create_dir(Path(out) if out else root0, "" if out else "figures")
apply_figure_style()

ORDER = ["L2-3 IT", "L4", "L5 IT", "L5 PT", "L6 IT", "L6 CT", "L6b", "NP",
         "Pvalb", "Sst", "Vip", "Lamp5", "Sncg", "Serpinf1", "Meis2",
         "Astro", "Oligo", "Macrophage", "VLMC", "Endo", "SMC", "Peri"]
CLASS = {**{c: "Glutamatergic" for c in ORDER[:8]}, **{c: "GABAergic" for c in ORDER[8:15]},
         **{c: "Non-neuronal" for c in ORDER[15:19]}, **{c: "Vascular" for c in ORDER[19:]}}
label = lambda c: c.replace("L2-3", "L2/3")          # display name

# ---------------- 1. abundances -> fractions -> dominant type
ad = sc.read_h5ad(c2l_in)
fn = list(ad.uns["mod"]["factor_names"])
q05 = ad.obsm["q05_cell_abundance_w_sf"].copy(); q05.columns = fn; q05 = q05[ORDER]
frac = q05.div(q05.sum(1), axis=0)
dom = frac.idxmax(1); dom_frac = frac.max(1)
ad.obs["dominant_type"] = pd.Categorical(dom, categories=ORDER)
ad.obs["dominant_frac"] = dom_frac
ad.obs["n_cells_est"] = q05.sum(1)
MIX = 0.5
ad.obs["dominant_conf"] = np.where(dom_frac >= MIX, dom.astype(str), "mixed")
for c in ORDER:
    ad.obs[f"c2l_{c}"] = q05[c].values          # prefixed: Pvalb/Sst are also gene names

pdwritecsv(q05, "c2l_abundance_q05.tsv", root_result, index=True)
pdwritecsv(frac.assign(dominant_type=dom, dominant_frac=dom_frac, cluster=ad.obs["cluster"]),
           "c2l_fractions.tsv", root_result, index=True)
summ = {"spots": ad.n_obs, "median_cells_per_spot": float(ad.obs.n_cells_est.median()),
        f"spots_dominant_frac>={MIX}": int((dom_frac >= MIX).sum()),
        "dominant_frac_median": float(dom_frac.median())}
print(summ, flush=True)

# ---------------- 1b. model-estimated capture efficiency vs observed UMI depth
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from matplotlib.colors import TwoSlopeNorm
ad.obs["detection_y_s"] = np.asarray(ad.uns["mod"]["post_sample_means"]["detection_y_s"]).ravel()
umi = np.asarray(ad.X.sum(1)).ravel()
r = np.log2(umi / pd.Series(umi, index=ad.obs_names).groupby(ad.obs["cluster"], observed=True).transform("median").values)
nb = cKDTree(ad.obsm["spatial"]).query_ball_point(ad.obsm["spatial"], r=300)
ad.obs["log2_umi_vs_cluster_smooth"] = [r[i].mean() for i in nb]
ad.obs["log2_detection_rel"] = np.log2(ad.obs["detection_y_s"] / ad.obs["detection_y_s"].median())
rho_det = spearmanr(ad.obs["log2_detection_rel"], ad.obs["log2_umi_vs_cluster_smooth"])[0]
rho_cells = spearmanr(ad.obs["n_cells_est"], umi)[0]
print({"rho(detection_y_s, smoothed UMI ratio)": round(rho_det, 3), "rho(total est. cells, UMI)": round(rho_cells, 3),
       "detection_y_s 2.5-97.5% (x median)": np.round(np.percentile(ad.obs["detection_y_s"] / ad.obs["detection_y_s"].median(), [2.5, 97.5]), 2).tolist()}, flush=True)
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.2))
sq.pl.spatial_scatter(ad, color="log2_umi_vs_cluster_smooth", ax=axs[0], cmap="RdBu_r", size=1.3,
                      norm=TwoSlopeNorm(0, -0.6, 0.6), title="A  observed: UMIs vs cluster median (log2, smoothed)")
sq.pl.spatial_scatter(ad, color="log2_detection_rel", ax=axs[1], cmap="RdBu_r", size=1.3,
                      norm=TwoSlopeNorm(0, -0.6, 0.6), title="B  model: capture efficiency y_s (log2 vs median)")
sq.pl.spatial_scatter(ad, color="n_cells_est", ax=axs[2], cmap="magma", size=1.3, title="C  model: total estimated cells per spot")
for ax in axs: ax.set_xlabel(""); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(root_figure / "c2l_detection_vs_umi.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ---------------- 2. spatial maps of selected cell types (q05 abundance)
show = ["L2-3 IT", "L4", "L5 PT", "L6 CT", "Pvalb", "Sst", "Astro", "Oligo", "Macrophage", "VLMC", "Endo", "SMC"]
fig, axs = plt.subplots(3, 4, figsize=(13, 10))
for ax, c in zip(axs.ravel(), show):
    sq.pl.spatial_scatter(ad, color=f"c2l_{c}", ax=ax, cmap="magma", size=1.3, colorbar=True, title=label(c),
                          vmax=float(np.quantile(ad.obs[f"c2l_{c}"], 0.992)))
    ax.set_xlabel(""); ax.set_ylabel("")
fig.suptitle("cell2location: estimated cells per spot (q05 posterior)", y=0.995)
fig.tight_layout(); fig.savefig(root_figure / "c2l_celltype_maps.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ---------------- 3. anatomical cluster x cell type (mean fraction)
cl = frac.groupby(ad.obs["cluster"], observed=True).mean()
cl = cl.loc[sorted(cl.index, key=lambda s: (s.split("_")[0], s))]
pdwritecsv(cl, "c2l_cluster_by_celltype.tsv", root_result, index=True)
fig, ax = plt.subplots(figsize=(9.5, 5.2))
im = ax.imshow(cl.values, aspect="auto", cmap="viridis", vmin=0, vmax=float(np.quantile(cl.values, 0.98)))
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([label(c) for c in ORDER], rotation=90)
ax.set_yticks(range(len(cl))); ax.set_yticklabels(cl.index)
for x in (7.5, 14.5, 18.5): ax.axvline(x, color="w", lw=1)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.set_label("mean fraction of spot")
ax.set_title("Cell-type composition of anatomical clusters (cell2location)")
fig.savefig(root_figure / "c2l_cluster_by_celltype_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- 4. dominant-type map
ad.obs["dominant_conf"] = pd.Categorical(ad.obs["dominant_conf"],
                                         categories=[c for c in ORDER if c in set(ad.obs.dominant_conf)] + ["mixed"])
cmap = plt.get_cmap("tab20").colors + plt.get_cmap("tab20b").colors
ad.uns["dominant_conf_colors"] = [matplotlib.colors.to_hex(cmap[ORDER.index(c)]) if c != "mixed" else "#d9d9d9"
                                  for c in ad.obs["dominant_conf"].cat.categories]
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
sq.pl.spatial_scatter(ad, color="dominant_conf", ax=axs[0], size=1.3, title=f"dominant type (grey: fraction < {MIX})")
sq.pl.spatial_scatter(ad, color="dominant_frac", ax=axs[1], size=1.3, cmap="viridis", title="fraction of dominant type")
for ax in axs: ax.set_xlabel(""); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(root_figure / "c2l_dominant_celltype_map.png", dpi=250, bbox_inches="tight"); plt.close(fig)

# ---------------- 5a. LIANA local LR scores vs cell-type abundance (Spearman across spots)
lr = sc.read_h5ad(root_data / "visium_hne_liana_bivariate.h5ad")
common = lr.obs_names.intersection(ad.obs_names)
L = lr[common].X; L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
A = frac.loc[common].values      # fractions: removes the shared 'more cells per spot' signal
rk = lambda M: np.apply_along_axis(rankdata, 0, M)
Lr, Ar = rk(L), rk(A)
Lr = (Lr - Lr.mean(0)) / (Lr.std(0) + 1e-12); Ar = (Ar - Ar.mean(0)) / (Ar.std(0) + 1e-12)
rho = pd.DataFrame(Lr.T @ Ar / len(common), index=lr.var_names, columns=ORDER)
rho = rho.join(lr.var[["morans"]]).sort_values("morans", ascending=False)
pdwritecsv(rho, "lr_vs_celltype_spearman.tsv", root_result, index=True)
top = rho[rho.morans > 0].drop(columns="morans")
top = top.loc[top.abs().max(1).sort_values(ascending=False).index[:40]]
fig, ax = plt.subplots(figsize=(8.5, 10))
im = ax.imshow(top.values, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([label(c) for c in ORDER], rotation=90)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=6)
for x in (7.5, 14.5, 18.5): ax.axvline(x, color="k", lw=0.6)
cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01); cb.set_label("Spearman rho (spots)")
ax.set_title("Local LR co-localisation (LIANA cosine) vs cell-type fraction\n40 LR pairs with the strongest |rho|")
fig.savefig(root_figure / "lr_vs_celltype_spearman_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------------- 5b. LIANA rank_aggregate on confident dominant types
vis = sc.read_h5ad(root_data / "visium_hne.h5ad")              # .X = log-normalised
vis = vis[ad.obs_names]
vis.obs["dominant_conf"] = ad.obs["dominant_conf"].astype(str).values
keep_types = vis.obs["dominant_conf"].value_counts()
keep_types = keep_types[(keep_types.index != "mixed") & (keep_types >= 20)].index
sub = vis[vis.obs["dominant_conf"].isin(keep_types)].copy()
print("LIANA groups (spots):", sub.obs["dominant_conf"].value_counts().to_dict(), flush=True)
if len(keep_types) >= 2:
    li.mt.rank_aggregate(sub, groupby="dominant_conf", resource_name="mouseconsensus",
                         expr_prop=0.1, use_raw=False, n_perms=1000, verbose=False)
    pdwritecsv(sub.uns["liana_res"], "liana_dominant_celltype_results.tsv", root_result)

ad.write_h5ad(Path(out) / "visium_hne_c2l_annot.h5ad" if out else root_data / "visium_hne_c2l_annot.h5ad")
print("done", flush=True)

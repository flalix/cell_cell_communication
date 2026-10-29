"""
cell2location -> LIANA+ cell-cell communication (CCC) helpers.

Input: the ccc01_tutorial.ipynb

(`results/lymph_nodes_analysis/cell2location_map/sp.h5ad`):
    X                         raw Visium counts (Ensembl var_names, symbols in var['symbol'])
    obsm['q05_cell_abundance_w_sf'] ...  posterior cell abundance per spot (34 cell types)
    layers[<cell type>]       expected counts of every gene attributed to that cell type
                              (compute_expected_per_cell_type, q05 posterior); the 34 layers
                              sum to X, i.e. they split each spot's counts across cell types.

Functions
---------
load_c2l_spots()            spots AnnData without the 34 big layers: symbols as var_names,
                            counts layer, log1p(CP10k) X, abundances / fractions in obsm.
region_names_by_enrichment()   name c2l regions by their most enriched cell types.
row_normalise()             row-stochastic spatial weights (niche mean).
spatial_colocalisation()    abundance(A) at a spot vs. neighbourhood abundance(B) correlation.
read_c2l_layers()           lazy read of selected genes from the 34 cell-type layers (h5py).
celltype_expr()             spots x cell types x genes, log1p(CP10k) of the attributed counts.
celltype_resolved_lr()      sender cell type x receiver cell type score for each LR pair,
                            with a spot-permutation null (z, p, BH-FDR).
inflow_source_enrichment()  LIANA inflow (source^L^R) -> share of each source around the
                            LR hotspots relative to its mean fraction (log2 enrichment).
niche_celltype_attribution()   which cell types carry the ligand / receptor of niche LRs
                            (center pool vs partners) - links niche_lr results to c2l.
"""
from __future__ import annotations
from fileinput import filename

import h5py
import numpy as np
import pandas as pd
import time
from pathlib import Path

import anndata as ad
import scanpy as sc
import squidpy as sq
import liana as li

from scipy import sparse, stats
import scipy.sparse as sps
from scipy.stats import rankdata

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

rcParams['pdf.fonttype']= 42 # enables correct plotting of text for PDFs

colors = ("#16466d", "#ebd246", "#f30707")
cmap_byr = LinearSegmentedColormap.from_list("blue_yellow_red", list(colors))

import programs.niche_lr as nl
import programs.programs as pg
import programs.c2l_liana as cl

from libs.Basic import pdwritecsv, pdreadcsv

def _plain(df):        # niche_lr keeps numpy arrays in DataFrame.attrs -> breaks repr / pd.concat
    d = df.copy()
    d.attrs = {}
    return d


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_c2l_spots(path, summary: str = "q05", region_labels=None, target_sum: float = 1e4):
    """Read sp.h5ad without the per-cell-type layers (they are ~5 GB; use read_c2l_layers).

    Returns AnnData with
        var_names = gene symbols (made unique), var['gene_ids'] = Ensembl
        layers['counts'] = raw counts, X = log1p(CP10k)
        obsm['c2l_abund'] = <summary> cell abundance (DataFrame, clean cell-type names)
        obsm['c2l_frac']  = abundance / total per spot
        obs['c2l_total_cells'], obs['region_cluster'] (+ 'region_name' from region_labels)
        uns['c2l_factor_names'], uns['spatial']
    """
    from anndata.io import read_elem
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        X = read_elem(f["X"])
        abund = read_elem(f[f"obsm/{summary}_cell_abundance_w_sf"])
        spatial = read_elem(f["obsm/spatial"])
        uns_spatial = read_elem(f["uns/spatial"])
        factor_names = [str(x) for x in read_elem(f["uns/mod/factor_names"])]
    abund = pd.DataFrame(np.asarray(abund), index=obs.index) if not isinstance(abund, pd.DataFrame) else abund
    abund.columns = factor_names
    keep_obs = [c for c in ["in_tissue", "array_row", "array_col", "sample", "region_cluster"] if c in obs]
    sym = ad.utils.make_index_unique(pd.Index(var["symbol"].astype(str).values))     # MATR3 is duplicated
    adata_vis = ad.AnnData(X=sparse.csr_matrix(X, dtype=np.float32), obs=obs[keep_obs].copy(),
                     var=pd.DataFrame({"gene_ids": var.index.values, "symbol": var["symbol"].astype(str).values},
                                      index=sym),
                     obsm={"spatial": np.asarray(spatial)}, uns={"spatial": uns_spatial})
    adata_vis.layers["counts"] = adata_vis.X.copy()
    tot = np.asarray(adata_vis.X.sum(1)).ravel()
    adata_vis.obs["total_counts"] = tot
    adata_vis.X = sparse.csr_matrix(sparse.diags(target_sum / np.maximum(tot, 1)) @ adata_vis.X)
    adata_vis.X.data = np.log1p(adata_vis.X.data)
    abund.index = adata_vis.obs_names
    adata_vis.obsm["c2l_abund"] = abund
    adata_vis.obsm["c2l_frac"] = abund.div(abund.sum(1), axis=0)
    adata_vis.obs["c2l_total_cells"] = abund.sum(1).values
    adata_vis.uns["c2l_factor_names"] = factor_names
    adata_vis.uns["c2l_summary"] = summary

    if region_labels is not None:
        lab = pd.read_csv(region_labels, index_col=0) if not isinstance(region_labels, pd.DataFrame) else region_labels
        lab = lab.reindex(adata_vis.obs_names)
        adata_vis.obs["region_cluster"] = lab["region_cluster"].astype(str).astype("category").values
        if "region_name" in lab:
            adata_vis.obs["region_name_tutorial"] = lab["region_name"].astype(str).values
    return adata_vis


def region_names_by_enrichment(adata_pos: ad.AnnData, df_frac: pd.DataFrame, regions: pd.Series, 
                               n: int = 2, min_frac: float = 0.02, root_liana: Path = Path("."),
                               verbose: bool = False, force: bool = False) -> pd.DataFrame:
    """
    Region -> 'typeA, typeB': cell types with the highest log2(mean fraction in region /
    mean fraction overall), among those making up >= min_frac of the region (avoids names
    driven by rare cell types, as happens with a max-scaled ranking).
    """

    fname = 'c2l_region_names.tsv'
    filename = root_liana / fname

    if filename.exists() and not force:
         df_reg = pdreadcsv(fname, root_liana, verbose=verbose)
         return df_reg


    regions = pd.Series(np.asarray(regions).astype(str), index=df_frac.index)
   
    df_mu = df_frac.groupby(regions).mean()
    
    df_enr = np.log2((df_mu + 1e-4) / (df_frac.mean() + 1e-4)).where(df_mu >= min_frac)
    
    reg_names = {r: ", ".join(df_enr.loc[r].dropna().nlargest(n).index) for r in df_mu.index}

    adata_pos.obs['region_cluster'] = pd.Categorical(adata_pos.obs['region_cluster'].astype(str), categories=order_reg)
    adata_pos.obs['region_name'] = adata_pos.obs['region_cluster'].astype(str).map(lambda r: f"{r}: {reg_names[r]}")

    dic = {'region_cluster': order_reg, 'region_name': [reg_names[r] for r in order_reg],
                            'n_spots': adata_pos.obs['region_cluster'].value_counts().reindex(order_reg).values,
                            'name_tutorial': adata_pos.obs.groupby('region_cluster', observed=True)['region_name_tutorial'].first().reindex(order_reg).values}
    df_reg = pd.DataFrame(dic)

    _ = pdwritecsv(df_reg, fname, root_liana, verbose=verbose)

    return df_reg


# --------------------------------------------------------------------------- #
# Spatial weights and co-localisation                                         #
# --------------------------------------------------------------------------- #
def row_normalise(W) -> sparse.csr_matrix:
    W = sparse.csr_matrix(W, dtype=np.float64)
    s = np.asarray(W.sum(1)).ravel()
    s[s == 0] = 1
    return sparse.csr_matrix(sparse.diags(1 / s) @ W)


def spatial_colocalisation(abund: pd.DataFrame, W) -> pd.DataFrame:
    """Symmetric cell-type co-localisation: Pearson r between abundance of A at a spot and the
    W-weighted (niche) abundance of B, averaged over (A,B) and (B,A)."""
    A = abund.values.astype(float)
    WA = row_normalise(W) @ A
    Az = (A - A.mean(0)) / (A.std(0) + 1e-12)
    WAz = (WA - WA.mean(0)) / (WA.std(0) + 1e-12)
    C = Az.T @ WAz / len(A)
    C = (C + C.T) / 2
    return pd.DataFrame(C, index=abund.columns, columns=abund.columns)


# --------------------------------------------------------------------------- #
# Cell-type-attributed expression from the cell2location layers               #
# --------------------------------------------------------------------------- #
def read_c2l_layers(path, gene_idx, celltypes) -> dict:
    """{cell type: csr (n_spots x len(gene_idx))} read lazily from sp.h5ad layers.
    gene_idx: integer column positions (same order as the file's var)."""
    gene_idx = np.asarray(gene_idx)
    order = np.argsort(gene_idx)
    inv = np.argsort(order)
    out = {}
    with h5py.File(path, "r") as f:
        for c in celltypes:
            M = ad.io.sparse_dataset(f["layers"][c])[:, gene_idx[order]]
            out[c] = sparse.csr_matrix(M)[:, inv]
    return out


def celltype_expr(filename, adata_vis, genes, celltypes=None, target_sum: float = 1e4):
    """Array (n_spots, n_celltypes, n_genes) = log1p(CP10k) of the counts that cell2location
    attributes to each cell type; CP10k uses the spot's total counts, so the values add the
    cell type's share of the spot to its expression level. Returns (E, genes_found, celltypes)."""
    celltypes = list(celltypes or adata_vis.uns["c2l_factor_names"])
    genes = [g for g in dict.fromkeys(genes) if g in adata_vis.var_names]
    idx = adata_vis.var_names.get_indexer(genes)
    L = read_c2l_layers(filename, idx, celltypes)
    tot = np.asarray(adata_vis.obs["total_counts"]).astype(float)
    E = np.stack([L[c].toarray() for c in celltypes], axis=1)
    E = np.log1p(E / np.maximum(tot, 1)[:, None, None] * target_sum)
    return E, genes, celltypes


def _entity(E, gpos, entity, sep="_"):
    subs = entity.split(sep)
    if not all(s in gpos for s in subs):
        return None
    return np.min(np.stack([E[:, :, gpos[s]] for s in subs], -1), -1)   # complex = min subunit


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def celltype_resolved_lr(E, genes, celltypes, lr: pd.DataFrame, W, n_perms: int = 200,
                         seed: int = 0, sep: str = "_") -> pd.DataFrame:
    """For every LR pair (ligand, receptor columns; complexes joined by '_'):
        score[A, B] = mean_s ( (W_norm L_A)(s) * R_B(s) )
    L_A = ligand attributed to sender type A (niche-smoothed with row-normalised W),
    R_B = receptor attributed to receiver type B at the spot.
    Null: receptor spots permuted (n_perms) -> z, empirical p (p_emp), p (Gaussian tail of z
    when < 5 permutations reach the observed score) and BH-FDR of p over all tests.
    The score is high when A carries the ligand where B carries the receptor nearby; it
    inherits the reference signatures (who can express L/R) and the c2l abundances (where)."""
    gpos = {g: i for i, g in enumerate(genes)}
    Wn = row_normalise(W)
    n, k = E.shape[0], E.shape[1]
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(n_perms)]
    S, R = np.meshgrid(celltypes, celltypes, indexing="ij")
    rows = []
    for _, r in lr.iterrows():
        Lk, Rk = _entity(E, gpos, r.ligand, sep), _entity(E, gpos, r.receptor, sep)
        if Lk is None or Rk is None:
            continue
        WL = Wn @ Lk
        obs = WL.T @ Rk / n
        null = np.stack([WL.T @ Rk[p] / n for p in perms])
        z = (obs - null.mean(0)) / (null.std(0) + 1e-12)
        n_ge = (null >= obs).sum(0)
        pe = (1 + n_ge) / (1 + n_perms)
        # beyond the permutation range use the Gaussian tail of z (as in niche_lr.score_niche_lr)
        p = np.where(n_ge < 5, np.minimum(pe, np.maximum(stats.norm.sf(z), 1e-12)), pe)
        rows.append(pd.DataFrame({"interaction": f"{r.ligand}^{r.receptor}", "ligand": r.ligand,
                                  "receptor": r.receptor, "sender": S.ravel(), "receiver": R.ravel(),
                                  "score": obs.ravel(), "z": z.ravel(), "p_emp": pe.ravel(), "p": p.ravel()}))
    df = pd.concat(rows, ignore_index=True)
    df["fdr"] = _bh(df.p.values)
    df["score_share"] = df.score / df.groupby("interaction").score.transform("sum")
    df["rank_in_lr"] = df.groupby("interaction").score.rank(ascending=False, method="first").astype(int)
    return df.sort_values(["interaction", "rank_in_lr"]).reset_index(drop=True)


def lr_matrix(df: pd.DataFrame, interaction: str, value: str = "score") -> pd.DataFrame:
    """sender x receiver matrix of one interaction from celltype_resolved_lr()."""
    return df[df.interaction == interaction].pivot(index="sender", columns="receiver", values=value)


# --------------------------------------------------------------------------- #
# LIANA inflow post-processing                                                 #
# --------------------------------------------------------------------------- #
def inflow_source_enrichment(inf, frac: pd.DataFrame, xy_sep: str = "^") -> pd.DataFrame:
    """inflow var 'source^ligand^receptor' -> global mean, share of the LR total and
    log2(share / mean fraction of the source). Because inflow multiplies spot ligand
    expression by the source proportion, raw scores track abundance; the enrichment asks
    whether the source is over-represented around the places where L->R is high."""
    X = inf.X
    gm = np.asarray(X.mean(0)).ravel()
    parts = inf.var_names.str.split(xy_sep, regex=False)
    df = pd.DataFrame({"feature": inf.var_names, "source": [p[0] for p in parts],
                       "ligand": [p[1] for p in parts], "receptor": [p[2] for p in parts], "global_mean": gm})
    df["interaction"] = df.ligand + xy_sep + df.receptor
    df["share"] = df.global_mean / df.groupby("interaction").global_mean.transform("sum")
    df["mean_frac"] = df.source.map(frac.mean())
    df["log2_enrichment"] = np.log2(df.share / df.mean_frac)
    return df.sort_values(["interaction", "log2_enrichment"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# niche_lr <-> cell2location                                                   #
# --------------------------------------------------------------------------- #
def niche_celltype_attribution(path, adata_vis, res, n: int = 15, celltypes=None, sep: str = "_") -> pd.DataFrame:
    """For the top-n LRs of a niche_lr NicheResult: fraction of the ligand counts (sender side)
    and of the receptor counts (receiver side) carried by each cell type, from the c2l layers.
    Sender side = pooled center for 'center->neighbors', partners for 'neighbors->center'.
    Complexes: mean share over subunits. Returns top sender / receiver with their shares."""
    celltypes = list(celltypes or adata_vis.uns["c2l_factor_names"])
    top = res.lr.head(n)
    genes = sorted({s for e in list(top.ligand) + list(top.receptor) for s in e.split(sep)})
    genes = [g for g in genes if g in adata_vis.var_names]
    L = read_c2l_layers(path, adata_vis.var_names.get_indexer(genes), celltypes)
    gpos = {g: i for i, g in enumerate(genes)}
    pool = np.asarray(res.lr.attrs.get("pooled_idx", [res.center]))
    part = np.setdiff1d(res.neighbors.idx.values, pool)
    def share(entity, spots):
        subs = [s for s in entity.split(sep) if s in gpos]
        if not subs:
            return pd.Series(np.nan, index=celltypes)
        M = np.array([[L[c][spots, gpos[s]].sum() for s in subs] for c in celltypes])   # k x subs
        M = M / np.maximum(M.sum(0, keepdims=True), 1e-12)
        return pd.Series(M.mean(1), index=celltypes)
    rows = []
    for _, r in top.iterrows():
        fwd = r.direction.startswith("center")
        sL = share(r.ligand, pool if fwd else part)
        sR = share(r.receptor, part if fwd else pool)
        rows.append({"ligand": r.ligand, "receptor": r.receptor, "direction": r.direction, "z": r.z, "fdr": r.fdr,
                     "top_sender": sL.idxmax(), "sender_share": sL.max(),
                     "sender_2nd": sL.drop(sL.idxmax()).idxmax(),
                     "top_receiver": sR.idxmax(), "receiver_share": sR.max(),
                     "receiver_2nd": sR.drop(sR.idxmax()).idxmax()})
    return pd.DataFrame(rows)


def niche_composition(adata_vis, res, bg_mask=None, key: str = "c2l_frac", n_background: int = 2000, seed: int = 0) -> pd.DataFrame:
    """c2l cell-type fractions of the niche (center + neighbours) vs random niches of the same
    size (z and percentile)."""
    from scipy.spatial import KDTree
    F = adata_vis.obsm[key]
    coords = np.asarray(adata_vis.obsm["spatial"])[:, :2]
    niche = np.r_[res.center, res.neighbors.idx.values]
    pool = np.arange(adata_vis.n_obs) if bg_mask is None else np.where(np.asarray(bg_mask, bool))[0]
    rng = np.random.default_rng(seed)
    bg = rng.choice(pool, size=min(n_background, len(pool)), replace=False)
    _, iB = KDTree(coords).query(coords[bg], k=len(niche))
    V = F.values
    obs = V[niche].mean(0)
    B = V[iB].mean(1)
    return pd.DataFrame({"cell_type": F.columns, "niche_frac": obs, "bg_mean": B.mean(0),
                         "z": (obs - B.mean(0)) / (B.std(0) + 1e-12),
                         "percentile": (B < obs).mean(0) * 100}).sort_values("z", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plot styling                                                                 #
# --------------------------------------------------------------------------- #
def style_clustermap(g, xlabel: str = "", ylabel: str = "", title: str | None = None,
                     cbar_label: str | None = None, fontsize: int = 12, pad: float = 0.09,
                     cbar_width: float = 0.015, cbar_height: float = 0.40):
    """Restyle a seaborn ClusterMap: font sizes for ticks / axis labels / title and the colorbar
    moved to the far right, beyond the row labels, vertically centred on the heatmap.
    Font sizes are set *before* measuring label extents, so the colorbar never overlaps them."""
    ax = g.ax_heatmap
    ax.set_xlabel(xlabel, fontsize=fontsize + 4)
    ax.set_ylabel(ylabel, fontsize=fontsize + 4)
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    g.ax_cbar.tick_params(labelsize=fontsize)
    if cbar_label:
        g.ax_cbar.set_ylabel(cbar_label, fontsize=fontsize + 2)
    g.ax_cbar.yaxis.set_ticks_position("right")
    g.ax_cbar.yaxis.set_label_position("right")
    if title:
        g.fig.suptitle(title, fontsize=fontsize + 6, y=1.02)

    fig = g.fig
    fig.canvas.draw()                                   # label extents are known only after a draw
    renderer = fig.canvas.get_renderer()
    bboxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
    bboxes.append(ax.yaxis.label.get_window_extent(renderer))
    x_right = fig.transFigure.inverted().transform((max(b.x1 for b in bboxes), 0))[0]
    hm = ax.get_position()
    g.ax_cbar.set_position([x_right + pad, hm.y0 + (1 - cbar_height) / 2 * hm.height,
                            cbar_width, cbar_height * hm.height])
    return g


def build_abund_perc(adata_pos: sc.AnnData) -> pd.DataFrame:

    df_abund_perc = adata_pos.obsm['c2l_abund'].copy()

    df_abund_perc = df_abund_perc.sum().sort_values(ascending=False).round(0).reset_index()
    df_abund_perc.columns = ['cell_type', 'abundance']
    df_abund_perc.abundance = df_abund_perc.abundance.astype(int)
    total = df_abund_perc.abundance.sum()
    df_abund_perc['perc'] = df_abund_perc.abundance / total

    return df_abund_perc

def plot_abundance(adata_pos: sc.AnnData, sel_cell_types: list, 
                   title='cell2location: q05 cell abundance per spot (colour capped at the 99.2th percentile)',
                   root_fig_liana: Path=Path('.'), cmap:LinearSegmentedColormap=cmap_byr, dpi: int=300, figsize: tuple=(17, 10.5)):

    lib_id = list(adata_pos.uns['spatial'].keys())[0]

    fig, axs = plt.subplots(3, 5, figsize=figsize)

    for ax, c in zip(axs.flat, sel_cell_types):
        v = adata_pos.obs[f'c2l_{c}']
        sq.pl.spatial_scatter(adata_pos, color=f'c2l_{c}', library_id=lib_id, img_res_key='hires', img_alpha=0.5,
                            size=1.3, cmap=cmap, vmax=float(np.quantile(v, 0.992)), title=c, ax=ax, colorbar=True)

    sq.pl.spatial_scatter(adata_pos, color='region_cluster', library_id=lib_id, img_res_key='hires', img_alpha=0.5,
                          size=1.3, title='c2l regions', ax=axs.flat[len(sel_cell_types)], legend_loc='on data', legend_fontsize=6)

    for ax in axs.flat[len(sel_cell_types) + 1:]:
        ax.axis('off')

    for ax in axs.flat:
        ax.set_xlabel(''); ax.set_ylabel('')

    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    filefig = root_fig_liana / 'c2l_abundance_maps.png'
    fig.savefig(filefig, dpi=dpi, bbox_inches='tight')
    plt.show()


def calc_liana_spacing(adata_pos: sc.AnnData, bandwidth:int = 150) -> pd.DataFrame:
    # nl = niche_lr lib
    spacing = nl.spot_spacing(adata_pos.obsm['spatial'])
    um_per_px = 100 / spacing
    
    li.ut.spatial_neighbors(adata_pos, bandwidth=bandwidth, cutoff=0.1, kernel='gaussian', set_diag=True)
    W = adata_pos.obsp['spatial_connectivities'].tocsr()
    W.eliminate_zeros()

    i = int(np.argmin(np.linalg.norm(adata_pos.obsm['spatial'] - np.median(adata_pos.obsm['spatial'], 0), axis=1)))   # a central spot
    d = np.linalg.norm(adata_pos.obsm['spatial'][W[i].indices] - adata_pos.obsm['spatial'][i], axis=1)

    dic = {'dist_um': (d * um_per_px).round(0), 'weight': W[i].data.round(3)}
    df_w = pd.DataFrame(dic)
    df_w = df_w.groupby('dist_um').agg(n=('weight', 'size'), weight=('weight', 'mean')).reset_index()

    print(f"spacing {spacing:.1f} px = 100 um -> {um_per_px:.3f} um/px; neighbours per spot (incl. self): {np.diff(W.indptr).mean():.1f}")

    return df_w

    
def calc_liana_bivariate(root_liana: Path = Path('.'), force: bool = False, verbose: bool = False) -> tuple[pd.DataFrame, sc.AnnData]:

    filename_libiv = root_liana / 'visium_lymph_node_liana_bivariate.h5ad'
    fname = 'liana_bivariate_global.tsv'
    filename = root_liana / fname

    if filename_libiv.exists() and not force:
        adata_lr = ad.read_h5ad(filename_libiv)
        if verbose: print(f"read {filename_libiv}")
    else:
        t0 = time.time()
        adata_lr = li.mt.bivariate(adata_pos, resource_name='consensus', local_name='cosine', global_name='morans',
                                n_perms=100, mask_negatives=False, add_categories=True, nz_prop=0.05,
                                use_raw=False, verbose=True)
        adata_lr.write_h5ad(filename_libiv)
        if verbose: print(f"Bivariate calc done in {time.time() - t0:.0f} s -> {filename_libiv}")

    if filename.exists() and not force:
        df_inter = pdreadcsv(fname, root_liana, verbose=verbose)
        df_inter.set_index('interaction', inplace=True)
        return df_inter, adata_lr

    df_inter = adata_lr.var.sort_values('morans', ascending=False).copy()
    df_inter['morans_fdr'] = nl._bh(df_inter.morans_pvals.values)          # BH over all tested pairs

    ret = pdwritecsv(df_inter.reset_index(), fname, root_liana, verbose=verbose)
    return df_inter, adata_lr



def plot_liana_bivariate(df_inter: pd.DataFrame, adata_lr: sc.AnnData, adata_pos: sc.AnnData, 
                         N:int=6, root_fig_liana: Path=Path('.'), 
                         cmap: LinearSegmentedColormap=cmap_byr, figsize: tuple=(16, 7.5)):

    lib_id = list(adata_pos.uns['spatial'].keys())[0]

    top_show = df_inter.index[:N].tolist()

    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(2, 4, width_ratios=[1.25, 1, 1, 1])

    ax = fig.add_subplot(gs[:, 0])
    t25 = df_inter.head(25)[::-1]

    ax.barh(t25.index, t25.morans, color='#4c72b0')
    ax.set_xlabel("global Moran's R (bivariate)")
    ax.set_title('top 25 LR pairs')
    ax.tick_params(axis='y', labelsize=6)

    for k, lr_id in enumerate(top_show):
        ax = fig.add_subplot(gs[k // 3, 1 + k % 3])
        v = adata_lr[:, lr_id].X
        v = np.asarray(v.toarray() if sps.issparse(v) else v).ravel()
        # local cosine saturates near 1 where L and R are both broadly expressed -> percentile limits for contrast
        sq.pl.spatial_scatter(adata_lr, color=lr_id, library_id=lib_id, img_res_key='hires', img_alpha=0.4,
                            size=1.3, cmap=cmap, vmin=float(np.quantile(v, 0.02)), vmax=float(np.quantile(v, 0.995)),
                            title=f"{lr_id}  (local cosine, 2-99.5th pct)", ax=ax)
        
        ax.set_xlabel(''); ax.set_ylabel('')

    fig.tight_layout()
    plt.show()

    filefig = root_fig_liana / 'liana_bivariate_top.png'
    fig.savefig(filefig, dpi=300, bbox_inches='tight')

def _rz(M):
    R = np.apply_along_axis(rankdata, 0, M)
    return (R - R.mean(0)) / (R.std(0) + 1e-12)


def calc_liana_spearman_correlation(df_inter_sig: pd.DataFrame, adata_lr: sc.AnnData, 
                              df_frac: pd.DataFrame, celltypes: list,
                              root_liana: Path=Path('.'), force: bool=False, verbose: bool=False) -> pd.DataFrame:

    fname = 'liana_local_vs_c2l_fraction_spearman.tsv'
    filename = root_liana / fname

    if filename.exists() and not force:
        df_rho = pdreadcsv(fname, root_liana, verbose=verbose)
        df_rho.set_index('interaction', inplace=True)
        return df_rho

    lr_top = df_inter_sig.index[:40]
    Xl = adata_lr[:, lr_top].X
    Xl = Xl.toarray() if sps.issparse(Xl) else np.asarray(Xl)

    df_rho = pd.DataFrame(_rz(Xl).T @ _rz(df_frac.values) / len(Xl), index=lr_top, columns=celltypes)

    pdwritecsv(df_rho.reset_index(names='interaction'), fname, root_liana, verbose=verbose)

    return df_rho


def plot_liana_spearman_correlation(df_rho: pd.DataFrame,
                              cmap: str|LinearSegmentedColormap='RdBu_r', 
                              fontsize:int=12, dpi:int=300,
                              root_fig_liana: Path=Path('.')):

    g = sns.clustermap(df_rho, cmap=cmap, center=0, vmin=-0.8, vmax=0.8, figsize=(15, 14),
                    dendrogram_ratio=(0.06, 0.06), 
                    # cbar_kws={'label': 'Spearman rho'},
                    xticklabels=True, yticklabels=True)

    ax = g.ax_heatmap

    # Place legend at the extreme right outside the axes
    # Place colorbar at the extreme right, outside the axes (beyond the row labels)
    fig = g.fig
    fig.canvas.draw()                                   # needed so label extents are known
    renderer = fig.canvas.get_renderer()

    bboxes = [t.get_window_extent(renderer) for t in ax.get_yticklabels()]
    bboxes.append(ax.yaxis.label.get_window_extent(renderer))
    x_right_px = max(b.x1 for b in bboxes)
    x_right = fig.transFigure.inverted().transform((x_right_px, 0))[0]   # pixels -> figure coords

    hm = ax.get_position()
    pad, width = 0.09, 0.015

    # left, bottom (vertically centred), width, height
    g.ax_cbar.set_position([x_right + pad, hm.y0 + 0.30 * hm.height, width,  0.40 * hm.height])

    # Tick font size (heatmap + colorbar) — must come before measuring label extents
    g.ax_cbar.tick_params(labelsize=fontsize)
    g.ax_cbar.yaxis.set_ticks_position('right')
    g.ax_cbar.yaxis.set_label_position('right')

    ax.set_xlabel('c2l cell type (fraction)')
    ax.set_ylabel('LR pair (local cosine)')
    ax.xaxis.label.set_size(fontsize + 4)
    ax.yaxis.label.set_size(fontsize + 4)

    ax.tick_params(axis='x', labelsize=fontsize)    # cell-type names
    ax.tick_params(axis='y', labelsize=fontsize)   # LR-pair names

    g.fig.suptitle("Spearman rho heatmap", fontsize=fontsize + 6, y=1.02)
    plt.show()

    filefig = root_fig_liana / 'liana_local_vs_c2l_fraction.png'
    g.savefig(filefig, dpi=dpi, bbox_inches='tight')

    return


def open_lr_biological_functions(root_liana: Path = Path('.'), verbose: bool = False):
    fname = 'ccc_lr_biological_functions.tsv'
    
    df_lr_bio = pdreadcsv(fname, root_liana)

    return df_lr_bio


def save_liana_inflow(adata_pos: ad.AnnData, df_inter: pd.DataFrame, df_frac: pd.DataFrame,
                      root_liana: Path=Path('.'), force: bool=False, verbose: bool=False):

    fname = 'liana_inflow_source_enrichment.tsv'
    filename = root_liana / fname
    lr150 = list(zip(df_inter.ligand[:150], df_inter.receptor[:150]))

    if filename.exists() and not force:
        df_inf = pdreadcsv(fname, root_liana, sep='\t', verbose=verbose)
    else:
        inf = li.mt.inflow(adata_pos, obsm_key='c2l_frac', interactions=lr150, use_raw=False)
        df_inf = cl.inflow_source_enrichment(inf, df_frac)
        pdwritecsv(df_inf, fname, root_liana, verbose=verbose)
        del inf

    return df_inf


def plot_liana_inflow(df_inter: pd.DataFrame, df_inf: pd.DataFrame, fontsize: int = 12,
                      bar_range: tuple = (-1.5, 1.5), root_fig_liana: Path = Path('.'), dpi: int = 300):
    
    lr_inf = df_inter.index[:25]
    E_inf = (df_inf[df_inf.interaction.isin(lr_inf)]
            .pivot(index='interaction', columns='source', values='log2_enrichment').reindex(lr_inf))

    g = sns.clustermap(E_inf.fillna(0), cmap='RdBu_r', center=0, vmin=bar_range[0], vmax=bar_range[1], figsize=(15, 10),
                    row_cluster=False, dendrogram_ratio=(0.06, 0.08), xticklabels=True, yticklabels=True)
    
    cl.style_clustermap(g, xlabel='source cell type (c2l proportion)', ylabel="LR pair (ordered by Moran's R)",
                        title='LIANA inflow: source enrichment', cbar_label='log2 enrichment', fontsize=fontsize)
    plt.show()

    filefig = root_fig_liana / 'liana_inflow_source_enrichment.png'
    g.savefig(filefig, dpi=dpi, bbox_inches='tight')


def calc_cell2location_resolved_sender_receiver(file_sp: Path,adata_pos: ad.AnnData, df_inter_sig: pd.DataFrame,
                                                celltypes: list, root_liana: Path = Path('.'),
                                                force: bool = False, verbose: bool = False) -> pd.DataFrame:

    fname = 'c2l_celltype_resolved_lr.tsv'
    filename = root_liana / fname
    lr100 = df_inter_sig.head(100)[['ligand', 'receptor']]

    if filename.exists() and not force:
        df_ct = pdreadcsv(fname, root_liana, verbose=verbose)
        return df_ct

    W = adata_pos.obsp['spatial_connectivities'].tocsr()
    W.eliminate_zeros()

    t0 = time.time()
    genes_lr = sorted({s for e in list(lr100.ligand) + list(lr100.receptor) for s in e.split('_')})
    E, genes_E, _ = celltype_expr(file_sp, adata_pos, genes_lr, celltypes)        # spots x cell types x genes
    df_ct = celltype_resolved_lr(E, genes_E, celltypes, lr100, W, n_perms=200, seed=0)
    _ = pdwritecsv(df_ct, fname, root_liana, verbose=verbose)
    del E

    if verbose: 
        print(f"{df_ct.interaction.nunique()} LR pairs scored in {time.time() - t0:.0f} s")

    return df_ct


def plot_sender_receiver_heatmap(df_ct: pd.DataFrame, lista: list,
                                 root_fig_liana: Path = Path('.'), 
                                 cmap: str = "RdBu_r", height:float=5, width:float=17, 
                                 fontsize_title:int=14, fontsize_label:int=12, 
                                 fontsize_ticks:int=11, dpi: int = 300):

    lr_panels = [x for x in lista if x in set(df_ct.interaction)]

    if not lr_panels:
        print("No matching LR pairs found.")
        return

    n = len(lr_panels)
    nrows = int(np.ceil(n / 2))

    height *= nrows
    figsize = (width, height)

    fig, axs = plt.subplots(nrows, 2, figsize=figsize)
    for ax, lr_id in zip(axs.flat, lr_panels):
        Mx = cl.lr_matrix(df_ct, lr_id, 'score')
        s_top = Mx.sum(1).nlargest(10).index
        r_top = Mx.sum(0).nlargest(10).index

        # cmap 'rocket_r'
        sns.heatmap(Mx.loc[s_top, r_top], ax=ax, cmap=cmap, cbar_kws={'label': 'score', 'shrink': .7},
                    xticklabels=True, yticklabels=True)

        # x/y tick label size (and optional rotation)
        ax.tick_params(axis='x', labelsize=fontsize_ticks, labelrotation=90)
        ax.tick_params(axis='y', labelsize=fontsize_ticks, labelrotation=0)

        # colorbar ticks + label, optional
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=fontsize_ticks)
        cbar.set_label('score', fontsize=fontsize_label)

        ax.set_title(lr_id, fontsize=fontsize_title)
        ax.set_ylabel('sender (ligand)', fontsize=fontsize_label)
        ax.set_xlabel('receiver (receptor)', fontsize=fontsize_label)

    for ax in axs.flat[len(lr_panels):]:
        ax.axis('off')

    fig.suptitle('cell2location-resolved CCC: top {n} senders x receivers per LR pair', y=1.0, fontsize=fontsize_title)
    fig.tight_layout()

    filefig = root_fig_liana / 'c2l_celltype_resolved_lr_panels.png'
    fig.savefig(filefig, dpi=dpi, bbox_inches='tight')

    plt.show()


def dotplot_sender_receiver(df_ct: pd.DataFrame, zmin: float=5, nlargest:int=40,
                            figsize=(7.5, 10), fontsize:int=6,
                            cmap: str|LinearSegmentedColormap='viridis',
                            root_fig_liana: Path=Path('.'), dpi: int=300):

    # strongest sender -> receiver links over all LR pairs (top 40 by score, z > 5)
    top_ = df_ct[df_ct.z > zmin].nlargest(nlargest, 'score').iloc[::-1]

    lab = top_.sender + ' -> ' + top_.receiver + '   ' + top_.interaction.str.replace('^', ' -> ', regex=False)

    fig, ax = plt.subplots(figsize=figsize)

    sc_ = ax.scatter(top_.score, np.arange(len(top_)), c=top_.z, s=40 + 300 * top_.score_share, cmap=cmap)
    
    ax.set_yticks(np.arange(len(top_)))
    ax.set_yticklabels(lab, fontsize=fontsize)
    
    ax.set_xlabel('score  (mean of niche ligand x receptor)')
    ax.set_title('top sender -> receiver links (size = share of LR total)')
    
    plt.colorbar(sc_, ax=ax, shrink=.5, label='z vs spot permutation')

    fig.tight_layout()

    filefig = root_fig_liana / 'c2l_celltype_resolved_top_links.png'
    fig.savefig(filefig, dpi=dpi, bbox_inches='tight')
    plt.show()

def calc_liana_rank_aggregate_reference(file_ref: Path, root_liana: Path, 
                                        force: bool=False, verbose: bool=False) -> pd.DataFrame:

    fname = 'liana_rank_aggregate_reference.tsv'
    filename = root_liana / fname

    if filename.exists() and not force:
        df_liref = pdreadcsv(fname, root_liana, verbose=verbose)
        return df_liref

    t0 = time.time()
    adata_refb = ad.read_h5ad(file_ref, backed='r')
    df_obs_ref = adata_refb.obs[['Subset', 'Method', 'Sample', 'Tissue']]

    rng = np.random.default_rng(0)
    keep = np.sort(np.concatenate([rng.choice(ix, size=min(400, len(ix)), replace=False)
                                for ix in df_obs_ref.groupby('Subset', observed=True).indices.values()]))

    adata_ref = adata_refb[keep].to_memory()
    adata_refb.file.close()

    cols = ['Subset', 'Method', 'Sample', 'Tissue']
    adata_ref = ad.AnnData(X=adata_ref.X.astype(np.float32), obs=adata_ref.obs[cols].copy(),
                    var=pd.DataFrame(index=adata_ref.var_names))          # var_names = symbols in the original file

    sc.pp.normalize_total(adata_ref, target_sum=1e4); sc.pp.log1p(adata_ref)
    print(adata_ref.shape, adata_ref.obs.Subset.value_counts().min(), 'cells in the smallest type')
    
    li.mt.rank_aggregate(adata_ref, groupby='Subset', resource_name='consensus', 
                            expr_prop=0.1, use_raw=False, n_perms=100, n_jobs=8, verbose=False)
    
    df_liref = adata_ref.uns['liana_res'].copy()
    del adata_ref
    pdwritecsv(df_liref, fname, root_liana, verbose=verbose)

    print(f"rank_aggregate in {time.time() - t0:.0f} s")

    return df_liref

def save_celltype_colocalisation(adata_pos, df_abund, df_liref: pd.DataFrame, root_liana: Path,
                                 verbose: bool=False, force: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:

    fname = 'c2l_celltype_colocalisation.tsv'
    filename = root_liana / fname

    fname2 = 'liana_rank_aggregate_spatially_supported.tsv'
    filename2 = root_liana / fname2

    if filename.exists() and filename2. exists() and not force:

        df_coloc = pdreadcsv(fname, root_liana, verbose=verbose)
        df_liref_sup = pdreadcsv(fname2, root_liana, verbose=verbose)

        return df_coloc, df_liref_sup

    W = adata_pos.obsp['spatial_connectivities'].tocsr()
    W.eliminate_zeros()

    df_coloc = spatial_colocalisation(df_abund, W)

    pdwritecsv(df_coloc.reset_index(names='cell_type'), fname, root_liana, verbose=verbose)

    df_liref['df_coloc'] = [df_coloc.at[s, t] for s, t in zip(df_liref.source, df_liref.target)]

    df_liref_sel = df_liref[(df_liref.specificity_rank <= 0.05) & (df_liref.magnitude_rank <= 0.05)]
    df_liref_sup = df_liref_sel[df_liref_sel.df_coloc > 0.2].sort_values('specificity_rank')
    df_liref_sup = df_liref_sup.sort_values('magnitude_rank')

    pdwritecsv(df_liref_sup, fname2, root_liana, verbose=verbose)

    return df_coloc, df_liref_sup

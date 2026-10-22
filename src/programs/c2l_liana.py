"""
cell2location -> LIANA+ cell-cell communication (CCC) helpers.

Input: the cell2location spatial output written by `cell2location01a_tutorial.ipynb`
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

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, stats


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
    vis = ad.AnnData(X=sparse.csr_matrix(X, dtype=np.float32), obs=obs[keep_obs].copy(),
                     var=pd.DataFrame({"gene_ids": var.index.values, "symbol": var["symbol"].astype(str).values},
                                      index=sym),
                     obsm={"spatial": np.asarray(spatial)}, uns={"spatial": uns_spatial})
    vis.layers["counts"] = vis.X.copy()
    tot = np.asarray(vis.X.sum(1)).ravel()
    vis.obs["total_counts"] = tot
    vis.X = sparse.csr_matrix(sparse.diags(target_sum / np.maximum(tot, 1)) @ vis.X)
    vis.X.data = np.log1p(vis.X.data)
    abund.index = vis.obs_names
    vis.obsm["c2l_abund"] = abund
    vis.obsm["c2l_frac"] = abund.div(abund.sum(1), axis=0)
    vis.obs["c2l_total_cells"] = abund.sum(1).values
    vis.uns["c2l_factor_names"] = factor_names
    vis.uns["c2l_summary"] = summary
    if region_labels is not None:
        lab = pd.read_csv(region_labels, index_col=0) if not isinstance(region_labels, pd.DataFrame) else region_labels
        lab = lab.reindex(vis.obs_names)
        vis.obs["region_cluster"] = lab["region_cluster"].astype(str).astype("category").values
        if "region_name" in lab:
            vis.obs["region_name_tutorial"] = lab["region_name"].astype(str).values
    return vis


def region_names_by_enrichment(frac: pd.DataFrame, regions, n: int = 2, min_frac: float = 0.02) -> dict:
    """Region -> 'typeA, typeB': cell types with the highest log2(mean fraction in region /
    mean fraction overall), among those making up >= min_frac of the region (avoids names
    driven by rare cell types, as happens with a max-scaled ranking)."""
    regions = pd.Series(np.asarray(regions).astype(str), index=frac.index)
    m = frac.groupby(regions).mean()
    enr = np.log2((m + 1e-4) / (frac.mean() + 1e-4)).where(m >= min_frac)
    return {r: ", ".join(enr.loc[r].dropna().nlargest(n).index) for r in m.index}


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


def celltype_expr(path, vis, genes, celltypes=None, target_sum: float = 1e4):
    """Array (n_spots, n_celltypes, n_genes) = log1p(CP10k) of the counts that cell2location
    attributes to each cell type; CP10k uses the spot's total counts, so the values add the
    cell type's share of the spot to its expression level. Returns (E, genes_found, celltypes)."""
    celltypes = list(celltypes or vis.uns["c2l_factor_names"])
    genes = [g for g in dict.fromkeys(genes) if g in vis.var_names]
    idx = vis.var_names.get_indexer(genes)
    L = read_c2l_layers(path, idx, celltypes)
    tot = np.asarray(vis.obs["total_counts"]).astype(float)
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
def niche_celltype_attribution(path, vis, res, n: int = 15, celltypes=None, sep: str = "_") -> pd.DataFrame:
    """For the top-n LRs of a niche_lr NicheResult: fraction of the ligand counts (sender side)
    and of the receptor counts (receiver side) carried by each cell type, from the c2l layers.
    Sender side = pooled center for 'center->neighbors', partners for 'neighbors->center'.
    Complexes: mean share over subunits. Returns top sender / receiver with their shares."""
    celltypes = list(celltypes or vis.uns["c2l_factor_names"])
    top = res.lr.head(n)
    genes = sorted({s for e in list(top.ligand) + list(top.receptor) for s in e.split(sep)})
    genes = [g for g in genes if g in vis.var_names]
    L = read_c2l_layers(path, vis.var_names.get_indexer(genes), celltypes)
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


def niche_composition(vis, res, bg_mask=None, key: str = "c2l_frac", n_background: int = 2000, seed: int = 0) -> pd.DataFrame:
    """c2l cell-type fractions of the niche (center + neighbours) vs random niches of the same
    size (z and percentile)."""
    from scipy.spatial import KDTree
    F = vis.obsm[key]
    coords = np.asarray(vis.obsm["spatial"])[:, :2]
    niche = np.r_[res.center, res.neighbors.idx.values]
    pool = np.arange(vis.n_obs) if bg_mask is None else np.where(np.asarray(bg_mask, bool))[0]
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

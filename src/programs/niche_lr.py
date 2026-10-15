"""
Spatial niche ligand-receptor analysis with program-level interpretation.

Workflow
--------
1. select_center()        pick a central cell/spot (barcode, coordinate, region, or
                          the hotspot of a gene / program score).
2. neighborhood()         k <= 20 nearest neighbours in 2-D (KD-tree), with distance,
                          angle and ring (distance in units of spot/cell spacing).
3. score_niche_lr()       for every LR pair of the resource, two directions:
                            center->neighbors  (ligand in center, receptor around)
                            neighbors->center  (ligand around, receptor in center)
                          score = distance-weighted mean of sqrt(L_sender * R_receiver).
                          Specificity: the same score is computed for a background of
                          random niches across the tissue -> z-score, empirical p, BH-FDR.
4. niche_program_activity()  program scores of the niche vs. background niches.
5. infer_functions()      combine LR evidence (programs containing L or R) with niche
                          program activity -> ranked candidate biological functions.
6. collinear_cells() / lr_gradient()
                          for each top LR, 3-6 collinear cells starting at the center
                          and heading to the best partner neighbour; ligand and receptor
                          gradients (OLS slope, Spearman rho) and source->sink pattern.

Expected input: AnnData with log-normalised (non-negative) expression in X or a
layer, and coordinates in adata.obsm['spatial'] (Squidpy / Scanpy convention).
For Visium, one "cell" is a 55 um spot (1-10 cells) with 100 um centre spacing;
on a hex grid the 6 + 12 spots of rings 1-2 give 18 neighbours, so k=20 ~ 2 rings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.spatial import KDTree

from .programs import PROGRAMS, flatten

# --------------------------------------------------------------------------- #
# Expression helpers                                                          #
# --------------------------------------------------------------------------- #
def _matrix(adata, layer=None):
    X = adata.layers[layer] if layer else adata.X
    return X


def gene_expr(adata, genes, layer=None) -> pd.DataFrame:
    """Dense cells x genes DataFrame for the genes present in adata."""
    genes = [g for g in dict.fromkeys(genes) if g in adata.var_names]
    idx = adata.var_names.get_indexer(genes)
    X = _matrix(adata, layer)[:, idx]
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return pd.DataFrame(X, index=adata.obs_names, columns=genes)


def load_lr(resource: str = "consensus", path: str | None = None) -> pd.DataFrame:
    """LIANA LR resource (ligand, receptor; complexes joined by '_').
    resource: 'consensus' (human symbols) or 'mouseconsensus' (mouse symbols)."""
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"lr_{resource}.csv")
    lr = pd.read_csv(path)[["ligand", "receptor"]].drop_duplicates().reset_index(drop=True)
    return lr


def complex_expr(adata, entities, layer=None) -> pd.DataFrame:
    """Expression of (possibly heteromeric) entities: complex = min over subunits.
    Entities with a missing subunit are dropped."""
    subunits = {e: e.split("_") for e in entities}
    allg = {g for s in subunits.values() for g in s}
    E = gene_expr(adata, allg, layer)
    out = {e: E[s].min(axis=1).values for e, s in subunits.items() if all(g in E for g in s)}
    return pd.DataFrame(out, index=adata.obs_names)


def spot_spacing(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance (= centre-to-centre spacing on Visium)."""
    d, _ = KDTree(coords).query(coords, k=2)
    return float(np.median(d[:, 1]))


# --------------------------------------------------------------------------- #
# 1-2. Center and neighbourhood                                                #
# --------------------------------------------------------------------------- #
def select_center(adata, *, barcode=None, xy=None, mask=None, genes=None, program=None,
                  programs=PROGRAMS, smooth_k=7, spatial_key="spatial", layer=None) -> int:
    """Return the integer index of the central cell/spot. Give exactly one of:
    barcode : obs name
    xy      : (x, y) coordinate -> nearest spot
    mask    : boolean array / obs column name defining a region -> spot nearest its centroid
    genes   : list of genes  -> hotspot of the (kNN-smoothed) mean z-score
    program : program name (e.g. 'EMT' or 'tissue_remodeling:EMT') -> hotspot
    """
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    if barcode is not None:
        return int(adata.obs_names.get_loc(barcode))
    if xy is not None:
        return int(KDTree(coords).query(np.asarray(xy))[1])
    if mask is not None:
        m = adata.obs[mask].values.astype(bool) if isinstance(mask, str) else np.asarray(mask, bool)
        return int(KDTree(coords).query(coords[m].mean(0))[1])
    if program is not None:
        flat = flatten(programs)
        key = program if program in flat else next(k for k in flat if k.split(":")[1] == program)
        genes = flat[key]
    if genes is None:
        raise ValueError("give barcode, xy, mask, genes or program")
    s = _zscore_mean(gene_expr(adata, genes, layer))
    _, nn = KDTree(coords).query(coords, k=smooth_k)          # smooth to find a *region*
    return int(np.argmax(s[nn].mean(1)))


def neighborhood(adata, center: int, k: int = 20, max_radius: float | None = None,
                 spatial_key: str = "spatial") -> pd.DataFrame:
    """k nearest neighbours of `center` (self excluded), optionally within max_radius
    (same units as coordinates). Columns: idx, barcode, dx, dy, dist, ring, angle_deg."""
    k = min(k, 20)
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    tree = KDTree(coords)
    d, i = tree.query(coords[center], k=k + 1)
    d, i = d[1:], i[1:]
    if max_radius is not None:
        keep = d <= max_radius
        d, i = d[keep], i[keep]
    sp = spot_spacing(coords)
    dxy = coords[i] - coords[center]
    return pd.DataFrame({
        "idx": i, "barcode": adata.obs_names[i], "dx": dxy[:, 0], "dy": dxy[:, 1],
        "dist": d, "ring": np.rint(d / sp).astype(int),
        "angle_deg": np.degrees(np.arctan2(dxy[:, 1], dxy[:, 0])) % 360,
    })


# --------------------------------------------------------------------------- #
# 3. LR scoring with a spatial background                                      #
# --------------------------------------------------------------------------- #
def _gauss(d, sigma):
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def _bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def _background_centers(n: int, n_background: int, bg_mask=None, seed: int = 0) -> np.ndarray:
    pool = np.arange(n) if bg_mask is None else np.where(np.asarray(bg_mask, bool))[0]
    rng = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n_background, len(pool)), replace=False)


def score_niche_lr(adata, center: int, nbrs: pd.DataFrame, lr: pd.DataFrame | None = None, *,
                   layer=None, spatial_key="spatial", min_expr: float = 0.1,
                   min_frac_nbrs: float = 0.1, center_pool: int = 0, sigma: float | None = None,
                   n_background: int = 2000, bg_mask=None, seed: int = 0) -> pd.DataFrame:
    """Score every LR pair in both directions for the niche (center + nbrs).

    center_pool : pool the center with its `center_pool` nearest spots (e.g. 6 = ring 1 on a
                  Visium grid) -> a central *region*; the remaining neighbours are the partners.
                  Pooling damps single-spot count noise. 0 = single central cell/spot.
    Filters (applied before the background, so fewer tests): the sender ligand and receiver
    receptor must reach `min_expr` (pooled center: mean; neighbours: in >= min_frac_nbrs of them).
    Background: same statistic for `n_background` random niches of identical geometry ->
    z, empirical p (floor 1/(n_background+1)) and BH-FDR over the tested pairs.
    bg_mask     : boolean array (n_obs) of spots allowed as background centers. None = whole
                  tissue ("what is special about this niche"); e.g. cluster != own cluster
                  ("this region vs the other regions").
    """
    lr = load_lr() if lr is None else lr
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    nbrs = nbrs.sort_values("dist").reset_index(drop=True)
    k = len(nbrs)
    if center_pool >= k:
        raise ValueError("center_pool must be smaller than the number of neighbours")
    sp = spot_spacing(coords)
    sigma = sigma or sp                       # weight 0.61 at 1 spacing, 0.14 at 2
    pool = np.r_[center, nbrs.idx.values[:center_pool]]
    part = nbrs.iloc[center_pool:]
    nb, d_nb = part.idx.values, part.dist.values
    w = _gauss(d_nb, sigma)
    w = w / w.sum()

    E = complex_expr(adata, pd.unique(lr[["ligand", "receptor"]].values.ravel()), layer)
    lr = lr[lr.ligand.isin(E.columns) & lr.receptor.isin(E.columns)].reset_index(drop=True)
    EL, ER = E.loc[:, lr.ligand].values, E.loc[:, lr.receptor].values
    Lc, Rc = EL[pool].mean(0), ER[pool].mean(0)
    Ln, Rn = EL[nb], ER[nb]                                  # m x P

    keep_f = (Lc >= min_expr) & ((Rn >= min_expr).mean(0) >= min_frac_nbrs)
    keep_r = (Rc >= min_expr) & ((Ln >= min_expr).mean(0) >= min_frac_nbrs)
    cols = np.where(keep_f | keep_r)[0]
    lr, EL, ER = lr.iloc[cols].reset_index(drop=True), EL[:, cols], ER[:, cols]
    Lc, Rc, Ln, Rn = Lc[cols], Rc[cols], Ln[:, cols], Rn[:, cols]
    keep_f, keep_r = keep_f[cols], keep_r[cols]

    fwd = (w[:, None] * np.sqrt(Lc[None] * Rn)).sum(0)       # center -> neighbours
    rev = (w[:, None] * np.sqrt(Ln * Rc[None])).sum(0)       # neighbours -> center

    # background niches with the same geometry (pool + k - pool partners)
    bg = _background_centers(len(coords), n_background, bg_mask, seed)
    dB, iB = KDTree(coords).query(coords[bg], k=k + 1)
    poolB, iP, dP = iB[:, :center_pool + 1], iB[:, center_pool + 1:], dB[:, center_pool + 1:]
    wB = _gauss(dP, sigma)
    wB = wB / wB.sum(1, keepdims=True)
    bf = np.empty((len(bg), len(lr)))
    br = np.empty_like(bf)
    for j in range(len(bg)):
        lc, rc = EL[poolB[j]].mean(0), ER[poolB[j]].mean(0)
        ww = wB[j][:, None]
        bf[j] = (ww * np.sqrt(lc[None] * ER[iP[j]])).sum(0)
        br[j] = (ww * np.sqrt(EL[iP[j]] * rc[None])).sum(0)

    rows = []
    for direction, obs, B, Ls, Rr, keep in [
        ("center->neighbors", fwd, bf, np.broadcast_to(Lc, Rn.shape), Rn, keep_f),
        ("neighbors->center", rev, br, Ln, np.broadcast_to(Rc, Ln.shape), keep_r),
    ]:
        mu, sd = B.mean(0), B.std(0) + 1e-9
        df = lr.copy()
        df["direction"] = direction
        df["score"] = obs
        df["bg_mean"] = mu
        df["z"] = (obs - mu) / sd
        n_ge = (B >= obs[None]).sum(0)
        df["p_emp"] = (1 + n_ge) / (1 + len(B))
        # beyond the background range the empirical p hits its floor; there use the Gaussian
        # tail of z (approximate, background scores are right-skewed) so that extreme pairs
        # are not all tied at 1/(n+1). `fdr_emp` keeps the purely empirical version.
        p_tail = stats.norm.sf(df["z"].values)
        df["p"] = np.where(n_ge < 5, np.minimum(df["p_emp"].values, np.maximum(p_tail, 1e-12)), df["p_emp"].values)
        df["ligand_sender"] = Lc if direction.startswith("center") else Ln.mean(0)
        df["receptor_receiver"] = Rn.mean(0) if direction.startswith("center") else Rc
        df["best_partner_idx"] = nb[np.argmax(w[:, None] * np.sqrt(Ls * Rr), 0)]
        rows.append(df[keep])
    res = pd.concat(rows, ignore_index=True)
    res["fdr"] = _bh(res.p.values)
    res["fdr_emp"] = _bh(res.p_emp.values)
    res.attrs.update(center_pool=center_pool, pooled_idx=pool, n_background=len(bg))
    return res.sort_values(["fdr", "z"], ascending=[True, False]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4-5. Programs                                                                #
# --------------------------------------------------------------------------- #
def _zscore_mean(E: pd.DataFrame) -> np.ndarray:
    X = E.values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X.mean(1) if X.shape[1] else np.zeros(len(E))


def niche_program_activity(adata, center: int, nbrs: pd.DataFrame, programs=PROGRAMS, *,
                           layer=None, spatial_key="spatial", min_genes: int = 3,
                           n_background: int = 2000, bg_mask=None, seed: int = 0) -> pd.DataFrame:
    """Mean gene z-score per program, averaged over center + neighbours, versus the same
    quantity for random niches. Returns z vs background and empirical percentile."""
    flat = flatten(programs)
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    k = len(nbrs)
    niche = np.r_[center, nbrs.idx.values]
    bg = _background_centers(len(coords), n_background, bg_mask, seed)
    _, iB = KDTree(coords).query(coords[bg], k=k + 1)
    allg = {g for gs in flat.values() for g in gs}
    E = gene_expr(adata, allg, layer)
    Z = (E - E.mean()) / (E.std() + 1e-9)
    rows = []
    for name, genes in flat.items():
        g = [x for x in genes if x in Z.columns]
        if len(g) < min_genes:
            continue
        s = Z[g].values.mean(1)
        obs = s[niche].mean()
        B = s[iB].mean(1)
        cat, prog = name.split(":")
        rows.append({"category": cat, "program": prog, "n_genes": len(g),
                     "niche_score": obs, "center_score": s[center],
                     "z": (obs - B.mean()) / (B.std() + 1e-9),
                     "percentile": (B < obs).mean() * 100})
    return pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)


def annotate_lr_programs(lr_res: pd.DataFrame, programs=PROGRAMS) -> pd.DataFrame:
    """Add the programs that contain the ligand and/or receptor (subunits included)."""
    flat = flatten(programs)
    g2p: dict[str, set] = {}
    for name, genes in flat.items():
        for g in genes:
            g2p.setdefault(g, set()).add(name.split(":")[1])
    def progs(entity):
        return sorted({p for s in entity.split("_") for p in g2p.get(s, ())})
    out = lr_res.copy()
    out["ligand_programs"] = out.ligand.map(lambda e: ", ".join(progs(e)))
    out["receptor_programs"] = out.receptor.map(lambda e: ", ".join(progs(e)))
    out["shared_programs"] = [", ".join(sorted(set(progs(l)) & set(progs(r))))
                              for l, r in zip(out.ligand, out.receptor)]
    return out


def infer_functions(lr_res: pd.DataFrame, activity: pd.DataFrame, programs=PROGRAMS,
                    fdr: float = 0.1, w_lr: float = 0.5) -> pd.DataFrame:
    """
    Rank candidate functions of the niche.
    lr_support = sum over significant LRs of -log10(p),
       for pairs whose L or R belongs to the program (shared L+R membership counts double). 
    Final score = w_lr * rank-pct(lr_support)
    + (1 - w_lr) * rank-pct(activity z).
    """
    sig = annotate_lr_programs(lr_res[lr_res.fdr <= fdr], programs)
    sup, who = {}, {}
    for _, r in sig.iterrows():
        lp = set(filter(None, r.ligand_programs.split(", ")))
        rp = set(filter(None, r.receptor_programs.split(", ")))
        for p in lp | rp:
            wt = -np.log10(r.p) * (2 if p in lp & rp else 1)
            sup[p] = sup.get(p, 0) + wt
            who.setdefault(p, []).append(f"{r.ligand}->{r.receptor} ({'C>N' if r.direction.startswith('center') else 'N>C'})")
    df = activity.copy()
    df["lr_support"] = df.program.map(sup).fillna(0)
    df["n_lr"] = df.program.map(lambda p: len(who.get(p, [])))
    df["top_lr"] = df.program.map(lambda p: "; ".join(who.get(p, [])[:5]))
    df["score"] = w_lr * df.lr_support.rank(pct=True) + (1 - w_lr) * df.z.rank(pct=True)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Collinear cells and gradients                                            #
# --------------------------------------------------------------------------- #
def collinear_cells(adata, center: int, *, target: int | None = None, angle_deg: float | None = None,
                    n_cells: int = 6, min_cells: int = 3, tol: float = 0.35, max_gap: float = 1.9,
                    spatial_key: str = "spatial") -> pd.DataFrame:
    """Cells lying on the ray from `center` towards `target` (or along angle_deg).
    tol / max_gap are in units of spot spacing (perpendicular tolerance, max step).
    Returns idx, barcode, t (distance along the ray, coordinate units), perp; center first.
    On a Visium hex grid, rays along 0/60/120 deg have steps of 1 spacing; rays along 30/90/150
    deg have steps of sqrt(3) = 1.73 spacings (hence max_gap 1.9)."""
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    sp = spot_spacing(coords)
    c = coords[center]
    if target is not None:
        u = coords[target] - c
    elif angle_deg is not None:
        u = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    else:
        raise ValueError("give target or angle_deg")
    u = u / np.linalg.norm(u)
    v = coords - c
    t = v @ u
    perp = np.abs(v[:, 0] * u[1] - v[:, 1] * u[0])
    cand = np.where((t >= -1e-9) & (perp <= tol * sp) & (t <= (n_cells + 1) * max_gap * sp))[0]
    cand = cand[np.argsort(t[cand])]
    line = [center]
    for i in cand:
        if i == center:
            continue
        if t[i] - t[line[-1]] > max_gap * sp:
            break
        if t[i] - t[line[-1]] < 0.5 * sp:       # same position (dense single-cell data): keep closest to axis
            if perp[i] < perp[line[-1]] and line[-1] != center:
                line[-1] = i
            continue
        line.append(i)
        if len(line) == n_cells:
            break
    if len(line) < min_cells:
        return pd.DataFrame(columns=["idx", "barcode", "t", "perp"])
    line = np.array(line)
    return pd.DataFrame({"idx": line, "barcode": adata.obs_names[line], "t": t[line], "perp": perp[line]})


def lr_gradient(adata, line: pd.DataFrame, ligand: str, receptor: str, direction: str,
                layer=None, um_per_unit: float | None = None, min_expr: float = 0.1,
                min_rho: float = 0.6) -> dict:
    """Ligand/receptor profiles along a collinear line (center first).

    direction 'center->neighbors': center is the sender -> expected ligand decreasing
        away from center (slope < 0) and receptor present/increasing downstream.
    direction 'neighbors->center': center is the receiver -> expected receptor highest at
        center (slope < 0) and ligand increasing towards the sender (slope > 0).
    Slopes are per unit of distance (um if um_per_unit given, else coordinate units).
    A slope is called a trend only if |Spearman rho| >= min_rho (with 3-6 points, descriptive)."""
    E = complex_expr(adata, [ligand, receptor], layer)
    d = line.t.values * (um_per_unit or 1.0)
    L = E[ligand].values[line.idx.values]
    R = E[receptor].values[line.idx.values]
    def fit(y):
        if np.ptp(y) == 0:
            return 0.0, 0.0, 1.0
        sl = stats.linregress(d, y).slope
        rho, p = stats.spearmanr(d, y)
        return float(sl), float(np.nan_to_num(rho)), float(np.nan_to_num(p, nan=1.0))
    sL, rL, pL = fit(L)
    sR, rR, pR = fit(R)
    # a trend counts only if monotonic enough (|Spearman rho| >= min_rho)
    tL = np.sign(sL) if abs(rL) >= min_rho else 0
    tR = np.sign(sR) if abs(rR) >= min_rho else 0
    # sender side should hold the ligand maximum, receiver side should express the receptor
    if direction.startswith("center"):
        consistent = (tL < 0) and (L[0] >= L[1:].mean()) and (R[1:].max() >= min_expr)
    else:
        consistent = (tL > 0) and (R[0] >= R[1:].mean()) and (R[0] >= min_expr)
    pattern = {(-1, 1): "opposing (source->sink)", (1, -1): "opposing (sink->source)",
               (-1, -1): "co-decreasing", (1, 1): "co-increasing"}.get((tL, tR),
              "ligand gradient only" if tL else "receptor gradient only" if tR else "flat")
    return {"ligand": ligand, "receptor": receptor, "direction": direction, "n_cells": len(line),
            "unit": "um" if um_per_unit else "coord units",
            "distance": d, "L": L, "R": R, "barcodes": line.barcode.tolist(),
            "L_slope": sL, "L_rho": rL, "L_p": pL, "R_slope": sR, "R_rho": rR, "R_p": pR,
            "LR_corr": float(np.corrcoef(L, R)[0, 1]) if np.ptp(L) and np.ptp(R) else np.nan,
            "pattern": pattern, "consistent_with_direction": bool(consistent)}


# --------------------------------------------------------------------------- #
# Wrapper                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class NicheResult:
    center: int
    center_barcode: str
    neighbors: pd.DataFrame
    lr: pd.DataFrame
    activity: pd.DataFrame
    functions: pd.DataFrame
    gradients: pd.DataFrame
    gradient_profiles: dict = field(default_factory=dict)


GRADIENT_COLUMNS = ["key", "ligand", "receptor", "direction", "n_cells", "L_slope", "L_rho", "L_p",
                    "R_slope", "R_rho", "R_p", "LR_corr", "pattern", "consistent_with_direction", "z", "fdr"]


def analyze_niche(adata, center: int, *, k: int = 20, center_pool: int = 0, lr: pd.DataFrame | None = None,
                  programs=PROGRAMS, layer=None, spatial_key="spatial", fdr: float = 0.1,
                  n_top_gradients: int = 15, n_line_cells: int = 6, um_per_unit: float | None = None,
                  n_background: int = 2000, bg_mask=None, seed: int = 0, **lr_kw) -> NicheResult:
    """
    Full pipeline for one center. `um_per_unit`: for Visium in full-res pixels use
    100 / spot_spacing(adata.obsm['spatial']) (100 um centre-to-centre).
    center_pool: pool the center with its nearest spots (6 = ring 1 on Visium) into a central
    region; partners are then the remaining k - center_pool neighbours.
    """
    nbrs = neighborhood(adata, center, k=k, spatial_key=spatial_key)
    lr_res = score_niche_lr(adata, center, nbrs, lr, layer=layer, spatial_key=spatial_key,
                            center_pool=center_pool, n_background=n_background, bg_mask=bg_mask,
                            seed=seed, **lr_kw)
    lr_res = annotate_lr_programs(lr_res, programs)
    act = niche_program_activity(adata, center, nbrs, programs, layer=layer, spatial_key=spatial_key,
                                 n_background=n_background, bg_mask=bg_mask, seed=seed)
    fun = infer_functions(lr_res, act, programs, fdr=fdr)

    grows, profiles = [], {}
    for _, r in lr_res[lr_res.fdr <= fdr].head(n_top_gradients).iterrows():
        line = collinear_cells(adata, center, target=int(r.best_partner_idx),
                               n_cells=n_line_cells, spatial_key=spatial_key)
        if line.empty:
            continue
        g = lr_gradient(adata, line, r.ligand, r.receptor, r.direction, layer, um_per_unit)
        key = f"{r.ligand}->{r.receptor}|{r.direction}"
        profiles[key] = g
        grows.append({k_: v for k_, v in g.items() if k_ not in ("distance", "L", "R", "barcodes", "unit")}
                     | {"key": key, "z": r.z, "fdr": r.fdr})
    grad = pd.DataFrame(grows, columns=GRADIENT_COLUMNS) if grows else pd.DataFrame(columns=GRADIENT_COLUMNS)
    return NicheResult(center, adata.obs_names[center], nbrs, lr_res, act, fun,
                       grad[GRADIENT_COLUMNS], profiles)


# --------------------------------------------------------------------------- #
# Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_niche(adata, res: NicheResult, gradient_keys=None, spatial_key="spatial",
               zoom: float = 6.0, ax=None):
    import matplotlib.pyplot as plt
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    sp = spot_spacing(coords)
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    c = coords[res.center]
    m = np.all(np.abs(coords - c) <= zoom * sp, axis=1)
    ax.scatter(*coords[m].T, s=40, c="#d9d9d9", lw=0)
    nb = res.neighbors
    sc = ax.scatter(*coords[nb.idx].T, s=60, c=nb.dist, cmap="viridis_r", lw=0)
    ax.scatter(*c, s=180, marker="*", c="crimson", zorder=5, label="center")
    for key in (gradient_keys or list(res.gradient_profiles)[:5]):
        b = res.gradient_profiles[key]["barcodes"]
        xy = coords[adata.obs_names.get_indexer(b)]
        ax.plot(*xy.T, lw=1.5, alpha=.8, label=key.split("|")[0])
    plt.colorbar(sc, ax=ax, shrink=.7, label="distance")
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.25, 1)); ax.set_title(f"niche of {res.center_barcode}")
    return ax


def plot_gradients(res: NicheResult, keys=None, ncols: int = 3):
    import matplotlib.pyplot as plt
    keys = keys or list(res.gradient_profiles)[:6]
    n = len(keys)
    fig, axs = plt.subplots(int(np.ceil(n / ncols)), ncols, figsize=(4 * ncols, 3 * np.ceil(n / ncols)),
                            squeeze=False)
    for ax, key in zip(axs.flat, keys):
        g = res.gradient_profiles[key]
        ax.plot(g["distance"], g["L"], "o-", c="#d95f02", label=f"L {g['ligand']}")
        ax.plot(g["distance"], g["R"], "s--", c="#1b9e77", label=f"R {g['receptor']}")
        ax.set_title(f"{key.split('|')[0]}\n{g['direction']} · {g['pattern']}", fontsize=9)
        ax.set_xlabel(f"distance from center ({g.get('unit', '')})"); ax.set_ylabel("log-expr"); ax.legend(fontsize=7)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Several niches / cross-check with LIANA bivariate                            #
# --------------------------------------------------------------------------- #
def scan_niches(adata, centers: dict, *, value: str = "score", exclude_own_group: str | None = None,
                **kw) -> tuple[pd.DataFrame, dict]:
    """
    Run analyze_niche for several centers, e.g. {'Hippocampus': idx, ...}.
    exclude_own_group: obs column; if given, each niche is compared with niches centred
    outside its own group (region vs. rest).
    Returns (programs x centers matrix of `value` from res.functions, {name: NicheResult}).
    """
    results, cols = {}, {}
    for name, c in centers.items():
        if exclude_own_group is not None:
            g = adata.obs[exclude_own_group].astype(str).values
            kw["bg_mask"] = g != g[int(c)]
        r = analyze_niche(adata, int(c), **kw)
        results[name] = r
        cols[name] = r.functions.set_index("program")[value]
    return pd.DataFrame(cols), results


def centers_by_group(adata, groupby: str, spatial_key: str = "spatial", min_spots: int = 10) -> dict:
    """One center per group, chosen in the group's interior: maximal fraction of same-group
    spots among the 20 nearest neighbours, not on the tissue edge (full first ring), and, among
    those, the one closest to the group's median position."""
    coords = np.asarray(adata.obsm[spatial_key])[:, :2]
    lab = np.asarray(adata.obs[groupby].astype(str), dtype=object)
    sp = spot_spacing(coords)
    tree = KDTree(coords)
    d, nn = tree.query(coords, k=21)
    purity = (lab[nn[:, 1:]] == lab[:, None]).mean(1)
    ring1 = (d[:, 1:] <= 1.2 * sp).sum(1)
    full = ring1 >= np.percentile(ring1, 90)            # interior spots have a complete first ring
    out = {}
    for g in pd.unique(lab):
        idx = np.where(lab == g)[0]
        if len(idx) < min_spots:
            continue
        score = purity[idx] + full[idx]
        best = idx[score >= score.max() - 1e-9]
        med = np.median(coords[idx], 0)
        out[g] = int(best[np.argmin(np.linalg.norm(coords[best] - med, axis=1))])
    return out


def liana_local_check(lrdata, res: NicheResult, n: int = 20) -> pd.DataFrame:
    """Look up the top niche LR pairs in a LIANA+ `li.mt.bivariate` object (spots x 'L^R').
    Reports the local score at the center, the mean over its neighbours, and the global
    rank of the pair (by Moran's R if present)."""
    top = res.lr.head(n)
    var = pd.Index(lrdata.var_names)
    X = lrdata.X.toarray() if sparse.issparse(lrdata.X) else np.asarray(lrdata.X)
    cidx = lrdata.obs_names.get_loc(res.center_barcode)
    nidx = lrdata.obs_names.get_indexer(res.neighbors.barcode)
    rank = None
    for col in ("morans", "global_mean", "morans_r"):
        if col in lrdata.var:
            rank = lrdata.var[col].rank(ascending=False)
            break
    rows = []
    for _, r in top.iterrows():
        key = f"{r.ligand}^{r.receptor}"
        if key in var:
            j = var.get_loc(key)
            rows.append({"lr": key, "direction": r.direction, "niche_z": r.z,
                         "liana_local_center": X[cidx, j], "liana_local_neighbors": X[nidx, j].mean(),
                         "liana_global_rank": None if rank is None else rank.iloc[j]})
        else:
            rows.append({"lr": key, "direction": r.direction, "niche_z": r.z,
                         "liana_local_center": np.nan, "liana_local_neighbors": np.nan, "liana_global_rank": np.nan})
    return pd.DataFrame(rows)

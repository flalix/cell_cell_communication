#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stat_lib2 -- consolidated statistics library (companion of the dr-spock skill)

@author: Flavio Lichtenstein
Created on 2010/06/10; rewritten 2026/11/23 (consolidation of cell_cell_communication,
neural_crest, gwas and perturb_agent statistical code)

@email:  flalix@gmail.com, flavio.lichtenstein@butantan.gov.br
@local:  Instituto Butantan / CENTD / Molecular Biology / Bioinformatics & Systems Biology

Conventions (every public function)
-----------------------------------
* Inputs are array-likes; NaN / inf are dropped (paired tests drop incomplete pairs).
* SD is the *sample* SD (ddof=1) everywhere.
* Results are plain ``dict`` (one test) or ``pd.DataFrame`` (many tests). Common keys:
  ``test, statistic, df, p, effect, effect_name, ci_low, ci_high, n, note``.
* ``report(res)`` turns a result dict into a one-line, journal-style sentence.
* A non-significant p means "H0 not rejected" -- never "H0 accepted".

Sections
--------
 0 helpers ............ clean, stars, fmt_p, report, groups_from_df
 1 descriptive ........ describe, describe_groups, iqr_outliers, remove_outliers,
                        ci_mean, ci_from_stats, standardize
 2 assumptions ........ normality, equal_variance
 3 two groups ......... compare_two, one_sample, ttest_from_stats, compare_two_matrix,
                        hedges_g
 4 k groups ........... compare_groups, welch_anova, posthoc (tukey, games_howell,
                        dunnett, dunn)
 5 categorical ........ contingency, crosstab_test, goodness_of_fit, mcnemar,
                        enrichment, enrichment_table
 6 correlation ........ correlation, corr_columns, linear_trend, ols
 7 multiple testing ... p_adjust, n_comparisons, lambda_gc, perm_pvalue
 8 modality ........... bimodality_coefficient, kde_modes, gmm_modality, kmeans_modes,
                        dip_test, modality, modality_table, split_high_low, assign_modes
 9 resampling ......... bootstrap_ci, permutation_two
10 power .............. power_ttest, sample_size_ttest, gwas_power, gwas_detectable_or
11 error propagation .. propagate_ratio, pool_measurements
12 test selection ..... choose_test
13 legacy aliases ..... fdr, fdr_df, chi2_or_fisher_exact_test, chisquare_2by2,
                        stat_asteristics, stat_list_asteristics

Old -> new map (removed functions)
----------------------------------
calc_ttest, ttest, ttest_2vals, ttest_2series, ttest_gender01_female_male  -> compare_two
ttest_1val -> one_sample;  ttest_mu_std(_calc) -> ttest_from_stats
calc_mannwhitneyu -> compare_two(test="mwu")
test_normality(_params/_desc/_desc_eng) -> normality;  test_one_way_ANOVA -> compare_groups
calc_stat_chi2(_mat), calc_fisher_exact_test, chisquare_gender01_female_male,
chisquare_2series -> contingency / crosstab_test
calc_params, calc_params_outliers(_new), calc_params_all_values, how_get_params_* -> describe /
describe_groups;  clean_df_outliers, filter_outlier(_groups) -> remove_outliers
calc_confidence_interval(_param) -> ci_mean / ci_from_stats;  calc_vc -> describe()["cv"]
get_dunnett_test(_old), scipy_dunnet, stat_dunnett_test -> posthoc(method="dunnett")
scipy_dunn(_ref), stat_dunn_test -> posthoc(method="dunn");  scipy_kruskal(_mat) -> compare_groups
results_summary_to_dataframe -> posthoc(method="tukey");  calc_ols_regression -> ols
propag_errors_mult_div_ref/_list -> propagate_ratio;  calc_normal_dist_errors -> pool_measurements
how_many_comparisons -> n_comparisons;  poisson -> scipy.stats.poisson.pmf
standardize_table, standardize_log2_x2_table -> standardize
project copies: _bh (x7) -> p_adjust; count_modes -> kde_modes; best_kmeans -> kmeans_modes;
axis_modality -> modality_table; discretize_axes -> split_high_low; analyse_gene -> assign_modes;
response_heterogeneity BC -> bimodality_coefficient; enrich / enrichment_test -> enrichment;
_ttest_matrix, deg_vs_normal core -> compare_two_matrix; _safe_spearman -> correlation;
gene_corr -> corr_columns; boot_median_gap -> bootstrap_ci; approx_power -> power_ttest;
lambda_gc, detectable_or, power_for (gwas) -> lambda_gc, gwas_detectable_or, gwas_power;
permutation p (1 + n_ge)/(1 + n_perm) -> perm_pvalue; harmonize_scores levene -> equal_variance
"""

from __future__ import annotations

import warnings
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# =============================================================================
# 0. helpers
# =============================================================================
def clean(x) -> np.ndarray:
    """1-D float array with NaN/inf removed."""
    try:
        a = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):                 # pd.NA, strings, mixed objects
        a = pd.to_numeric(pd.Series(np.ravel(np.asarray(x, dtype=object))), errors="coerce").to_numpy(float)
    return a[np.isfinite(a)]


def stars(p: float, alpha: float = ALPHA, ns: str = "NS") -> str:
    """NS / * / ** / *** / **** for p < alpha, alpha/5, alpha/50, alpha/500 (0.05 -> .01 .001 .0001).
    NaN or negative p -> 'NA'."""
    if p is None or not np.isfinite(p) or p < 0:
        return "NA"
    if p >= alpha:
        return ns
    if p >= alpha / 5:
        return "*"
    if p >= alpha / 50:
        return "**"
    if p >= alpha / 500:
        return "***"
    return "****"


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p = NA"
    if p < 1e-300:
        return "p < 1e-300"
    return f"p = {p:.2g}" if p < 0.001 else f"p = {p:.3f}"


def report(res: dict, digits: int = 3) -> str:
    """One-line summary: 'Welch t-test: t(8.0) = 4.47, p = 0.002 **; Hedges g = 2.54; diff = 2.00 [0.97, 3.03]; n = 5 vs 5'."""
    if not res:
        return ""
    f = lambda v: "NA" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}g}"
    s = res.get("test", "test")
    if res.get("statistic") is not None:
        df = res.get("df")
        dfs = "" if df is None else (f"({f(df)})" if np.isscalar(df) else f"({', '.join(f(d) for d in df)})")
        s += f": {res.get('stat_name', 'stat')}{dfs} = {f(res['statistic'])}"
    s += f", {fmt_p(res.get('p'))} {stars(res.get('p', np.nan))}"
    if res.get("effect") is not None:
        s += f"; {res.get('effect_name', 'effect')} = {f(res['effect'])}"
    if res.get("estimate") is not None:
        s += f"; {res.get('estimate_name', 'estimate')} = {f(res['estimate'])}"
        if res.get("ci_low") is not None:
            s += f" [{f(res['ci_low'])}, {f(res['ci_high'])}]"
    n = res.get("n")
    if n is not None:
        s += "; n = " + (" vs ".join(str(int(v)) for v in n) if np.ndim(n) else str(int(n)))
    if res.get("note"):
        s += f"  ({res['note']})"
    return s


def groups_from_df(df: pd.DataFrame, value: str, by: str, order=None):
    """-> (list of arrays, list of labels) from a long DataFrame."""
    labels = list(order) if order is not None else list(pd.unique(df[by].dropna()))
    return [clean(df.loc[df[by] == g, value]) for g in labels], labels


def _out(**kw) -> dict:
    base = dict(test=None, stat_name=None, statistic=None, df=None, p=np.nan, effect=None,
                effect_name=None, estimate=None, estimate_name=None, ci_low=None, ci_high=None,
                n=None, note="")
    base.update(kw)
    base["stars"] = stars(base["p"])
    return base


def _note(*msgs) -> str:
    return "; ".join(m for m in msgs if m)


# =============================================================================
# 1. descriptive
# =============================================================================
def iqr_outliers(x, k: float = 1.5) -> np.ndarray:
    """Boolean mask (same length as x): True = outside [Q1 - k*IQR, Q3 + k*IQR]. NaN -> False."""
    a = np.asarray(x, dtype=float)
    ok = np.isfinite(a)
    if ok.sum() < 4:
        return np.zeros(a.shape, bool)
    q1, q3 = np.percentile(a[ok], [25, 75])
    iqr = q3 - q1
    if iqr == 0:                                   # degenerate: do not flag anything
        return np.zeros(a.shape, bool)
    return ok & ((a < q1 - k * iqr) | (a > q3 + k * iqr))


def describe(x, ci: float = 0.95, remove_outliers: bool = False, k: float = 1.5) -> dict:
    """n, mean, sd, sem, t-CI of the mean, median, q1, q3, iqr, min, max, cv, skew, excess kurtosis.
    remove_outliers=True drops Tukey-fence outliers first (reported in n_removed)."""
    raw = np.asarray(x, dtype=float)
    a = clean(raw)
    n_nan = int(raw.size - a.size)
    n_removed = 0
    if remove_outliers:
        m = iqr_outliers(a, k)
        n_removed, a = int(m.sum()), a[~m]
    n = a.size
    d = dict(n=n, n_nan=n_nan, n_removed=n_removed)
    if n == 0:
        return d
    mean, sd = float(a.mean()), float(a.std(ddof=1)) if n > 1 else np.nan
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    sem = sd / np.sqrt(n) if n > 1 else np.nan
    lo, hi = (stats.t.interval(ci, n - 1, loc=mean, scale=sem) if n > 1 and sem > 0 else (np.nan, np.nan))
    d.update(mean=mean, sd=sd, sem=sem, ci_low=float(lo), ci_high=float(hi), median=float(med),
             q1=float(q1), q3=float(q3), iqr=float(q3 - q1), min=float(a.min()), max=float(a.max()),
             cv=sd / mean if mean != 0 and n > 1 else np.nan,
             skew=float(stats.skew(a, bias=False)) if n > 2 and sd > 0 else np.nan,
             excess_kurtosis=float(stats.kurtosis(a, bias=False)) if n > 3 and sd > 0 else np.nan)
    return d


def describe_groups(df: pd.DataFrame, value: str, by, ci: float = 0.95,
                    remove_outliers: bool = False, with_normality: bool = True) -> pd.DataFrame:
    """describe() per group (by = column or list of columns) + Shapiro p (descriptive only)."""
    rows = []
    for key, sub in df.groupby(by, sort=False, dropna=False):
        d = describe(sub[value], ci=ci, remove_outliers=remove_outliers)
        if with_normality:
            nr = normality(sub[value])
            d.update(normality_p=nr["p"], normal=nr["normal"])
        keys = key if isinstance(key, tuple) else (key,)
        cols = by if isinstance(by, (list, tuple)) else [by]
        rows.append({**dict(zip(cols, keys)), **d})
    return pd.DataFrame(rows)


def remove_outliers(df: pd.DataFrame, value: str, by=None, k: float = 1.5) -> pd.DataFrame:
    """Drop rows whose `value` is a Tukey-fence outlier (within each `by` group if given) or NaN."""
    if by is None:
        m = iqr_outliers(df[value].to_numpy(float), k)
    else:
        m = df.groupby(by, group_keys=False)[value].transform(lambda s: iqr_outliers(s.to_numpy(float), k)).astype(bool)
    return df.loc[~np.asarray(m) & df[value].notna()]


def ci_mean(x, ci: float = 0.95, two_sided: bool = True):
    """-> (mean, ci_low, ci_high) t-interval with sample SD."""
    a = clean(x)
    if a.size < 2:
        return (float(a.mean()) if a.size else np.nan), np.nan, np.nan
    lo, hi, _ = ci_from_stats(a.mean(), a.std(ddof=1), a.size, ci, two_sided)
    return float(a.mean()), lo, hi


def ci_from_stats(mean: float, sd: float, n: int, ci: float = 0.95, two_sided: bool = True):
    """-> (ci_low, ci_high, sem). One-sided returns the lower bound and +inf."""
    sem = sd / np.sqrt(n)
    if two_sided:
        e = stats.t.ppf(1 - (1 - ci) / 2, n - 1) * sem
        return float(mean - e), float(mean + e), float(sem)
    return float(mean - stats.t.ppf(ci, n - 1) * sem), np.inf, float(sem)


def standardize(df: pd.DataFrame, cols, log2: bool = False, square: bool = False) -> pd.DataFrame:
    """Add std_<col> (z-score, ddof=1); optionally log2_<col> (NaN for x <= 0) and x2_<col>."""
    out = df.copy()
    for c in [cols] if isinstance(cols, str) else cols:
        out["std_" + c] = (out[c] - out[c].mean()) / out[c].std()
        if log2:
            out["log2_" + c] = np.log2(out[c].where(out[c] > 0))
        if square:
            out["x2_" + c] = out[c] ** 2
    return out


# =============================================================================
# 2. assumption checks (descriptive diagnostics -- do NOT use them to pick the test)
# =============================================================================
def normality(x, alpha: float = ALPHA) -> dict:
    """Shapiro-Wilk (3 <= n <= 5000) or D'Agostino-Pearson (n > 5000).
    With large n tiny deviations are 'significant'; with small n nothing is. Prefer a QQ-plot."""
    a = clean(x)
    n = a.size
    if n < 3 or np.ptp(a) == 0:
        return _out(test="normality", p=np.nan, n=n, normal=False,
                    note="n < 3" if n < 3 else "constant values")
    if n <= 5000:
        r, name = stats.shapiro(a), "Shapiro-Wilk"
    else:
        r, name = stats.normaltest(a), "D'Agostino-Pearson"
    p = float(r.pvalue)
    return _out(test=name, stat_name="W" if n <= 5000 else "K2", statistic=float(r.statistic), p=p,
                n=n, normal=bool(p >= alpha))


def equal_variance(*groups, center: str = "median") -> dict:
    """Brown-Forsythe (center='median', robust) or Levene (center='mean'); also SD ratio max/min."""
    g = [clean(x) for x in groups]
    g = [x for x in g if x.size > 1]
    if len(g) < 2:
        return _out(test="equal variance", note="need >= 2 groups with n > 1")
    r = stats.levene(*g, center=center)
    sds = [x.std(ddof=1) for x in g]
    return _out(test="Brown-Forsythe" if center == "median" else "Levene", stat_name="W",
                statistic=float(r.statistic), p=float(r.pvalue), n=[x.size for x in g],
                effect=float(max(sds) / max(min(sds), 1e-300)), effect_name="SD ratio max/min")


# =============================================================================
# 3. two groups
# =============================================================================
def hedges_g(a, b) -> float:
    """Bias-corrected standardized mean difference (a - b) / pooled SD."""
    a, b = clean(a), clean(b)
    n1, n2 = a.size, b.size
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))) if sp > 0 else np.nan


def _min_p_rank(n1: int, n2: int | None = None, alternative: str = "two-sided") -> float:
    """Smallest attainable exact p of Mann-Whitney (n1, n2) or Wilcoxon signed-rank (n1 pairs)."""
    p = 1 / comb(n1 + n2, n1) if n2 is not None else 1 / 2 ** n1
    return min(1.0, 2 * p if alternative == "two-sided" else p)


def compare_two(a, b, test: str = "welch", alternative: str = "two-sided", ci: float = 0.95) -> dict:
    """Two-sample comparison of a vs b (effect sign = a - b).

    test : 'welch' (default; unequal variances), 'student' (pooled), 'mwu' (Mann-Whitney U),
           'paired_t', 'wilcoxon' (signed-rank; paired). Paired tests drop incomplete pairs.
    alternative : 'two-sided' | 'greater' (a > b) | 'less'.
    Effects: Welch/Student -> mean diff + CI, Hedges g; paired_t -> mean diff + CI, Cohen dz;
    MWU -> Hodges-Lehmann shift, rank-biserial r; Wilcoxon -> median diff, matched-pairs r.
    """
    test = test.lower()
    if test in ("paired_t", "wilcoxon"):
        a0, b0 = np.asarray(a, float), np.asarray(b, float)
        if a0.shape != b0.shape:
            raise ValueError("paired tests need equal-length a and b")
        ok = np.isfinite(a0) & np.isfinite(b0)
        a, b = a0[ok], b0[ok]
        d = a - b
        n = int(d.size)
        if n < 2:
            return _out(test=test, n=n, note="n < 2 pairs")
        if test == "paired_t":
            r = stats.ttest_rel(a, b, alternative=alternative)
            lo, hi = r.confidence_interval(ci)
            sd = d.std(ddof=1)
            return _out(test="paired t-test", stat_name="t", statistic=float(r.statistic), df=float(r.df),
                        p=float(r.pvalue), effect=float(d.mean() / sd) if sd > 0 else np.nan,
                        effect_name="Cohen dz", estimate=float(d.mean()), estimate_name="mean diff",
                        ci_low=float(lo), ci_high=float(hi), n=n)
        dn = d[d != 0]
        if dn.size == 0:
            return _out(test="Wilcoxon signed-rank", n=n, p=1.0, note="all differences are zero")
        r = stats.wilcoxon(a, b, alternative=alternative)
        rk = stats.rankdata(np.abs(dn))
        rb = float((rk[dn > 0].sum() - rk[dn < 0].sum()) / rk.sum())
        mp = _min_p_rank(dn.size, None, alternative)
        return _out(test="Wilcoxon signed-rank", stat_name="W", statistic=float(r.statistic), p=float(r.pvalue),
                    effect=rb, effect_name="rank-biserial r", estimate=float(np.median(d)),
                    estimate_name="median diff", n=n,
                    note=f"min attainable p = {mp:.3g} > {ALPHA}" if mp > ALPHA else "")

    a, b = clean(a), clean(b)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return _out(test=test, n=[n1, n2], note="each group needs n >= 2")
    if test in ("welch", "student"):
        if np.ptp(np.r_[a, b]) == 0:
            return _out(test=test, n=[n1, n2], note="all values identical")
        r = stats.ttest_ind(a, b, equal_var=(test == "student"), alternative=alternative)
        lo, hi = r.confidence_interval(ci)
        return _out(test="Welch t-test" if test == "welch" else "Student t-test", stat_name="t",
                    statistic=float(r.statistic), df=float(r.df), p=float(r.pvalue),
                    effect=hedges_g(a, b), effect_name="Hedges g", estimate=float(a.mean() - b.mean()),
                    estimate_name="mean diff", ci_low=float(lo), ci_high=float(hi), n=[n1, n2])
    if test == "mwu":
        if np.ptp(np.r_[a, b]) == 0:
            return _out(test="Mann-Whitney U", n=[n1, n2], p=1.0, note="all values identical")
        r = stats.mannwhitneyu(a, b, alternative=alternative)
        diffs = (a[:, None] - b[None, :]).ravel() if n1 * n2 <= 4_000_000 else \
            np.random.default_rng(0).choice(a, 2_000_000) - np.random.default_rng(1).choice(b, 2_000_000)
        mp = _min_p_rank(n1, n2, alternative)
        return _out(test="Mann-Whitney U", stat_name="U", statistic=float(r.statistic), p=float(r.pvalue),
                    effect=float(2 * r.statistic / (n1 * n2) - 1), effect_name="rank-biserial r",
                    estimate=float(np.median(diffs)), estimate_name="Hodges-Lehmann shift", n=[n1, n2],
                    note=f"min attainable p = {mp:.3g} > {ALPHA}" if mp > ALPHA else "")
    raise ValueError("test must be welch | student | mwu | paired_t | wilcoxon")


def one_sample(x, mu0: float = 0.0, test: str = "t", alternative: str = "two-sided", ci: float = 0.95) -> dict:
    """One-sample t-test (test='t') or Wilcoxon signed-rank vs mu0 (test='wilcoxon')."""
    a = clean(x)
    if test == "t":
        if a.size < 2 or np.ptp(a) == 0:
            return _out(test="one-sample t-test", n=a.size, note="n < 2 or constant")
        r = stats.ttest_1samp(a, mu0, alternative=alternative)
        lo, hi = r.confidence_interval(ci)
        sd = a.std(ddof=1)
        return _out(test="one-sample t-test", stat_name="t", statistic=float(r.statistic), df=float(r.df),
                    p=float(r.pvalue), effect=float((a.mean() - mu0) / sd), effect_name="Cohen d",
                    estimate=float(a.mean()), estimate_name="mean", ci_low=float(lo), ci_high=float(hi), n=a.size)
    return compare_two(a, np.full(a.size, mu0), test="wilcoxon", alternative=alternative) | {"test": "one-sample Wilcoxon"}


def ttest_from_stats(mean1, sd1, n1, mean2, sd2, n2, equal_var: bool = False,
                     alternative: str = "two-sided") -> dict:
    """t-test from summary statistics (sd = sample SD, ddof=1). Welch by default."""
    r = stats.ttest_ind_from_stats(mean1, sd1, n1, mean2, sd2, n2, equal_var=equal_var, alternative=alternative)
    if equal_var:
        df = n1 + n2 - 2
    else:
        v1, v2 = sd1 ** 2 / n1, sd2 ** 2 / n2
        df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return _out(test="Student t-test" if equal_var else "Welch t-test", stat_name="t",
                statistic=float(r.statistic), df=float(df), p=float(r.pvalue),
                estimate=float(mean1 - mean2), estimate_name="mean diff", n=[n1, n2])


def compare_two_matrix(A, B, test: str = "welch", adjust: str = "bh", log1p: bool = False,
                       min_n: int = 2) -> pd.DataFrame:
    """Vectorised per-feature test. A, B: DataFrames features x samples (same feature index);
    paired_t needs the same sample columns order. Returns diff (mean A - mean B), stat, p, fdr,
    mean_a, mean_b, n_a, n_b, frac_zero_a/b. Degenerate features get p = NaN (excluded from FDR)."""
    A, B = pd.DataFrame(A), pd.DataFrame(B)
    idx = A.index.intersection(B.index)
    a, b = A.loc[idx].to_numpy(float), B.loc[idx].to_numpy(float)
    za, zb = (a == 0).mean(1), (b == 0).mean(1)
    if log1p:
        a, b = np.log1p(a), np.log1p(b)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if test in ("welch", "student"):
            r = stats.ttest_ind(a, b, axis=1, equal_var=(test == "student"), nan_policy="omit")
        elif test == "paired_t":
            r = stats.ttest_rel(a, b, axis=1, nan_policy="omit")
        elif test == "mwu":
            r = stats.mannwhitneyu(a, b, axis=1, nan_policy="omit")
        else:
            raise ValueError("test must be welch | student | paired_t | mwu")
        stat, p = np.asarray(r.statistic, float), np.asarray(r.pvalue, float)
        na, nb = np.isfinite(a).sum(1), np.isfinite(b).sum(1)
        ma, mb = np.nanmean(a, 1), np.nanmean(b, 1)
    deg = (na < min_n) | (nb < min_n) | (np.nanmax(np.c_[a, b], 1) == np.nanmin(np.c_[a, b], 1))
    p[deg] = np.nan
    out = pd.DataFrame({"diff": ma - mb, "stat": stat, "p": p, "mean_a": ma, "mean_b": mb,
                        "n_a": na, "n_b": nb, "frac_zero_a": za, "frac_zero_b": zb}, index=idx)
    out.insert(3, "fdr", p_adjust(out["p"].to_numpy(), adjust))
    out.attrs.update(test=test, log1p=log1p)
    return out.sort_values("p")


# =============================================================================
# 4. k groups
# =============================================================================
def welch_anova(*groups) -> dict:
    """Welch's heteroscedastic one-way ANOVA."""
    g = [clean(x) for x in groups]
    k = len(g)
    n = np.array([x.size for x in g], float)
    m = np.array([x.mean() for x in g])
    v = np.array([x.var(ddof=1) for x in g])
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    tmp = ((1 - w / W) ** 2 / (n - 1)).sum()
    F = ((w * (m - mw) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp)
    df2 = (k ** 2 - 1) / (3 * tmp)
    return _out(test="Welch ANOVA", stat_name="F", statistic=float(F), df=(k - 1, float(df2)),
                p=float(stats.f.sf(F, k - 1, df2)), n=[int(x) for x in n])


def compare_groups(groups, test: str = "welch_anova", labels=None) -> dict:
    """Omnibus test for k >= 2 independent groups.
    test: 'welch_anova' (default), 'anova' (equal variances), 'kruskal'.
    Effects: eta^2 and omega^2 (ANOVA), epsilon^2 (Kruskal). Follow with posthoc()."""
    g = [clean(x) for x in groups]
    keep = [i for i, x in enumerate(g) if x.size >= 2]
    if len(keep) < 2:
        return _out(test=test, note="need >= 2 groups with n >= 2")
    note = "" if len(keep) == len(g) else f"dropped {len(g) - len(keep)} group(s) with n < 2"
    g = [g[i] for i in keep]
    allv = np.concatenate(g)
    N, k = allv.size, len(g)
    ssb = sum(x.size * (x.mean() - allv.mean()) ** 2 for x in g)
    sst = ((allv - allv.mean()) ** 2).sum()
    msw = (sst - ssb) / (N - k)
    if test == "kruskal":
        r = stats.kruskal(*g)
        res = _out(test="Kruskal-Wallis", stat_name="H", statistic=float(r.statistic), df=k - 1,
                   p=float(r.pvalue), effect=float(r.statistic / (N - 1)), effect_name="epsilon^2")
    elif test == "anova":
        r = stats.f_oneway(*g)
        res = _out(test="one-way ANOVA", stat_name="F", statistic=float(r.statistic), df=(k - 1, N - k),
                   p=float(r.pvalue))
    elif test == "welch_anova":
        res = welch_anova(*g)
    else:
        raise ValueError("test must be welch_anova | anova | kruskal")
    if test != "kruskal" and sst > 0:
        res.update(effect=float(ssb / sst), effect_name="eta^2",
                   omega2=float((ssb - (k - 1) * msw) / (sst + msw)))
    res.update(n=[x.size for x in g], note=_note(note, res.get("note")),
               labels=[labels[i] for i in keep] if labels is not None else keep)
    return res


def posthoc(groups, method: str = "tukey", labels=None, control=0, p_adjust_method: str = "holm",
            ci: float = 0.95) -> pd.DataFrame:
    """Pairwise post-hoc tests -> long DataFrame (group1, group2, diff, stat, p_adj, ci_low, ci_high, stars).

    method: 'tukey' (ANOVA, equal var), 'games_howell' (Welch ANOVA), 'dunnett' (each vs `control`,
            normal), 'dunn' (Kruskal follow-up, rank z with tie correction; p adjusted by
            `p_adjust_method`), 'dunn_control' (Dunn, only vs control).
    control: index or label of the control group. diff = group1 - group2 (group2 = control for Dunnett).
    """
    g = [clean(x) for x in groups]
    labels = list(labels) if labels is not None else list(range(len(g)))
    ci_idx = labels.index(control) if control in labels else int(control)
    rows = []
    if method == "tukey":
        r = stats.tukey_hsd(*g)
        c = r.confidence_interval(ci)
        for i, j in combinations(range(len(g)), 2):
            rows.append((labels[i], labels[j], g[i].mean() - g[j].mean(), r.statistic[i, j], r.pvalue[i, j],
                         c.low[i, j], c.high[i, j]))
    elif method == "games_howell":
        k = len(g)
        for i, j in combinations(range(k), 2):
            vi, vj = g[i].var(ddof=1) / g[i].size, g[j].var(ddof=1) / g[j].size
            se, diff = np.sqrt(vi + vj), g[i].mean() - g[j].mean()
            df = (vi + vj) ** 2 / (vi ** 2 / (g[i].size - 1) + vj ** 2 / (g[j].size - 1))
            t = diff / se
            p = stats.studentized_range.sf(abs(t) * np.sqrt(2), k, df)
            q = stats.studentized_range.ppf(ci, k, df) / np.sqrt(2) * se
            rows.append((labels[i], labels[j], diff, t, min(p, 1.0), diff - q, diff + q))
    elif method == "dunnett":
        others = [i for i in range(len(g)) if i != ci_idx]
        r = stats.dunnett(*[g[i] for i in others], control=g[ci_idx], random_state=0)  # QMC: seeded
        c = r.confidence_interval(ci)
        for n_, i in enumerate(others):
            rows.append((labels[i], labels[ci_idx], g[i].mean() - g[ci_idx].mean(), r.statistic[n_],
                         r.pvalue[n_], c.low[n_], c.high[n_]))
    elif method in ("dunn", "dunn_control"):
        allv = np.concatenate(g)
        N = allv.size
        rk = stats.rankdata(allv)
        _, t = np.unique(allv, return_counts=True)
        s2 = N * (N + 1) / 12 - (t ** 3 - t).sum() / (12 * (N - 1))
        cuts = np.cumsum([0] + [x.size for x in g])
        mr = [rk[cuts[i]:cuts[i + 1]].mean() for i in range(len(g))]
        pairs = ([(i, ci_idx) for i in range(len(g)) if i != ci_idx] if method == "dunn_control"
                 else list(combinations(range(len(g)), 2)))
        raw = []
        for i, j in pairs:
            z = (mr[i] - mr[j]) / np.sqrt(s2 * (1 / g[i].size + 1 / g[j].size))
            raw.append((labels[i], labels[j], np.median(g[i]) - np.median(g[j]), z, 2 * stats.norm.sf(abs(z))))
        padj = p_adjust([r_[4] for r_ in raw], p_adjust_method)
        rows = [(*r_[:4], pa, np.nan, np.nan) for r_, pa in zip(raw, padj)]
    else:
        raise ValueError("method must be tukey | games_howell | dunnett | dunn | dunn_control")
    df = pd.DataFrame(rows, columns=["group1", "group2", "diff", "stat", "p_adj", "ci_low", "ci_high"])
    df["stars"] = [stars(p) for p in df.p_adj]
    df.attrs["method"] = method
    return df


# =============================================================================
# 5. categorical
# =============================================================================
def contingency(table, test: str = "auto", correction: bool = True, n_mc: int = 0, seed: int = 0) -> dict:
    """Association in an r x c table of counts.

    test='auto': 2x2 with any expected count < 5 -> Fisher exact; otherwise chi-square (Yates
    correction for 2x2 if `correction`). Cochran rule checked for r x c (no E < 1, <= 20% E < 5);
    if violated, a note is added and, with n_mc > 0, a Monte-Carlo permutation p is returned.
    Effects: Cramer's V; for 2x2 also odds ratio with 95% CI.
    """
    T = np.asarray(table, dtype=float)
    T = T[T.sum(1) > 0][:, T.sum(0) > 0]                      # drop empty rows / cols
    if T.ndim != 2 or min(T.shape) < 2:
        return _out(test="contingency", note="table needs >= 2 non-empty rows and columns")
    E = stats.contingency.expected_freq(T)
    is22 = T.shape == (2, 2)
    cochran_ok = (E.min() >= 1) and ((E < 5).mean() <= 0.2)
    use_fisher = test == "fisher" or (test == "auto" and is22 and E.min() < 5)
    V = float(stats.contingency.association(T.astype(int), method="cramer")) if T.sum() > 0 else np.nan
    extra = dict(expected_min=float(E.min()), cramers_v=V, table=T.astype(int))
    if is22:
        orr = stats.contingency.odds_ratio(T.astype(int), kind="sample")
        oc = orr.confidence_interval(0.95)
        extra.update(estimate=float(orr.statistic), estimate_name="odds ratio",
                     ci_low=float(oc.low), ci_high=float(oc.high))
    if use_fisher:
        if not is22:
            raise ValueError("Fisher exact here is 2x2 only; use n_mc > 0 for r x c")
        r = stats.fisher_exact(T.astype(int))
        return _out(test="Fisher exact", p=float(r.pvalue), n=int(T.sum()), effect=V,
                    effect_name="Cramer's V", **extra)
    r = stats.chi2_contingency(T, correction=correction)
    res = _out(test="chi-square" + (" (Yates)" if is22 and correction else ""), stat_name="chi2",
               statistic=float(r.statistic), df=int(r.dof), p=float(r.pvalue), n=int(T.sum()),
               effect=V, effect_name="Cramer's V", **extra)
    if not cochran_ok:
        res["note"] = "Cochran rule violated (small expected counts)"
        if n_mc > 0:
            rng = np.random.default_rng(seed)
            rows = np.repeat(np.arange(T.shape[0]), T.sum(1).astype(int))
            cols = np.repeat(np.arange(T.shape[1]), T.sum(0).astype(int))
            obs = stats.chi2_contingency(T, correction=False).statistic
            ge = 0
            for _ in range(n_mc):
                Tp = np.zeros_like(T)
                np.add.at(Tp, (rows, rng.permutation(cols)), 1)
                ge += ((Tp - E) ** 2 / E).sum() >= obs - 1e-12
            res.update(p=perm_pvalue_count(ge, n_mc), test="chi-square (Monte-Carlo)",
                       note=f"Monte-Carlo p, {n_mc} permutations")
            res["stars"] = stars(res["p"])
    return res


def crosstab_test(df: pd.DataFrame, row: str, col: str, **kw) -> dict:
    """contingency() on pd.crosstab(df[row], df[col]); the crosstab is returned in res['crosstab']."""
    ct = pd.crosstab(df[row], df[col])
    res = contingency(ct.to_numpy(), **kw)
    res["crosstab"] = ct
    return res


def goodness_of_fit(observed, expected_probs=None) -> dict:
    """Chi-square goodness of fit (uniform if expected_probs is None)."""
    o = np.asarray(observed, float)
    e = None if expected_probs is None else np.asarray(expected_probs, float) / np.sum(expected_probs) * o.sum()
    r = stats.chisquare(o, e)
    return _out(test="chi-square GOF", stat_name="chi2", statistic=float(r.statistic), df=o.size - 1,
                p=float(r.pvalue), n=int(o.sum()))


def mcnemar(table) -> dict:
    """Exact McNemar test for a paired 2x2 table [[a, b], [c, d]] (binomial on discordant b, c)."""
    T = np.asarray(table, int)
    b, c = int(T[0, 1]), int(T[1, 0])
    p = float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0
    return _out(test="McNemar (exact)", stat_name="b", statistic=b, p=p, n=int(T.sum()),
                estimate=b / c if c > 0 else np.inf, estimate_name="discordant ratio b/c")


def enrichment(hits, members, background) -> dict:
    """One-sided over-representation of `members` among `hits` (hypergeometric = Fisher 'greater').
    All sets are intersected with `background` first; empty inputs give NaN, not 'no enrichment'."""
    bg = set(background)
    hits, members = set(hits) & bg, set(members) & bg
    M, n, N = len(bg), len(members), len(hits)
    ov = hits & members
    k = len(ov)
    if min(M, n, N) == 0:
        return dict(k=k, N=N, n=n, M=M, expected=np.nan, fold=np.nan, odds=np.nan, p=np.nan, overlap=ov)
    odds = stats.contingency.odds_ratio([[k, N - k], [n - k, M - n - N + k]], kind="sample").statistic
    return dict(k=k, N=N, n=n, M=M, expected=N * n / M, fold=(k / N) / (n / M), odds=float(odds),
                p=float(stats.hypergeom.sf(k - 1, M, n, N)), overlap=ov)


def enrichment_table(hits, gene_sets: dict, background, adjust: str = "bh") -> pd.DataFrame:
    """enrichment() for every set in a {name: members} dict, with FDR."""
    rows = []
    for name, mem in gene_sets.items():
        e = enrichment(hits, mem, background)
        e["overlap"] = ";".join(sorted(map(str, e["overlap"])))
        rows.append({"set": name, **e})
    df = pd.DataFrame(rows)
    df["fdr"] = p_adjust(df["p"].to_numpy(), adjust)
    return df.sort_values("p")


# =============================================================================
# 6. correlation / regression
# =============================================================================
def correlation(x, y, method: str = "spearman", min_n: int = 4, ci: float = 0.95) -> dict:
    """Pearson / Spearman / Kendall on complete pairs; constant input -> NaN (not 0).
    CI by Fisher z (Spearman: Bonett-Wright SE)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(m.sum())
    if n < min_n or np.ptp(x) == 0 or np.ptp(y) == 0:
        return _out(test=method, n=n, effect=np.nan, effect_name="r", note="n < min_n or constant input")
    f = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    r = f(x, y)
    rho = float(r.statistic)
    lo = hi = np.nan
    if n > 3 and method != "kendall" and abs(rho) < 1:
        se = (np.sqrt(1 + rho ** 2 / 2) if method == "spearman" else 1.0) / np.sqrt(n - 3)
        zc = stats.norm.ppf(1 - (1 - ci) / 2)
        lo, hi = np.tanh(np.arctanh(rho) - zc * se), np.tanh(np.arctanh(rho) + zc * se)
    return _out(test=f"{method.capitalize()} correlation", stat_name="r", statistic=rho, p=float(r.pvalue),
                effect=rho, effect_name="r", estimate=rho, estimate_name="r", ci_low=float(lo),
                ci_high=float(hi), n=n)


def corr_columns(A: pd.DataFrame, B: pd.DataFrame, method: str = "pearson", min_n: int = 3) -> pd.DataFrame:
    """Per-column correlation between two samples x features matrices, aligned on shared rows/cols.
    Columns with zero variance or any missing value get NaN (not 0)."""
    idx, cols = A.index.intersection(B.index), A.columns.intersection(B.columns)
    if len(idx) < min_n:
        raise ValueError(f"only {len(idx)} shared rows (need >= {min_n})")
    if len(cols) == 0:
        raise ValueError("no shared columns -- check ID space (symbol vs Ensembl)")
    a, b = A.loc[idx, cols].astype(float), B.loc[idx, cols].astype(float)
    if method == "spearman":
        a, b = a.rank(), b.rank()
    sa, sb = a.std(), b.std()
    ok = (sa > 0) & (sb > 0) & a.notna().all() & b.notna().all()
    r = ((a - a.mean()) / sa.where(sa > 0) * (b - b.mean()) / sb.where(sb > 0)).sum() / (len(idx) - 1)
    r = r.where(ok)
    n = len(idx)
    t = r * np.sqrt((n - 2) / (1 - r.clip(-0.999999999, 0.999999999) ** 2))
    p = 2 * stats.t.sf(np.abs(t), n - 2)
    out = pd.DataFrame({"r": r, "p": p, "n": n})
    out["fdr"] = p_adjust(out["p"].to_numpy())
    return out


def linear_trend(x, y) -> dict:
    """OLS slope/intercept + Pearson r/p (linregress) and Spearman rho/p; constant y -> slope 0, p 1."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if m.sum() < 3 or np.ptp(y) == 0 or np.ptp(x) == 0:
        return dict(slope=0.0, intercept=float(np.mean(y)) if y.size else np.nan, r=0.0, p=1.0,
                    rho=0.0, p_rho=1.0, n=int(m.sum()))
    lr = stats.linregress(x, y)
    sp = stats.spearmanr(x, y)
    return dict(slope=float(lr.slope), intercept=float(lr.intercept), slope_se=float(lr.stderr),
                r=float(lr.rvalue), p=float(lr.pvalue), rho=float(sp.statistic), p_rho=float(sp.pvalue),
                n=int(m.sum()))


def ols(df: pd.DataFrame, formula: str, robust: str | None = None) -> dict:
    """statsmodels OLS via formula ('y ~ x + C(group)'). robust='HC3' for heteroscedasticity-robust SE.
    Returns r2, adj_r2, F, p_F, aic, bic, n, residual normality p, coef DataFrame and the fit."""
    from statsmodels.formula.api import ols as smols
    fit = smols(formula, data=df).fit(cov_type=robust or "nonrobust")
    ci = fit.conf_int()
    coef = pd.DataFrame({"coef": fit.params, "se": fit.bse, "t": fit.tvalues, "p": fit.pvalues,
                         "ci_low": ci[0], "ci_high": ci[1]})
    return dict(r2=float(fit.rsquared), adj_r2=float(fit.rsquared_adj), F=float(fit.fvalue),
                p_F=float(fit.f_pvalue), aic=float(fit.aic), bic=float(fit.bic), n=int(fit.nobs),
                resid_normality_p=normality(fit.resid)["p"], coef=coef, fit=fit)


# =============================================================================
# 7. multiple testing
# =============================================================================
def p_adjust(p, method: str = "bh") -> np.ndarray:
    """Adjusted p-values; NaN stay NaN and are not counted in m.
    method: 'bh' (Benjamini-Hochberg FDR), 'by' (Benjamini-Yekutieli, any dependence),
            'holm' (FWER, step-down), 'bonferroni'."""
    p = np.asarray(p, dtype=float).copy()
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = pv.size
    if m == 0:
        return out
    o = np.argsort(pv)
    method = method.lower()
    if method in ("bh", "fdr", "fdr_bh", "by", "fdr_by"):
        q = pv[o] * m / np.arange(1, m + 1)
        if method in ("by", "fdr_by"):
            q *= (1 / np.arange(1, m + 1)).sum()
        q = np.minimum.accumulate(q[::-1])[::-1]
    elif method == "holm":
        q = np.maximum.accumulate(pv[o] * (m - np.arange(m)))
    elif method == "bonferroni":
        q = pv[o] * m
    else:
        raise ValueError("method must be bh | by | holm | bonferroni")
    tmp = np.empty(m)
    tmp[o] = np.clip(q, 0, 1)
    out[ok] = tmp
    return out


def n_comparisons(k: int) -> int:
    """Number of pairwise comparisons among k groups."""
    return k * (k - 1) // 2


def lambda_gc(p) -> float:
    """Genomic inflation factor: median observed chi2(1) / 0.4549. ~1 = no inflation."""
    p = clean(p)
    chi2 = stats.chi2.isf(np.clip(p, 1e-300, 1), df=1)
    return float(np.median(chi2) / stats.chi2.ppf(0.5, df=1))


def perm_pvalue_count(n_extreme: int, n_perm: int) -> float:
    """(1 + #null >= observed) / (1 + n_perm) -- never exactly 0 (Phipson & Smyth 2010)."""
    return (1 + n_extreme) / (1 + n_perm)


def perm_pvalue(observed: float, null, alternative: str = "greater") -> float:
    """Permutation p from a null sample; alternative 'greater' | 'less' | 'two-sided' (|x|)."""
    null = clean(null)
    if alternative == "greater":
        k = (null >= observed).sum()
    elif alternative == "less":
        k = (null <= observed).sum()
    else:
        k = (np.abs(null) >= abs(observed)).sum()
    return perm_pvalue_count(int(k), null.size)


# =============================================================================
# 8. modality / bimodality
# =============================================================================
def bimodality_coefficient(x) -> float:
    """Sarle's BC = (g1^2 + 1) / (g2 + 3(n-1)^2/((n-2)(n-3))) with bias-corrected skew g1 and excess
    kurtosis g2. BC > 5/9 (0.555, uniform) hints at bi/multimodality; heavy skew alone can exceed it."""
    a = clean(x)
    n = a.size
    if n < 4 or np.ptp(a) == 0:
        return np.nan
    g1, g2 = stats.skew(a, bias=False), stats.kurtosis(a, bias=False)
    return float((g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def kde_modes(x, grid: int = 512, min_prominence: float = 0.05, bw=None) -> dict:
    """Number of Gaussian-KDE peaks whose normalised prominence >= min_prominence.
    Returns n_modes, mode_x (locations), xs, density (max-normalised)."""
    from scipy.signal import find_peaks
    a = clean(x)
    if a.size < 5 or np.ptp(a) == 0:
        return dict(n_modes=1 if a.size else 0, mode_x=np.array([a.mean()]) if a.size else np.array([]),
                    xs=None, density=None)
    xs = np.linspace(a.min(), a.max(), grid)
    dens = stats.gaussian_kde(a, bw_method=bw)(xs)
    dens /= dens.max()
    # pad so that peaks at the edges are detectable
    pk, _ = find_peaks(np.r_[0, dens, 0], prominence=min_prominence)
    pk = pk - 1
    return dict(n_modes=len(pk), mode_x=xs[pk], xs=xs, density=dens)


def gmm_modality(x, min_weight: float = 0.15, seed: int = 0) -> dict:
    """1- vs 2-component Gaussian mixture. delta_bic = BIC1 - BIC2 (> 10 strong for 2 comps).
    Ashman D = |mu1 - mu2| / sqrt((s1^2 + s2^2)/2): D > 2 needed for two visibly separate modes
    (an equal mixture with D <= 2 is unimodal). Also returns the 2-comp threshold & labels."""
    from sklearn.mixture import GaussianMixture
    a = clean(x)
    if a.size < 20 or np.ptp(a) == 0:
        return dict(n=a.size, delta_bic=np.nan, verdict="too few samples" if a.size < 20 else "constant")
    X = a.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=seed).fit(X)
    g2 = GaussianMixture(2, random_state=seed, n_init=5).fit(X)
    mu, sd, w = g2.means_.ravel(), np.sqrt(g2.covariances_.ravel()), g2.weights_
    o = np.argsort(mu)
    mu, sd, w = mu[o], sd[o], w[o]
    D = float(abs(mu[1] - mu[0]) / np.sqrt((sd[0] ** 2 + sd[1] ** 2) / 2))
    lab = (g2.predict(X) == o[1]).astype(int)                 # 1 = high component
    thr = float((a[lab == 0].max() + a[lab == 1].min()) / 2) if lab.min() == 0 and lab.max() == 1 else np.nan
    dbic = float(g1.bic(X) - g2.bic(X))
    verdict = ("bimodal" if dbic > 10 and w.min() >= min_weight and D > 2 else
               "shoulder / weak" if dbic > 2 and w.min() >= min_weight else
               "outlier-driven" if dbic > 2 else "unimodal")
    return dict(n=a.size, delta_bic=dbic, means=mu, sds=sd, weights=w, min_weight=float(w.min()),
                ashman_d=D, threshold=thr, labels=lab, verdict=verdict)


def kmeans_modes(x, ks=(2, 3, 4, 5), sil_threshold: float = 0.65, seed: int = 0):
    """1-D k-means; best k by silhouette, k = 1 if no k reaches sil_threshold. -> (k, silhouette, labels)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    a = clean(x)
    X = a.reshape(-1, 1)
    best_k, best_s, best_lab = 1, np.nan, np.zeros(a.size, int)
    for k in ks:
        if k >= a.size:
            continue
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(X, lab)
        if np.isnan(best_s) or s > best_s:
            best_k, best_s, best_lab = k, s, lab
    if best_k > 1 and best_s < sil_threshold:
        best_k, best_lab = 1, np.zeros(a.size, int)
    return best_k, float(best_s), best_lab


def dip_test(x) -> dict:
    """Hartigan's dip test of unimodality (needs `pip install diptest`); p < alpha rejects unimodality."""
    try:
        import diptest
    except ImportError:
        return dict(dip=np.nan, p=np.nan, note="pip install diptest")
    d, p = diptest.diptest(clean(x))
    return dict(dip=float(d), p=float(p), note="")


def modality(x, floor=None, floor_frac_max: float = 0.30, min_weight: float = 0.15, seed: int = 0) -> dict:
    """Combined verdict: is x one cloud or two?

    Evidence: Hartigan dip (if installed), KDE peak count, Sarle BC, 2-GMM delta_bic + Ashman D.
    `floor` (e.g. 0 for counts / psi / expression) -> frac_at_floor; when > floor_frac_max the
    verdict is flagged 'floor-inflated': truncation at a detection floor manufactures apparent
    bimodality (simulated dBIC > 1800 from structureless data), so do not split on it.
    """
    a = clean(x)
    fl = a.min() if floor is None and a.size else floor
    frac_floor = float(np.mean(a <= fl)) if a.size else np.nan
    g = gmm_modality(a, min_weight=min_weight, seed=seed)
    kd = kde_modes(a)
    bc = bimodality_coefficient(a)
    dp = dip_test(a)
    votes = [g.get("verdict") == "bimodal", kd["n_modes"] >= 2, bool(bc > 5 / 9) if np.isfinite(bc) else False]
    if np.isfinite(dp["p"]):
        votes.append(dp["p"] < ALPHA)
    if a.size < 20:
        verdict = "too few samples"
    elif sum(votes) >= max(2, len(votes) - 1):
        verdict = "bimodal"
    elif sum(votes) >= 1:
        verdict = "ambiguous"
    else:
        verdict = "unimodal"
    if frac_floor > floor_frac_max and verdict != "unimodal":
        verdict += " (floor-inflated: likely truncation artefact)"
    return dict(n=a.size, verdict=verdict, votes=int(sum(votes)), n_votes=len(votes), dip_p=dp["p"],
                n_kde_modes=kd["n_modes"], kde_mode_x=kd["mode_x"], bc=bc, delta_bic=g.get("delta_bic"),
                ashman_d=g.get("ashman_d"), min_weight=g.get("min_weight"), gmm_verdict=g.get("verdict"),
                threshold=g.get("threshold"), frac_at_floor=frac_floor)


def modality_table(df: pd.DataFrame, cols=None, **kw) -> pd.DataFrame:
    """modality() for each column of df (e.g. program/axis scores, genes)."""
    cols = list(df.columns) if cols is None else cols
    rows = []
    for c in cols:
        r = modality(df[c], **kw)
        r.pop("kde_mode_x", None)
        rows.append({"feature": c, **r})
    return pd.DataFrame(rows)


def split_high_low(x, method: str = "median", min_frac: float = 0.20, seed: int = 0) -> pd.Series:
    """'high'/'low' labels. method='median' (balanced) or 'gmm' (warns if the smaller component
    < min_frac -> the axis is unimodal and a hard cut invents a group). NaN -> NaN."""
    s = pd.Series(x) if not isinstance(x, pd.Series) else x
    v = s.dropna()
    if method == "median":
        lab = np.where(v > v.median(), "high", "low")
    elif method == "gmm":
        g = gmm_modality(v.to_numpy(), seed=seed)
        if "labels" not in g:
            raise ValueError(f"gmm split impossible: {g.get('verdict')}")
        lab = np.where(g["labels"] == 1, "high", "low")
        frac = min((lab == "high").mean(), (lab == "low").mean())
        if frac < min_frac or g["verdict"] != "bimodal":
            warnings.warn(f"GMM split: smaller group {frac:.0%}, verdict '{g['verdict']}' -- prefer the continuous score")
    else:
        raise ValueError("method must be median | gmm")
    return pd.Series(lab, index=v.index).reindex(s.index)


def assign_modes(x, central: str = "zero", **kw) -> pd.DataFrame:
    """Split x into k-means modes (kmeans_modes) and label them relative to a central mode:
    offset 0 = central ('zero': mean closest to 0; 'largest': most populated), <0 down, >0 up.
    Returns one row per mode: offset, mode_type, mean, median, sd, n, members (index labels)."""
    s = pd.Series(x).dropna() if not isinstance(x, pd.Series) else x.dropna()
    k, sil, lab = kmeans_modes(s.to_numpy(float), **kw)
    modes = sorted([(s[lab == m]) for m in np.unique(lab)], key=lambda v: v.mean())
    c = int(np.argmax([len(m) for m in modes])) if central == "largest" else int(np.argmin([abs(m.mean()) for m in modes]))
    rows = []
    for i, m in enumerate(modes):
        off = i - c
        rows.append(dict(k=k, silhouette=sil, offset=off,
                         mode_type="central" if off == 0 else ("down" if off < 0 else "up"),
                         mean=m.mean(), median=m.median(), sd=m.std(), n=len(m), members=list(m.index)))
    return pd.DataFrame(rows)


# =============================================================================
# 9. resampling
# =============================================================================
def bootstrap_ci(a, b=None, stat=np.median, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap CI of stat(a) or of stat(a) - stat(b) (vectorised; stat must accept axis=)."""
    rng = np.random.default_rng(seed)
    a = clean(a)
    ia = rng.integers(0, a.size, (n_boot, a.size))
    bs = stat(a[ia], axis=1)
    est = stat(a)
    if b is not None:
        b = clean(b)
        ib = rng.integers(0, b.size, (n_boot, b.size))
        bs = bs - stat(b[ib], axis=1)
        est = est - stat(b)
    lo, hi = np.percentile(bs, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return dict(estimate=float(est), ci_low=float(lo), ci_high=float(hi), n_boot=n_boot,
                n=[a.size, b.size] if b is not None else a.size)


def permutation_two(a, b, stat: str = "mean", n_perm: int = 9999, alternative: str = "two-sided",
                    seed: int = 0) -> dict:
    """Two-sample permutation test of stat(a) - stat(b) (stat 'mean' or 'median')."""
    f = np.mean if stat == "mean" else np.median
    r = stats.permutation_test((clean(a), clean(b)), lambda x, y, axis: f(x, axis=axis) - f(y, axis=axis),
                               n_resamples=n_perm, alternative=alternative, random_state=seed, vectorized=True)
    return _out(test=f"permutation ({stat} diff)", stat_name="diff", statistic=float(r.statistic),
                p=float(r.pvalue), n=[clean(a).size, clean(b).size])


# =============================================================================
# 10. power
# =============================================================================
def power_ttest(d: float, n1: int, n2: int | None = None, alpha: float = ALPHA,
                alternative: str = "two-sided") -> float:
    """Exact power of the two-sample t-test (noncentral t) for standardized effect d.
    For log2FC: d = log2fc / sd."""
    n2 = n1 if n2 is None else n2
    df = n1 + n2 - 2
    nc = d * np.sqrt(n1 * n2 / (n1 + n2))
    if alternative == "two-sided":
        tc = stats.t.ppf(1 - alpha / 2, df)
        return float(stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc))
    tc = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(tc, df, abs(nc)))


def sample_size_ttest(d: float, power: float = 0.80, alpha: float = ALPHA, ratio: float = 1.0,
                      alternative: str = "two-sided", n_max: int = 100000) -> int:
    """Smallest n1 (n2 = ceil(ratio * n1)) reaching `power`."""
    for n1 in range(2, n_max):
        if power_ttest(d, n1, int(np.ceil(ratio * n1)), alpha, alternative) >= power:
            return n1
    return n_max


def gwas_power(odds_ratio: float, maf: float, n: int, case_frac: float, alpha: float = 5e-8) -> float:
    """Power of a per-allele logistic (Wald) test; asymptotic, optimistic for rare variants."""
    var_beta = 1 / (2 * n * maf * (1 - maf) * case_frac * (1 - case_frac))
    z = abs(np.log(odds_ratio)) / np.sqrt(var_beta)
    return float(stats.norm.sf(stats.norm.isf(alpha / 2) - z))


def gwas_detectable_or(maf: float, n: int, case_frac: float, alpha: float = 5e-8, power: float = 0.80) -> float:
    """Per-allele OR detectable at `power` (inverse of gwas_power)."""
    var_beta = 1 / (2 * n * maf * (1 - maf) * case_frac * (1 - case_frac))
    return float(np.exp((stats.norm.isf(alpha / 2) + stats.norm.ppf(power)) * np.sqrt(var_beta)))


# =============================================================================
# 11. error propagation
# =============================================================================
def propagate_ratio(mu_a: float, sd_a: float, mu_b: float, sd_b: float, n: int | None = None,
                    ci: float = 0.95) -> dict:
    """ratio = a / b with relative errors in quadrature (first-order, independent a and b).
    Also valid for products. With n, returns SEM and t-based CI half-width."""
    ratio = mu_a / mu_b
    sd = abs(ratio) * np.sqrt((sd_a / mu_a) ** 2 + (sd_b / mu_b) ** 2)
    out = dict(ratio=float(ratio), sd=float(sd))
    if n is not None and n > 1:
        sem = sd / np.sqrt(n)
        out.update(sem=float(sem), error=float(stats.t.ppf(1 - (1 - ci) / 2, n - 1) * sem))
    return out


def pool_measurements(means, sds, ci: float = 0.95) -> dict:
    """Combine n replicate measurements each with its own SD: mean of means, RMS SD
    (sqrt(mean variance)), SEM and t-CI half-width, plus Shapiro p of the means."""
    m, s = np.asarray(means, float), np.asarray(sds, float)
    ok = np.isfinite(m) & np.isfinite(s)
    m, s = m[ok], s[ok]
    n = m.size
    if n < 2:
        return dict(n=n)
    sd = float(np.sqrt(np.mean(s ** 2)))
    sem = sd / np.sqrt(n)
    return dict(n=n, mean=float(m.mean()), median=float(np.median(m)), sd=sd, sem=sem,
                error=float(stats.t.ppf(1 - (1 - ci) / 2, n - 1) * sem),
                cv=sd / m.mean() if m.mean() != 0 else np.nan, normality_p=normality(m)["p"])


# =============================================================================
# 12. test selection (Pandey flow chart, corrected)
# =============================================================================
def choose_test(outcome: str = "continuous", groups: int | str = 2, paired: bool = False,
                skewed: bool = False, n_per_group: int | None = None, control: bool = False,
                many_features: bool = False) -> dict:
    """Recommend a test from the study design (NOT from a normality pre-test of the same data).

    outcome : 'continuous' | 'ordinal' | 'categorical' | 'survival'
    groups  : 1, 2, k (int > 2) or 'continuous' (continuous predictor)
    paired  : repeated / matched measurements
    skewed  : prior knowledge that the outcome is heavy-tailed / ordinal-like (or use a log scale)
    control : k groups compared only against a control
    many_features : one test per gene/feature -> vectorise + FDR
    """
    notes, k = [], groups
    if outcome == "survival":
        rec = dict(test="Kaplan-Meier + log-rank" if k != "continuous" else "Cox regression",
                   function="lifelines (KaplanMeierFitter / logrank_test / CoxPHFitter)", post_hoc=None)
    elif outcome == "categorical":
        if paired:
            rec = dict(test="McNemar (exact)", function="mcnemar(table)", post_hoc=None)
        elif k == 1:
            rec = dict(test="binomial / chi-square goodness of fit",
                       function="scipy.stats.binomtest | goodness_of_fit", post_hoc=None)
        elif k == "continuous":
            rec = dict(test="logistic regression", function="statsmodels.formula.api.logit", post_hoc=None)
        else:
            rec = dict(test="chi-square or Fisher exact (auto by expected counts)", function="contingency(table)",
                       post_hoc="pairwise contingency() on 2x2 sub-tables + p_adjust(method='holm')")
            notes.append("report Cramer's V / odds ratio, not only p")
    else:
        rank = skewed or outcome == "ordinal"
        if k == "continuous":
            rec = dict(test="Spearman correlation" if rank else "Pearson correlation / OLS",
                       function="correlation(x, y, 'spearman')" if rank else "correlation(x, y, 'pearson') | ols(df, 'y ~ x')",
                       post_hoc=None)
            notes.append("regression assumes normal residuals, not normal x or y: check ols()['resid_normality_p']")
        elif k == 1:
            rec = dict(test="one-sample Wilcoxon" if rank else "one-sample t-test",
                       function=f"one_sample(x, mu0, test='{'wilcoxon' if rank else 't'}')", post_hoc=None)
        elif k == 2:
            t = ("wilcoxon" if rank else "paired_t") if paired else ("mwu" if rank else "welch")
            rec = dict(test={"wilcoxon": "Wilcoxon signed-rank", "paired_t": "paired t-test",
                             "mwu": "Mann-Whitney U", "welch": "Welch t-test"}[t],
                       function=f"compare_two(a, b, test='{t}')", post_hoc=None)
            if not paired and not rank:
                notes.append("Welch by default: same power as Student when variances are equal, "
                             "correct size when they are not")
        else:
            if paired:
                rec = dict(test="Friedman" if rank else "repeated-measures ANOVA / mixed model",
                           function="scipy.stats.friedmanchisquare" if rank else
                           "statsmodels AnovaRM | mixedlm('y ~ cond', groups=subject)",
                           post_hoc="pairwise compare_two(paired) + p_adjust('holm')")
            elif rank:
                rec = dict(test="Kruskal-Wallis", function="compare_groups(groups, 'kruskal')",
                           post_hoc="posthoc(groups, 'dunn_control' if control else 'dunn')")
            else:
                rec = dict(test="Welch ANOVA", function="compare_groups(groups, 'welch_anova')",
                           post_hoc="posthoc(groups, 'dunnett', control=...)" if control else
                           "posthoc(groups, 'games_howell')  # or 'tukey' if variances similar")
            notes.append(f"{n_comparisons(k)} pairwise comparisons: never run them as raw t-tests")
    if n_per_group is not None and n_per_group <= 5 and outcome in ("continuous", "ordinal") and groups == 2:
        mp = _min_p_rank(n_per_group, None if paired else n_per_group)
        notes.append(f"n = {n_per_group}/group: smallest attainable rank-test p = {mp:.3g}; "
                     "prefer Welch on a sensible scale (e.g. log) and report the CI")
    if many_features:
        notes.append("many features: use compare_two_matrix(...) (vectorised) and control FDR with "
                     "p_adjust('bh'); for counts use limma-voom / DESeq2 / edgeR instead")
    rec.update(outcome=outcome, groups=groups, paired=paired, notes=notes)
    return rec


# =============================================================================
# 13. legacy aliases (kept for callers in GDC_lib, cBioPortal_lib, enricher_lib)
# =============================================================================
def fdr(p_vals) -> np.ndarray:
    """Benjamini-Hochberg q-values (legacy name). NaN p stay NaN (previously coerced to 1)."""
    return p_adjust(p_vals, "bh")


def fdr_df(df: pd.DataFrame, col: str) -> pd.Series:
    """BH q-values of df[col], aligned to df.index (legacy)."""
    return pd.Series(p_adjust(df[col].to_numpy(float), "bh"), index=df.index, name="fdr")


def chi2_or_fisher_exact_test(vals0, vals1):
    """Legacy 2-row table test -> (text, statistic, pvalue, dof, expected).
    Fisher for 2x2 tables with an expected count < 5, else chi-square."""
    T = np.array([list(vals0), list(vals1)], dtype=float)
    try:
        r = contingency(T)
    except ValueError as e:
        return f"Error: {e}", -1, -1, 0, []
    if r["statistic"] is None and r["test"] != "Fisher exact":
        return f"Error: {r['note']}", -1, -1, 0, []
    if r["test"] == "Fisher exact":
        return (f"Fisher's exact test OR = {r['estimate']:.3f}, p-value = {r['p']:.2e}",
                r["estimate"], r["p"], 1, [])
    E = stats.contingency.expected_freq(r["table"])
    return (f"chi-square statistics = {r['statistic']:5.3f}, p-value = {r['p']:.2e} dof = {r['df']}",
            r["statistic"], r["p"], r["df"], E)


def chisquare_2by2(df: pd.DataFrame, alpha: float = ALPHA, verbose: bool = False):
    """Legacy -> (reject, dof, statistic, pvalue, text) on the first two rows of df."""
    r = contingency(df.iloc[:2].to_numpy(float), test="chi2")
    txt = f"{r['test']} pval {r['p']:.2e} {r['stars']}"
    if verbose:
        print(txt)
    return bool(r["p"] < alpha), r["df"], r["statistic"], r["p"], txt


def stat_asteristics(pval: float, threshold: float = ALPHA, NS: str = "NS") -> str:
    """Legacy name of stars()."""
    return stars(pval, threshold, NS)


def stat_list_asteristics(pvals, threshold: float = ALPHA, NS: str = "NS") -> list:
    return [stars(p, threshold, NS) for p in pvals]

"""Validation of stat_lib2 against scipy / statsmodels / scikit-posthocs / pingouin references.
Run from src/: python -m pytest -q libs/test_stat_lib2.py   (reference packages only needed for the tests)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

try:
    from libs import stat_lib2 as sl          # project layout: run from src/
except ImportError:
    import stat_lib2 as sl

rng = np.random.default_rng(42)
A = rng.normal(5, 1, 12)
B = rng.normal(6, 2, 15)
C = rng.normal(5.5, 1.5, 10)


# ---------- multiple testing
@pytest.mark.parametrize("m,ref", [("bh", "fdr_bh"), ("by", "fdr_by"), ("holm", "holm"), ("bonferroni", "bonferroni")])
def test_p_adjust_matches_statsmodels(m, ref):
    from statsmodels.stats.multitest import multipletests
    p = rng.uniform(0, 0.2, 200)
    assert np.allclose(sl.p_adjust(p, m), multipletests(p, method=ref)[1])


def test_p_adjust_nan_excluded():
    from statsmodels.stats.multitest import multipletests
    q = sl.p_adjust([0.001, np.nan, 0.01, np.nan])
    assert np.isnan(q[1]) and np.isnan(q[3])
    assert np.allclose(q[[0, 2]], multipletests([0.001, 0.01], method="fdr_bh")[1])


def test_legacy_fdr_is_bh():
    from statsmodels.stats.multitest import multipletests
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.049, 0.2, 0.0101])
    assert np.allclose(sl.fdr(p), multipletests(p, method="fdr_bh")[1])


# ---------- two groups
def test_pandey_example():
    r = sl.compare_two([3, 4, 4, 4, 5], [1, 2, 2, 2, 3], test="student")
    assert round(r["statistic"], 2) == 4.47 and r["df"] == 8 and abs(r["p"] - 0.002077) < 1e-5


@pytest.mark.parametrize("alt", ["two-sided", "greater", "less"])
def test_welch_and_alternative(alt):
    r = sl.compare_two(A, B, "welch", alternative=alt)
    ref = stats.ttest_ind(A, B, equal_var=False, alternative=alt)
    assert np.isclose(r["p"], ref.pvalue) and np.isclose(r["df"], ref.df)


def test_ttest_from_stats_two_sided_not_halved():
    r = sl.ttest_from_stats(4, 0.7071, 5, 2, 0.7071, 5)
    assert np.isclose(r["p"], stats.ttest_ind_from_stats(4, 0.7071, 5, 2, 0.7071, 5, equal_var=False).pvalue)


def test_mwu_effect_and_minp_note():
    r = sl.compare_two([1, 2, 3], [4, 5, 6], "mwu")
    assert np.isclose(r["p"], 0.1) and r["effect"] == -1 and "min attainable" in r["note"]
    assert r["estimate"] == -3


def test_paired_and_wilcoxon():
    a, b = A[:10], A[:10] + rng.normal(0.5, 0.5, 10)
    r = sl.compare_two(a, b, "paired_t")
    assert np.isclose(r["p"], stats.ttest_rel(a, b).pvalue)
    w = sl.compare_two(a, b, "wilcoxon")
    assert np.isclose(w["p"], stats.wilcoxon(a, b).pvalue)
    a2 = a.copy(); a2[0] = np.nan                                   # incomplete pair dropped
    assert sl.compare_two(a2, b, "paired_t")["n"] == 9


def test_hedges_g_vs_pingouin():
    import pingouin as pg
    assert np.isclose(sl.hedges_g(A, B), pg.compute_effsize(A, B, eftype="hedges"))


def test_compare_two_matrix():
    X = pd.DataFrame(rng.normal(0, 1, (50, 6)))
    Y = pd.DataFrame(rng.normal(0.5, 1, (50, 7)))
    X.iloc[0] = 3.0; Y.iloc[0] = 3.0                                 # constant feature
    out = sl.compare_two_matrix(X, Y, "welch")
    assert np.isnan(out.loc[0, "p"]) and np.isnan(out.loc[0, "fdr"])
    i = 5
    assert np.isclose(out.loc[i, "p"], stats.ttest_ind(X.loc[i], Y.loc[i], equal_var=False).pvalue)


# ---------- k groups
def test_welch_anova_vs_statsmodels():
    from statsmodels.stats.oneway import anova_oneway
    r = sl.welch_anova(A, B, C)
    ref = anova_oneway([A, B, C], use_var="unequal")
    assert np.isclose(r["statistic"], ref.statistic) and np.isclose(r["p"], ref.pvalue)


def test_compare_groups_kruskal_anova():
    assert np.isclose(sl.compare_groups([A, B, C], "kruskal")["p"], stats.kruskal(A, B, C).pvalue)
    assert np.isclose(sl.compare_groups([A, B, C], "anova")["p"], stats.f_oneway(A, B, C).pvalue)


def test_posthoc_tukey_vs_statsmodels():
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    y = np.r_[A, B, C]; g = np.repeat(["A", "B", "C"], [A.size, B.size, C.size])
    ref = pairwise_tukeyhsd(y, g).pvalues
    assert np.allclose(sl.posthoc([A, B, C], "tukey", labels=["A", "B", "C"]).p_adj, ref, atol=1e-3)


def test_posthoc_games_howell_vs_pingouin():
    import pingouin as pg
    df = pd.DataFrame({"y": np.r_[A, B, C], "g": np.repeat(["A", "B", "C"], [A.size, B.size, C.size])})
    ref = pg.pairwise_gameshowell(df, dv="y", between="g")
    out = sl.posthoc([A, B, C], "games_howell", labels=["A", "B", "C"])
    assert np.allclose(out.p_adj, ref["pval"], atol=1e-4)


def test_posthoc_dunn_vs_scikit_posthocs():
    import scikit_posthocs as sp
    ref = sp.posthoc_dunn([A, B, C], p_adjust="holm").to_numpy()
    out = sl.posthoc([A, B, C], "dunn", p_adjust_method="holm")
    assert np.allclose(out.p_adj, [ref[0, 1], ref[0, 2], ref[1, 2]])


def test_posthoc_dunnett_vs_scipy():
    out = sl.posthoc([A, B, C], "dunnett", labels=["ctl", "B", "C"], control="ctl")
    assert np.allclose(out.p_adj, stats.dunnett(B, C, control=A).pvalue, atol=1e-3)  # QMC integration


# ---------- categorical
def test_contingency_auto_fisher_and_legacy():
    r = sl.contingency([[1, 9], [8, 2]])
    assert r["test"] == "Fisher exact" and np.isclose(r["p"], stats.fisher_exact([[1, 9], [8, 2]]).pvalue)
    s, stat, p, dof, _ = sl.chi2_or_fisher_exact_test([1, 9], [8, 2])
    assert s.startswith("Fisher") and np.isclose(p, r["p"])


def test_contingency_chi2_large():
    T = [[30, 20, 25], [20, 30, 35]]
    r = sl.contingency(T)
    assert np.isclose(r["p"], stats.chi2_contingency(T).pvalue) and 0 < r["effect"] < 1


def test_contingency_monte_carlo_small():
    r = sl.contingency([[3, 0, 1], [0, 4, 1]], n_mc=2000)
    assert "Monte-Carlo" in r["test"] and 0 < r["p"] < 0.2


def test_enrichment():
    bg = range(1000); hits = range(50); mem = list(range(30)) + list(range(500, 520))
    e = sl.enrichment(hits, mem, bg)
    assert e["k"] == 30 and np.isclose(e["p"], stats.fisher_exact([[30, 20], [20, 930]], alternative="greater").pvalue)


# ---------- correlation
def test_correlation_constant_nan():
    assert np.isnan(sl.correlation([1, 2, 3, 4, 5], [2, 2, 2, 2, 2])["effect"])
    r = sl.correlation(A[:10], C, "pearson")
    assert np.isclose(r["p"], stats.pearsonr(A[:10], C).pvalue) and r["ci_low"] < r["effect"] < r["ci_high"]


def test_corr_columns():
    X = pd.DataFrame(rng.normal(size=(20, 3)), columns=list("abc"))
    Y = X + rng.normal(scale=0.5, size=(20, 3)); Y["c"] = 1.0
    r = sl.corr_columns(X, Y)
    assert np.isclose(r.loc["a", "r"], np.corrcoef(X.a, Y.a)[0, 1]) and np.isnan(r.loc["c", "r"])
    assert np.isclose(r.loc["a", "p"], stats.pearsonr(X.a, Y.a).pvalue)


# ---------- describe / CI
def test_describe_ci_uses_sample_sd():
    d = sl.describe([1, 2, 3, 4, 5])
    lo, hi = stats.t.interval(0.95, 4, loc=3, scale=stats.sem([1, 2, 3, 4, 5]))
    assert np.isclose(d["ci_low"], lo) and np.isclose(d["sd"], np.std([1, 2, 3, 4, 5], ddof=1))


def test_stars_nan():
    assert sl.stars(np.nan) == "NA" and sl.stars(0.03) == "*" and sl.stars(0.004) == "**" and sl.stars(1e-5) == "****"


# ---------- modality
def test_modality_bimodal_vs_unimodal():
    bi = np.r_[rng.normal(0, 1, 300), rng.normal(5, 1, 300)]
    uni = rng.normal(0, 1, 600)
    assert sl.modality(bi)["verdict"].startswith("bimodal")
    assert sl.modality(uni)["verdict"] == "unimodal"
    assert sl.gmm_modality(bi)["ashman_d"] > 4


def test_modality_shoulder_not_bimodal():
    # equal mixture with D = 1.5 is unimodal in shape
    sh = np.r_[rng.normal(0, 1, 2000), rng.normal(1.5, 1, 2000)]
    assert not sl.modality(sh)["verdict"].startswith("bimodal")


def test_modality_floor_flag():
    x = np.clip(rng.normal(0, 1, 500), 0, None)           # ~50% at floor 0
    assert "floor-inflated" in sl.modality(x, floor=0)["verdict"] or sl.modality(x, floor=0)["verdict"] == "unimodal"


def test_bimodality_coefficient_uniform_and_normal():
    assert abs(sl.bimodality_coefficient(rng.uniform(size=100000)) - 5 / 9) < 0.01
    assert sl.bimodality_coefficient(rng.normal(size=100000)) < 0.4


# ---------- power
def test_power_vs_statsmodels():
    from statsmodels.stats.power import TTestIndPower
    assert np.isclose(sl.power_ttest(0.8, 20), TTestIndPower().power(0.8, 20, 0.05), atol=1e-6)
    assert sl.sample_size_ttest(0.5) == int(np.ceil(TTestIndPower().solve_power(0.5, power=0.8, alpha=0.05)))


def test_choose_test():
    assert sl.choose_test("continuous", 2)["test"] == "Welch t-test"
    assert sl.choose_test("continuous", 4, control=True)["post_hoc"].startswith("posthoc(groups, 'dunnett'")
    assert sl.choose_test("categorical", 2, paired=True)["test"].startswith("McNemar")
    assert any("attainable" in n for n in sl.choose_test("continuous", 2, n_per_group=3)["notes"])

"""Tests for the rank-agreement statistics.

These run today, without any benchmark data. If they pass, the numbers that go
into Figures 2, 3 and 5 are trustworthy.
"""
import numpy as np
import pandas as pd
import pytest

from bra.analysis import ranks


def _frame(scores_by_bench, metric="mcc"):
    rows = []
    for b, scores in scores_by_bench.items():
        for m, s in scores.items():
            rows.append({"benchmark": b, "model_id": m, "metric": metric, "score": s})
    return pd.DataFrame(rows)


def test_identical_rankings_give_tau_one():
    df = _frame({"X": {"M01": 0.1, "M02": 0.2, "M03": 0.3},
                 "Y": {"M01": 0.4, "M02": 0.5, "M03": 0.6}})
    rm = ranks.rank_matrix(df, "mcc")
    assert ranks.mean_pairwise_tau(rm) == pytest.approx(1.0)


def test_reversed_rankings_give_tau_minus_one():
    df = _frame({"X": {"M01": 0.1, "M02": 0.2, "M03": 0.3},
                 "Y": {"M01": 0.3, "M02": 0.2, "M03": 0.1}})
    rm = ranks.rank_matrix(df, "mcc")
    assert ranks.mean_pairwise_tau(rm) == pytest.approx(-1.0)


def test_tau_b_handles_ties():
    """tau-a would mishandle this; tau-b is why we chose it."""
    a = pd.Series([1.5, 1.5, 3.0], index=list("abc"))
    b = pd.Series([1.0, 2.0, 3.0], index=list("abc"))
    t = ranks.tau_b(a, b)
    assert -1.0 <= t <= 1.0 and not np.isnan(t)


def test_null_sd_matches_closed_form_at_n12():
    """About 0.22 at n=12 -- the number that keeps us honest on Figure 2."""
    assert ranks.tau_null_sd(12) == pytest.approx(0.2210, abs=1e-3)


def test_permutation_null_is_centred_on_zero():
    null = ranks.permutation_null(n_models=12, n_benchmarks=5, n_perm=400, seed=0)
    assert abs(null.mean()) < 0.05


def test_top1_flip_rate_detects_disagreement():
    df = _frame({"X": {"M01": 0.9, "M02": 0.1},
                 "Y": {"M01": 0.1, "M02": 0.9}})
    rm = ranks.rank_matrix(df, "mcc")
    flip = ranks.top1_flip_rate(rm)
    assert flip["n_flips"] == 1 and flip["rate"] == pytest.approx(1.0)


def test_inversions_counts_reversals():
    a = pd.Series([1, 2, 3], index=["M01", "M02", "M03"])
    b = pd.Series([3, 2, 1], index=["M01", "M02", "M03"])
    assert ranks.inversions(a, b)["inversions"] == 3


def test_rbo_is_top_weighted():
    """A swap at the top must cost more than a swap at the bottom."""
    base = ["a", "b", "c", "d", "e"]
    top_swap = ["b", "a", "c", "d", "e"]
    bottom_swap = ["a", "b", "c", "e", "d"]
    assert ranks.rbo(base, top_swap) < ranks.rbo(base, bottom_swap)


def test_footrule_is_zero_for_identical():
    s = pd.Series([1, 2, 3])
    assert ranks.footrule(s, s) == 0.0


# --------------------------------------------------------------------------- #
# Agreement ceiling
# --------------------------------------------------------------------------- #

def _seeded_frame(bench_scores, metric="mcc"):
    """bench -> {seed -> {model -> score}}"""
    rows = []
    for b, by_seed in bench_scores.items():
        for seed, scores in by_seed.items():
            for m, s in scores.items():
                rows.append({"benchmark": b, "seed": seed, "model_id": m,
                             "metric": metric, "score": s})
    return pd.DataFrame(rows)


def test_ceiling_is_one_when_seeds_agree_perfectly():
    """No seed noise means the ceiling is 1.0 and disagreement is all protocol."""
    scores = {"M01": 0.1, "M02": 0.5, "M03": 0.9}
    df = _seeded_frame({"X": {0: scores, 1: scores, 2: scores}})
    out = ranks.ceiling_from_seeds(df, "mcc")
    assert out.loc[0, "ceiling_tau_b"] == pytest.approx(1.0)
    assert out.loc[0, "n_seed_pairs"] == 3


def test_ceiling_drops_when_seeds_disagree():
    """Seed noise alone reorders models, so the ceiling falls below 1.0."""
    df = _seeded_frame({"X": {
        0: {"M01": 0.1, "M02": 0.5, "M03": 0.9},
        1: {"M01": 0.9, "M02": 0.5, "M03": 0.1},   # fully reversed by noise
    }})
    out = ranks.ceiling_from_seeds(df, "mcc")
    assert out.loc[0, "ceiling_tau_b"] == pytest.approx(-1.0)


def test_normalised_agreement_is_one_when_benchmarks_match_their_own_ceiling():
    """Two benchmarks that agree with each other exactly as well as a benchmark
    agrees with itself must normalise to 1.0 -- the headline H1 quantity."""
    a = {"M01": 0.1, "M02": 0.5, "M03": 0.9}
    df = _seeded_frame({"X": {0: a, 1: a}, "Y": {0: a, 1: a}})
    out = ranks.normalised_agreement(df, "mcc")
    assert out["within_benchmark_ceiling"] == pytest.approx(1.0)
    assert out["between_benchmark_tau_b"] == pytest.approx(1.0)
    assert out["normalised_agreement"] == pytest.approx(1.0)


def test_normalised_agreement_exceeds_raw_tau_when_ceiling_is_below_one():
    """The whole point: judging against 1.0 overstates disagreement."""
    noisy_a = {"M01": 0.10, "M02": 0.50, "M03": 0.90, "M04": 0.52}
    noisy_b = {"M01": 0.10, "M02": 0.52, "M03": 0.90, "M04": 0.50}  # M02/M04 swap
    df = _seeded_frame({"X": {0: noisy_a, 1: noisy_b},
                        "Y": {0: noisy_b, 1: noisy_a}})
    out = ranks.normalised_agreement(df, "mcc")
    assert out["within_benchmark_ceiling"] < 1.0
    assert out["normalised_agreement"] >= out["between_benchmark_tau_b"]


# --------------------------------------------------------------------------- #
# Regressions found by scripts/smoke_analysis.py on 2026-10-05
# --------------------------------------------------------------------------- #

def test_mean_pairwise_tau_returns_nan_for_single_column():
    """A one-column rank matrix has nothing to compare. Announce it, do not warn."""
    rm = pd.DataFrame({"only": [1.0, 2.0, 3.0]}, index=list("abc"))
    assert np.isnan(ranks.mean_pairwise_tau(rm))


def test_bootstrap_produces_usable_resamples():
    """Regression: the groupby.apply version yielded n_boot=0 under pandas 3 and a
    bare except hid it. The CI must be finite and ordered."""
    rng = np.random.default_rng(0)
    rows = []
    for b in ["X", "Y", "Z"]:
        offset = {m: rng.normal(0, 0.3) for m in [f"M{i:02d}" for i in range(1, 9)]}
        for seed in range(4):
            for m, o in offset.items():
                rows.append({"benchmark": b, "model_id": m, "seed": seed,
                             "metric": "mcc",
                             "score": 0.5 + o + rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)
    out = ranks.bootstrap_mean_tau(df, "mcc", n_boot=60, seed=1)
    assert out["n_boot"] == 60
    assert np.isfinite(out["lo95"]) and np.isfinite(out["hi95"])
    assert out["lo95"] <= out["point"] <= out["hi95"]


def test_bootstrap_raises_on_unknown_metric():
    """Fail loudly rather than returning an empty result."""
    df = pd.DataFrame({"benchmark": ["X"], "model_id": ["M01"], "seed": [0],
                       "metric": ["mcc"], "score": [0.5]})
    with pytest.raises(ValueError):
        ranks.bootstrap_mean_tau(df, "auroc", n_boot=5)


# --------------------------------------------------------------------------- #
# The flip rate must be read against its own ceiling, like tau-b
# --------------------------------------------------------------------------- #
def _flip_grid(winner_of, n_seeds=4, noise=0.0, seed=0):
    """Grid where `winner_of(level, seed)` names the intended best pipeline."""
    rng = np.random.default_rng(seed)
    rows = []
    for lvl in ["a", "b", "c"]:
        for s in range(n_seeds):
            w = winner_of(lvl, s)
            for m in [f"M{i:02d}" for i in range(1, 6)]:
                rows.append({"model_id": m, "negative_rule": lvl, "seed": s,
                             "metric": "mcc",
                             "score": (1.0 if m == w else 0.2) + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_flip_ceiling_is_zero_when_the_winner_never_moves():
    d = _flip_grid(lambda lvl, s: "M01")
    r = ranks.flip_ceiling_matched(d, "negative_rule", n_boot=400)
    assert r["flip_across_levels"] == 0.0
    assert r["flip_ceiling_seeds"] == 0.0
    assert r["difference"] == 0.0


def test_flip_excess_is_one_when_only_the_level_moves_the_winner():
    """Winner depends on the level but not the seed: the ceiling is 0, the
    observed rate is 1, and the whole available range is used."""
    d = _flip_grid(lambda lvl, s: {"a": "M01", "b": "M02", "c": "M03"}[lvl])
    r = ranks.flip_ceiling_matched(d, "negative_rule", n_boot=400)
    assert r["flip_across_levels"] == 1.0
    assert r["flip_ceiling_seeds"] == 0.0
    assert r["normalised_flip_excess"] == pytest.approx(1.0)


def test_flip_excess_is_zero_when_only_the_seed_moves_the_winner():
    """The case the manuscript correction turns on: a factor that looks unstable
    only because reruns are unstable must show no excess over its ceiling."""
    # three seeds, three distinct winners, so every seed pair flips
    d = _flip_grid(lambda lvl, s: f"M{s + 1:02d}", n_seeds=3)
    r = ranks.flip_ceiling_matched(d, "negative_rule", n_boot=400)
    assert r["flip_across_levels"] == 0.0
    assert r["flip_ceiling_seeds"] == 1.0
    assert r["difference"] == -1.0


def test_flip_ceiling_reports_pair_counts_and_an_interval():
    d = _flip_grid(lambda lvl, s: "M01" if (s + ord(lvl)) % 2 else "M02")
    r = ranks.flip_ceiling_matched(d, "negative_rule", n_boot=800, seed=1)
    assert r["n_level_pairs"] > 0 and r["n_seed_pairs"] > 0
    assert r["diff_lo95"] <= r["difference"] <= r["diff_hi95"]

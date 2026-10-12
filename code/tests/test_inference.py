"""Tests for the inferential layer. These guard the numbers that carry the
paper's claims, so a silent error here would be worse than no intervals."""
import numpy as np
import pandas as pd
import pytest

from bra.analysis import inference as inf


def _frame(n_pipe=12, n_level=5, protocol_sd=0.0, seed=0, metric="mcc"):
    """Synthetic scores with a controllable level-specific effect."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n_pipe)
    rows = []
    for L in range(n_level):
        eff = rng.normal(0, protocol_sd, n_pipe) if protocol_sd else np.zeros(n_pipe)
        for i in range(n_pipe):
            rows.append({"level": f"L{L}", "model_id": f"M{i:02d}", "metric": metric,
                         "score": float(theta[i] + eff[i] + rng.normal(0, 0.01))})
    return pd.DataFrame(rows)


def test_ci_brackets_the_point_estimate():
    df = _frame(protocol_sd=0.5)
    r = inf.bootstrap_rank_stats(df, "level", n_boot=400, seed=1)
    assert r["tau_lo95"] <= r["tau_point"] <= r["tau_hi95"]
    assert r["flip_lo95"] <= r["flip_point"] <= r["flip_hi95"]


def test_identical_levels_give_tau_near_one_with_tight_ci():
    df = _frame(protocol_sd=0.0)
    r = inf.bootstrap_rank_stats(df, "level", n_boot=400, seed=2)
    assert r["tau_point"] > 0.95
    assert r["tau_hi95"] - r["tau_lo95"] < 0.25


def test_strong_level_effect_lowers_tau_and_ci_excludes_one():
    df = _frame(protocol_sd=2.0)
    r = inf.bootstrap_rank_stats(df, "level", n_boot=400, seed=3)
    assert r["tau_point"] < 0.7
    assert r["tau_hi95"] < 1.0


def test_compare_arms_handles_different_level_counts():
    """Regression: the arms have 10 and 5 levels. An earlier row-swapping
    permutation test crashed on the shape mismatch."""
    a = _frame(n_pipe=12, n_level=10, protocol_sd=1.0, seed=40)
    b = _frame(n_pipe=12, n_level=5, protocol_sd=1.0, seed=41)
    out = inf.compare_arms(a, "level", b, "level", n_boot=300, seed=7)
    assert np.isfinite(out["difference"])
    assert out["diff_lo95"] <= out["difference"] <= out["diff_hi95"]


def test_permutation_test_finds_no_difference_when_arms_match():
    a = _frame(protocol_sd=1.0, seed=10)
    b = _frame(protocol_sd=1.0, seed=10)       # identical construction
    out = inf.compare_arms(a, "level", b, "level", n_boot=600, seed=4)
    assert out["p_two_sided"] > 0.2, out


def test_permutation_test_detects_a_real_difference():
    a = _frame(protocol_sd=0.05, seed=20)      # high agreement
    b = _frame(protocol_sd=3.0, seed=21)       # low agreement
    out = inf.compare_arms(a, "level", b, "level", n_boot=800, seed=5)
    assert out["difference"] > 0
    assert out["p_two_sided"] < 0.05, out


def test_compare_arms_uses_only_shared_pipelines():
    a = _frame(n_pipe=12, protocol_sd=1.0, seed=30)
    b = _frame(n_pipe=8, protocol_sd=1.0, seed=31)
    out = inf.compare_arms(a, "level", b, "level", n_boot=200, seed=6)
    assert out["n_shared_pipelines"] == 8


def test_jackknife_ci_contains_its_point_estimate():
    """Regression: resampling cells broke the balanced design and produced
    intervals that excluded their own point estimates."""
    rng = np.random.default_rng(0)
    rows = []
    for rule in ["r1", "r2", "r3"]:
        for thr in ["t1", "t2"]:
            for sd in range(5):
                for i in range(10):
                    rows.append({"negative_rule": rule, "cluster_threshold": thr,
                                 "seed": sd, "model_id": f"M{i:02d}", "metric": "mcc",
                                 "score": float(i * 0.05 + rng.normal(0, 0.02))})
    df = pd.DataFrame(rows)
    t = inf.jackknife_variance_components(df)
    assert not t.empty
    bad = t[(t.pct_rank_var < t.lo95 - 1e-6) | (t.pct_rank_var > t.hi95 + 1e-6)]
    assert bad.empty, f"CI excludes point estimate:\n{bad}"

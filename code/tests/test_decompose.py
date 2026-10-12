"""Tests for the variance decomposition.

The planted-effect tests are the important ones: if a factor we injected does
not dominate the decomposition, the estimator is wrong and H3/H4 would be
answered by a bug rather than by the data.
"""
import numpy as np
import pandas as pd
import pytest

from bra.analysis import decompose


def _grid(effect_sizes, n_seeds=3, noise=0.01, seed=0):
    """Synthetic Arm-B-shaped grid with known, planted effects."""
    rng = np.random.default_rng(seed)
    rows = []
    for model in [f"M{i:02d}" for i in range(1, 7)]:
        for thr in ["random", "0.9", "0.7", "0.4"]:
            for neg in ["random_uniprot", "length_matched", "shuffled_positives"]:
                for s in range(n_seeds):
                    score = (0.5
                             + effect_sizes.get("model", 0) * (int(model[1:]) / 6)
                             + effect_sizes.get("threshold", 0) * ["random", "0.9", "0.7", "0.4"].index(thr)
                             + effect_sizes.get("negative", 0) * ["random_uniprot", "length_matched", "shuffled_positives"].index(neg)
                             + rng.normal(0, noise))
                    rows.append({"model_id": model, "cluster_threshold": thr,
                                 "negative_rule": neg, "seed": s,
                                 "metric": "mcc", "score": score})
    return pd.DataFrame(rows)


def test_planted_negative_effect_dominates():
    df = _grid({"negative": 0.20, "threshold": 0.01, "model": 0.02})
    vc = decompose.variance_components(df).set_index("component")
    assert vc.loc["negative_rule", "pct_variance"] > vc.loc["cluster_threshold", "pct_variance"]


def test_planted_threshold_effect_dominates():
    df = _grid({"negative": 0.01, "threshold": 0.20, "model": 0.02})
    vc = decompose.variance_components(df).set_index("component")
    assert vc.loc["cluster_threshold", "pct_variance"] > vc.loc["negative_rule", "pct_variance"]


def test_seed_component_is_small_when_noise_is_small():
    df = _grid({"model": 0.20}, noise=0.001)
    vc = decompose.variance_components(df).set_index("component")
    assert vc.loc["seed", "pct_variance"] < 5.0


def test_components_sum_to_about_one_hundred():
    df = _grid({"model": 0.1, "threshold": 0.05, "negative": 0.08})
    vc = decompose.variance_components(df)
    assert vc["pct_variance"].sum() == pytest.approx(100.0, abs=1.0)


def test_rank_response_is_added_per_cell():
    df = _grid({"model": 0.1})
    out = decompose.add_rank_response(df)
    cell = ["metric", "cluster_threshold", "negative_rule", "seed"]
    # Within any cell, ranks must be a permutation of 1..n_models
    g = out.groupby(cell)["rank"]
    assert g.min().eq(1.0).all()
    assert g.max().eq(out["model_id"].nunique()).all()


def test_decompose_both_returns_both_responses():
    df = _grid({"model": 0.1, "negative": 0.05})
    out = decompose.decompose_both(df)
    assert {"pct_score_var", "pct_rank_var", "rank_minus_score"} <= set(out.columns)


def test_logit_is_monotone_and_finite_at_bounds():
    v = decompose.logit(np.array([0.0, 0.5, 1.0]))
    assert np.all(np.isfinite(v)) and v[0] < v[1] < v[2]


def test_h5_probe_works_when_protocol_columns_are_constant():
    """Regression: in Arm A, cluster_threshold and negative_rule are both
    "native", so the old cell selection gave one pseudo-cell and a silent NaN.
    The benchmark is the varying factor there."""
    rng = np.random.default_rng(0)
    rows = []
    for b in ["B1", "B2", "B3"]:
        for m in range(1, 13):
            rows.append({
                "benchmark": b, "model_id": f"M{m:02d}", "seed": 0,
                "metric": "mcc", "score": float(rng.normal(0.6, 0.1)),
                "cluster_threshold": "native", "negative_rule": "native",
                "representation": f"R{m % 4}", "learner": f"L{m % 3}",
            })
    df = pd.DataFrame(rows)
    out = decompose.representation_vs_learner_stability(df, "mcc")
    assert "error" not in out
    assert np.isfinite(out["representation_mean_tau_b"])
    assert np.isfinite(out["learner_mean_tau_b"])


def test_h5_probe_reports_error_when_nothing_varies():
    df = pd.DataFrame([{
        "benchmark": "B1", "model_id": "M01", "seed": 0, "metric": "mcc",
        "score": 0.5, "cluster_threshold": "native", "negative_rule": "native",
        "representation": "R1", "learner": "L1",
    }])
    assert "error" in decompose.representation_vs_learner_stability(df, "mcc")


# --------------------------------------------------------------------------- #
# Exact method-of-moments components, and the structural zero they expose
# --------------------------------------------------------------------------- #
FACTORS = ["model_id", "negative_rule", "cluster_threshold", "seed"]


def test_mom_recovers_a_planted_variance_ordering():
    """The planted negative-rule effect must dominate as a VARIANCE component,
    not merely as a sum-of-squares share, which is level-count dependent."""
    df = _grid({"negative": 0.20, "threshold": 0.01, "model": 0.02})
    v = decompose.mom_variance_components(df, FACTORS).set_index("component")
    assert v.loc["negative_rule", "sigma2"] > v.loc["cluster_threshold", "sigma2"]
    assert v.loc["negative_rule", "pct_sigma2"] > 50


def test_mom_ss_shares_sum_to_one_hundred():
    """The orthogonal decomposition must be exhaustive: if the shares do not sum
    to 100% the effect arrays are not orthogonal and every component is wrong."""
    df = _grid({"negative": 0.10, "threshold": 0.05, "model": 0.05})
    v = decompose.mom_variance_components(df, FACTORS)
    assert v["pct_ss"].sum() == pytest.approx(100.0, abs=1e-6)


def test_mom_rejects_an_incomplete_design():
    df = _grid({"negative": 0.1}).iloc[:-1]
    with pytest.raises(ValueError, match="not complete"):
        decompose.mom_variance_components(df, FACTORS)


def test_rank_main_effects_are_structurally_zero():
    """Regression guard for the claim corrected in the manuscript.

    Ranks are assigned within each (metric, threshold, rule, seed) cell, so every
    cell holds each rank exactly once and every cell-defining factor has group
    mean (K+1)/2. Its between-group sum of squares is therefore identically zero
    for ANY data -- which is why a zero here must never be reported as a measured
    null. Planting a huge protocol effect must still produce zero.
    """
    df = decompose.add_rank_response(_grid({"negative": 5.0, "threshold": 3.0}))
    k = df.groupby(["metric", "cluster_threshold", "negative_rule", "seed"]).size().iloc[0]
    for f in ["negative_rule", "cluster_threshold", "seed"]:
        assert df.groupby(f)["rank"].mean().round(10).nunique() == 1
        assert df.groupby(f)["rank"].mean().iloc[0] == pytest.approx((k + 1) / 2)
    vc = decompose.variance_components(df, "rank").set_index("component")
    for f in ["negative_rule", "cluster_threshold", "seed"]:
        assert vc.loc[f, "pct_variance"] == pytest.approx(0.0, abs=1e-9)


def test_rank_interaction_is_not_structurally_zero():
    """The counterpart: pipeline x factor interactions ARE free to vary, so the
    rank decomposition is not vacuous -- it is only the main effects that are."""
    rng = np.random.default_rng(0)
    rows = []
    for m in [f"M{i:02d}" for i in range(1, 7)]:
        for neg in ["a", "b", "c"]:
            for thr in ["t1", "t2"]:
                for s in range(3):
                    # the ordering of pipelines is reversed under rule "c"
                    base = int(m[1:]) / 6
                    score = (1 - base) if neg == "c" else base
                    rows.append({"model_id": m, "negative_rule": neg,
                                 "cluster_threshold": thr, "seed": s,
                                 "metric": "mcc", "score": score + rng.normal(0, 1e-4)})
    d = decompose.add_rank_response(pd.DataFrame(rows))
    vc = decompose.variance_components(d, "rank").set_index("component")
    assert vc.loc["model_id x negative_rule", "pct_variance"] > 5.0

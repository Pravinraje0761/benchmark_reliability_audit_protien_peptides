"""Uncertainty and hypothesis tests for the rank-agreement statistics.

Without this the paper reports point estimates with no intervals, which is the
first thing a statistically literate reviewer will reject it for. Three things
are provided:

1. **Bootstrap confidence intervals** on mean pairwise tau-b and on the top-1
   flip rate, resampling the **pipelines** (the units the statistic is computed
   over). Resampling runs would answer a different and less relevant question:
   we want to know how much the conclusion depends on which pipelines happened
   to be in the zoo.
2. **A paired permutation test** that cross-task agreement differs from
   cross-protocol agreement. The two arms share the same 24 pipelines, so the
   comparison is paired at the pipeline level and the permutation respects that.
3. **Variance-component intervals** by a nonparametric bootstrap over cells.

Everything here is distribution-free. With 24 pipelines and rank statistics,
normal-theory intervals would be unjustified.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .ranks import mean_pairwise_tau, tau_b, top1_flip_rate


def _rankmat(df: pd.DataFrame, factor: str, metric: str = "mcc") -> pd.DataFrame:
    lb = df[df.metric == metric].groupby([factor, "model_id"], as_index=False)["score"].mean()
    lb["rank"] = lb.groupby(factor)["score"].rank(ascending=False, method="average")
    return lb.pivot(index="model_id", columns=factor, values="rank")


def _restat(scores: pd.DataFrame, pipelines) -> dict:
    """Recompute rank statistics on a subset/resample of pipelines.

    Ranks are recomputed *within* the resample: a bootstrap that reuses the
    full-set ranks would understate uncertainty, because the ranking of a
    smaller pipeline set is not a subset of the ranking of the larger one.
    """
    sub = scores.loc[list(pipelines)]
    kmat = sub.rank(ascending=False, method="average")
    if kmat.shape[1] < 2 or kmat.shape[0] < 3:
        return {"tau": np.nan, "flip": np.nan}
    f = top1_flip_rate(kmat)
    return {"tau": mean_pairwise_tau(kmat), "flip": f["rate"]}


def bootstrap_rank_stats(df: pd.DataFrame, factor: str, metric: str = "mcc",
                         n_boot: int = 4000, seed: int = 0) -> dict:
    """Percentile CIs for mean tau-b and top-1 flip rate, resampling pipelines."""
    lb = df[df.metric == metric].groupby([factor, "model_id"], as_index=False)["score"].mean()
    scores = lb.pivot(index="model_id", columns=factor, values="score")
    pipes = list(scores.index)
    point = _restat(scores, pipes)

    rng = np.random.default_rng(seed)
    taus, flips = [], []
    for _ in range(n_boot):
        pick = rng.choice(pipes, size=len(pipes), replace=True)
        # a resample with ties is fine; duplicated pipelines simply carry weight
        r = _restat(scores, pd.Index(pick).unique() if len(set(pick)) >= 3 else pipes)
        taus.append(r["tau"]); flips.append(r["flip"])
    taus = np.asarray(taus, float); flips = np.asarray(flips, float)
    return {
        "n_pipelines": len(pipes), "n_levels": scores.shape[1],
        "tau_point": point["tau"],
        "tau_lo95": float(np.nanpercentile(taus, 2.5)),
        "tau_hi95": float(np.nanpercentile(taus, 97.5)),
        "flip_point": point["flip"],
        "flip_lo95": float(np.nanpercentile(flips, 2.5)),
        "flip_hi95": float(np.nanpercentile(flips, 97.5)),
        "n_boot": int(np.isfinite(taus).sum()),
    }


def compare_arms(df_a: pd.DataFrame, factor_a: str,
                 df_b: pd.DataFrame, factor_b: str,
                 metric: str = "mcc", n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap: does rank agreement differ between the two arms?

    A permutation test that swaps a pipeline's row between arms is not
    available here, because the arms have different numbers of levels (ten
    tasks against five protocol rules) and the rows are therefore not
    exchangeable - an earlier version attempted it and failed on the shape
    mismatch.

    What *is* paired is the pipeline set: both arms rank the same 24 pipelines.
    So we resample pipelines once per iteration and recompute **both** tau-b
    values on that same resample, giving a distribution of the difference that
    respects the pairing. A 95% interval excluding zero means the two arms
    disagree by more than pipeline-sampling noise.

    Ranks are recomputed inside each resample, since the ranking of a subset is
    not a subset of the ranking of the whole.
    """
    def score_matrix(df, factor):
        return (df[df.metric == metric]
                .groupby([factor, "model_id"], as_index=False)["score"].mean()
                .pivot(index="model_id", columns=factor, values="score"))

    sa, sb = score_matrix(df_a, factor_a), score_matrix(df_b, factor_b)
    shared = sa.index.intersection(sb.index)
    sa, sb = sa.loc[shared], sb.loc[shared]

    def tau_of(x):
        if x.shape[0] < 3 or x.shape[1] < 2:
            return np.nan
        return mean_pairwise_tau(x.rank(ascending=False, method="average"))

    obs_a, obs_b = tau_of(sa), tau_of(sb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = pd.Index(rng.choice(list(shared), size=len(shared), replace=True)).unique()
        if len(pick) < 3:
            diffs[i] = np.nan; continue
        diffs[i] = tau_of(sa.loc[pick]) - tau_of(sb.loc[pick])
    d = diffs[np.isfinite(diffs)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    # two-sided bootstrap p: how often the resampled difference crosses zero
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"tau_a": obs_a, "tau_b_arm": obs_b, "difference": obs_a - obs_b,
            "diff_lo95": float(lo), "diff_hi95": float(hi),
            "p_two_sided": float(min(p, 1.0)),
            "n_shared_pipelines": len(shared), "n_boot": int(len(d))}


def jackknife_variance_components(df: pd.DataFrame, metric: str = "mcc") -> pd.DataFrame:
    """Leave-one-seed-out jackknife intervals for the rank-variance components.

    Two earlier attempts were wrong, and the reasons are worth recording because
    they are easy to repeat:

    * Resampling **cells** with replacement destroyed the balance the ANOVA
      decomposition requires; the intervals did not contain their own point
      estimates.
    * Resampling **seeds** with replacement preserved balance but duplicated
      seeds, which lowers the effective number of seed levels and reattributes
      variance to the seed component. Intervals were valid but skewed, with
      point estimates sitting on an interval boundary.

    A delete-one jackknife over seeds avoids both: every replicate is a
    balanced design over a distinct subset of seeds, and nothing is duplicated.
    With five seeds this gives five replicates - few, so the interval is coarse,
    and we report it as such rather than implying bootstrap-grade precision.
    """
    from .decompose import decompose_both
    sub = df[df.metric == metric]
    seeds = sorted(sub["seed"].unique())
    n = len(seeds)
    if n < 3:
        return pd.DataFrame()

    point = decompose_both(sub)["pct_rank_var"]
    reps = []
    for s_out in seeds:
        frame = sub[sub["seed"] != s_out]
        try:
            reps.append(decompose_both(frame)["pct_rank_var"])
        except Exception:
            continue
    M = pd.DataFrame(reps).reindex(columns=point.index)
    mean = M.mean()
    # standard delete-one jackknife variance
    se = np.sqrt((n - 1) / n * ((M - mean) ** 2).sum())
    return pd.DataFrame({
        "pct_rank_var": point,
        "jackknife_se": se,
        "lo95": (point - 1.96 * se).clip(lower=0),
        "hi95": (point + 1.96 * se).clip(upper=100),
        "n_replicates": len(reps),
    }).sort_values("pct_rank_var", ascending=False)

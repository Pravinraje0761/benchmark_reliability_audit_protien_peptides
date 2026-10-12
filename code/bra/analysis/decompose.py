"""Variance decomposition: the mechanism half of the BRA paper.

Answers H3 (negative-set rule vs clustering threshold) and H4 (seed noise is
small relative to protocol design), on two responses -- score and rank -- because
a factor can move every score without reordering anything, or reorder the top
while barely moving the scores. Only the second is fatal to a leaderboard.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .ranks import mean_pairwise_tau, rank_models

FACTORS = ["model_id", "negative_rule", "cluster_threshold", "metric", "seed"]
INTERACTIONS = [
    ("cluster_threshold", "negative_rule"),   # must be fitted: shuffled negatives are
    ("model_id", "cluster_threshold"),        # never homologous, so strict clustering
    ("model_id", "negative_rule"),            # on them is near-trivial
]


def add_rank_response(df: pd.DataFrame) -> pd.DataFrame:
    """Rank models within each (metric, threshold, negative_rule, seed) cell."""
    cell = ["metric", "cluster_threshold", "negative_rule", "seed"]
    out = df.copy()
    out["rank"] = out.groupby(cell)["score"].transform(lambda s: rank_models(s, True))
    return out


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Bounded metrics compress near the ceiling, which distorts additive variance.

    Decompose AUROC/accuracy on the logit scale as well as the raw scale and
    report both; if they disagree, say so rather than picking the friendlier one.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def variance_components(
    df: pd.DataFrame,
    response: str = "score",
    factors: list[str] | None = None,
    interactions: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Crossed random-effects variance components by the ANOVA (moment) method.

    The Arm B grid is balanced by construction, so this is exact and is the
    pre-registered primary. A REML fit is a sensitivity check, not the headline:
    statsmodels MixedLM handles crossed random effects awkwardly, and disagreement
    between the two is a reason to trust the balanced design, not to panic.

    Returns one row per component with its variance and share of the total.
    """
    factors = factors or [f for f in FACTORS if f in df.columns]
    interactions = interactions if interactions is not None else [
        (a, b) for a, b in INTERACTIONS if a in df.columns and b in df.columns
    ]

    y = df[response].to_numpy(dtype=float)
    grand = y.mean()
    total_ss = float(((y - grand) ** 2).sum())
    n = len(y)

    rows = []

    def _component(name: str, keys: list[str]) -> float:
        g = df.groupby(keys)[response]
        means, counts = g.mean(), g.size()
        return float((counts * (means - grand) ** 2).sum())

    main_ss = {}
    for f in factors:
        ss = _component(f, [f])
        main_ss[f] = ss
        rows.append({"component": f, "type": "main",
                     "df": df[f].nunique() - 1, "ss": ss})

    for a, b in interactions:
        ss_joint = _component(f"{a}:{b}", [a, b])
        ss = max(ss_joint - main_ss.get(a, 0.0) - main_ss.get(b, 0.0), 0.0)
        rows.append({"component": f"{a} x {b}", "type": "interaction",
                     "df": (df[a].nunique() - 1) * (df[b].nunique() - 1), "ss": ss})

    explained = sum(r["ss"] for r in rows)
    rows.append({"component": "residual", "type": "residual",
                 "df": max(n - 1 - sum(r["df"] for r in rows), 1),
                 "ss": max(total_ss - explained, 0.0)})

    out = pd.DataFrame(rows)
    out["pct_variance"] = 100.0 * out["ss"] / total_ss if total_ss > 0 else np.nan
    out["ms"] = out["ss"] / out["df"].clip(lower=1)
    return out.sort_values("pct_variance", ascending=False).reset_index(drop=True)


def decompose_both(df: pd.DataFrame) -> pd.DataFrame:
    """Score-level and rank-level components side by side. This is Figure 4."""
    d = add_rank_response(df)
    s = variance_components(d, "score").set_index("component")["pct_variance"]
    r = variance_components(d, "rank").set_index("component")["pct_variance"]
    out = pd.concat([s.rename("pct_score_var"), r.rename("pct_rank_var")], axis=1)
    out["rank_minus_score"] = out["pct_rank_var"] - out["pct_score_var"]
    return out.sort_values("pct_rank_var", ascending=False)


def tau_ablation(df: pd.DataFrame, metric: str = "mcc") -> pd.DataFrame:
    """Pin each factor to one level, recompute mean tau-b across the rest.

    The rise in tau-b when a factor is held fixed is that factor's contribution to
    disagreement expressed in the units of the headline statistic -- far more legible
    to a reader than a variance percentage, and the most quotable panel in the paper.
    """
    sub = df[df["metric"] == metric].copy()
    results = []

    def _mean_tau(frame: pd.DataFrame, group_cols: list[str]) -> float:
        """Treat each combination of group_cols as a pseudo-benchmark, then compare."""
        piv = (frame.groupby(group_cols + ["model_id"])["score"].mean()
                    .reset_index())
        piv["pseudo"] = piv[group_cols].astype(str).agg("|".join, axis=1)
        rm = (piv.pivot(index="model_id", columns="pseudo", values="score")
                 .rank(ascending=False, method="average"))
        return mean_pairwise_tau(rm) if rm.shape[1] >= 2 else float("nan")

    varying = [f for f in ["cluster_threshold", "negative_rule", "seed"] if f in sub.columns]

    baseline = _mean_tau(sub, varying)
    results.append({"pinned": "none (baseline)", "level": "-",
                    "mean_tau_b": baseline, "delta_vs_baseline": 0.0})

    for f in varying:
        rest = [v for v in varying if v != f]
        if not rest:
            continue
        for lvl in sorted(sub[f].astype(str).unique()):
            frame = sub[sub[f].astype(str) == lvl]
            t = _mean_tau(frame, rest)
            results.append({"pinned": f, "level": lvl, "mean_tau_b": t,
                            "delta_vs_baseline": t - baseline})

    return pd.DataFrame(results)


def representation_vs_learner_stability(df: pd.DataFrame, metric: str = "mcc") -> dict:
    """H5: are representation ranks less stable across protocols than learner ranks?

    Marginalise over one axis, rank the other, and compare mean tau-b across
    protocol cells. Mirrors the single-task precedent (Tian et al. 2026) so the
    numbers are directly comparable.
    """
    sub = df[df["metric"] == metric]

    # Pick cell columns that actually VARY. In Arm A the protocol columns are
    # constant ("native"), so using them yields a single pseudo-cell and a tau-b
    # over a one-column matrix -- which returned a silent NaN before this fix.
    # There, the benchmark itself is the thing being varied.
    candidates = ["cluster_threshold", "negative_rule", "benchmark"]
    cell = [c for c in candidates if c in sub.columns and sub[c].nunique() > 1]
    if not cell:
        return {"error": "no varying protocol or benchmark column; nothing to compare"}

    out = {}
    for axis in ["representation", "learner"]:
        if axis not in sub.columns:
            continue
        m = (sub.groupby(cell + [axis])["score"].mean().reset_index())
        m["pseudo"] = m[cell].astype(str).agg("|".join, axis=1)
        rm = (m.pivot(index=axis, columns="pseudo", values="score")
                .rank(ascending=False, method="average"))
        out[f"{axis}_mean_tau_b"] = mean_pairwise_tau(rm)
    if len(out) == 2:
        out["h5_supported"] = out["representation_mean_tau_b"] < out["learner_mean_tau_b"]
    return out


def mom_variance_components(df: pd.DataFrame, factors: list[str],
                            response: str = "score") -> pd.DataFrame:
    """Exact method-of-moments variance components for a COMPLETE balanced factorial.

    `variance_components` above returns each term's share of the total sum of
    squares. That is an eta-squared, not a variance component: it weights a
    factor by how many levels it happens to have, so a 16-level factor and a
    5-level factor are not on the same footing. The distinction matters because
    the manuscript's claim is about how much of the *variance* a design choice
    owns, and a referee with mixed-model experience will ask for the real thing.

    Here the Arm B grid is a complete balanced factorial with exactly one
    observation per cell, so the orthogonal decomposition is exact and the
    expected-mean-square equations solve in closed form, top-down:

        E[MS_A] = sum over C containing A of  (N / prod_{f in C} n_f) * sigma2_C

    with the highest-order interaction absorbing the residual. No optimiser, no
    convergence failure -- REML on this design hits the parameter boundary and
    does not converge, which is why we do not use it as the primary.

    Moment estimators can return small negative variances where the true value
    is zero; we report them unclipped in `sigma2` and clip only for the
    percentage, because silently clipping is how a structural zero gets
    misreported as a precise measurement.
    """
    cats = {f: sorted(df[f].astype(str).unique()) for f in factors}
    shape = [len(cats[f]) for f in factors]
    n_cells = int(np.prod(shape))

    d = df.copy()
    for f in factors:
        d[f] = pd.Categorical(d[f].astype(str), categories=cats[f])
    cell = d.groupby(factors, observed=True)[response].mean().sort_index()
    if len(cell) != n_cells:
        raise ValueError(f"design is not complete: {len(cell)} of {n_cells} cells present")
    y = cell.to_numpy(dtype=float).reshape(shape)

    grand = y.mean()
    total_ss = float(((y - grand) ** 2).sum())

    def _pure(subset: tuple[str, ...]) -> np.ndarray:
        """Orthogonal effect array: average over the factors outside the subset,
        then centre along each factor inside it."""
        out_axes = tuple(i for i, f in enumerate(factors) if f not in subset)
        a = y.mean(axis=out_axes, keepdims=True) if out_axes else y.copy()
        for i, f in enumerate(factors):
            if f in subset:
                a = a - a.mean(axis=i, keepdims=True)
        return a

    subsets = [s for r in range(1, len(factors) + 1)
               for s in itertools.combinations(factors, r)]
    rep = {s: n_cells / np.prod([len(cats[f]) for f in s]) for s in subsets}
    ss = {s: float(rep[s] * (_pure(s) ** 2).sum()) for s in subsets}
    dof = {s: int(np.prod([len(cats[f]) - 1 for f in s])) for s in subsets}
    ms = {s: ss[s] / dof[s] for s in subsets}

    sigma2: dict[tuple[str, ...], float] = {}
    for r in range(len(factors), 0, -1):
        for s in itertools.combinations(factors, r):
            higher = sum(rep[c] * sigma2[c] for c in subsets
                         if len(c) > r and set(s) <= set(c))
            sigma2[s] = (ms[s] - higher) / rep[s]

    denom = sum(max(v, 0.0) for v in sigma2.values())
    rows = [{
        "component": " x ".join(s),
        "order": len(s),
        "df": dof[s],
        "ss": ss[s],
        "pct_ss": 100.0 * ss[s] / total_ss if total_ss > 0 else np.nan,
        "sigma2": sigma2[s],
        "pct_sigma2": 100.0 * max(sigma2[s], 0.0) / denom if denom > 0 else np.nan,
        "negative_estimate": sigma2[s] < 0,
    } for s in subsets]
    return (pd.DataFrame(rows)
            .sort_values("pct_sigma2", ascending=False)
            .reset_index(drop=True))

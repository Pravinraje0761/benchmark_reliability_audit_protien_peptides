"""Rank-agreement statistics: the symptom half of the BRA paper.

Fully implemented and testable without any benchmark data. Every number in
Figures 2, 3 and 5 comes from here.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def rank_models(scores: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Rank 1 = best. Ties take the average rank, which is why we use tau-b."""
    return scores.rank(ascending=not higher_is_better, method="average")


def leaderboard(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    """Mean score per (benchmark, model) over seeds and folds, plus rank within benchmark."""
    agg = (
        df[df["metric"] == metric]
        .groupby(["benchmark", "model_id"], as_index=False)["score"]
        .mean()
    )
    agg["rank"] = agg.groupby("benchmark")["score"].transform(
        lambda s: rank_models(s, higher_is_better)
    )
    return agg


def rank_matrix(df: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    """models x benchmarks matrix of ranks. The input to everything below."""
    lb = leaderboard(df, metric, higher_is_better)
    return lb.pivot(index="model_id", columns="benchmark", values="rank")


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #

def tau_b(a, b) -> float:
    """Kendall tau-b. Ties corrected; tau-a would be wrong here."""
    return float(stats.kendalltau(a, b, variant="b").statistic)


def tau_null_sd(n: int) -> float:
    """SD of tau under the null of two independent random rankings.

    At n=12 this is about 0.22. Plot it as a band on Figure 2 -- without it a
    reader will over-interpret a tau of 0.35 as agreement.
    """
    return math.sqrt(2.0 * (2 * n + 5) / (9.0 * n * (n - 1)))


def pairwise_tau(rm: pd.DataFrame) -> pd.DataFrame:
    """Symmetric benchmark x benchmark matrix of tau-b."""
    cols = list(rm.columns)
    out = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    for i, j in itertools.combinations(cols, 2):
        sub = rm[[i, j]].dropna()
        t = tau_b(sub[i], sub[j]) if len(sub) >= 3 else np.nan
        out.loc[i, j] = out.loc[j, i] = t
    return out


def mean_pairwise_tau(rm: pd.DataFrame) -> float:
    """Mean of the off-diagonal tau-b values. NaN if there is nothing to compare.

    Returns NaN explicitly for a single-column matrix rather than letting
    np.nanmean warn on an empty slice -- a silent NaN that looks like a result
    is worse than one that announces itself.
    """
    if rm.shape[1] < 2:
        return float("nan")
    pt = pairwise_tau(rm)
    vals = pt.values[np.triu_indices_from(pt.values, k=1)]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else float("nan")


# --------------------------------------------------------------------------- #
# The agreement ceiling
# --------------------------------------------------------------------------- #

def ceiling_from_seeds(df: pd.DataFrame, metric: str,
                       higher_is_better: bool = True) -> pd.DataFrame:
    """Within-benchmark rank reproducibility across seeds: the agreement CEILING.

    Why this matters. tau-b cannot reach 1.0 even between two runs of the SAME
    benchmark under the SAME protocol, because seed noise alone reorders
    near-tied models. The design simulation puts that ceiling at 0.96 for 5%
    run noise and 0.91 for 10% (results/tables/power_followup_ceiling.csv).

    So a between-benchmark tau-b of 0.85 does not mean "15% disagreement" -- it
    may already be at the ceiling, i.e. as much agreement as the measurement
    permits. Comparing observed tau-b against 1.0 overstates disagreement.

    Because Arm A and Arm B both run 5 seeds, the ceiling comes for free: rank
    models within each (benchmark, seed), then take tau-b between seed pairs of
    the same benchmark. The protocol is held constant, so what remains is run noise.

    Returns one row per benchmark with its ceiling and the number of seed pairs.
    """
    sub = df[df["metric"] == metric]
    rows = []
    for bench, g in sub.groupby("benchmark"):
        per_seed = (g.groupby(["seed", "model_id"], as_index=False)["score"].mean()
                     .pivot(index="model_id", columns="seed", values="score")
                     .rank(ascending=not higher_is_better, method="average"))
        seeds = list(per_seed.columns)
        taus = [tau_b(per_seed[a], per_seed[b])
                for a, b in itertools.combinations(seeds, 2)
                if per_seed[[a, b]].dropna().shape[0] >= 3]
        rows.append({
            "benchmark": bench,
            "ceiling_tau_b": float(np.nanmean(taus)) if taus else np.nan,
            "ceiling_sd": float(np.nanstd(taus, ddof=1)) if len(taus) > 1 else np.nan,
            "n_seed_pairs": len(taus),
            "n_seeds": len(seeds),
        })
    return pd.DataFrame(rows)


def normalised_agreement(df: pd.DataFrame, metric: str,
                         higher_is_better: bool = True) -> dict:
    """Between-benchmark agreement as a fraction of the within-benchmark ceiling.

        normalised = mean between-benchmark tau-b / mean within-benchmark ceiling

    Near 1.0 means the benchmarks agree about as well as a benchmark agrees with
    itself -- they are effectively interchangeable. Near 0 means the protocol,
    not run noise, drives the ranking. This is the quantity H1 should be stated
    in; reporting raw tau-b alone invites the comparison against 1.0 that
    overstates disagreement.
    """
    rm = rank_matrix(df, metric, higher_is_better)
    between = mean_pairwise_tau(rm)
    ceil_df = ceiling_from_seeds(df, metric, higher_is_better)
    ceiling = float(np.nanmean(ceil_df["ceiling_tau_b"]))
    ok = bool(ceiling) and np.isfinite(ceiling)
    return {
        "between_benchmark_tau_b": between,
        "within_benchmark_ceiling": ceiling,
        "normalised_agreement": between / ceiling if ok else np.nan,
        "per_benchmark_ceiling": ceil_df,
    }


def footrule(a, b) -> float:
    """Spearman footrule: total absolute rank displacement, in positions."""
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def rbo(a_order: list, b_order: list, p: float = 0.9) -> float:
    """Rank-biased overlap: top-weighted agreement.

    Leaderboards are read from the top, so a swap at ranks 11/12 should not
    count the same as a swap at 1/2. `a_order` and `b_order` are lists of
    model ids, best first.
    """
    if not a_order or not b_order:
        return 0.0
    k = min(len(a_order), len(b_order))
    total, a_set, b_set = 0.0, set(), set()
    for d in range(1, k + 1):
        a_set.add(a_order[d - 1])
        b_set.add(b_order[d - 1])
        total += (p ** (d - 1)) * (len(a_set & b_set) / d)
    return float((1 - p) * total)


def inversions(a: pd.Series, b: pd.Series, top_k: int | None = None) -> dict:
    """Count pairwise order reversals between two rankings.

    With top_k, restrict to pairs where either model is in the top-k of either
    ranking -- the subset a practitioner would actually notice.
    """
    models = list(a.index)
    if top_k is not None:
        keep = set(a.nsmallest(top_k).index) | set(b.nsmallest(top_k).index)
    else:
        keep = set(models)

    n_inv = n_pairs = 0
    for x, y in itertools.combinations(models, 2):
        if x not in keep and y not in keep:
            continue
        n_pairs += 1
        if (a[x] - a[y]) * (b[x] - b[y]) < 0:
            n_inv += 1
    return {"inversions": n_inv, "pairs": n_pairs,
            "fraction": n_inv / n_pairs if n_pairs else float("nan")}


def top1_flip_rate(rm: pd.DataFrame) -> dict:
    """Fraction of benchmark pairs that name a different best model.

    Crude, robust at small n, and the number that belongs in the abstract.
    """
    winners = {c: rm[c].idxmin() for c in rm.columns}
    pairs = list(itertools.combinations(rm.columns, 2))
    flips = [(i, j) for i, j in pairs if winners[i] != winners[j]]
    return {
        "winners": winners,
        "n_pairs": len(pairs),
        "n_flips": len(flips),
        "rate": len(flips) / len(pairs) if pairs else float("nan"),
        "flipped_pairs": flips,
    }


def rank_range(rm: pd.DataFrame) -> pd.DataFrame:
    """Per model: best rank, worst rank, spread. Feeds the bump chart."""
    return pd.DataFrame({
        "best": rm.min(axis=1),
        "worst": rm.max(axis=1),
        "range": rm.max(axis=1) - rm.min(axis=1),
        "mean": rm.mean(axis=1),
    }).sort_values("mean")


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #

def bootstrap_mean_tau(
    df: pd.DataFrame,
    metric: str,
    n_boot: int = 2000,
    seed: int = 0,
    higher_is_better: bool = True,
) -> dict:
    """Percentile CI for mean pairwise tau-b, resampling run rows within benchmark.

    Note: this estimates evaluation-set uncertainty, which is exactly the thing a
    leaderboard reader implicitly assumes away. For the pre-registered BCa interval
    over test *sequences*, resample sequences upstream and feed per-resample scores
    in here -- the estimator below is unchanged.
    """
    rng = np.random.default_rng(seed)
    sub = df[df["metric"] == metric].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"no rows with metric={metric!r}")

    # Resample row positions per benchmark with plain numpy. The earlier
    # groupby.apply version silently produced zero usable resamples under
    # pandas 3 and a bare `except: continue` hid it -- the exact silent-failure
    # pattern this project is written to avoid. Failures now surface.
    groups = {b: g.to_numpy() for b, g in sub.groupby("benchmark").groups.items()}
    vals = np.empty(n_boot)
    first_error: Exception | None = None
    n_failed = 0

    for i in range(n_boot):
        take = np.concatenate([rng.choice(idx, size=len(idx), replace=True)
                               for idx in groups.values()])
        try:
            vals[i] = mean_pairwise_tau(
                rank_matrix(sub.iloc[take], metric, higher_is_better))
        except Exception as e:          # noqa: BLE001 - recorded, not swallowed
            vals[i] = np.nan
            n_failed += 1
            first_error = first_error or e

    n_ok = int(np.isfinite(vals).sum())
    if n_ok == 0:
        raise RuntimeError(
            f"all {n_boot} bootstrap resamples failed; first error was "
            f"{type(first_error).__name__}: {first_error}"
        )
    if n_failed:
        print(f"[warn] {n_failed}/{n_boot} bootstrap resamples failed "
              f"({type(first_error).__name__}: {first_error})")

    return {
        "point": mean_pairwise_tau(rank_matrix(sub, metric, higher_is_better)),
        "lo95": float(np.nanpercentile(vals, 2.5)),
        "hi95": float(np.nanpercentile(vals, 97.5)),
        "n_boot": n_ok,
    }


def permutation_null(n_models: int, n_benchmarks: int, n_perm: int = 5000, seed: int = 0) -> np.ndarray:
    """Empirical null distribution of mean pairwise tau-b. Draws the band on Figure 2.

    Pure numpy: pandas construction inside this loop made the design simulation
    unusably slow (~35k studies), so the hot path avoids DataFrames entirely.
    """
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(range(n_benchmarks), 2))
    out = np.empty(n_perm)
    for i in range(n_perm):
        rk = np.array([rng.permutation(n_models) + 1 for _ in range(n_benchmarks)])
        out[i] = np.mean([tau_b(rk[a], rk[b]) for a, b in pairs])
    return out


def flip_ceiling_matched(df: pd.DataFrame, level_factor: str,
                         hold: str | None = None, metric: str = "mcc",
                         higher_is_better: bool = True,
                         n_boot: int = 20000, seed: int = 0) -> dict:
    """Top-1 flip rate against its own measured CEILING, matched on seed.

    `top1_flip_rate` compares the winner across levels and is quoted against an
    implicit floor of zero. That is the same mistake `ceiling_from_seeds` exists
    to prevent for tau-b: reseeding alone reorders near-tied pipelines, so some
    winner flips are run noise rather than protocol sensitivity. A referee who
    accepts our ceiling argument for tau-b will ask for it here, and should.

    The contrast is matched rather than marginal:

      * across levels, holding the seed fixed -- the quantity of interest;
      * across seeds, holding the level fixed -- the ceiling.

    Averaging over seeds before taking the winner (as `top1_flip_rate` does)
    suppresses run noise in the first term but not in the second, so the two
    are not comparable; fixing the seed in both makes them so. We report the
    difference with a bootstrap interval over pairs, and the *normalised flip
    excess*, (observed - ceiling) / (1 - ceiling), which is the share of the
    available room above the noise floor that the factor actually uses.
    """
    sub = df[df["metric"] == metric]
    pick = (lambda g: g.loc[g["score"].idxmax(), "model_id"]) if higher_is_better \
        else (lambda g: g.loc[g["score"].idxmin(), "model_id"])

    def _winners(frame: pd.DataFrame, by: str) -> pd.Series:
        return frame.groupby(by)[["score", "model_id"]].apply(pick)

    def _flips(group_cols: list[str], by: str) -> list[int]:
        out = []
        for _, g in sub.groupby(group_cols, observed=True):
            w = _winners(g, by)
            out += [0 if w[a] == w[b] else 1
                    for a, b in itertools.combinations(list(w.index), 2)]
        return out

    across_levels = np.array(_flips([c for c in [hold, "seed"] if c], level_factor))
    across_seeds = np.array(_flips([c for c in [hold, level_factor] if c], "seed"))
    if across_levels.size == 0 or across_seeds.size == 0:
        return {"error": "not enough levels or seeds to form pairs"}

    rng = np.random.default_rng(seed)
    obs, ceil = float(across_levels.mean()), float(across_seeds.mean())
    boot = (rng.choice(across_levels, (n_boot, across_levels.size)).mean(axis=1)
            - rng.choice(across_seeds, (n_boot, across_seeds.size)).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "factor": level_factor,
        "flip_across_levels": obs, "n_level_pairs": int(across_levels.size),
        "flip_ceiling_seeds": ceil, "n_seed_pairs": int(across_seeds.size),
        "difference": obs - ceil, "diff_lo95": float(lo), "diff_hi95": float(hi),
        "p_two_sided": float(max(p, 1.0 / n_boot)),
        "normalised_flip_excess": (obs - ceil) / (1.0 - ceil) if ceil < 1 else np.nan,
        "n_boot": n_boot,
    }

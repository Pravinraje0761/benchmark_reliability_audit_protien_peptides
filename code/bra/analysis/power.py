"""Design simulation: can this study actually detect what it claims to?

The pre-registration commits to rejecting H1 when the upper CI bound on mean
pairwise Kendall tau-b falls below 0.8, with 5 benchmarks and 12 models. Nothing
in the literature says whether that is resolvable at those sample sizes. Since
tau-b over 12 items is noisy (null SD 0.22), this has to be checked BEFORE the
threshold is locked, not after the result disappoints.

Generative model
----------------
Each model m has a latent quality theta_m shared across benchmarks. Each
benchmark b scores it as

    score[b, m] = theta_m + sigma_p * eta[b, m] + sigma_e * eps[b, m]

where eta is a benchmark-specific *protocol* effect (the thing BRA is about: a
negative-set rule that favours cationic-charge models, a clustering threshold
that starves data-hungry ones) and eps is run noise. sigma_p = 0 means the
benchmarks are interchangeable and tau-b is 1 up to noise; large sigma_p means
each benchmark measures its own protocol.

Caveat, stated plainly: this captures sampling variability over models and
protocol effects, which is the dominant term for a rank statistic computed on 12
items. It does not model test-set resampling, which is what the pre-registered
BCa bootstrap covers. The two are complementary, and the simulation is the
conservative one for threshold-setting because protocol variance is the larger term.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as sps

# --------------------------------------------------------------------------- #
# Generative model
# --------------------------------------------------------------------------- #

# Hot path is pure numpy. The first version of this module built a DataFrame per
# simulated study, which cost ~15 ms and made the full sweep (~35k studies)
# unusable. Nothing below the public API touches pandas.

def _simulate_ranks(n_models: int, n_benchmarks: int, sigma_p: float,
                    sigma_e: float, rng: np.random.Generator) -> np.ndarray:
    """(n_benchmarks, n_models) array of ranks, 1 = best, ties averaged."""
    theta = rng.normal(0.0, 1.0, size=n_models)
    eta = rng.normal(0.0, 1.0, size=(n_benchmarks, n_models))
    eps = rng.normal(0.0, 1.0, size=(n_benchmarks, n_models))
    scores = theta[None, :] + sigma_p * eta + sigma_e * eps
    return sps.rankdata(-scores, axis=1)


def _pairwise_taus(ranks: np.ndarray) -> np.ndarray:
    n_b = ranks.shape[0]
    return np.array([sps.kendalltau(ranks[a], ranks[b], variant="b").statistic
                     for a, b in itertools.combinations(range(n_b), 2)])


def _top1_flips(ranks: np.ndarray) -> float:
    winners = ranks.argmin(axis=1)
    pairs = list(itertools.combinations(range(len(winners)), 2))
    if not pairs:
        return float("nan")
    return sum(winners[a] != winners[b] for a, b in pairs) / len(pairs)


def simulate_study(
    n_models: int = 12,
    n_benchmarks: int = 5,
    sigma_p: float = 0.3,
    sigma_e: float = 0.05,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """models x benchmarks matrix of RANKS (1 = best). For inspection/plotting."""
    rng = rng or np.random.default_rng()
    rk = _simulate_ranks(n_models, n_benchmarks, sigma_p, sigma_e, rng)
    return pd.DataFrame(rk.T,
                        index=[f"M{i + 1:02d}" for i in range(n_models)],
                        columns=[f"B{j + 1}" for j in range(n_benchmarks)])


@lru_cache(maxsize=1024)
def population_tau(sigma_p: float, sigma_e: float = 0.05,
                   n_models: int = 20000, seed: int = 12345) -> float:
    """Expected pairwise tau-b at this protocol sensitivity.

    Estimated with a very large model set so the item-sampling noise that
    plagues n=12 is negligible. This is the truth the n=12 estimator is chasing,
    and it makes every result below interpretable in tau units rather than in
    units of an invented sigma. Cached: called repeatedly with identical args.
    """
    rng = np.random.default_rng(seed)
    rk = _simulate_ranks(n_models, 2, sigma_p, sigma_e, rng)
    return float(sps.kendalltau(rk[0], rk[1], variant="b").statistic)


def sigma_for_tau(target: float, cal: pd.DataFrame | None = None) -> float:
    """Invert the calibration: which protocol sensitivity gives this tau-b?"""
    cal = cal if cal is not None else calibrate()
    i = int(np.argmin(np.abs(cal["population_tau_b"].to_numpy() - target)))
    return float(cal["sigma_p"].iloc[i])


def calibrate(sigma_grid: np.ndarray | None = None, **kw) -> pd.DataFrame:
    """Map protocol sensitivity to population tau-b, so results are interpretable
    in tau units rather than in units of an invented sigma."""
    sigma_grid = sigma_grid if sigma_grid is not None else np.concatenate(
        [[0.0], np.geomspace(0.02, 3.0, 24)])
    return pd.DataFrame({
        "sigma_p": sigma_grid,
        "population_tau_b": [population_tau(s, **kw) for s in sigma_grid],
    })


# --------------------------------------------------------------------------- #
# Estimator behaviour
# --------------------------------------------------------------------------- #

def _within_study_se(taus: np.ndarray, n_benchmarks: int) -> float:
    """SE of the mean of pairwise tau values, from the spread of those pairs.

    This is what we can compute from a single real study, so the simulation must
    use the same quantity -- otherwise the power estimate flatters a procedure we
    could not actually run.
    """
    vals = taus[np.isfinite(taus)]
    if len(vals) < 2:
        return float("nan")
    # Pairwise taus are correlated (they share benchmarks), so the naive
    # independent-sample SE understates. Divide by sqrt(n_benchmarks - 1) rather
    # than sqrt(n_pairs): the conservative round-robin correction.
    return float(vals.std(ddof=1) / np.sqrt(max(n_benchmarks - 1, 1)))


def estimator_distribution(
    n_models: int = 12,
    n_benchmarks: int = 5,
    sigma_p: float = 0.3,
    sigma_e: float = 0.05,
    n_studies: int = 1000,
    threshold: float = 0.8,
    seed: int = 0,
) -> dict:
    """Sampling distribution of mean pairwise tau-b, plus power at the threshold."""
    rng = np.random.default_rng(seed)
    ests = np.empty(n_studies)
    uppers = np.empty(n_studies)
    flips = np.empty(n_studies)

    for i in range(n_studies):
        rk = _simulate_ranks(n_models, n_benchmarks, sigma_p, sigma_e, rng)
        taus = _pairwise_taus(rk)
        est = float(np.nanmean(taus))
        se = _within_study_se(taus, n_benchmarks)
        ests[i] = est
        uppers[i] = est + 1.96 * se if np.isfinite(se) else np.nan
        flips[i] = _top1_flips(rk)

    truth = population_tau(sigma_p, sigma_e)
    return {
        "n_models": n_models,
        "n_benchmarks": n_benchmarks,
        "sigma_p": sigma_p,
        "population_tau_b": truth,
        "est_mean": float(np.nanmean(ests)),
        "est_sd": float(np.nanstd(ests, ddof=1)),
        "est_q025": float(np.nanpercentile(ests, 2.5)),
        "est_q975": float(np.nanpercentile(ests, 97.5)),
        # Power = P(we correctly declare disagreement) under this truth
        "power_reject_H1": float(np.nanmean(uppers < threshold)),
        "mean_top1_flip_rate": float(np.nanmean(flips)),
        "top1_flip_sd": float(np.nanstd(flips, ddof=1)),
        "bias": float(np.nanmean(ests) - truth),
    }


def power_curve(
    taus_wanted=(0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
    n_models: int = 12,
    n_benchmarks: int = 5,
    n_studies: int = 1000,
    threshold: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """Power to reject H1 across a grid of true population tau-b values."""
    cal = calibrate()
    rows = []
    for t in taus_wanted:
        r = estimator_distribution(n_models, n_benchmarks, sigma_for_tau(t, cal),
                                   n_studies=n_studies, threshold=threshold,
                                   seed=seed)
        r["target_tau"] = t
        rows.append(r)
    return pd.DataFrame(rows)


def design_sweep(
    model_counts=(8, 12, 16, 20),
    benchmark_counts=(3, 4, 5, 6),
    target_tau: float = 0.6,
    n_studies: int = 500,
    threshold: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """How much does the estimator tighten if we add models, or benchmarks?

    Decision-relevant and asymmetric in cost: adding a model is one more pipeline
    in the zoo; adding a benchmark means acquisition, a loader and a reproduction
    gate. Worth knowing which one buys precision before paying for either.
    """
    s = sigma_for_tau(target_tau)
    rows = []
    for m in model_counts:
        for b in benchmark_counts:
            rows.append(estimator_distribution(m, b, s, n_studies=n_studies,
                                               threshold=threshold, seed=seed))
    return pd.DataFrame(rows)

"""End-to-end smoke test of the analysis layer on a synthetic results table.

Exercises exactly the code path that will consume the real Arm A parquet, and
needs no sklearn -- so it runs despite the Application Control block recorded in
E01. Run this after any change to bra/analysis/.

    python scripts/smoke_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.analysis import decompose, ranks  # noqa: E402

MODELS = [f"M{i:02d}" for i in range(1, 21)]
BENCHES = ["B1_genpept", "B3_ppb", "B4_ampbenchmark", "B5_autopeptideml", "B6_abp"]


def synthetic_results(protocol_sd: float = 0.45, run_sd: float = 0.05,
                      seed: int = 0) -> pd.DataFrame:
    """Arm-A-shaped results with a known protocol effect planted per benchmark."""
    rng = np.random.default_rng(seed)
    theta = dict(zip(MODELS, rng.normal(0, 1, len(MODELS))))
    rows = []
    for b in BENCHES:
        protocol = {m: rng.normal(0, protocol_sd) for m in MODELS}
        for sd in range(5):
            for m in MODELS:
                score = 0.60 + 0.08 * (theta[m] + protocol[m] + rng.normal(0, run_sd))
                rows.append({
                    "benchmark": b, "model_id": m, "seed": sd, "metric": "mcc",
                    "score": score,
                    "cluster_threshold": "native", "negative_rule": "native",
                    "representation": f"R{int(m[1:]) % 10}",
                    "learner": f"L{int(m[1:]) % 7}",
                })
    return pd.DataFrame(rows)


def main() -> None:
    df = synthetic_results()
    rm = ranks.rank_matrix(df, "mcc")
    print(f"rank matrix              : {rm.shape[0]} models x {rm.shape[1]} benchmarks")
    print(f"mean pairwise tau-b      : {ranks.mean_pairwise_tau(rm):.3f}")

    na = ranks.normalised_agreement(df, "mcc")
    print(f"within-benchmark ceiling : {na['within_benchmark_ceiling']:.3f}")
    print(f"NORMALISED AGREEMENT     : {na['normalised_agreement']:.3f}")

    flip = ranks.top1_flip_rate(rm)
    print(f"top-1 flip rate          : {flip['n_flips']}/{flip['n_pairs']}")
    print(f"largest rank range       : {ranks.rank_range(rm)['range'].max():.0f} positions")
    print(f"H5 probe                 : {decompose.representation_vs_learner_stability(df, 'mcc')}")

    boot = ranks.bootstrap_mean_tau(df, "mcc", n_boot=120, seed=1)
    print(f"bootstrap mean tau-b     : {boot['point']:.3f} "
          f"[{boot['lo95']:.3f}, {boot['hi95']:.3f}]  (n_boot={boot['n_boot']})")

    # Sanity: a no-protocol-effect world must look interchangeable.
    clean = synthetic_results(protocol_sd=0.0, seed=2)
    na_clean = ranks.normalised_agreement(clean, "mcc")
    print(f"\nzero-protocol-effect control:")
    print(f"  tau-b {na_clean['between_benchmark_tau_b']:.3f} "
          f"ceiling {na_clean['within_benchmark_ceiling']:.3f} "
          f"normalised {na_clean['normalised_agreement']:.3f}  <- should be near 1.0")

    assert na_clean["normalised_agreement"] > 0.9, "control world should look interchangeable"
    assert na["normalised_agreement"] < na_clean["normalised_agreement"]
    print("\nSMOKE TEST PASS")


if __name__ == "__main__":
    main()

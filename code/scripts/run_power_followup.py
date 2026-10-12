"""Follow-up: what zoo size rescues the pre-registered 0.8 threshold, and how
much does the achievable tau-b ceiling depend on run noise?

Run after run_power_analysis.py. Writes results/tables/power_followup_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.analysis import power  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 200)

FMT = lambda v: f"{v:.3f}"


def main() -> None:
    print("=" * 78)
    print("[A] Ceiling: max achievable population tau-b vs run noise (sigma_p = 0)")
    print("=" * 78)
    rows = []
    for se in [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]:
        rows.append({"sigma_e": se, "ceiling_tau_b": power.population_tau(0.0, se)})
    ceil = pd.DataFrame(rows)
    ceil.to_csv(OUT / "power_followup_ceiling.csv", index=False)
    print(ceil.to_string(index=False, float_format=FMT))
    print("\n  Interpretation: tau-b cannot reach 1.0 even with identical protocols,")
    print("  because run noise alone reorders near-tied models. The 0.8 decision")
    print("  threshold must be read against THIS ceiling, not against 1.0.")

    print("\n" + "=" * 78)
    print("[B] Power at the pre-registered 0.8 threshold, by zoo size")
    print("=" * 78)
    cal = power.calibrate()
    rows = []
    for target in [0.85, 0.80, 0.75, 0.70]:
        s = power.sigma_for_tau(target, cal)
        for m in [12, 16, 20, 24, 30, 40]:
            r = power.estimator_distribution(m, 5, s, n_studies=600, seed=7)
            rows.append({"target_tau": target,
                         "population_tau_b": r["population_tau_b"],
                         "n_models": m,
                         "est_sd": r["est_sd"],
                         "power_reject_H1": r["power_reject_H1"]})
    zoo = pd.DataFrame(rows)
    zoo.to_csv(OUT / "power_followup_zoo.csv", index=False)
    print("\n  power to reject H1 (upper 95% CI < 0.80), n_benchmarks = 5:")
    print(zoo.pivot(index="n_models", columns="target_tau",
                    values="power_reject_H1").to_string(float_format=FMT))
    print("\n  estimator SD:")
    print(zoo.pivot(index="n_models", columns="target_tau",
                    values="est_sd").to_string(float_format=FMT))

    print("\n" + "=" * 78)
    print("[C] Minimum detectable disagreement at 80% power")
    print("=" * 78)
    rows = []
    for m in [12, 16, 20, 24, 30]:
        found = None
        for target in np.arange(0.84, 0.30, -0.02):
            s = power.sigma_for_tau(float(target), cal)
            r = power.estimator_distribution(m, 5, s, n_studies=400, seed=11)
            if r["power_reject_H1"] >= 0.80:
                found = r["population_tau_b"]
                break
        rows.append({"n_models": m, "mdd_tau_b_at_80pct_power": found})
    mdd = pd.DataFrame(rows)
    mdd.to_csv(OUT / "power_followup_mdd.csv", index=False)
    print(mdd.to_string(index=False, float_format=FMT))
    print("\n  Read as: with this zoo size, we can reliably confirm disagreement")
    print("  only when the TRUE mean tau-b is at or below this value.")

    print(f"\nTables written to {OUT}")


if __name__ == "__main__":
    main()

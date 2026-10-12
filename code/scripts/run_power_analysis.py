"""Run the BRA design simulation and write the tables the pre-registration needs.

    python scripts/run_power_analysis.py

Outputs to ../results/tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra.analysis import power, ranks  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)


def main() -> None:
    print("=" * 78)
    print("BRA design simulation")
    print("=" * 78)

    # ---------------------------------------------------------------- null band
    print("\n[1] Null reference: mean pairwise tau-b when benchmarks are unrelated")
    for m, b in [(12, 5), (12, 2), (20, 5)]:
        null = ranks.permutation_null(m, b, n_perm=4000, seed=1)
        print(f"  n_models={m:>3} n_benchmarks={b}: "
              f"mean {null.mean():+.3f}  sd {null.std(ddof=1):.3f}  "
              f"95% [{np.percentile(null, 2.5):+.3f}, {np.percentile(null, 97.5):+.3f}]")

    # --------------------------------------------------------------- calibration
    print("\n[2] Calibration: protocol sensitivity -> population tau-b")
    cal = power.calibrate()
    cal.to_csv(OUT / "power_calibration.csv", index=False)
    print(cal.iloc[::3].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -------------------------------------------------------------- power curve
    print("\n[3] Power to reject H1 (upper 95% CI < 0.80) at n_models=12, n_benchmarks=5")
    pc = power.power_curve(n_models=12, n_benchmarks=5, n_studies=1500, seed=2)
    cols = ["target_tau", "population_tau_b", "est_mean", "est_sd",
            "est_q025", "est_q975", "power_reject_H1", "mean_top1_flip_rate"]
    pc[cols].to_csv(OUT / "power_curve.csv", index=False)
    print(pc[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------------- design sweep
    print("\n[4] Design sweep: estimator SD by zoo size and benchmark count (sigma_p=0.3)")
    ds = power.design_sweep(n_studies=600, seed=3)
    ds.to_csv(OUT / "power_design_sweep.csv", index=False)
    piv = ds.pivot(index="n_models", columns="n_benchmarks", values="est_sd")
    print("\n  estimator SD:")
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    pw = ds.pivot(index="n_models", columns="n_benchmarks", values="power_reject_H1")
    print("\n  power to reject H1:")
    print(pw.to_string(float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------- top-1 flip linkage
    print("\n[5] Is top-1 flip rate a better-behaved headline than tau-b?")
    rows = []
    for t in [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]:
        i = int(np.abs(cal["population_tau_b"] - t).argmin())
        s = float(cal["sigma_p"].iloc[i])
        r = power.estimator_distribution(12, 5, s, n_studies=800, seed=4)
        rows.append({"population_tau_b": r["population_tau_b"],
                     "tau_est_sd": r["est_sd"],
                     "mean_top1_flip_rate": r["mean_top1_flip_rate"]})
    flip = pd.DataFrame(rows)
    flip.to_csv(OUT / "power_flip_rate.csv", index=False)
    print(flip.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nTables written to {OUT}")


if __name__ == "__main__":
    main()

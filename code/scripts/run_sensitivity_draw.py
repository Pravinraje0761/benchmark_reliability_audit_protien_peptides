"""Sensitivity: how much does the NEGATIVE DRAW itself move the ranking?

The main factorial fixes the negative-set realisation per rule (seed 0) and lets
the seed vary only the partition and model initialisation, because embedding
129,750 distinct negatives was out of reach on CPU. That leaves a gap the Methods
section states openly: draw-to-draw variability is not inside the reported seed
component.

This closes it. We re-run five independent draws per rule on the
**embedding-free** pipelines only, where no ESM cache is needed and the cost is
minutes rather than hours. If ranking disagreement across draws is small
relative to disagreement across rules, the main result is not an artefact of
having fixed the draw -- which is the thing a reviewer will and should ask.

    python scripts/run_sensitivity_draw.py

Writes results/tables/sensitivity_negative_draw.csv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra.analysis import ranks  # noqa: E402
from bra.armb import NEG_RULES, THRESHOLDS, ZOO_B, Pipe, data_root, run_cell  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

# Embedding-free subset: everything whose representation is computed from the
# sequence directly. Keeps the null baseline, which is the most protocol-sensitive.
ZOO_CHEAP: list[Pipe] = [p for p in ZOO_B if not p.rep.startswith("esm2")]

DRAWS = [0, 1, 2, 3, 4]
FIXED_SEED = 0              # hold partition/init fixed so the draw is the only thing varying
THRESHOLD = "0.7"           # one representative threshold; the grid covers the rest


def main() -> None:
    print(f"pipelines (embedding-free): {len(ZOO_CHEAP)} -> "
          f"{[p.pid for p in ZOO_CHEAP]}")
    print(f"rules {len(NEG_RULES)} x draws {len(DRAWS)} = "
          f"{len(NEG_RULES)*len(DRAWS)} cells at threshold {THRESHOLD}\n")

    corpus = data_root() / "corpus"

    # One file per cell, written the moment that cell finishes, and reused on a
    # later run. An earlier version accumulated everything in memory and wrote
    # once at the end, so a run stopped twenty minutes in produced nothing at
    # all. Each cell costs an MMseqs2 clustering plus a leakage search, which
    # makes discarding completed cells the one clearly avoidable waste here.
    cells = OUT / "sensitivity_draw_cells"
    cells.mkdir(parents=True, exist_ok=True)

    rows = []
    for rule in NEG_RULES:
        for draw in DRAWS:
            f = corpus / f"armB_{rule}_seed{draw}.parquet"
            if not f.exists():
                print(f"  missing {f.name}; skipping")
                continue

            part = cells / f"{rule}_draw{draw}.csv"
            if part.exists():
                rows += pd.read_csv(part).to_dict(orient="records")
                print(f"  {rule:22s} draw {draw}  resumed from cache", flush=True)
                continue

            df = pd.read_parquet(f)
            r = run_cell(df, THRESHOLD, rule, FIXED_SEED, zoo=ZOO_CHEAP)
            for rec in r:
                rec["negative_draw"] = draw
            # write before extending `rows`, so an interruption mid-loop leaves a
            # complete file for every cell that actually finished
            pd.DataFrame(r).to_csv(part, index=False)
            rows += r
            mcc = [x["score"] for x in r if x["metric"] == "mcc"]
            print(f"  {rule:22s} draw {draw}  mcc {min(mcc):.3f}-{max(mcc):.3f}",
                  flush=True)

    res = pd.DataFrame(rows)
    res = res[res.metric == "mcc"]
    res.to_csv(OUT / "sensitivity_negative_draw_raw.csv", index=False)

    # Disagreement ACROSS DRAWS within a rule: the quantity in question.
    print("\n" + "=" * 70)
    print("Rank agreement across independent negative DRAWS, within each rule")
    print("=" * 70)
    out = []
    for rule, g in res.groupby("negative_rule"):
        piv = (g.groupby(["negative_draw", "model_id"], as_index=False)["score"].mean()
                .pivot(index="model_id", columns="negative_draw", values="score")
                .rank(ascending=False, method="average"))
        taus = [ranks.tau_b(piv[a], piv[b])
                for a, b in itertools.combinations(piv.columns, 2)]
        out.append({"negative_rule": rule, "mean_tau_b_across_draws": float(np.mean(taus)),
                    "min_tau_b": float(np.min(taus)), "n_pairs": len(taus)})
    across_draws = pd.DataFrame(out)
    print(across_draws.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Disagreement ACROSS RULES, same pipelines, for the comparison that matters.
    piv_rule = (res.groupby(["negative_rule", "model_id"], as_index=False)["score"].mean()
                  .pivot(index="model_id", columns="negative_rule", values="score")
                  .rank(ascending=False, method="average"))
    tau_rules = ranks.mean_pairwise_tau(piv_rule)
    mean_draws = float(across_draws["mean_tau_b_across_draws"].mean())

    print(f"\nmean tau-b across DRAWS (within rule) : {mean_draws:.3f}")
    print(f"mean tau-b across RULES               : {tau_rules:.3f}")
    print(f"ratio                                 : {tau_rules/mean_draws:.3f}")
    print("\nReading: if agreement across draws is high while agreement across")
    print("rules is low, then fixing the draw in the main factorial did not")
    print("manufacture the disagreement -- the rule did.")

    across_draws.to_csv(OUT / "sensitivity_negative_draw.csv", index=False)
    pd.DataFrame([{"mean_tau_b_across_draws": mean_draws,
                   "mean_tau_b_across_rules": tau_rules,
                   "n_pipelines": piv_rule.shape[0],
                   "threshold": THRESHOLD}]).to_csv(
        OUT / "sensitivity_draw_summary.csv", index=False)
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

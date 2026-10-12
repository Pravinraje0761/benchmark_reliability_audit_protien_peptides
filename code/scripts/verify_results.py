"""Pre-flight verification of the Arm B output. Run BEFORE reading any result.

    python scripts/verify_results.py

Every check that fails prints FAIL and the script exits non-zero. The point is
that no number reaches the manuscript from an output that has not passed this.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.armb import NEG_RULES, SEEDS, THRESHOLDS, ZOO_B, data_root  # noqa: E402

FAILURES: list[str] = []
EXPECTED_METRICS = {"auroc", "auprc", "mcc", "bacc", "f1", "accuracy"}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def main() -> None:
    src = data_root() / "runs" / "all_runs_armB.parquet"
    if not src.exists():
        raise SystemExit(f"no results at {src}; run the grid and merge first")
    df = pd.read_parquet(src)
    print(f"loaded {len(df)} rows\n")

    print("1. Design completeness")
    n_cells_expected = len(THRESHOLDS) * len(NEG_RULES) * len(SEEDS)
    cells = df.groupby(["cluster_threshold", "negative_rule", "seed"]).ngroups
    check("all cells present", cells == n_cells_expected,
          f"{cells} of {n_cells_expected}")
    check("all pipelines present", df.model_id.nunique() == len(ZOO_B),
          f"{df.model_id.nunique()} of {len(ZOO_B)}")
    check("all thresholds present", set(df.cluster_threshold) == set(THRESHOLDS))
    check("all rules present", set(df.negative_rule) == set(NEG_RULES))
    check("all seeds present", set(df.seed) == set(SEEDS))

    print("\n2. Balance (the decomposition assumes it)")
    counts = df[df.metric == "mcc"].groupby(
        ["cluster_threshold", "negative_rule", "seed"]).size()
    check("every cell scored by every pipeline",
          counts.nunique() == 1 and counts.iloc[0] == len(ZOO_B),
          f"cell sizes {sorted(counts.unique())}")

    print("\n3. Failures and missing scores")
    n_failed = int((df.metric == "failed").sum())
    check("no failed pipeline-cells", n_failed == 0, f"{n_failed} failed")
    if n_failed:
        print(df.loc[df.metric == "failed", ["model_id", "error"]]
                .drop_duplicates().to_string(index=False))
    live = df[df.metric != "failed"]
    n_nan = int(live.score.isna().sum())
    check("no NaN scores among live rows", n_nan == 0, f"{n_nan} NaN")

    print("\n4. Metrics")
    check("expected metrics present", EXPECTED_METRICS <= set(live.metric.unique()),
          f"found {sorted(live.metric.unique())}")
    for m in ["auroc", "auprc", "bacc", "f1", "accuracy"]:
        sub = live[live.metric == m]["score"]
        if len(sub):
            check(f"{m} within [0,1]", bool(sub.between(0, 1).all()),
                  f"range {sub.min():.3f}-{sub.max():.3f}")
    mcc = live[live.metric == "mcc"]["score"]
    check("mcc within [-1,1]", bool(mcc.between(-1, 1).all()),
          f"range {mcc.min():.3f}-{mcc.max():.3f}")

    print("\n5. Sanity of the learning problem")
    check("mean AUROC above chance",
          float(live[live.metric == "auroc"].score.mean()) > 0.6,
          f"mean {live[live.metric=='auroc'].score.mean():.3f}")
    check("no perfect scores (would suggest leakage)",
          float(live[live.metric == "auroc"].score.max()) < 0.999,
          f"max {live[live.metric=='auroc'].score.max():.4f}")

    print("\n6. Fairness invariants recorded by the runner")
    per_cell_idx = df.groupby(["cluster_threshold", "negative_rule", "seed"])[
        "train_idx_hash"].nunique()
    check("one training split per cell, shared by all pipelines",
          bool((per_cell_idx == 1).all()),
          f"max distinct hashes in a cell {int(per_cell_idx.max())}")
    per_cell_lbl = df.groupby(["cluster_threshold", "negative_rule", "seed"])[
        "label_hash"].nunique()
    check("one label vector per cell",
          bool((per_cell_lbl == 1).all()),
          f"max {int(per_cell_lbl.max())}")

    print("\n7. Class balance per partition")
    bal = df[df.metric == "mcc"].copy()
    bal["pos_frac_train"] = bal.n_pos_train / bal.n_train
    bal["pos_frac_test"] = bal.n_pos_test / bal.n_test
    check("train positive fraction within 0.3-0.7",
          bool(bal.pos_frac_train.between(0.3, 0.7).all()),
          f"range {bal.pos_frac_train.min():.3f}-{bal.pos_frac_train.max():.3f}")
    check("test positive fraction within 0.2-0.8",
          bool(bal.pos_frac_test.between(0.2, 0.8).all()),
          f"range {bal.pos_frac_test.min():.3f}-{bal.pos_frac_test.max():.3f}")
    print("    per threshold, test positive fraction:")
    print(bal.groupby("cluster_threshold")["pos_frac_test"]
             .agg(["mean", "min", "max"]).to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n8. Test-partition size")
    sz = bal.n_test / (bal.n_train + bal.n_test)
    check("test fraction near 0.2", bool(sz.between(0.10, 0.35).all()),
          f"range {sz.min():.3f}-{sz.max():.3f}")

    print("\n9. Cluster ratios (clustering-bottleneck diagnostic)")
    cr = bal.groupby("cluster_threshold")["cluster_ratio"].mean()
    print(cr.to_string(float_format=lambda v: f"{v:.3f}"))
    check("random level has ratio 1.0", abs(cr.get("random", 1.0) - 1.0) < 1e-6)
    non_random = [t for t in THRESHOLDS if t != "random"]
    if all(t in cr.index for t in non_random):
        mono = all(cr[non_random[i]] >= cr[non_random[i + 1]] - 1e-9
                   for i in range(len(non_random) - 1))
        check("cluster ratio falls as the threshold tightens", mono,
              "a looser threshold should not yield MORE clusters")

    print("\n10. Null baseline behaves like a null somewhere")
    lb = (live[live.metric == "mcc"].groupby(["negative_rule", "model_id"],
                                             as_index=False)["score"].mean())
    lb["rank"] = lb.groupby("negative_rule")["score"].rank(ascending=False,
                                                           method="average")
    p16 = lb[lb.model_id == "P16"]
    check("null baseline is not best on every rule",
          bool((p16["rank"] > 1).any()),
          f"ranks {sorted(p16['rank'].tolist())}")
    print("    null-baseline rank per rule:")
    print(p16[["negative_rule", "score", "rank"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"VERIFICATION FAILED: {len(FAILURES)} problem(s)")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("VERIFICATION PASSED -- results are safe to interpret")


if __name__ == "__main__":
    main()

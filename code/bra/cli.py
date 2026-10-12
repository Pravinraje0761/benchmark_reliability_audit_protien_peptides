"""Command line entry point.

    python -m bra.cli selftest
    python -m bra.cli run-arm-a --out results/runs/arm_a.parquet
    python -m bra.cli run-arm-b --out results/runs/arm_b.parquet
    python -m bra.cli analyse   --runs results/runs/all_runs.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import data as D
from .analysis import decompose, ranks
from .runner import Cell, records_to_frame, run_cell


def cmd_selftest(args) -> int:
    """E01 gate. Synthetic data with a planted motif, end to end.

    The shuffled-label check is the one that matters: a harness scoring above
    chance on random labels has a bug, and finding it here costs minutes.
    """
    df = D.synthetic(400, seed=0)
    from .models import ModelSpec
    mini = [ModelSpec("M01", "aac", "logreg", ""),
            ModelSpec("M12", "length_charge", "logreg", "")]

    cell = Cell("SYNTH", "amp_binary", "A", "random", "native", 0)
    recs = run_cell(df, cell, zoo=mini, n_folds=3, verify_leakage=False)
    out = records_to_frame(recs)
    auroc = out[(out.model_id == "M01") & (out.metric == "auroc")]["score"].mean()
    print(f"planted-signal AUROC (M01): {auroc:.3f}   expected > 0.9")

    shuffled = df.copy()
    shuffled["label"] = shuffled["label"].sample(frac=1.0, random_state=1).to_numpy()
    recs2 = run_cell(shuffled, cell, zoo=mini, n_folds=3, verify_leakage=False)
    out2 = records_to_frame(recs2)
    null_auroc = out2[(out2.model_id == "M01") & (out2.metric == "auroc")]["score"].mean()
    print(f"shuffled-label AUROC (M01): {null_auroc:.3f}   expected ~ 0.50")

    ok = auroc > 0.9 and abs(null_auroc - 0.5) < 0.10
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_run_arm_a(args) -> int:
    """Arm A: every benchmark under its own native protocol."""
    frames = []
    for name in args.benchmarks:
        df = D.load(name)
        cell = Cell(name, df["task"].iloc[0], "A", "native_split", "native", args.seed)
        frames.append(records_to_frame(run_cell(df, cell)))
    out = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"wrote {len(out)} rows -> {args.out}")
    return 0


def cmd_run_arm_b(args) -> int:
    """Arm B: pooled corpus, crossed factorial. 12 x 4 x 5 x 5 = 1,200 runs."""
    raise NotImplementedError(
        "Build the pooled corpus first (milestone M1/M2), then iterate "
        "Cell(threshold, negative_rule, seed) over configs/grid.yaml."
    )


def cmd_analyse(args) -> int:
    df = pd.read_parquet(args.runs)
    metric = args.metric

    rm = ranks.rank_matrix(df, metric)
    print("\n=== rank matrix ===")
    print(rm.to_string())

    print("\n=== pairwise tau-b ===")
    print(ranks.pairwise_tau(rm).round(3).to_string())

    n_models, n_bench = rm.shape
    print(f"\nmean pairwise tau-b : {ranks.mean_pairwise_tau(rm):.3f}")
    print(f"null SD at n={n_models}   : {ranks.tau_null_sd(n_models):.3f}"
          "   <- SINGLE-PAIR null SD; the mean over pairs is much tighter")
    print("MDD at 80% power    : tau-b <= 0.646 (20 models x 5 benchmarks, E00)")
    print("                      a result in 0.65-0.80 is INCONCLUSIVE, not a near-miss")

    # Agreement ceiling: how much agreement is even achievable? tau-b between
    # two seeds of the SAME benchmark is not 1.0, so comparing against 1.0
    # overstates disagreement. See E00 Design simulation.
    if df["seed"].nunique() > 1:
        na = ranks.normalised_agreement(df, metric)
        print("\n=== agreement ceiling ===")
        print(na["per_benchmark_ceiling"].round(3).to_string(index=False))
        print(f"\nbetween-benchmark tau-b : {na['between_benchmark_tau_b']:.3f}")
        print(f"within-benchmark ceiling: {na['within_benchmark_ceiling']:.3f}")
        print(f"NORMALISED AGREEMENT    : {na['normalised_agreement']:.3f}"
              "   <- co-primary; < 0.85 supports H1")
    else:
        print("\n[note] only one seed present: agreement ceiling not estimable")

    flip = ranks.top1_flip_rate(rm)
    print(f"top-1 flip rate     : {flip['n_flips']}/{flip['n_pairs']} = {flip['rate']:.2f}")
    print(f"winners             : {flip['winners']}")

    print("\n=== rank range per model ===")
    print(ranks.rank_range(rm).round(2).to_string())

    if df["cluster_threshold"].nunique() > 1:
        print("\n=== variance components (score vs rank) ===")
        print(decompose.decompose_both(df).round(2).to_string())
        print("\n=== tau-b ablation ===")
        print(decompose.tau_ablation(df, metric).round(3).to_string())
        print("\n=== H5 representation vs learner ===")
        print(decompose.representation_vs_learner_stability(df, metric))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bra", description="Benchmark Reliability Audit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)

    a = sub.add_parser("run-arm-a")
    a.add_argument("--benchmarks", nargs="+",
                   default=["B1_genpept", "B3_ppb", "B4_ampbenchmark",
                            "B5_autopeptideml", "B6_abp_framework"])
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="results/runs/arm_a.parquet")
    a.set_defaults(fn=cmd_run_arm_a)

    b = sub.add_parser("run-arm-b")
    b.add_argument("--out", default="results/runs/arm_b.parquet")
    b.set_defaults(fn=cmd_run_arm_b)

    an = sub.add_parser("analyse")
    an.add_argument("--runs", default="results/runs/all_runs.parquet")
    an.add_argument("--metric", default="mcc")
    an.set_defaults(fn=cmd_analyse)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

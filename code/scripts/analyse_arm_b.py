"""Analysis of the Arm B factorial. Produces every number in the paper.

    python scripts/analyse_arm_b.py

Reads $BRA_DATA/runs/all_runs_armB.parquet, writes CSVs to results/tables/ and
prints a readable report. Deterministic given the input.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra.analysis import decompose, ranks  # noqa: E402
from bra.armb import data_root  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
F3 = lambda v: f"{v:.3f}"
F1 = lambda v: f"{v:.1f}"
PRIMARY = "mcc"


def leaderboard(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    sub = df[df["metric"] == metric]
    agg = sub.groupby(by + ["model_id"], as_index=False)["score"].mean()
    agg["rank"] = agg.groupby(by)["score"].rank(ascending=False, method="average")
    return agg


def rankmat_over(df: pd.DataFrame, factor: str, metric: str) -> pd.DataFrame:
    """models x levels-of-factor rank matrix, averaging over everything else."""
    lb = leaderboard(df, [factor], metric)
    return lb.pivot(index="model_id", columns=factor, values="rank")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main(runs: str | None = None) -> None:
    src = Path(runs) if runs else data_root() / "runs" / "all_runs_armB.parquet"
    df = pd.read_parquet(src)
    df = df[df["metric"] != "failed"]
    print(f"loaded {len(df)} rows from {src}")
    print(f"models {df.model_id.nunique()}  rules {df.negative_rule.nunique()}  "
          f"thresholds {df.cluster_threshold.nunique()}  seeds {df.seed.nunique()}  "
          f"metrics {df.metric.nunique()}")

    section("1. Absolute performance by protocol cell (primary metric: MCC)")
    piv = (df[df.metric == PRIMARY]
           .groupby(["negative_rule", "cluster_threshold"])["score"]
           .mean().unstack())
    print(piv.to_string(float_format=F3))
    piv.to_csv(OUT / "armB_mean_mcc_by_cell.csv")
    print(f"\nSpread across cells: {piv.values.min():.3f} to {piv.values.max():.3f} "
          f"(range {piv.values.max()-piv.values.min():.3f} MCC)")

    section("2. Do rankings agree ACROSS NEGATIVE RULES?  (the H1 analogue)")
    kmat_rule = rankmat_over(df, "negative_rule", PRIMARY)
    print(kmat_rule.to_string(float_format=F1))
    pt = ranks.pairwise_tau(kmat_rule)
    print("\npairwise tau-b:")
    print(pt.to_string(float_format=F3))
    mt_rule = ranks.mean_pairwise_tau(kmat_rule)
    flip = ranks.top1_flip_rate(kmat_rule)
    print(f"\nmean pairwise tau-b        : {mt_rule:.3f}")
    print(f"top-1 flip rate            : {flip['n_flips']}/{flip['n_pairs']} = {flip['rate']:.2f}")
    print(f"winners per rule           : {flip['winners']}")
    rr = ranks.rank_range(kmat_rule)
    print(f"largest rank range         : {rr['range'].max():.0f} of {kmat_rule.shape[0]} positions")
    print("\nrank range per model:")
    print(rr.to_string(float_format=F1))
    kmat_rule.to_csv(OUT / "armB_rank_matrix_by_rule.csv")
    pt.to_csv(OUT / "armB_tau_by_rule.csv")
    rr.to_csv(OUT / "armB_rank_range_by_rule.csv")

    section("3. Do rankings agree ACROSS CLUSTERING THRESHOLDS?")
    kmat_thr = rankmat_over(df, "cluster_threshold", PRIMARY)
    print(kmat_thr.to_string(float_format=F1))
    pt_thr = ranks.pairwise_tau(kmat_thr)
    print("\npairwise tau-b:")
    print(pt_thr.to_string(float_format=F3))
    mt_thr = ranks.mean_pairwise_tau(kmat_thr)
    flip_thr = ranks.top1_flip_rate(kmat_thr)
    print(f"\nmean pairwise tau-b        : {mt_thr:.3f}")
    print(f"top-1 flip rate            : {flip_thr['n_flips']}/{flip_thr['n_pairs']}")
    kmat_thr.to_csv(OUT / "armB_rank_matrix_by_threshold.csv")
    pt_thr.to_csv(OUT / "armB_tau_by_threshold.csv")

    section("4. The agreement ceiling, measured from seed replicates")
    cell = ["negative_rule", "cluster_threshold"]
    sub = df[df.metric == PRIMARY]
    taus = []
    for keys, g in sub.groupby(cell):
        per_seed = (g.groupby(["seed", "model_id"], as_index=False)["score"].mean()
                     .pivot(index="model_id", columns="seed", values="score")
                     .rank(ascending=False, method="average"))
        for a, b in itertools.combinations(per_seed.columns, 2):
            taus.append({"negative_rule": keys[0], "cluster_threshold": keys[1],
                         "seed_a": a, "seed_b": b,
                         "tau_b": ranks.tau_b(per_seed[a], per_seed[b])})
    ceil = pd.DataFrame(taus)
    ceiling = float(ceil["tau_b"].mean())
    print(f"within-cell seed-pair tau-b (the ceiling): mean {ceiling:.3f}  "
          f"sd {ceil['tau_b'].std(ddof=1):.3f}  n={len(ceil)}")
    print("\nper negative rule:")
    print(ceil.groupby("negative_rule")["tau_b"].agg(["mean", "std", "count"]).to_string(float_format=F3))
    ceil.to_csv(OUT / "armB_ceiling_seed_pairs.csv", index=False)
    print(f"\nNORMALISED AGREEMENT (between-rule / ceiling)      : {mt_rule/ceiling:.3f}")
    print(f"NORMALISED AGREEMENT (between-threshold / ceiling) : {mt_thr/ceiling:.3f}")
    print("\nJudging tau-b against 1.0 rather than this ceiling would overstate")
    print("disagreement. Pre-registered: normalised < 0.85 supports H1.")

    section("5. Variance decomposition (H3, H4) -- score AND rank")
    both = decompose.decompose_both(df[df.metric == PRIMARY])
    print(both.to_string(float_format=lambda v: f"{v:.2f}"))
    both.to_csv(OUT / "armB_variance_components.csv")

    section("5b. Variance decomposition with metric as an explicit factor")
    allm = decompose.decompose_both(df)
    print(allm.to_string(float_format=lambda v: f"{v:.2f}"))
    allm.to_csv(OUT / "armB_variance_components_allmetrics.csv")

    section("6. tau-b ablation: pin a factor, see how much agreement returns")
    ab = decompose.tau_ablation(df, PRIMARY)
    print(ab.to_string(index=False, float_format=F3))
    ab.to_csv(OUT / "armB_tau_ablation.csv", index=False)

    section("7. H5: are representation ranks less stable than learner ranks?")
    h5 = decompose.representation_vs_learner_stability(df, PRIMARY)
    for k, v in h5.items():
        print(f"  {k}: {v}")
    pd.DataFrame([h5]).to_csv(OUT / "armB_h5.csv", index=False)

    section("8. Null-baseline probe: where does length+charge (P16) land?")
    lb = leaderboard(df, ["negative_rule"], PRIMARY)
    p16 = lb[lb.model_id == "P16"][["negative_rule", "score", "rank"]]
    print(p16.to_string(index=False, float_format=F3))
    print(f"\n(of {df.model_id.nunique()} pipelines; rank 1 = best)")
    print("Mid-pack or better means that rule is separable by length and charge")
    print("alone: the rule has a shortcut, and its other numbers are suspect.")
    p16.to_csv(OUT / "armB_null_baseline.csv", index=False)

    section("9. Clustering bottleneck: cluster-count / sequence-count ratio")
    cr = df.groupby("cluster_threshold")["cluster_ratio"].agg(["mean", "min", "max"])
    print(cr.to_string(float_format=F3))
    print("\nA ratio near 1.0 means clustering degenerated to roughly one")
    print("sequence per cluster, so the homology control is cosmetic there.")
    cr.to_csv(OUT / "armB_cluster_ratios.csv")

    section("9b. Residual leakage: does cluster-disjoint mean pairwise-disjoint?")
    if "leak_fraction" in df.columns:
        lk = (df[df.metric == PRIMARY]
              .groupby("cluster_threshold")[["leak_fraction", "n_test_with_hit"]]
              .mean())
        lk["pct_test_leaked"] = 100 * lk["leak_fraction"]
        print(lk[["pct_test_leaked", "n_test_with_hit"]].to_string(float_format=F3))
        print()
        print("Greedy clustering does not form equivalence classes, so two")
        print("sequences can be in different clusters and still exceed the")
        print("identity threshold. A STRICTER nominal threshold leaves MORE")
        print("residual leakage, because loose thresholds admit far more")
        print("qualifying pairs and short peptides clear 40% identity easily.")
        lk.to_csv(OUT / "armB_residual_leakage.csv")
    else:
        print("  (no leak_fraction column; results predate the leakage measurement)")

    section("10. Metric choice alone: does it reorder models?")
    lbm = df.groupby(["metric", "model_id"], as_index=False)["score"].mean()
    lbm["rank"] = lbm.groupby("metric")["score"].rank(ascending=False, method="average")
    kmat_metric = lbm.pivot(index="model_id", columns="metric", values="rank")
    print(kmat_metric.to_string(float_format=F1))
    pt_m = ranks.pairwise_tau(kmat_metric)
    print("\nrank agreement BETWEEN METRICS (protocol averaged out):")
    print(pt_m.to_string(float_format=F3))
    print(f"mean tau-b between metrics: {ranks.mean_pairwise_tau(kmat_metric):.3f}")
    pt_m.to_csv(OUT / "armB_tau_between_metrics.csv")

    print(f"\n\nAll tables written to {OUT}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None,
                    help="explicit results parquet (default: the merged Arm B file)")
    main(ap.parse_args().runs)

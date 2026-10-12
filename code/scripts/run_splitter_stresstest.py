"""Stress-test the balance-preserving splitter against transparent alternatives.

The manuscript introduces a quota-driven cluster-disjoint splitter because
largest-first assignment drives the test positive fraction to 0.995 on a corpus
that is 1:1 by construction. The per-cell audit then shows the correction is not
free: in several cells it *raises* residual pairwise leakage, because the
clusters that balance the classes are not the clusters that minimise
cross-partition similarity.

That observation invites the obvious question, which a referee asked: is any
splitter good at all three of test size, class balance and cross-partition
similarity simultaneously? This script answers it by running four splitters over
the same cached cluster assignments and measuring all three quantities per cell.

The four are deliberately simple, because the point is a trade-off curve rather
than a new method:

    largest_first   cluster size, descending   (the pre-registered splitter)
    balanced        remaining per-class quota  (the manuscript's splitter)
    random          clusters in random order   (transparent control)
    leak_aware      least-bridging clusters first, greedily

Clusters stay atomic in all four; that is what makes them comparable. Only the
*order* of selection differs, which is the whole argument of the paper's
splitter section.

Cost note: clustering is computed once per (rule, threshold) and reused by all
four splitters, so the expense is the leakage search, one per splitter per cell.
The unclustered `random` threshold is skipped because every splitter degenerates
to the same random split there, which would pad the table with ties.

    python scripts/run_splitter_stresstest.py

Writes results/tables/splitter_stresstest.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra.armb import NEG_RULES, data_root  # noqa: E402
from bra.balanced_split import assign_test_clusters_balanced  # noqa: E402
from bra.splits import cluster_mmseqs, measure_leakage  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.9, 0.7, 0.4]      # clustered cells only
TEST_FRACTION = 0.2
SEED = 0


# --------------------------------------------------------------------------- #
# the four selection orders. each returns a boolean mask over SEQUENCES.
# --------------------------------------------------------------------------- #
def _take_until(order, sizes, inv, target):
    """Take whole clusters in `order` until the test fraction is reached."""
    chosen, total = [], 0
    for c in order:
        if total >= target:
            break
        chosen.append(int(c))
        total += sizes[c]
    mask = np.zeros(len(sizes), dtype=bool)
    mask[chosen] = True
    return mask[inv]


def split_largest_first(cluster_ids, labels, seed=SEED):
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uniq))
    order = np.argsort(-sizes)
    return _take_until(order, sizes, inv, TEST_FRACTION * len(labels))


def split_balanced(cluster_ids, labels, seed=SEED):
    return assign_test_clusters_balanced(cluster_ids, labels,
                                         test_fraction=TEST_FRACTION, seed=seed)


def split_random(cluster_ids, labels, seed=SEED):
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uniq))
    order = np.random.default_rng(seed).permutation(len(uniq))
    return _take_until(order, sizes, inv, TEST_FRACTION * len(labels))


def split_leak_aware(cluster_ids, labels, seqs, threshold, seed=SEED):
    """Prefer clusters that bridge least to the rest of the corpus.

    Re-clustering at the same identity but looser coverage merges strict clusters
    that are near-neighbours; a strict cluster sharing its loose cluster with many
    others is one whose members are most likely to exceed the threshold against
    sequences left in training. Taking the least-bridging clusters first should
    therefore reduce residual leakage. It is a cheap proxy, not an exact pair
    count, which is the point of including it as a transparent alternative rather
    than proposing it as a method.
    """
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    sizes = np.bincount(inv, minlength=len(uniq))
    loose = cluster_mmseqs(seqs, min_seq_id=threshold, coverage=0.3)
    df = pd.DataFrame({"strict": inv, "loose": loose})
    shared = df.groupby("loose")["strict"].nunique()
    bridge = df.assign(b=df["loose"].map(shared)).groupby("strict")["b"].mean()
    score = bridge.reindex(range(len(uniq))).fillna(1.0).to_numpy(dtype=float)
    jitter = np.random.default_rng(seed).random(len(uniq)) * 1e-6
    order = np.argsort(score + jitter)          # least bridging first
    return _take_until(order, sizes, inv, TEST_FRACTION * len(labels))


SPLITTERS = ["largest_first", "balanced", "random", "leak_aware"]


def main() -> None:
    corpus = data_root() / "corpus"
    rows = []

    for rule in NEG_RULES:
        f = corpus / f"armB_{rule}_seed{SEED}.parquet"
        if not f.exists():
            print(f"  missing {f.name}; skipping")
            continue
        df = pd.read_parquet(f)
        seqs = df["sequence"].astype(str).tolist()
        labels = df["label"].to_numpy(dtype=float)
        corpus_pos = float(labels.mean())

        for thr in THRESHOLDS:
            cl = cluster_mmseqs(seqs, min_seq_id=thr, coverage=0.8)
            n_clusters = int(len(np.unique(cl)))

            for name in SPLITTERS:
                if name == "largest_first":
                    is_test = split_largest_first(cl, labels, SEED)
                elif name == "balanced":
                    is_test = split_balanced(cl, labels, SEED)
                elif name == "random":
                    is_test = split_random(cl, labels, SEED)
                else:
                    is_test = split_leak_aware(cl, labels, seqs, thr, SEED)

                tr = [s for s, t in zip(seqs, is_test) if not t]
                te = [s for s, t in zip(seqs, is_test) if t]
                if not te or not tr:
                    print(f"  {rule} {thr} {name}: degenerate split; skipping")
                    continue

                leak = measure_leakage(tr, te, min_seq_id=thr, coverage=0.8)
                pos = float(labels[is_test].mean())
                rows.append({
                    "negative_rule": rule,
                    "threshold": thr,
                    "splitter": name,
                    "seed": SEED,
                    "test_fraction": float(is_test.mean()),
                    "test_pos_frac": pos,
                    "balance_error": abs(pos - corpus_pos),
                    "leak_fraction": leak["leak_fraction"],
                    "n_test": int(is_test.sum()),
                    "n_clusters": n_clusters,
                    "n_test_clusters": int(len(np.unique(cl[is_test]))),
                })
                print(f"  {rule:22s} thr {thr}  {name:14s} "
                      f"test={rows[-1]['test_fraction']:.3f} "
                      f"pos={pos:.3f} leak={leak['leak_fraction']:.3f}", flush=True)

                pd.DataFrame(rows).to_csv(OUT / "splitter_stresstest.csv", index=False)

    res = pd.DataFrame(rows)
    if res.empty:
        print("no cells completed")
        return

    print("\n" + "=" * 78)
    print("Mean and worst case over the clustered cells: does any splitter win on all three?")
    print("=" * 78)
    summ = (res.groupby("splitter")[["test_fraction", "balance_error", "leak_fraction"]]
               .agg(["mean", "max"]).round(4))
    print(summ.to_string())
    print("\nTarget test_fraction is 0.200; lower is better for balance_error and")
    print("leak_fraction. A splitter best in every column would make the paper's")
    print("trade-off claim wrong, so this table is reported whichever way it falls.")
    print(f"\nwritten to {OUT / 'splitter_stresstest.csv'}")


if __name__ == "__main__":
    main()

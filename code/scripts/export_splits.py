"""Persist the exact train/test partitions every reported number was computed on.

The splits are deterministic given the corpus, the threshold, the seed and the
splitter, so in principle "rerun the code" reproduces them. In practice that is
a weaker guarantee than it sounds: it holds only if the reader's MMseqs2 build,
its default parameters and the corpus bytes all match ours. Shipping the
partitions themselves turns a claim into something checkable.

Sequences are NOT redistributed here -- the Arm B corpus derives from DBAASP and
UniProt, whose terms govern that. Instead:

  Arm B  test-set ROW INDICES into the canonical corpus ordering, plus a SHA-256
         fingerprint of that ordering. A reader who rebuilds the corpus checks
         the fingerprint first; if it matches, the indices define exactly our
         partitions, and if it does not, nothing downstream is comparable and
         they know immediately.
  Arm A  per-sequence SHA-1, the split as the publishers released it, and
         whether we scored it. The only thing we changed is dropping the
         test-side copy of cross-split duplicates, and this records precisely
         which rows that removed.

    python scripts/export_splits.py

Writes results/splits/.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra.arma import ARMA_TASKS, TASK_DESC, load_task                    # noqa: E402
from bra.armb import (NEG_RULES, SEEDS, THRESHOLDS, data_root,           # noqa: E402
                      split_cell)

OUT = Path(__file__).resolve().parents[2] / "results" / "splits"
OUT.mkdir(parents=True, exist_ok=True)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def export_arm_a() -> None:
    rows = []
    for task in ARMA_TASKS:
        raw = load_task(task, drop_cross_split_duplicates=False)
        kept = load_task(task, drop_cross_split_duplicates=True)
        kept_keys = set(zip(kept.sequence.astype(str), kept.split))
        for s, lab, sp in zip(raw.sequence.astype(str), raw.label, raw.split):
            rows.append({"task": task, "property": TASK_DESC[task],
                         "seq_sha1": sha1(s), "label": float(lab),
                         "split_as_released": sp,
                         "used_in_scoring": (s, sp) in kept_keys})
        print(f"  {task}: released {len(raw)}, scored {len(kept)}, "
              f"dropped {len(raw) - len(kept)}", flush=True)
    pd.DataFrame(rows).to_parquet(OUT / "armA_splits.parquet", index=False)
    print(f"  wrote armA_splits.parquet ({len(rows)} rows)")


def export_arm_b() -> None:
    corpus = data_root() / "corpus"
    rows, fingerprints = [], []

    for rule in NEG_RULES:
        f = corpus / f"armB_{rule}_seed0.parquet"
        if not f.exists():
            print(f"  missing {f.name}; skipping")
            continue
        df = pd.read_parquet(f)
        seqs = df["sequence"].astype(str).tolist()
        fp = hashlib.sha256("\n".join(seqs).encode()).hexdigest()
        fingerprints.append({"negative_rule": rule, "n_sequences": len(seqs),
                             "sha256_ordered_sequences": fp})
        print(f"  {rule}: n={len(seqs)} fingerprint={fp[:16]}...", flush=True)

        for thr in THRESHOLDS:
            for seed in SEEDS:
                for splitter, balanced in (("largest_first", False),
                                           ("balance_preserving", True)):
                    _, te_idx, _ = split_cell(df, thr, seed, balanced=balanced)
                    for i in np.sort(np.asarray(te_idx)):
                        rows.append({"splitter": splitter, "negative_rule": rule,
                                     "cluster_threshold": str(thr), "seed": int(seed),
                                     "test_index": int(i)})
            print(f"    thr {thr}: done", flush=True)
        # write as we go, so an interrupted export still leaves usable rules
        pd.DataFrame(rows).to_parquet(OUT / "armB_splits.parquet", index=False)
        pd.DataFrame(fingerprints).to_csv(OUT / "armB_fingerprint.csv", index=False)

    print(f"  wrote armB_splits.parquet ({len(rows)} test-row assignments)")


def main() -> int:
    print("=== Arm A: native partitions, and the rows we dropped ===")
    export_arm_a()
    print("\n=== Arm B: generated partitions, both splitters ===")
    export_arm_b()
    print(f"\nwritten to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

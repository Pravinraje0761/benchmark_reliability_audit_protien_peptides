"""Run the SIX Arm A datasets the main analysis excluded, for cutoff sensitivity.

The ranking analysis uses the ten tasks with at least 900 training sequences.
That cutoff is defensible -- a 24-pipeline leaderboard on a few hundred
sequences is dominated by test-set noise -- but a reader is entitled to ask
whether the conclusion depends on it, and justifying an exclusion is weaker
than showing what including them does.

So this runs the excluded six (AOX 698, APP 482, QS 349, MRSA 237, AMAP 221,
BBP 191 training sequences) with the identical zoo, seeds and procedure, into a
SEPARATE output directory. Nothing here touches the ten-task records: the main
result stays exactly as reported, and the sensitivity analysis is assembled by
reading both directories.

    python scripts/run_arm_a_excluded.py --stage embed   # ~2k new sequences
    python scripts/run_arm_a_excluded.py --stage run     # 6 x 24 x 5 = 720 fits

Writes $BRA_DATA/runs/armA_excluded/task_<ID>.parquet, one per task, skipping
any that already exist so the job is resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra import features as F                                         # noqa: E402
from bra.arma import ARMA_TASKS, TASK_DESC, load_task                 # noqa: E402
from bra.armb import ESM, ZOO_B, build_learner, data_root, featurise  # noqa: E402
from bra.metrics import classification_scores                         # noqa: E402

OUTDIR = data_root() / "runs" / "armA_excluded"
SEEDS = [0, 1, 2, 3, 4]

# everything in the released collection that the ranking analysis left out
EXCLUDED = [t for t in TASK_DESC if t not in ARMA_TASKS]


def stage_embed() -> None:
    cache = F.EmbeddingCache(data_root() / "embeddings")
    seqs = sorted({s for t in EXCLUDED for s in load_task(t).sequence})
    print(f"excluded tasks: {EXCLUDED}")
    print(f"unique sequences to embed: {len(seqs)}")
    for rep in sorted({p.rep for p in ZOO_B} & set(ESM)):
        t0 = time.time()
        F.embed_plm(pd.Series(seqs), ESM[rep], cache)
        print(f"  {rep}: {time.time() - t0:.0f}s", flush=True)


def stage_run() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = F.EmbeddingCache(data_root() / "embeddings")

    for task in EXCLUDED:
        out = OUTDIR / f"task_{task}.parquet"
        if out.exists():
            print(f"skip {task} (done)")
            continue
        df = load_task(task)
        tr, te = df[df.split == "train"], df[df.split == "test"]
        y_tr, y_te = tr.label.to_numpy(), te.label.to_numpy()
        idx_hash = hashlib.sha1(
            np.ascontiguousarray(tr.index.to_numpy()).tobytes()
        ).hexdigest()[:12]
        print(f"\n{task} ({TASK_DESC[task]}): train={len(tr)} test={len(te)}", flush=True)

        rows, t0 = [], time.time()
        for p in ZOO_B:
            for seed in SEEDS:
                try:
                    Xtr, Xte = featurise(p.rep, tr.sequence, te.sequence, cache)
                    est = build_learner(p.learner, seed)
                    est.fit(Xtr, y_tr)
                    scores = classification_scores(y_te, est.predict_proba(Xte)[:, 1])
                    err = ""
                except Exception as exc:       # a failure is recorded, never dropped
                    scores, err = {}, f"{type(exc).__name__}: {exc}"
                    print(f"    [FAIL] {p.pid} seed={seed}: {err}", flush=True)
                base = {
                    "benchmark": f"armA_{task}", "task": task, "arm": "A_excluded",
                    "model_id": p.pid, "representation": p.rep, "learner": p.learner,
                    "cluster_threshold": "native", "negative_rule": "native",
                    "seed": seed, "fold": 0, "n_train": len(tr), "n_test": len(te),
                    "train_idx_hash": idx_hash, "error": err,
                }
                if scores:
                    rows += [{**base, "metric": m, "score": v} for m, v in scores.items()]
                else:
                    rows.append({**base, "metric": "failed", "score": float("nan")})
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"  wrote {out.name} in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["embed", "run", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("embed", "all"):
        stage_embed()
    if args.stage in ("run", "all"):
        stage_run()
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Arm A: the fixed zoo across 10 peptide tasks under their native splits.

    python scripts/run_arm_a.py --stage embed
    python scripts/run_arm_a.py --stage run      # resumable, one parquet per task
    python scripts/run_arm_a.py --stage merge
"""
from __future__ import annotations

import argparse, hashlib, sys, time
from pathlib import Path

import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra import features as F                                   # noqa: E402
from bra.arma import ARMA_TASKS, load_task                      # noqa: E402
from bra.armb import ESM, ZOO_B, build_learner, data_root, featurise  # noqa: E402
from bra.metrics import classification_scores                   # noqa: E402

OUTDIR = data_root() / "runs" / "armA"


def all_sequences():
    s = set()
    for t in ARMA_TASKS:
        s |= set(load_task(t).sequence)
    return sorted(s)


def stage_embed():
    cache = F.EmbeddingCache(data_root() / "embeddings")
    seqs = all_sequences()
    print(f"unique Arm A sequences: {len(seqs)}")
    for rep in sorted({p.rep for p in ZOO_B} & set(ESM)):
        t0 = time.time()
        F.embed_plm(pd.Series(seqs), ESM[rep], cache)
        print(f"  {rep}: {time.time()-t0:.0f}s", flush=True)


SEEDS = [0, 1, 2, 3, 4]


def stage_run():
    """Score every task with every pipeline, replicated across seeds.

    The benchmark ships a fixed partition, so the seed cannot vary the split as
    it does in Arm B; it varies model initialisation and any stochastic learner
    component. That is what measures the **agreement ceiling** for this arm -
    tau-b between two seeds of the same task under the same protocol. Without
    it, cross-task tau-b cannot be normalised or compared to Arm B.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = F.EmbeddingCache(data_root() / "embeddings")
    for task in ARMA_TASKS:
        out = OUTDIR / f"task_{task}.parquet"
        if out.exists():
            print(f"skip {task} (done)"); continue
        df = load_task(task)
        n_dup = df.attrs.get("n_cross_split_duplicates", 0)
        tr = df[df.split == "train"]; te = df[df.split == "test"]
        y_tr, y_te = tr.label.to_numpy(), te.label.to_numpy()
        idx_hash = hashlib.sha1(np.ascontiguousarray(tr.index.to_numpy()).tobytes()).hexdigest()[:12]
        rows, t0 = [], time.time()
        for p in ZOO_B:
          for seed in SEEDS:
            t1 = time.time()
            try:
                Xtr, Xte = featurise(p.rep, tr.sequence, te.sequence, cache)
                est = build_learner(p.learner, seed)
                est.fit(Xtr, y_tr)
                sc = est.predict_proba(Xte)[:, 1]
                scores, err = classification_scores(y_te, sc), ""
            except Exception as e:
                scores, err = {}, f"{type(e).__name__}: {e}"
                print(f"    [FAIL] {p.pid} seed{seed}: {err}", flush=True)
            base = {"benchmark": f"armA_{task}", "task": task, "arm": "A",
                    "model_id": p.pid, "representation": p.rep, "learner": p.learner,
                    "cluster_threshold": "native", "negative_rule": "native",
                    "seed": seed, "fold": 0, "n_train": len(tr), "n_test": len(te),
                    "n_pos_train": int(y_tr.sum()), "n_pos_test": int(y_te.sum()),
                    "cross_split_duplicates_removed": n_dup,
                    "fit_seconds": round(time.time() - t1, 2),
                    "train_idx_hash": idx_hash, "error": err}
            rows += ([{**base, "metric": m, "score": v} for m, v in scores.items()]
                     if scores else [{**base, "metric": "failed", "score": float("nan")}])
        pd.DataFrame(rows).to_parquet(out)
        mcc = [r["score"] for r in rows if r["metric"] == "mcc"]
        print(f"{task:6s} n={len(tr)}/{len(te)}  {time.time()-t0:5.0f}s  "
              f"mcc {min(mcc):.3f}-{max(mcc):.3f}", flush=True)


def stage_merge():
    files = sorted(OUTDIR.glob("task_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dest = data_root() / "runs" / "all_runs_armA.parquet"
    df.to_parquet(dest)
    print(f"merged {len(files)} tasks -> {dest} ({len(df)} rows)")
    print(f"failures: {int((df.metric=='failed').sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["embed", "run", "merge"])
    a = ap.parse_args()
    {"embed": stage_embed, "run": stage_run, "merge": stage_merge}[a.stage]()

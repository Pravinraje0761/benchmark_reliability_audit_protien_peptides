"""Run the Arm B crossed factorial.

    python scripts/run_arm_b.py --stage embed      # once, expensive
    python scripts/run_arm_b.py --stage grid       # resumable
    python scripts/run_arm_b.py --stage merge

Writes one parquet per cell to $BRA_DATA/runs/armB/, so the job can be killed and
resumed without losing work. `merge` concatenates them into all_runs_armB.parquet.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra import features as F              # noqa: E402
from bra.armb import (NEG_RULES, SEEDS, THRESHOLDS, ZOO_B, data_root,  # noqa: E402
                      precompute_embeddings, run_cell)

CORPUS = data_root() / "corpus"
import os as _os
_SUFFIX = "_balanced" if _os.environ.get("BRA_BALANCED") == "1" else ""
OUTDIR = data_root() / "runs" / f"armB{_SUFFIX}"


def stage_embed() -> None:
    precompute_embeddings(CORPUS)


def stage_grid(limit: int | None = None, shard: int = 0, nshards: int = 1) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = F.EmbeddingCache(data_root() / "embeddings")

    # Negatives are drawn once per rule (seed 0); the seed varies the partition
    # and model initialisation. See the module docstring in bra/armb.py.
    corpora = {r: pd.read_parquet(CORPUS / f"armB_{r}_seed0.parquet") for r in NEG_RULES}
    for r, d in corpora.items():
        print(f"{r:22s} n={len(d)} pos={int(d.label.sum())}")

    cells = [(t, r, s) for r in NEG_RULES for t in THRESHOLDS for s in SEEDS]
    if limit:
        cells = cells[:limit]
    total_cells = len(cells)
    # Cells are independent and each writes its own parquet, so the grid shards
    # trivially across processes. Measured cost is ~212 s per cell single-process
    # (RF and MLP on 640-1280 dim embeddings dominate), which is 5.9 h serially
    # on 100 cells; sharding uses the idle cores instead.
    if nshards > 1:
        cells = cells[shard::nshards]
    print(f"\nshard {shard+1}/{nshards}: {len(cells)} of {total_cells} cells "
          f"x {len(ZOO_B)} pipelines = {len(cells)*len(ZOO_B)} fits\n")

    t_start = time.time()
    for i, (thr, rule, seed) in enumerate(cells, 1):
        out = OUTDIR / f"cell_{rule}_{thr}_s{seed}.parquet"
        if out.exists():
            print(f"[{i}/{len(cells)}] skip (done) {rule} thr={thr} seed={seed}")
            continue
        t0 = time.time()
        rows = run_cell(corpora[rule], thr, rule, seed, cache=cache)
        pd.DataFrame(rows).to_parquet(out)
        done_mcc = [r["score"] for r in rows if r["metric"] == "mcc"]
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(cells) - i)
        print(f"[{i}/{len(cells)}] {rule} thr={thr} seed={seed} "
              f"{time.time()-t0:5.0f}s  mcc range "
              f"{min(done_mcc):.3f}-{max(done_mcc):.3f}  ETA {eta/60:.0f}min",
              flush=True)


def stage_merge() -> None:
    files = sorted(OUTDIR.glob("cell_*.parquet"))
    if not files:
        raise SystemExit("no cell files; run --stage grid first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dest = data_root() / "runs" / f"all_runs_armB{_SUFFIX}.parquet"
    df.to_parquet(dest)
    print(f"merged {len(files)} cells -> {dest}")
    print(f"rows: {len(df)}")
    n_fail = int((df["metric"] == "failed").sum())
    print(f"failed pipeline-cells: {n_fail}")
    if n_fail:
        print(df.loc[df.metric == "failed", ["model_id", "error"]]
                .drop_duplicates().to_string(index=False))
    print("\ncells per factor:")
    for c in ["cluster_threshold", "negative_rule", "seed", "model_id"]:
        print(f"  {c}: {df[c].nunique()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["embed", "grid", "merge"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    {"embed": stage_embed,
     "grid": lambda: stage_grid(a.limit, a.shard, a.nshards),
     "merge": stage_merge}[a.stage]()

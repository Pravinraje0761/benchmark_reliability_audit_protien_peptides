"""Preview grid: the 8 embedding-free pipelines over all 100 cells.

Writes to a SEPARATE directory so it cannot be mistaken for, or collide with,
the full 16-pipeline grid. Purpose is twofold: use spare cores while the ESM
embeddings compute, and exercise the entire analysis path on real data before
the definitive run lands, so the final pass is a fill-in rather than a debug.

    python scripts/run_preview.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.armb import NEG_RULES, SEEDS, THRESHOLDS, ZOO_B, data_root, run_cell  # noqa: E402

ZOO_CHEAP = [p for p in ZOO_B if not p.rep.startswith("esm2")]
OUTDIR = data_root() / "runs" / "armB_preview"
CORPUS = data_root() / "corpus"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"preview zoo ({len(ZOO_CHEAP)}): {[p.pid for p in ZOO_CHEAP]}")
    corpora = {r: pd.read_parquet(CORPUS / f"armB_{r}_seed0.parquet") for r in NEG_RULES}
    cells = [(t, r, s) for r in NEG_RULES for t in THRESHOLDS for s in SEEDS]
    print(f"{len(cells)} cells x {len(ZOO_CHEAP)} = {len(cells)*len(ZOO_CHEAP)} fits\n")

    t_start = time.time()
    for i, (thr, rule, seed) in enumerate(cells, 1):
        out = OUTDIR / f"cell_{rule}_{thr}_s{seed}.parquet"
        if out.exists():
            continue
        t0 = time.time()
        rows = run_cell(corpora[rule], thr, rule, seed, zoo=ZOO_CHEAP)
        pd.DataFrame(rows).to_parquet(out)
        mcc = [r["score"] for r in rows if r["metric"] == "mcc"]
        el = time.time() - t_start
        print(f"[{i}/{len(cells)}] {rule} thr={thr} s{seed} {time.time()-t0:5.0f}s "
              f"mcc {min(mcc):.3f}-{max(mcc):.3f} ETA {el/i*(len(cells)-i)/60:.0f}min",
              flush=True)

    files = sorted(OUTDIR.glob("cell_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dest = data_root() / "runs" / "all_runs_armB_preview.parquet"
    df.to_parquet(dest)
    print(f"\nmerged {len(files)} cells -> {dest} ({len(df)} rows)")
    nf = int((df.metric == "failed").sum())
    print(f"failed pipeline-cells: {nf}")
    if nf:
        print(df.loc[df.metric == "failed", ["model_id", "error"]].drop_duplicates()
                .to_string(index=False))


if __name__ == "__main__":
    main()

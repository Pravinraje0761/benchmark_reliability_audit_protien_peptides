"""Materialise benchmark B2 (QMAP) and characterise it. Milestone M1, first loader.

    python scripts/materialise_b2.py

Writes the canonical table to $BRA_DATA/qmap/ and prints the facts that go into
Datasets/B2 QMAP.md. The train mask (18,033 x 3,650 pairwise alignments) is
cached by the loader, so re-runs are fast.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra.data.qmap_loader import _data_root, load_qmap  # noqa: E402
from bra.schema import validate_peptides  # noqa: E402


def characterise(df: pd.DataFrame, name: str) -> None:
    print(f"\n----- {name} -----")
    print(f"rows            : {len(df)}")
    print(df["split"].value_counts().to_string())
    for sp, g in df.groupby("split"):
        L = g["sequence"].map(len)
        print(f"  {sp:<5} len min/med/max {L.min()}/{int(L.median())}/{L.max()}"
              f"   label min/med/max {g['label'].min():.2f}/{g['label'].median():.2f}/{g['label'].max():.2f}")

    tr = set(df.loc[df["split"] == "train", "sequence"])
    te = set(df.loc[df["split"] == "test", "sequence"])
    dup = tr & te
    print(f"  exact train/test duplicates: {len(dup)}")
    if dup:
        print(f"    LEAKAGE example: {next(iter(dup))[:40]}")

    noncanon = df.loc[~df["sequence"].map(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))]
    print(f"  sequences with non-canonical residues: {len(noncanon)}")
    if len(noncanon):
        print(f"    e.g. {noncanon['sequence'].iloc[0][:40]}  (O=ornithine, B=DAB per QMAP docs)")


def main() -> None:
    t0 = time.time()
    print("loading B2 QMAP mic_reg (E. coli, split 0) ...")
    mic = load_qmap(split=0, task="mic_reg")
    print(f"  took {time.time() - t0:.0f}s")
    characterise(mic, "B2 QMAP / mic_reg / E. coli / split 0")

    t1 = time.time()
    hc = load_qmap(split=0, task="hc50_reg")
    print(f"\nloading hc50_reg took {time.time() - t1:.0f}s (mask cached)")
    characterise(hc, "B2 QMAP / hc50_reg / split 0")

    out = _data_root()
    for df, fn in [(mic, "B2_mic_ecoli_split0.parquet"), (hc, "B2_hc50_split0.parquet")]:
        # validate_peptides raises on non-canonical residues; QMAP legitimately
        # contains ornithine (O) and DAB (B), so record the decision explicitly
        # rather than letting the validator decide it for us.
        canon = df.loc[df["sequence"].map(lambda s: set(s) <= set("ACDEFGHIKLMNPQRSTVWY"))].copy()
        dropped = len(df) - len(canon)
        validate_peptides(canon)
        canon.to_parquet(out / fn)
        print(f"wrote {out / fn}  ({len(canon)} rows, dropped {dropped} non-canonical)")

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

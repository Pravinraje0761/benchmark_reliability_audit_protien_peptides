"""Substitute the manuscript placeholders with values read from results/tables/.

    python scripts/fill_results.py            # report what would change
    python scripts/fill_results.py --apply    # write it

Why this exists. Every number in the abstract is a transcription from a CSV, and
hand-copying is where published figures get corrupted. This reads the artefacts
and performs the substitution mechanically, so the manuscript cannot disagree
with the data it came from. Re-running after a re-analysis updates the prose.

Nothing is invented: if a required table is absent, the placeholder is left
alone and reported, and the submission gate keeps failing until it is resolved.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
MS = ROOT / "Projects" / "BRA" / "manuscript"


def read(name: str, **kw):
    f = TAB / name
    return pd.read_csv(f, **kw) if f.exists() else None


def fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float, np.floating)) else str(v)


def collect() -> tuple[dict, list]:
    vals, missing = {}, []

    cells = read("armB_mean_mcc_by_cell.csv", index_col=0)
    if cells is not None:
        arr = cells.to_numpy(dtype=float)
        vals["[X]"] = fmt(np.nanmin(arr))
        vals["[Y]"] = fmt(np.nanmax(arr))
    else:
        missing.append("armB_mean_mcc_by_cell.csv -> [X] [Y]")

    tau = read("armB_tau_by_rule.csv", index_col=0)
    if tau is not None:
        a = tau.to_numpy(dtype=float)
        off = a[np.triu_indices_from(a, k=1)]
        vals["[T]"] = fmt(float(np.nanmean(off)))
    else:
        missing.append("armB_tau_by_rule.csv -> [T]")

    ceil = read("armB_ceiling_seed_pairs.csv")
    if ceil is not None:
        c = float(ceil["tau_b"].mean())
        vals["[C]"] = fmt(c)
        if "[T]" in vals:
            vals["[N]"] = fmt(float(vals["[T]"]) / c)
    else:
        missing.append("armB_ceiling_seed_pairs.csv -> [C] [N]")

    kmat = read("armB_rank_matrix_by_rule.csv", index_col=0)
    if kmat is not None:
        winners = {c: kmat[c].idxmin() for c in kmat.columns}
        pairs = list(itertools.combinations(kmat.columns, 2))
        vals["[K]"] = str(sum(winners[i] != winners[j] for i, j in pairs))
        vals["[R]"] = str(int((kmat.max(axis=1) - kmat.min(axis=1)).max()))
    else:
        missing.append("armB_rank_matrix_by_rule.csv -> [K] [R]")

    vc = read("armB_variance_components.csv", index_col=0)
    if vc is not None and "pct_rank_var" in vc.columns:
        g = vc["pct_rank_var"]
        for key, comp in [("[A]", "negative_rule"), ("[B]", "cluster_threshold"),
                          ("[S]", "seed")]:
            if comp in g.index:
                vals[key] = fmt(float(g.loc[comp]), 1)
    else:
        missing.append("armB_variance_components.csv -> [A] [B] [S]")

    nb = read("armB_null_baseline.csv")
    if nb is not None and len(nb):
        best = nb.loc[nb["rank"].idxmin()]
        vals["[P]"] = f"{int(best['rank'])} of 16"
        vals["[rule]"] = str(best["negative_rule"]).replace("_", " ")
    else:
        missing.append("armB_null_baseline.csv -> [P] [rule]")

    return vals, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    vals, missing = collect()
    print(f"resolved {len(vals)} placeholder(s) from results/tables/:")
    for k, v in sorted(vals.items()):
        print(f"  {k:8s} -> {v}")
    if missing:
        print("\nnot yet available (left untouched):")
        for m in missing:
            print(f"  {m}")

    targets = [MS / "00 Abstract and claims.md", MS / "04 Results template.md",
               MS / "03 Discussion.md"]
    total = 0
    for f in targets:
        if not f.exists():
            continue
        s = orig = f.read_text()
        for k, v in vals.items():
            s = s.replace(k, v)
        n = sum(orig.count(k) for k in vals)
        if n:
            print(f"\n{f.name}: {n} substitution(s)" + ("" if args.apply else " (dry run)"))
            total += n
        if args.apply and s != orig:
            f.write_text(s)

    if not args.apply:
        print(f"\ndry run: {total} substitution(s) available. Re-run with --apply.")
    else:
        print(f"\napplied {total} substitution(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

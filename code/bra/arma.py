"""Arm A: the same fixed pipelines across independently curated peptide tasks,
each under its own native homology-aware split.

Source: the AutoPeptideML benchmark collection (Fernandez-Diaz et al. 2024),
16 peptide bioactivity datasets with predefined train/test partitions produced
by their homology-based partitioning procedure.

What this arm can and cannot claim. The 16 datasets were curated by one group
with one pipeline, so they are **not** 16 independently built benchmarks; they
are one protocol applied to 16 biological questions. The question Arm A answers
is therefore narrower and cleaner than "do benchmarks agree": it is **does the
ranking of methods transfer across tasks when the evaluation protocol is held
constant?** That is the complement of Arm B, which holds the task constant and
varies the protocol. Together they separate the two sources of ranking movement.

Tasks (train/test): AB antibacterial 13245/3311, AMP antimicrobial 10266/2566,
AV antiviral 4711/1177, Neuro neuropeptide 3881/969, TOX toxicity 3092/772,
ACE ACE-inhibitory 1685/421, AF antifungal 1591/395, ACP anticancer 1378/344,
DPPIV 1063/265, TTCA tumour T-cell antigen 948/236, plus six smaller sets.
All are balanced near 1:1 and fully canonical.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Tasks with enough training data to rank a 20+ pipeline zoo meaningfully.
ARMA_TASKS = ["AB", "AMP", "AV", "Neuro", "TOX", "ACE", "AF", "ACP", "DPPIV", "TTCA"]

TASK_DESC = {
    "AB": "antibacterial", "AMP": "antimicrobial", "AV": "antiviral",
    "Neuro": "neuropeptide", "TOX": "toxicity", "ACE": "ACE-inhibitory",
    "AF": "antifungal", "ACP": "anticancer", "DPPIV": "DPP-IV inhibitory",
    "TTCA": "tumour T-cell antigen", "AOX": "antioxidant", "APP": "anti-parasitic",
    "QS": "quorum sensing", "MRSA": "anti-MRSA", "AMAP": "anti-MAP",
    "BBP": "blood-brain barrier",
}


def arma_root() -> Path:
    return (Path(os.environ.get("BRA_DATA", Path.home() / "bra-data"))
            / "raw" / "armA" / "apml" / "PeptideBenchmarks")


def load_task(task: str, drop_cross_split_duplicates: bool = True) -> pd.DataFrame:
    """Canonical table for one task, using the benchmark's own split.

    `drop_cross_split_duplicates` removes sequences present in BOTH partitions.
    Several of these published sets contain them (AV 117, AB 77, TTCA 46,
    ACE 26, ACP 15, DPPIV 16, AF 8, Neuro 10), which is exact-duplicate leakage
    across a split that is otherwise homology-controlled. We remove the test-side
    copy so the held-out partition is clean, and report the count, rather than
    silently scoring on it.
    """
    d = arma_root() / task / "splits"
    tr = pd.read_csv(d / "train.csv"); tr["split"] = "train"
    te = pd.read_csv(d / "test.csv");  te["split"] = "test"

    n_dup = 0
    if drop_cross_split_duplicates:
        dup = set(tr.sequence) & set(te.sequence)
        n_dup = len(dup)
        te = te.loc[~te.sequence.isin(dup)]

    df = pd.concat([tr, te], ignore_index=True)
    out = pd.DataFrame({
        "sequence": df.sequence.astype(str).str.upper(),
        "label": df.Y.astype(float),
        "split": df.split,
    })
    out["source"] = "AutoPeptideML"
    out["benchmark"] = f"armA_{task}"
    out["task"] = task
    out.attrs["n_cross_split_duplicates"] = n_dup
    return out.astype({"sequence": "string", "split": "string",
                       "source": "string", "benchmark": "string", "task": "string"})


def duplicate_audit() -> pd.DataFrame:
    """Exact sequences appearing in both partitions of each published task."""
    rows = []
    for t in sorted(TASK_DESC):
        d = arma_root() / t / "splits"
        if not (d / "train.csv").exists():
            continue
        tr = pd.read_csv(d / "train.csv"); te = pd.read_csv(d / "test.csv")
        dup = set(tr.sequence) & set(te.sequence)
        rows.append({"task": t, "description": TASK_DESC[t],
                     "n_train": len(tr), "n_test": len(te),
                     "n_cross_split_duplicates": len(dup),
                     "pct_of_test": round(100 * len(dup) / max(len(te), 1), 2)})
    return pd.DataFrame(rows).sort_values("pct_of_test", ascending=False)

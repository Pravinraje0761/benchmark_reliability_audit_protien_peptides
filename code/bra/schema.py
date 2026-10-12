"""Canonical data structures for the Benchmark Reliability Audit.

Everything downstream reads these. If a column is not defined here, it does not
exist as far as the analysis is concerned.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import pandas as pd

# --------------------------------------------------------------------------- #
# Peptide table: what every benchmark loader must return
# --------------------------------------------------------------------------- #

PEPTIDE_COLUMNS = {
    "sequence": "string",   # uppercase, 20 canonical AA only
    "label": "float64",     # 0/1 for classification, continuous for regression
    "split": "string",      # train | valid | test | <NA> if the benchmark ships no fixed split
    "source": "string",     # originating database, for leave-source-out splitting
    "benchmark": "string",  # B1..B6
    "task": "string",       # amp_binary | hemolysis_binary | mic_reg | hc50_reg
}

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def validate_peptides(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly and early. A loader that cannot pass this is not finished."""
    missing = set(PEPTIDE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"loader output missing columns: {sorted(missing)}")

    if df["sequence"].isna().any():
        raise ValueError("null sequences present")

    bad = df.loc[~df["sequence"].map(lambda s: set(s) <= VALID_AA)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} sequences contain non-canonical residues, e.g. "
            f"{bad['sequence'].iloc[0][:40]!r}. Decide explicitly whether to drop "
            "or map them, and record the decision in the dataset note."
        )

    dupes = df["sequence"].duplicated().sum()
    if dupes:
        # Not fatal: some benchmarks legitimately repeat a sequence across tasks.
        # But it must be visible, because silent duplicates are leakage.
        print(f"[warn] {dupes} duplicated sequences in {df['benchmark'].iloc[0]}")

    return df


# --------------------------------------------------------------------------- #
# Results table: one row per (cell, model, fold)
# --------------------------------------------------------------------------- #

RESULT_COLUMNS = {
    "benchmark": "string",
    "task": "string",
    "arm": "string",            # A (as-published) | B (factorial)
    "model_id": "string",       # M01..M12
    "representation": "string",
    "learner": "string",
    "cluster_threshold": "string",   # random | 0.9 | 0.7 | 0.4 | kmer_jaccard | leave_source_out
    "negative_rule": "string",       # random_uniprot | length_matched | ... | native
    "seed": "int64",
    "fold": "int64",
    "metric": "string",
    "score": "float64",
    "n_train": "int64",
    "n_test": "int64",
    "cluster_ratio": "float64",      # n_clusters / n_sequences, the cosmetic-control detector
    "fit_seconds": "float64",
    "run_ts": "string",              # ISO-8601 UTC, e.g. 2026-10-04T13:05:22Z
    "code_version": "string",        # git sha, so a row is traceable to the code that made it
}


@dataclass(frozen=True)
class Cell:
    """One evaluation condition. All 12 pipelines see an identical Cell."""
    benchmark: str
    task: str
    arm: Literal["A", "B"]
    cluster_threshold: str
    negative_rule: str
    seed: int

    def key(self) -> str:
        return "|".join(
            [self.benchmark, self.task, self.arm,
             self.cluster_threshold, self.negative_rule, str(self.seed)]
        )


@dataclass
class RunRecord:
    cell: Cell
    model_id: str
    representation: str
    learner: str
    fold: int
    scores: dict[str, float]
    n_train: int
    n_test: int
    cluster_ratio: float
    fit_seconds: float
    run_ts: str
    code_version: str

    def to_rows(self) -> list[dict[str, Any]]:
        """One row per metric. Metrics are free: same predictions, different readout."""
        base = {
            "benchmark": self.cell.benchmark,
            "task": self.cell.task,
            "arm": self.cell.arm,
            "model_id": self.model_id,
            "representation": self.representation,
            "learner": self.learner,
            "cluster_threshold": self.cell.cluster_threshold,
            "negative_rule": self.cell.negative_rule,
            "seed": self.cell.seed,
            "fold": self.fold,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "cluster_ratio": self.cluster_ratio,
            "fit_seconds": self.fit_seconds,
            "run_ts": self.run_ts,
            "code_version": self.code_version,
        }
        return [{**base, "metric": m, "score": s} for m, s in self.scores.items()]

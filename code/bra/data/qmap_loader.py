"""QMAP loader (benchmark B2).

Protocol as published, established by inspecting the package API on 2026-10-05:

* `QMAPBenchmark(split=s)` for s in 0..4 is the **test set** (3,650 sequences at
  split 0). The benchmark ships five predefined test sets, so in Arm A the five
  BRA seeds map onto QMAP own five splits rather than onto re-partitioning.
* Training data is the full DBAASP pool, `DBAASPDataset()` -> 18,033 samples.
* **Leakage control works by filtering the training set, not by partitioning a
  corpus.** `benchmark.get_train_mask(sequences, threshold=0.6, matrix="blosum45")`
  returns a boolean mask over YOUR training sequences, dropping any with >0.6
  identity to any test sequence. That is a different mechanism AND a different
  threshold and tool (BLOSUM45 pairwise alignment, not MMseqs2) from every other
  benchmark in the inventory -- exactly the protocol variation BRA measures.

Task choices, recorded because they are judgement calls:

* `mic_reg` uses **log10 of the consensus MIC for one species**, default
  *Escherichia coli* (2,112 of 3,650 test sequences, the best covered of 310
  species; S. aureus 2,083 and P. aeruginosa 1,413 are the alternatives). QMAP
  reports per-target metrics, so a species must be named; pooling across 310
  species would blend assay conditions.
* `hc50_reg` uses log10 of the consensus HC50. Only **536 of 3,650** test
  sequences carry a value, so this task is underpowered and is reported as
  exploratory, consistent with the pre-registration.
* MIC spans 0.018 to 4,000 ug/mL at split 0, hence the log transform.

The expensive step is the train mask: 18,033 training by 3,650 test pairwise
alignments. It is cached to disk per (split, threshold), keyed so a changed
threshold invalidates it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import register

DEFAULT_SPECIES = "Escherichia coli"
QMAP_THRESHOLD = 0.6          # as published; do not change without a deviation
QMAP_MATRIX = "blosum45"


def _data_root() -> Path:
    """Bulk artefacts live on the Linux filesystem, not the Windows mount.

    Cross-mount I/O through /mnt/c is slow enough to matter for the alignment
    cache. Override with BRA_DATA.
    """
    root = Path(os.environ.get("BRA_DATA", Path.home() / "bra-data")) / "qmap"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _log10_positive(x: float | None) -> float:
    if x is None:
        return float("nan")
    x = float(x)
    return float(np.log10(x)) if x > 0 else float("nan")


def _train_mask(bench, sequences: list[str], split: int,
                threshold: float, num_threads: int | None) -> np.ndarray:
    cache = _data_root() / f"train_mask_split{split}_thr{threshold}_{QMAP_MATRIX}.npy"
    if cache.exists():
        mask = np.load(cache)
        if len(mask) == len(sequences):
            return mask
        print(f"[warn] cached mask length {len(mask)} != {len(sequences)}; recomputing")
    mask = bench.get_train_mask(sequences, threshold=threshold, matrix=QMAP_MATRIX,
                                num_threads=num_threads, show_progress=True)
    np.save(cache, mask)
    return mask


def load_qmap(split: int = 0, species: str = DEFAULT_SPECIES,
              task: str = "mic_reg", threshold: float = QMAP_THRESHOLD,
              num_threads: int | None = None,
              preprocess: str = "bra_canonical") -> pd.DataFrame:
    from qmap import DBAASPDataset, QMAPBenchmark

    if task not in {"mic_reg", "hc50_reg"}:
        raise ValueError(f"task must be mic_reg or hc50_reg, got {task!r}")

    bench = QMAPBenchmark(split=split)
    pool = DBAASPDataset()

    mask = _train_mask(bench, list(pool.sequences), split, threshold, num_threads)
    train_idx = np.flatnonzero(mask)

    def _rows(dataset, indices, split_label):
        out = []
        seqs = dataset.sequences
        targets = dataset.targets
        hc50s = dataset.hc50s
        for i in indices:
            if task == "mic_reg":
                hits = [t for t in targets[i] if t.name == species]
                label = _log10_positive(hits[0].consensus) if hits else float("nan")
            else:
                h = hc50s[i]
                label = _log10_positive(getattr(h, "consensus", None)) if h else float("nan")
            if not np.isfinite(label):
                continue
            out.append({"sequence": seqs[i], "label": label, "split": split_label})
        return out

    rows = _rows(pool, train_idx, "train")
    rows += _rows(bench, range(len(bench.sequences)), "test")

    df = pd.DataFrame(rows)
    df["source"] = "DBAASP"
    df["benchmark"] = "B2_qmap"
    df["task"] = task

    if preprocess == "native":
        pass
    elif preprocess == "bra_canonical":
        df = _bra_canonical(df)
    else:
        raise ValueError(f"preprocess must be native or bra_canonical, got {preprocess!r}")

    return df.astype({"sequence": "string", "split": "string", "source": "string",
                      "benchmark": "string", "task": "string"})


L_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")


def label_noise_report(df: pd.DataFrame) -> pd.DataFrame:
    """How much do repeated measurements of the SAME sequence disagree?

    This bounds the achievable score on the benchmark, independently of any model.
    At split 0, 56 of 63 duplicated test sequences carry disagreeing MIC values,
    median log10 spread 0.52 (a 3.3-fold disagreement) and max 2.12 (132-fold).

    Relevant to the QMAP headline claim of "limited progress over six years":
    part of that ceiling is irreducible label noise, not model failure. Worth
    reporting in the paper rather than attributing the whole gap to methods.
    """
    out = []
    for sp, g in df.groupby("split", observed=True):
        vc = g["sequence"].value_counts()
        dups = vc[vc > 1]
        spread = (g.groupby("sequence")["label"].agg(lambda s: s.max() - s.min()))
        spread = spread[spread > 0]
        out.append({
            "split": sp,
            "n_rows": len(g),
            "n_dup_sequences": int(len(dups)),
            "n_excess_rows": int(dups.sum() - len(dups)) if len(dups) else 0,
            "n_disagreeing": int(len(spread)),
            "median_log10_spread": float(spread.median()) if len(spread) else 0.0,
            "max_log10_spread": float(spread.max()) if len(spread) else 0.0,
        })
    return pd.DataFrame(out)


def _bra_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed preprocessing for the cross-benchmark comparison.

    Two decisions, both declared and both applied IDENTICALLY to every pipeline
    and every benchmark -- otherwise preprocessing silently becomes part of the
    model (the lesson from Soto-Garcia et al. 2026).

    1. **Keep L-canonical sequences only.** QMAP encodes D-amino acids as
       lowercase and uses X, O (ornithine) and B (DAB). At split 0 that is 1,193
       of 10,624 rows, of which 799 are purely a chirality issue. Upper-casing
       them would merge D and L forms into identical feature vectors with
       different labels -- manufacturing exactly the hidden-duplicate problem
       this project audits. So they are dropped, not coerced.
    2. **Collapse duplicate sequences to the median label.** The same peptide
       appears as several DBAASP records with different measured MICs. Keeping
       the duplicates would let the metric double-count those peptides and
       penalise models for inconsistent ground truth; the median is the robust
       summary. The discarded disagreement is quantified by
       `label_noise_report` and reported, not hidden.

    This is applied to BOTH splits, so it is NOT the as-published view. The
    reproduction gate (E03) must use `preprocess="native"` plus QMAP own
    `compute_metrics`, or it is not reproducing QMAP.
    """
    keep = df["sequence"].map(lambda x: set(x) <= L_CANONICAL)
    df = df.loc[keep].copy()
    agg = (df.groupby(["split", "sequence"], as_index=False, observed=True)
             .agg(label=("label", "median"), n_measurements=("label", "size")))
    agg["source"] = "DBAASP"
    agg["benchmark"] = "B2_qmap"
    agg["task"] = df["task"].iloc[0] if len(df) else "mic_reg"
    return agg


@register("B2_qmap")
def load_b2() -> pd.DataFrame:
    """Default Arm A view: MIC regression against E. coli, split 0."""
    return load_qmap(split=0, species=DEFAULT_SPECIES, task="mic_reg")


@register("B2_qmap_hc50")
def load_b2_hc50() -> pd.DataFrame:
    """Exploratory: HC50 regression. Only 536 test points -- underpowered."""
    return load_qmap(split=0, task="hc50_reg")

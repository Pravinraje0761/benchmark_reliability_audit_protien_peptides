"""Tests for the B2 preprocessing decisions.

Pure logic on synthetic frames -- no QMAP download needed, so these run in CI
and guard the two judgement calls recorded in Datasets/B2 QMAP.md.
"""
import numpy as np
import pandas as pd
import pytest

from bra.data.qmap_loader import _bra_canonical, label_noise_report


def _frame(pairs, split="train"):
    return pd.DataFrame([{"sequence": s, "label": v, "split": split,
                          "source": "DBAASP", "benchmark": "B2_qmap",
                          "task": "mic_reg"} for s, v in pairs])


def test_d_amino_acids_are_dropped_not_uppercased():
    """Upper-casing would merge D and L forms into identical features with
    different labels -- manufacturing the hidden-duplicate problem we audit."""
    df = _frame([("KLAKLAK", 1.0), ("kLAKLAK", 2.0)])
    out = _bra_canonical(df)
    assert list(out["sequence"]) == ["KLAKLAK"]


def test_non_standard_residues_are_dropped():
    """X (unknown), O (ornithine), B (DAB) have no slot in one-hot or AAC."""
    df = _frame([("KLAKLAK", 1.0), ("RRXXRF", 1.0), ("KLAOLAK", 1.0), ("KLABLAK", 1.0)])
    out = _bra_canonical(df)
    assert list(out["sequence"]) == ["KLAKLAK"]


def test_duplicates_collapse_to_median():
    df = _frame([("KLAKLAK", 1.0), ("KLAKLAK", 2.0), ("KLAKLAK", 9.0)])
    out = _bra_canonical(df)
    assert len(out) == 1
    assert out["label"].iloc[0] == pytest.approx(2.0)      # median, not mean
    assert out["n_measurements"].iloc[0] == 3


def test_duplicates_are_collapsed_within_split_not_across():
    """Collapsing across splits would move a test sequence into train."""
    df = pd.concat([_frame([("KLAKLAK", 1.0)], "train"),
                    _frame([("KLAKLAK", 5.0)], "test")])
    out = _bra_canonical(df)
    assert len(out) == 2
    assert set(out["split"]) == {"train", "test"}


def test_label_noise_report_quantifies_disagreement():
    df = _frame([("AAAA", 1.0), ("AAAA", 3.0), ("CCCC", 2.0)])
    rep = label_noise_report(df)
    row = rep.iloc[0]
    assert row["n_dup_sequences"] == 1
    assert row["n_disagreeing"] == 1
    assert row["max_log10_spread"] == pytest.approx(2.0)


def test_label_noise_report_reports_zero_when_clean():
    rep = label_noise_report(_frame([("AAAA", 1.0), ("CCCC", 2.0)]))
    assert rep.iloc[0]["n_dup_sequences"] == 0
    assert rep.iloc[0]["median_log10_spread"] == 0.0

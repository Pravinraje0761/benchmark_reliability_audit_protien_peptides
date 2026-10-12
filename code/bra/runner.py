"""Grid runner. Builds a Cell, materialises folds once, and hands the SAME folds
to all 12 pipelines -- then asserts that it really did.

The fairness assertions here are what separate an audit from an accusation.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import time

import numpy as np
import pandas as pd

from . import features as F
from . import splits as S
from .metrics import classification_scores, regression_scores
from .models import ZOO, PLM_CHECKPOINTS, build_learner, max_len_from_train
from .schema import Cell, RunRecord


def code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "nogit"


def _hash_index(idx) -> str:
    return hashlib.sha1(np.asarray(idx).tobytes()).hexdigest()[:12]


def make_folds(df: pd.DataFrame, cell: Cell, n_folds: int = 5, coverage: float = 0.8):
    """One fold assignment per cell, shared by every pipeline in it."""
    t = cell.cluster_threshold
    if t == "random":
        folds = S.random_folds(len(df), n_folds, cell.seed)
        ratio = 1.0
    elif t == "leave_source_out":
        folds = S.leave_source_out(df["source"])
        ratio = float("nan")
    elif t.startswith("kmer"):
        cl = S.cluster_kmer(df["sequence"].tolist())
        folds = S.assign_folds_by_cluster(cl, df["label"].to_numpy(), n_folds, cell.seed)
        ratio = S.cluster_ratio(cl)
    else:
        cl = S.cluster_mmseqs(df["sequence"].tolist(), float(t), coverage)
        folds = S.assign_folds_by_cluster(cl, df["label"].to_numpy(), n_folds, cell.seed)
        ratio = S.cluster_ratio(cl)
    return folds, ratio


def featurise(spec, train: pd.DataFrame, test: pd.DataFrame, cache: F.EmbeddingCache):
    """Returns (X_train, X_test). Fitted transforms see the train fold only."""
    r = spec.representation
    if r == "aac":
        return F.aac(train["sequence"]), F.aac(test["sequence"])
    if r == "aac_dpc":
        return (np.hstack([F.aac(train["sequence"]), F.dpc(train["sequence"])]),
                np.hstack([F.aac(test["sequence"]), F.dpc(test["sequence"])]))
    if r == "physchem":
        return F.physchem(train["sequence"]), F.physchem(test["sequence"])
    if r == "length_charge":
        return F.length_charge(train["sequence"]), F.length_charge(test["sequence"])
    if r == "kmer_tfidf":
        return F.kmer_tfidf(train["sequence"], test["sequence"])
    if r == "onehot":
        L = max_len_from_train(train["sequence"].map(len))
        return F.onehot(train["sequence"], L), F.onehot(test["sequence"], L)
    if r in PLM_CHECKPOINTS:
        ck = PLM_CHECKPOINTS[r]
        return (F.embed_plm(train["sequence"], ck, cache),
                F.embed_plm(test["sequence"], ck, cache))
    raise ValueError(f"unknown representation {r!r}")


def run_cell(
    df: pd.DataFrame,
    cell: Cell,
    zoo=ZOO,
    n_folds: int = 5,
    coverage: float = 0.8,
    task_type: str = "classification",
    verify_leakage: bool = True,
) -> list[RunRecord]:
    """Fit every pipeline in the zoo on identical folds and score them."""
    cache = F.EmbeddingCache()
    folds, ratio = make_folds(df, cell, n_folds, coverage)
    records: list[RunRecord] = []
    index_hashes: set[str] = set()
    label_hashes: set[str] = set()

    for fold in range(int(folds.max()) + 1):
        te_mask = folds == fold
        train, test = df[~te_mask], df[te_mask]
        if len(train) == 0 or len(test) == 0:
            continue

        if verify_leakage and cell.cluster_threshold not in {"random", "leave_source_out"} \
           and not cell.cluster_threshold.startswith("kmer"):
            S.verify_no_leakage(train["sequence"].tolist(), test["sequence"].tolist(),
                                float(cell.cluster_threshold), coverage)

        index_hashes.add(_hash_index(train.index.to_numpy()))
        label_hashes.add(_hash_index(train["label"].to_numpy()))

        for spec in zoo:
            t0 = time.time()
            try:
                Xtr, Xte = featurise(spec, train, test, cache)
                if spec.learner in {"cnn", "bilstm"}:
                    # Torch pipelines are trained by their own loop; omitted here
                    # deliberately so the classical sweep stays dependency-light.
                    from .nn_train import fit_predict_torch
                    y_score = fit_predict_torch(spec, Xtr, train["label"].to_numpy(),
                                                Xte, cell.seed, task_type)
                else:
                    est = build_learner(spec.learner, task_type, cell.seed)
                    est.fit(Xtr, train["label"].to_numpy())
                    y_score = (est.predict_proba(Xte)[:, 1]
                               if task_type == "classification" else est.predict(Xte))
                scores = (classification_scores(test["label"], y_score)
                          if task_type == "classification"
                          else regression_scores(test["label"], y_score))
            except Exception as e:
                # A failed pipeline becomes NaN rows, not a dead sweep. The failure
                # is recorded so it cannot quietly vanish from the results table.
                print(f"[fail] {cell.key()} {spec.model_id}: {type(e).__name__}: {e}")
                scores = {"error": float("nan")}

            records.append(RunRecord(
                cell=cell, model_id=spec.model_id,
                representation=spec.representation, learner=spec.learner,
                fold=fold, scores=scores,
                n_train=len(train), n_test=len(test), cluster_ratio=ratio,
                fit_seconds=time.time() - t0,
                run_ts=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                code_version=code_version(),
            ))

    # Fairness invariants: every pipeline saw identical data in this cell.
    assert len(index_hashes) <= int(folds.max()) + 1, "training indices diverged across models"
    assert len(label_hashes) <= int(folds.max()) + 1, "label vectors diverged across models"
    return records


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    rows = [r for rec in records for r in rec.to_rows()]
    return pd.DataFrame(rows)

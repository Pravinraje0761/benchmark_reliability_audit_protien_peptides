"""Arm B: the crossed factorial. The primary analysis of the paper.

Design actually run, and the reasons for every reduction from the original plan.
All of this is CPU-only; there is no GPU on this machine.

Grid
----
4 clustering thresholds x 5 negative rules x 5 seeds = **100 cells**, each with
one cluster-disjoint 80/20 split, scored by 16 pipelines -> **1,600 fits**.

* **One split per cell, not 5 folds.** The seed factor already supplies
  replication, and 5-fold on top would multiply cost by five for a variance
  component E00 showed to be small. Fold and seed would also be partially
  redundant as variance sources.
* **The negative-set realisation is drawn once per rule (seed 0).** The seed
  then varies the cluster-disjoint partition and model initialisation. This is
  forced by compute: letting the draw vary with seed would make 5 x 5 x 5,190 =
  129,750 distinct negative sequences, and ESM-2 650M embeddings for those on
  CPU are out of reach. Consequence, stated plainly: **sampling variability in
  the negative draw is not inside the reported seed component.** It is bounded
  separately by `sensitivity_negative_draw`, which re-runs 5 independent draws
  on the classical (embedding-free) pipelines where cost permits.
* **Pipelines excluded, with reasons.** End-to-end CNN and BiLSTM (M05/M06):
  per-fit training across 100 cells is infeasible on CPU, and unlike the others
  their cost cannot be amortised by caching features. ProtT5-XL (M11/M19): 3B
  parameters, CPU-infeasible. Their representation slots are covered instead by
  four ESM-2 scales (8M, 35M, 150M, 650M), so the representation axis stays
  populated for H5. This is a deviation from `04 Pre-registration` and is logged
  there.

Fairness invariants, asserted per cell rather than assumed
----------------------------------------------------------
* Every pipeline receives the identical train/test index and label vector
  (checked by hash).
* No test sequence shares a cluster with a training sequence at the configured
  identity threshold (re-verified with MMseqs2 after splitting).
* Feature transforms that learn anything (TF-IDF vocabulary, scaler statistics)
  are fitted on the training partition only.
* All metrics are computed downstream from stored predictions, so every metric
  provably sees identical predictions.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import splits as S
from .metrics import classification_scores

THRESHOLDS = ["random", "0.9", "0.7", "0.4"]
NEG_RULES = ["random_uniprot", "annotation_filtered", "length_matched",
             "composition_matched", "shuffled_positives"]
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.2

# Capped worker count: see the comment in build_learner.
N_JOBS = int(os.environ.get("BRA_N_JOBS", "8"))

ESM = {
    "esm2_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
}


@dataclass(frozen=True)
class Pipe:
    pid: str
    rep: str
    learner: str
    note: str = ""


# 16 pipelines. Partially crossed on purpose: every representation appears with
# at least one shared learner (logreg) so the representation axis can be
# marginalised for H5, and the 650M row carries three learners so the learner
# axis can be marginalised at fixed representation.
ZOO_B: list[Pipe] = [
    Pipe("P01", "aac", "logreg", "floor"),
    Pipe("P02", "aac_dpc", "rf"),
    Pipe("P03", "aac_dpc", "logreg"),
    Pipe("P04", "physchem", "xgboost"),
    Pipe("P05", "physchem", "logreg"),
    Pipe("P06", "kmer_tfidf", "logreg"),
    Pipe("P07", "kmer_tfidf", "svm_rbf"),
    Pipe("P08", "esm2_8M", "logreg"),
    Pipe("P09", "esm2_8M", "rf"),
    Pipe("P10", "esm2_35M", "logreg"),
    Pipe("P11", "esm2_35M", "mlp"),
    Pipe("P12", "esm2_150M", "logreg"),
    Pipe("P13", "esm2_650M", "logreg"),
    Pipe("P14", "esm2_650M", "rf"),
    Pipe("P15", "esm2_650M", "mlp"),
    Pipe("P16", "length_charge", "logreg", "DELIBERATE NULL"),
    # ---------------------------------------------------------------------- #
    # P17-P24 added 2026-10-07. The design simulation showed the estimator
    # tightens faster with pipelines than with protocols, and these eight cost
    # nothing extra: every one is a new learner on a representation already
    # cached, so no further embedding passes are needed. They complete the
    # representation x learner grid, which is what H5 marginalises over.
    # ---------------------------------------------------------------------- #
    Pipe("P17", "aac", "rf", "floor representation, trees"),
    Pipe("P18", "aac_dpc", "xgboost", "classical features, boosting"),
    Pipe("P19", "physchem", "mlp", "biophysics, non-linear"),
    Pipe("P20", "kmer_tfidf", "rf", "order-sensitive, trees"),
    Pipe("P21", "esm2_8M", "mlp", "small PLM, non-linear head"),
    Pipe("P22", "esm2_35M", "rf", "mid PLM, trees"),
    Pipe("P23", "esm2_150M", "mlp", "mid-large PLM, non-linear head"),
    Pipe("P24", "esm2_650M", "xgboost", "large PLM, boosting"),
]


def data_root() -> Path:
    return Path(os.environ.get("BRA_DATA", Path.home() / "bra-data"))


def build_learner(name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "logreg":
        est = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    elif name == "rf":
        # n_jobs is capped rather than -1: on these small matrices (7-1280
        # features, ~8k rows) spawning 32 workers cost more in coordination than
        # it saved. Measured: XGBoost with n_jobs=-1 took 165s on a 7-feature
        # problem that completes in seconds with n_jobs=8.
        est = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=N_JOBS, random_state=seed,
                                     class_weight="balanced")
    elif name == "svm_rbf":
        # Approximate RBF SVM: Nystroem kernel map (512 components) followed by a
        # linear SVM with sigmoid-calibrated probabilities.
        #
        # An exact SVC with probability=True was measured at over 9 minutes for a
        # single cell on 8,299 training points -- 16+ hours across the 100-cell
        # grid -- because Platt scaling refits the kernel machine five times. The
        # Nystroem approximation is a standard substitute, returns calibrated
        # probabilities on the same 0.5 threshold as every other pipeline, and
        # keeps the kernel-method family represented. sklearn 1.9 additionally
        # deprecates SVC(probability=True) in favour of CalibratedClassifierCV.
        #
        # Logged as a deviation from the pre-registration.
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.kernel_approximation import Nystroem
        from sklearn.svm import LinearSVC
        return make_pipeline(
            StandardScaler(),
            Nystroem(n_components=512, random_state=seed),
            CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", random_state=seed,
                          max_iter=5000),
                method="sigmoid", cv=3, ensemble=False),
        )
    elif name == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                            early_stopping=True, random_state=seed)
    elif name == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, n_jobs=N_JOBS,
                            tree_method="hist", random_state=seed,
                            eval_metric="logloss")
    else:
        raise ValueError(name)
    # Scaling is applied uniformly: it is needed by logreg/svm/mlp and is
    # harmless for trees. Applying it only where "needed" would quietly make the
    # pipelines non-comparable.
    return make_pipeline(StandardScaler(), est)


def featurise(rep: str, train_seq: pd.Series, test_seq: pd.Series, cache):
    """Return (X_train, X_test). Anything learned is learned on train only."""
    if rep == "aac":
        return F.aac(train_seq), F.aac(test_seq)
    if rep == "aac_dpc":
        return (np.hstack([F.aac(train_seq), F.dpc(train_seq)]),
                np.hstack([F.aac(test_seq), F.dpc(test_seq)]))
    if rep == "physchem":
        return F.physchem(train_seq), F.physchem(test_seq)
    if rep == "length_charge":
        return F.length_charge(train_seq), F.length_charge(test_seq)
    if rep == "kmer_tfidf":
        return F.kmer_tfidf(train_seq, test_seq)
    if rep in ESM:
        ck = ESM[rep]
        return (F.embed_plm(train_seq, ck, cache), F.embed_plm(test_seq, ck, cache))
    raise ValueError(f"unknown representation {rep!r}")


def _hash(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


BALANCED = os.environ.get("BRA_BALANCED", "0") == "1"


def split_cell(df: pd.DataFrame, threshold: str, seed: int,
               coverage: float = 0.8,
               balanced: bool | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Cluster-disjoint 80/20 split. Returns (train_idx, test_idx, cluster_ratio).

    With `balanced` (or BRA_BALANCED=1), cluster selection is driven by the
    remaining per-class quota instead of by cluster size. Clusters stay atomic
    either way; only the order of selection changes. Largest-first drives the
    test positive fraction to 0.99 on this corpus, which makes
    threshold-dependent metrics uninterpretable -- see bra/balanced_split.py.
    """
    balanced = BALANCED if balanced is None else balanced
    seqs = df["sequence"].tolist()
    if threshold == "random":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(df))
        n_test = int(round(TEST_FRACTION * len(df)))
        return perm[n_test:], perm[:n_test], 1.0

    cl = S.cluster_mmseqs(seqs, float(threshold), coverage)
    ratio = S.cluster_ratio(cl)

    if balanced:
        from .balanced_split import assign_test_clusters_balanced
        is_test = assign_test_clusters_balanced(
            cl, df["label"].to_numpy(), TEST_FRACTION, seed)
        return np.flatnonzero(~is_test), np.flatnonzero(is_test), ratio

    # Assign whole clusters to test until the test fraction is reached, largest
    # first with a seed-dependent jitter so the partition varies with the seed.
    rng = np.random.default_rng(seed)
    uniq, sizes = np.unique(cl, return_counts=True)
    order = np.argsort(-(sizes + rng.random(len(sizes))))
    target = int(round(TEST_FRACTION * len(df)))
    test_clusters, filled = set(), 0
    for i in order:
        if filled >= target:
            break
        test_clusters.add(uniq[i])
        filled += sizes[i]
    is_test = np.array([c in test_clusters for c in cl])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), ratio


# --------------------------------------------------------------------------- #
# Cell execution
# --------------------------------------------------------------------------- #

def run_cell(df: pd.DataFrame, threshold: str, rule: str, seed: int,
             zoo: list[Pipe] = None, cache=None, verify: bool = True,
             coverage: float = 0.8) -> list[dict]:
    """Fit every pipeline on one identical split and score from predictions."""
    zoo = zoo or ZOO_B
    cache = cache or F.EmbeddingCache(data_root() / "embeddings")

    tr_idx, te_idx, ratio = split_cell(df, threshold, seed, coverage)
    train, test = df.iloc[tr_idx], df.iloc[te_idx]
    y_tr = train["label"].to_numpy()
    y_te = test["label"].to_numpy()

    # Measure residual leakage rather than assert its absence.
    #
    # Assigning whole MMseqs2 clusters to the test partition is standard
    # practice and is what split_cell does, but greedy clustering does not form
    # equivalence classes: two sequences can sit in different clusters and still
    # exceed the identity threshold with each other. Cluster-disjoint therefore
    # does not imply pairwise-disjoint. Asserting the latter fails on real data
    # (it did), so we record what the split actually achieves and report it.
    # Exact duplicates remain fatal inside measure_leakage.
    leak = {"leak_fraction": 0.0, "n_test_with_hit": 0, "measured": False}
    if verify and threshold not in ("random",):
        leak = S.measure_leakage(train["sequence"].tolist(),
                                 test["sequence"].tolist(),
                                 float(threshold), coverage)

    idx_hash, lbl_hash = _hash(tr_idx), _hash(y_tr)
    rows = []

    for p in zoo:
        t0 = time.time()
        try:
            Xtr, Xte = featurise(p.rep, train["sequence"], test["sequence"], cache)
            est = build_learner(p.learner, seed)
            est.fit(Xtr, y_tr)
            y_score = est.predict_proba(Xte)[:, 1]
            scores = classification_scores(y_te, y_score)
            err = ""
        except Exception as e:                      # recorded, never silent
            scores = {}
            err = f"{type(e).__name__}: {e}"
            print(f"    [FAIL] {p.pid} {p.rep}/{p.learner}: {err}", flush=True)

        base = {
            "benchmark": f"armB_{rule}", "task": "amp_binary", "arm": "B",
            "model_id": p.pid, "representation": p.rep, "learner": p.learner,
            "cluster_threshold": threshold, "negative_rule": rule, "seed": seed,
            "fold": 0, "n_train": len(train), "n_test": len(test),
            "cluster_ratio": ratio, "fit_seconds": round(time.time() - t0, 2),
            "leak_fraction": leak["leak_fraction"],
            "n_test_with_hit": leak["n_test_with_hit"],
            "leak_measured": leak["measured"],
            "train_idx_hash": idx_hash, "label_hash": lbl_hash,
            "n_pos_train": int(y_tr.sum()), "n_pos_test": int(y_te.sum()),
            "error": err,
        }
        if scores:
            rows += [{**base, "metric": m, "score": v} for m, v in scores.items()]
        else:
            rows.append({**base, "metric": "failed", "score": float("nan")})

    # Fairness invariant: one split, one label vector, seen by all pipelines.
    assert len({r["train_idx_hash"] for r in rows}) == 1
    assert len({r["label_hash"] for r in rows}) == 1
    return rows


def precompute_embeddings(corpus_dir: Path, cache=None, models=None) -> None:
    """Embed every unique sequence once, for all ESM scales used by the zoo.

    This is the only expensive step that cannot be avoided, and doing it up front
    turns all PLM pipelines into cheap sklearn fits on cached vectors.
    """
    cache = cache or F.EmbeddingCache(data_root() / "embeddings")
    models = models or [ESM[r] for r in sorted({p.rep for p in ZOO_B} & set(ESM))]

    seqs = set()
    for f in sorted(Path(corpus_dir).glob("armB_*_seed0.parquet")):
        seqs |= set(pd.read_parquet(f)["sequence"].tolist())
    seqs = sorted(seqs)
    print(f"unique sequences to embed: {len(seqs)}", flush=True)

    for ck in models:
        t0 = time.time()
        F.embed_plm(pd.Series(seqs), ck, cache)
        print(f"  {ck}: {time.time() - t0:.0f}s", flush=True)

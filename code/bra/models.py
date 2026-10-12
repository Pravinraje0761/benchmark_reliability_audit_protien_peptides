"""The frozen model zoo: 12 pipelines, hyperparameters fixed before any
benchmark is downloaded. See Projects/BRA/02 Model zoo.md.

A pipeline is representation x learner. That factorisation is deliberate: it is
what makes H5 (representation ranks are less stable than learner ranks) testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    representation: str
    learner: str
    note: str


ZOO: list[ModelSpec] = [
    ModelSpec("M01", "aac", "logreg", "Floor. Anything that cannot beat it is noise."),
    ModelSpec("M02", "aac_dpc", "rf", "Workhorse of the 2015-2020 AMP literature."),
    ModelSpec("M03", "physchem", "xgboost", "Does hand-crafted biophysics still compete?"),
    ModelSpec("M04", "kmer_tfidf", "svm_rbf", "Text-style baseline; threshold-sensitive by construction."),
    ModelSpec("M05", "onehot", "cnn", "Canonical deep baseline (AMPScanner-class)."),
    ModelSpec("M06", "onehot", "bilstm", "Second canonical deep baseline."),
    ModelSpec("M07", "esm2_t6_8M", "logreg", "Smallest PLM -- is scale the thing?"),
    ModelSpec("M08", "esm2_t12_35M", "mlp", "Mid PLM."),
    ModelSpec("M09", "esm2_t33_650M", "logreg", "Large PLM, linear probe."),
    ModelSpec("M10", "esm2_t33_650M", "svm_rbf", "Same features as M09, different learner -> isolates the learner axis."),
    ModelSpec("M11", "prott5_xl", "mlp", "Different PLM family -- guards against ESM-specific conclusions."),
    ModelSpec("M12", "length_charge", "logreg", "DELIBERATE NULL. Mid-pack rank = the benchmark has a shortcut."),
    # ----------------------------------------------------------------------- #
    # M13-M20 added 2026-10-05 after the design simulation (E00). The zoo grew
    # from 12 to 20 because minimum detectable disagreement improves from
    # tau-b 0.575 to 0.646, and adding a model is far cheaper than adding a
    # benchmark (SD falls 46% going 8->20 models, only 21% going 3->6 benchmarks).
    # These 8 deliberately complete the representation x learner grid so H5 can
    # be answered by marginalising cleanly over either axis. All are cheap:
    # logreg/RF fits on features that are already cached.
    # ----------------------------------------------------------------------- #
    ModelSpec("M13", "aac_dpc", "logreg", "M02 features, linear learner."),
    ModelSpec("M14", "physchem", "rf", "M03 features, non-boosted trees."),
    ModelSpec("M15", "physchem", "logreg", "Biophysics under a linear learner."),
    ModelSpec("M16", "esm2_t6_8M", "rf", "Small PLM + trees."),
    ModelSpec("M17", "esm2_t12_35M", "logreg", "Mid PLM, linear probe."),
    ModelSpec("M18", "esm2_t33_650M", "rf", "Large PLM + trees; completes the 650M row."),
    ModelSpec("M19", "prott5_xl", "logreg", "ProtT5 linear probe, pairs with M09."),
    ModelSpec("M20", "kmer_tfidf", "logreg", "Linear counterpart to M04."),
]

PLM_CHECKPOINTS = {
    "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D",
    "prott5_xl": "Rostlab/prot_t5_xl_uniref50",
}

Task = Literal["classification", "regression"]


def build_learner(name: str, task: Task, seed: int):
    """Fixed hyperparameters. No per-benchmark tuning -- that would confound
    'this benchmark prefers this model' with 'I searched harder here'."""
    clf = task == "classification"

    if name == "logreg":
        est = (LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
               if clf else Ridge(alpha=1.0))
    elif name == "rf":
        est = (RandomForestClassifier if clf else RandomForestRegressor)(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=seed,
            **({"class_weight": "balanced"} if clf else {}),
        )
    elif name == "svm_rbf":
        est = (SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced",
                   random_state=seed)
               if clf else SVR(C=1.0, gamma="scale"))
    elif name == "mlp":
        est = (MLPClassifier if clf else MLPRegressor)(
            hidden_layer_sizes=(256,), max_iter=500, early_stopping=True,
            random_state=seed,
        )
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as e:
            raise ImportError("pip install xgboost, or drop M03 as a logged deviation") from e
        est = (XGBClassifier if clf else XGBRegressor)(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=seed,
            **({"eval_metric": "logloss"} if clf else {}),
        )
    elif name in {"cnn", "bilstm"}:
        from .nn import build_sequence_model
        return build_sequence_model(name, task, seed)
    else:
        raise ValueError(f"unknown learner {name!r}")

    # Scaling matters for logreg/svm/mlp and is harmless for trees, so applying it
    # uniformly keeps the pipelines comparable rather than quietly differentiated.
    return make_pipeline(StandardScaler(with_mean=True), est)


def max_len_from_train(train_lengths) -> int:
    """Data-derived parameters come from the TRAIN fold only, by a deterministic rule."""
    import numpy as np
    return int(np.percentile(list(train_lengths), 99))

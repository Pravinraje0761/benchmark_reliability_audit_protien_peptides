"""
Depth x activation ablation for the MLP pipelines of the benchmark reliability audit.

WHAT THIS IS FOR
----------------
The paper's zoo contains MLP learners fixed at `hidden_layer_sizes=(256,)` with
scikit-learn's default ReLU activation, and the design deliberately froze every
hyperparameter before the corpus was built so that protocol preference could not
be confounded with search effort. This script does NOT change that. It runs a
separate, clearly-labelled supplementary ablation that answers three questions
the main grid was never designed to answer:

  (a) does the MLP's single hidden layer cost it anything, at depths 3-7;
  (b) does the activation function change the result;
  (c) do either of those move the confusion matrix by more than reseeding does.

Report it as a supplementary sensitivity analysis. Folding it into the main
results would contradict the paper's own central methodological claim.

PLACEMENT
---------
Drop this file in `code/scripts/` of the benchmark_reliability_audit repo.
It imports the repo's own corpus construction, splitters, and featurisers, so
the ablation sees exactly the data the main grid saw.

USAGE
-----
    # Arm A (native benchmark splits, one task) -- cheapest, start here
    python scripts/run_depth_activation_ablation.py --arm A --task AMP \
        --rep esm2_35M --outdir results/ablation_depth/armA_AMP

    # Arm A across several tasks
    python scripts/run_depth_activation_ablation.py --arm A \
        --task AMP --task AB --task TOX --rep esm2_35M

    # Arm B (one protocol cell of the factorial)
    python scripts/run_depth_activation_ablation.py --arm B \
        --threshold 0.7 --neg-rule length_matched --rep esm2_35M

    # capacity-controlled sensitivity check
    python scripts/run_depth_activation_ablation.py --arm A --task AMP \
        --width-policy pyramid --outdir results/ablation_depth/armA_AMP_pyramid

Set BRA_DATA to your data root first, exactly as for the main pipeline, and
build the embedding cache before running (`run_arm_a.py --stage embed`) or the
first cell will pay the full embedding cost.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))

from ablation_common import (  # noqa: E402
    DEPTHS, SEEDS, RunRecord, count_params, hidden_sizes, key,
    metrics_from_confusion, write_outputs,
)

# scikit-learn's MLP offers these four. 'identity' is not padding -- it is the
# control that matters: a stack of identity-activated layers collapses to a
# single linear map, so its row tells you how much of the model's performance
# needed non-linearity at all. GELU is unavailable here; if the paper needs it,
# the fundus harness runs the torch equivalent.
ACTIVATIONS = ["relu", "tanh", "logistic", "identity"]


def build_mlp(hidden: tuple, activation: str, seed: int):
    """One MLP, scaled exactly as the repo scales its learners.

    Everything except hidden_layer_sizes and activation is held at the values
    build_learner() uses in models.py, so the only thing moving is the factor
    under study.
    """
    return make_pipeline(
        StandardScaler(with_mean=True),
        MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=activation,
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        ),
    )


def load_arm_a(task: str, rep: str):
    from bra import features as F
    from bra.arma import load_task
    from bra.armb import ESM, data_root, featurise

    df = load_task(task)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    cache = F.EmbeddingCache(data_root() / "embeddings")
    Xtr, Xte = featurise(rep, tr.sequence, te.sequence, cache)
    return Xtr, Xte, tr.label.to_numpy(), te.label.to_numpy(), f"armA:{task}"


def load_arm_b(threshold: str, neg_rule: str, rep: str, seed: int):
    """One Arm B cell. The split depends on the seed, as in the main grid."""
    from bra import features as F
    from bra import splits as S
    from bra.armb import ESM, Cell, build_corpus, data_root, featurise

    # build_corpus / Cell names follow armb.py; adjust the two lines below if
    # your local signature differs -- everything downstream is unchanged.
    df = build_corpus(neg_rule=neg_rule, seed=0)
    clusters = S.cluster_mmseqs(df["sequence"].tolist(), float(threshold), 0.8)
    folds = S.assign_folds_by_cluster(clusters, df["label"].to_numpy(), 5, seed)
    te_mask = folds == 0
    tr, te = df[~te_mask], df[te_mask]
    cache = F.EmbeddingCache(data_root() / "embeddings")
    Xtr, Xte = featurise(rep, tr.sequence, te.sequence, cache)
    return (Xtr, Xte, tr.label.to_numpy(), te.label.to_numpy(),
            f"armB:{threshold}:{neg_rule}")


def main() -> int:
    import warnings

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--task", action="append", default=None,
                    help="Arm A task id, repeatable (AMP, AB, TOX, ...)")
    ap.add_argument("--threshold", default="0.7", help="Arm B clustering threshold")
    ap.add_argument("--neg-rule", default="length_matched", help="Arm B negative-set rule")
    ap.add_argument("--rep", default="esm2_35M",
                    help="representation; must match a key the repo's featurise() knows")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--width-policy", choices=["constant", "pyramid"], default="constant")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--depths", type=int, nargs="+", default=DEPTHS)
    ap.add_argument("--include-baseline", action="store_true", default=True,
                    help="also run depth 1 (the published configuration) as the anchor")
    ap.add_argument("--outdir", default="results/ablation_depth")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    depths = list(args.depths)
    if args.include_baseline and 1 not in depths:
        depths = [1] + depths          # the published (256,) configuration

    tasks = args.task or ["AMP"]
    records: list[RunRecord] = []
    matrices: dict = {}

    for task in (tasks if args.arm == "A" else ["_"]):
        for seed in args.seeds:
            if args.arm == "A":
                Xtr, Xte, ytr, yte, label = load_arm_a(task, args.rep)
            else:
                Xtr, Xte, ytr, yte, label = load_arm_b(
                    args.threshold, args.neg_rule, args.rep, seed
                )

            for depth in depths:
                hidden = hidden_sizes(depth, args.width, args.width_policy)
                for act in ACTIVATIONS:
                    t0 = time.time()
                    est = build_mlp(hidden, act, seed)
                    converged, note = True, ""
                    try:
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always", ConvergenceWarning)
                            est.fit(Xtr, ytr)
                            converged = not any(
                                issubclass(x.category, ConvergenceWarning) for x in w
                            )
                        score = est.predict_proba(Xte)[:, 1]
                        pred = (score >= 0.5).astype(int)
                    except Exception as exc:          # a failure is a result, not a gap
                        note = f"{type(exc).__name__}: {exc}"[:200]
                        records.append(RunRecord(
                            model=f"mlp:{args.rep}:{label}", depth=depth, activation=act,
                            seed=seed, width_policy=args.width_policy,
                            hidden_sizes=str(hidden),
                            n_params=count_params(hidden, Xtr.shape[1], 2),
                            n_train=len(ytr), n_test=len(yte),
                            fit_seconds=time.time() - t0, converged=False, note=note,
                        ))
                        print(f"  FAIL d={depth} {act} seed={seed}: {note}")
                        continue

                    cm = confusion_matrix(yte, pred, labels=[0, 1])
                    m = metrics_from_confusion(cm)
                    try:
                        auroc = float(roc_auc_score(yte, score))
                    except Exception:
                        auroc = float("nan")

                    model_id = f"mlp:{args.rep}:{label}"
                    matrices[key(model_id, depth, act, seed)] = cm.tolist()
                    records.append(RunRecord(
                        model=model_id, depth=depth, activation=act, seed=seed,
                        width_policy=args.width_policy, hidden_sizes=str(hidden),
                        n_params=count_params(hidden, Xtr.shape[1], 2),
                        n_train=len(ytr), n_test=len(yte),
                        fit_seconds=time.time() - t0, converged=converged,
                        auroc=auroc, note=note, **m,
                    ))
                    print(f"  {label} d={depth:<2} {act:<9} seed={seed} "
                          f"MCC={m['mcc']:.4f} bacc={m['balanced_accuracy']:.4f} "
                          f"{'' if converged else '(not converged)'}", flush=True)

    write_outputs(
        Path(args.outdir), records, matrices, class_names=["negative", "positive"],
        context={
            "paper": "Benchmark reliability audit (peptide property prediction)",
            "arm": args.arm, "tasks": tasks, "representation": args.rep,
            "threshold": args.threshold if args.arm == "B" else None,
            "neg_rule": args.neg_rule if args.arm == "B" else None,
            "baseline_config": "hidden_layer_sizes=(256,), activation=relu "
                               "-- the published configuration, run here as depth 1",
            "caveat": "Supplementary sensitivity analysis. The main grid froze "
                      "all hyperparameters before corpus construction; this sweep "
                      "does not alter that and must not be merged into it.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

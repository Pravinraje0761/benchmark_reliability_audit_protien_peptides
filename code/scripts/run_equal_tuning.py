"""Equal tuning budget for every pipeline, to test whether fixed hyperparameters
favoured some pipeline families.

The main grid fixes hyperparameters across every cell. That isolates the protocol
effect cleanly -- protocol preference cannot be confounded with search effort --
but it leaves a fair objection open: a fixed setting may suit a random forest
better than an MLP, so the ranking might partly reflect whose defaults were
luckier rather than whose method is better.

This gives every pipeline the SAME budget: six candidate configurations for each
of the five learners, selected on a validation split taken from inside the
training partition only. Two properties matter and are enforced here:

  * the inner validation split is **cluster-disjoint**, using the same clustering
    that produced the outer split, so hyperparameters are not chosen on
    sequences homologous to those they are validated against;
  * the **test partition is never touched** during selection -- it is scored once,
    after refitting the chosen configuration on the full training partition.

Scope: the 5 negative rules at threshold 0.7, seed 0, the 16 pipelines the Arm B
grid used. 16 x (6 + 1) x 5 = 560 fits. One threshold and one seed, because the
question is whether tuning reorders the pipelines, not how tuning interacts with
the protocol.

    python scripts/run_equal_tuning.py

Writes results/tables/equal_tuning.csv and equal_tuning_summary.csv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra import features as F                                        # noqa: E402
from bra.analysis.ranks import mean_pairwise_tau, tau_b              # noqa: E402
from bra.armb import (N_JOBS, NEG_RULES, ZOO_B, data_root,           # noqa: E402
                      featurise, split_cell)
from bra.metrics import classification_scores                        # noqa: E402
from bra.splits import cluster_mmseqs                                # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

PIPES = [p for p in ZOO_B if p.pid <= "P16"]
THRESHOLD = "0.7"
SEED = 0
INNER_TEST_FRAC = 0.25

# Six configurations per learner. Equal budget is the point: no family gets a
# larger search than another, so a ranking change cannot be bought with compute.
GRIDS = {
    "logreg":  [{"C": c} for c in (0.003, 0.03, 0.3, 1.0, 3.0, 30.0)],
    "rf":      [{"n_estimators": n, "min_samples_leaf": m}
                for n, m in [(200, 1), (200, 4), (400, 1),
                             (400, 4), (800, 2), (800, 8)]],
    "svm_rbf": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
    "mlp":     [{"hidden_layer_sizes": h, "alpha": a}
                for h, a in itertools.product([(64,), (256,), (512,)], [1e-4, 1e-2])],
    "xgboost": [{"max_depth": d, "learning_rate": lr}
                for d, lr in itertools.product([3, 6, 9], [0.05, 0.2])],
}


def make_est(learner: str, params: dict, seed: int):
    """Same construction as bra.armb.build_learner, with the grid applied."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if learner == "logreg":
        est = LogisticRegression(max_iter=3000, class_weight="balanced", **params)
    elif learner == "rf":
        est = RandomForestClassifier(n_jobs=N_JOBS, random_state=seed,
                                     class_weight="balanced", **params)
    elif learner == "svm_rbf":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.kernel_approximation import Nystroem
        from sklearn.svm import LinearSVC
        return make_pipeline(
            StandardScaler(),
            Nystroem(n_components=512, random_state=seed),
            CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", random_state=seed,
                          max_iter=5000, **params),
                method="sigmoid", cv=3, ensemble=False),
        )
    elif learner == "mlp":
        est = MLPClassifier(max_iter=400, early_stopping=True,
                            random_state=seed, **params)
    elif learner == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(n_estimators=400, subsample=0.8, colsample_bytree=0.8,
                            n_jobs=N_JOBS, tree_method="hist", random_state=seed,
                            eval_metric="logloss", **params)
    else:
        raise ValueError(learner)
    return make_pipeline(StandardScaler(), est)


def inner_split(clusters_train: np.ndarray, seed: int) -> np.ndarray:
    """Cluster-disjoint holdout inside the training partition.

    Whole clusters go to the inner validation side, taken in random order rather
    than by size: the stress test showed size-ordered assignment is what wrecks
    class balance, and there is no reason to repeat that here.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters_train)
    order = rng.permutation(uniq)
    target = INNER_TEST_FRAC * len(clusters_train)
    chosen, total = [], 0
    for c in order:
        if total >= target:
            break
        chosen.append(c)
        total += int((clusters_train == c).sum())
    return np.isin(clusters_train, chosen)


def main() -> None:
    cache = F.EmbeddingCache(data_root() / "embeddings")
    corpus = data_root() / "corpus"
    rows = []

    for rule in NEG_RULES:
        f = corpus / f"armB_{rule}_seed{SEED}.parquet"
        if not f.exists():
            print(f"  missing {f.name}; skipping")
            continue
        df = pd.read_parquet(f)
        tr_idx, te_idx, _ = split_cell(df, THRESHOLD, SEED)
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()

        cl_all = cluster_mmseqs(df["sequence"].astype(str).tolist(),
                                min_seq_id=float(THRESHOLD), coverage=0.8)
        is_val = inner_split(cl_all[tr_idx], SEED)
        print(f"\n{rule}: train={len(tr_idx)} inner-val={int(is_val.sum())} "
              f"test={len(te_idx)} (test never seen during selection)", flush=True)

        for p in PIPES:
            try:
                Xtr, Xte = featurise(p.rep, train["sequence"], test["sequence"], cache)
            except Exception as e:
                print(f"    [FAIL feat] {p.pid}: {type(e).__name__}: {e}", flush=True)
                continue

            Xi_tr, Xi_val = Xtr[~is_val], Xtr[is_val]
            yi_tr, yi_val = y_tr[~is_val], y_tr[is_val]

            best, best_score = None, -np.inf
            for params in GRIDS[p.learner]:
                try:
                    est = make_est(p.learner, params, SEED)
                    est.fit(Xi_tr, yi_tr)
                    s = classification_scores(yi_val, est.predict_proba(Xi_val)[:, 1])
                    v = s.get("mcc", float("nan"))
                except Exception as e:
                    print(f"      [cfg fail] {p.pid} {params}: {type(e).__name__}",
                          flush=True)
                    continue
                if np.isfinite(v) and v > best_score:
                    best, best_score = params, v
            if best is None:
                print(f"    [FAIL tune] {p.pid}: no configuration fitted", flush=True)
                continue

            est = make_est(p.learner, best, SEED)
            est.fit(Xtr, y_tr)
            sc = classification_scores(y_te, est.predict_proba(Xte)[:, 1])
            rows.append({
                "negative_rule": rule, "model_id": p.pid,
                "representation": p.rep, "learner": p.learner,
                "chosen": str(best), "inner_mcc": round(float(best_score), 4),
                "tuned_mcc": sc.get("mcc"), "tuned_auroc": sc.get("auroc"),
            })
            print(f"    {p.pid} {p.learner:8s} {str(best):36s} "
                  f"inner {best_score:.3f} -> test {sc.get('mcc', float('nan')):.3f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / "equal_tuning.csv", index=False)

    res = pd.DataFrame(rows)
    if res.empty:
        print("nothing completed")
        return

    # the fixed-hyperparameter baseline: same rules, same threshold, same seed
    runs = pd.read_parquet(data_root() / "runs" / "all_runs_armB.parquet")
    base = runs[(runs.metric == "mcc") & (runs.cluster_threshold == THRESHOLD)
                & (runs.seed == SEED)
                & (runs.model_id.isin([p.pid for p in PIPES]))]
    base = base.groupby(["negative_rule", "model_id"], as_index=False)["score"].mean()
    res = res.merge(base.rename(columns={"score": "fixed_mcc"}),
                    on=["negative_rule", "model_id"], how="left")
    res["delta"] = res["tuned_mcc"] - res["fixed_mcc"]
    res.to_csv(OUT / "equal_tuning.csv", index=False)

    def rankmat(frame, col):
        return (frame.pivot(index="model_id", columns="negative_rule", values=col)
                     .rank(ascending=False, method="average"))

    rm_t, rm_f = rankmat(res, "tuned_mcc"), rankmat(res, "fixed_mcc")
    tau_t, tau_f = mean_pairwise_tau(rm_t), mean_pairwise_tau(rm_f)

    print("\n" + "=" * 78)
    print("Does an equal tuning budget change the ranking conclusions?")
    print("=" * 78)
    print(f"  mean MCC      fixed {res.fixed_mcc.mean():.3f}  tuned {res.tuned_mcc.mean():.3f}"
          f"   (mean gain {res.delta.mean():+.3f}, max {res.delta.max():+.3f})")
    print(f"  mean pairwise tau-b ACROSS RULES: fixed {tau_f:.3f}  tuned {tau_t:.3f}")

    per = [tau_b(rm_f[c], rm_t[c]) for c in rm_f.columns]
    print(f"  within-rule tau-b between fixed and tuned rankings: "
          f"mean {np.mean(per):.3f}  min {np.min(per):.3f}")
    w_f = {c: rm_f[c].idxmin() for c in rm_f.columns}
    w_t = {c: rm_t[c].idxmin() for c in rm_t.columns}
    changed = [c for c in w_f if w_f[c] != w_t[c]]
    print(f"  winner changes under tuning: {len(changed)} of {len(w_f)} rules {changed}")

    print("\n  mean gain by learner family (does tuning favour one?):")
    print(res.groupby("learner")["delta"].agg(["mean", "max", "count"]).round(3).to_string())

    pd.DataFrame([{
        "tau_across_rules_fixed": tau_f, "tau_across_rules_tuned": tau_t,
        "tau_fixed_vs_tuned_mean": float(np.mean(per)),
        "tau_fixed_vs_tuned_min": float(np.min(per)),
        "winner_changes": len(changed), "n_rules": len(w_f),
        "mean_gain": float(res.delta.mean()), "threshold": THRESHOLD, "seed": SEED,
    }]).to_csv(OUT / "equal_tuning_summary.csv", index=False)
    print(f"\nwritten to {OUT / 'equal_tuning.csv'}")


if __name__ == "__main__":
    main()

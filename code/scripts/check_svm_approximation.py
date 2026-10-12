"""Does the Nystroem approximation preserve what the exact RBF SVM would do?

The approximation is the only deviation in this study that changes a model
rather than its scheduling, so deferring its validation to "available on
request" is weaker than simply doing it.

Exact SVC with probability calibration is intractable at full corpus size (>9
minutes for one cell, 16+ hours over the grid). It is perfectly tractable on a
subsample, so we compare the two at a size where both run, across all five
negative rules and three seeds. If the approximation tracks the exact kernel
machine in both score and rank at n=2000, the substitution is not doing
anything interesting to the result.

    python scripts/check_svm_approximation.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.armb import NEG_RULES, build_learner, featurise, data_root, split_cell  # noqa: E402
from bra import features as F  # noqa: E402
from bra.metrics import classification_scores  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
N_TRAIN, N_TEST = 2000, 600


def exact_svc(seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(
        StandardScaler(),
        SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced",
            random_state=seed, cache_size=1000),
    )


def main() -> None:
    cache = F.EmbeddingCache(data_root() / "embeddings")
    rows = []
    for rule in NEG_RULES:
        df = pd.read_parquet(data_root() / f"corpus/armB_{rule}_seed0.parquet")
        for seed in range(3):
            tr, te, _ = split_cell(df, "0.7", seed)
            rng = np.random.default_rng(seed)
            tr = rng.choice(tr, size=min(N_TRAIN, len(tr)), replace=False)
            te = rng.choice(te, size=min(N_TEST, len(te)), replace=False)
            train, test = df.iloc[tr], df.iloc[te]
            y_tr, y_te = train.label.to_numpy(), test.label.to_numpy()
            Xtr, Xte = featurise("kmer_tfidf", train.sequence, test.sequence, cache)

            for name, est in [("nystroem", build_learner("svm_rbf", seed)),
                              ("exact_svc", exact_svc(seed))]:
                t0 = time.time()
                est.fit(Xtr, y_tr)
                sc = est.predict_proba(Xte)[:, 1]
                m = classification_scores(y_te, sc)
                rows.append({"rule": rule, "seed": seed, "variant": name,
                             "auroc": m["auroc"], "mcc": m["mcc"],
                             "secs": round(time.time() - t0, 1)})
            print(f"  {rule:22s} seed{seed} done", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "svm_approximation_check.csv", index=False)

    piv = t.pivot_table(index=["rule", "seed"], columns="variant",
                        values=["auroc", "mcc", "secs"])
    print("\n" + piv.to_string(float_format=lambda v: f"{v:.3f}"))

    for metric in ["auroc", "mcc"]:
        a = t[t.variant == "nystroem"].sort_values(["rule", "seed"])[metric].to_numpy()
        b = t[t.variant == "exact_svc"].sort_values(["rule", "seed"])[metric].to_numpy()
        r = stats.pearsonr(a, b)
        tau = stats.kendalltau(a, b, variant="b")
        print(f"\n{metric}: mean nystroem {a.mean():.3f} vs exact {b.mean():.3f}  "
              f"(mean abs diff {np.abs(a-b).mean():.3f}, max {np.abs(a-b).max():.3f})")
        print(f"  across the {len(a)} matched cells: pearson r={r.statistic:.3f}, "
              f"kendall tau-b={tau.statistic:.3f}")

    sa = t[t.variant == "nystroem"].secs.mean()
    sb = t[t.variant == "exact_svc"].secs.mean()
    print(f"\nmean fit time: nystroem {sa:.1f}s, exact {sb:.1f}s "
          f"({sb/max(sa,1e-9):.1f}x) at n_train={N_TRAIN}")
    print(f"\nwritten to {OUT/'svm_approximation_check.csv'}")


if __name__ == "__main__":
    main()

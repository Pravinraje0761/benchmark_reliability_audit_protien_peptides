"""Cluster-aware confidence intervals for the SCORES, not just the ranks.

Why this exists. Every interval reported in the manuscript so far resamples
either pipelines or evaluation levels, which is right for a rank-agreement
statistic. The underlying scores carry their own uncertainty, and the usual
sequence-level bootstrap is too optimistic for peptides: test sequences are
related to each other, so they are not independent draws. Resampling whole
similarity clusters respects that dependence, and the honest interval is the
wider one.

This cannot be computed from the stored run records. `RunRecord` keeps aggregate
metrics only -- no per-sequence predictions and no test cluster ids -- so the
fits have to be repeated with the predictions retained. We therefore scope it
deliberately rather than re-running the whole grid:

  Arm B  the 5 negative rules at threshold 0.7, seed 0, the 16 pipelines the
         Arm B grid actually used (P01-P16; P17-P24 were added after that grid
         ran, and mixing them would not match the reported numbers).
  Arm A  the 5 smallest tasks, where a sequence-level interval is most
         optimistic and where the referee asked specifically for uncertainty,
         with all 24 pipelines.

Bootstrap units. For Arm B the unit is the corpus cluster at the cell's own
identity threshold, which is the same clustering that defined the split, so the
resampling unit matches the partitioning unit. Arm A ships a native partition
with no published clustering, so its test set is clustered here at 40% identity
-- the coarse end, chosen because coarser clusters give wider and therefore more
conservative intervals, and stated rather than tuned.

    python scripts/run_cluster_intervals.py

Writes results/tables/cluster_intervals.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bra import features as F                                      # noqa: E402
from bra.arma import load_task                                     # noqa: E402
from bra.armb import (NEG_RULES, ZOO_B, build_learner, data_root,   # noqa: E402
                      featurise, split_cell)
from bra.metrics import classification_scores                      # noqa: E402
from bra.splits import cluster_mmseqs                              # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ARMB_PIPES = [p for p in ZOO_B if p.pid <= "P16"]   # the zoo the Arm B grid used
ARMB_THRESHOLD = "0.7"
SEED = 0
SMALL_TASKS = ["TTCA", "DPPIV", "ACP", "AF", "ACE"]
ARMA_CLUSTER_ID = 0.4
METRICS = ["mcc", "auroc"]
N_BOOT = 2000


def _ci(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size < 50:
        return float("nan"), float("nan")
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bootstrap_scores(y_true, y_score, clusters, n_boot=N_BOOT, seed=0):
    """Sequence-level and cluster-level percentile intervals for each metric.

    The cluster bootstrap resamples whole clusters with replacement and
    concatenates their members, so a cluster contributes all of its sequences or
    none. That is what makes it respect within-cluster dependence; it also means
    the resampled test set varies in size, which is expected and is why the
    interval is wider rather than merely shifted.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}

    seq_draws = {m: [] for m in METRICS}
    clu_draws = {m: [] for m in METRICS}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = classification_scores(y_true[idx], y_score[idx])
        for m in METRICS:
            seq_draws[m].append(s.get(m, float("nan")))

        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        s = classification_scores(y_true[idx], y_score[idx])
        for m in METRICS:
            clu_draws[m].append(s.get(m, float("nan")))

    point = classification_scores(y_true, y_score)
    out = {}
    for m in METRICS:
        slo, shi = _ci(seq_draws[m])
        clo, chi = _ci(clu_draws[m])
        out[m] = {
            "point": point.get(m, float("nan")),
            "seq_lo": slo, "seq_hi": shi, "seq_width": shi - slo,
            "clu_lo": clo, "clu_hi": chi, "clu_width": chi - clo,
            "width_ratio": (chi - clo) / (shi - slo) if (shi - slo) > 0 else float("nan"),
        }
    return out


def _fit_predict(p, train, test, y_tr, cache):
    Xtr, Xte = featurise(p.rep, train["sequence"], test["sequence"], cache)
    est = build_learner(p.learner, SEED)
    est.fit(Xtr, y_tr)
    return est.predict_proba(Xte)[:, 1]


def main() -> None:
    cache = F.EmbeddingCache(data_root() / "embeddings")
    rows = []

    # ---------------- Arm B ----------------
    corpus = data_root() / "corpus"
    for rule in NEG_RULES:
        f = corpus / f"armB_{rule}_seed{SEED}.parquet"
        if not f.exists():
            print(f"  missing {f.name}; skipping")
            continue
        df = pd.read_parquet(f)
        tr_idx, te_idx, _ = split_cell(df, ARMB_THRESHOLD, SEED)
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()

        # the clustering that defined the split, restricted to the test partition
        cl_all = cluster_mmseqs(df["sequence"].astype(str).tolist(),
                                min_seq_id=float(ARMB_THRESHOLD), coverage=0.8)
        cl_te = cl_all[te_idx]
        print(f"\n{rule}: n_test={len(te_idx)} test clusters={len(np.unique(cl_te))}",
              flush=True)

        for p in ARMB_PIPES:
            try:
                y_score = _fit_predict(p, train, test, y_tr, cache)
            except Exception as e:
                print(f"    [FAIL] {p.pid}: {type(e).__name__}: {e}", flush=True)
                continue
            res = bootstrap_scores(y_te, y_score, cl_te, seed=SEED)
            for m, d in res.items():
                rows.append({"arm": "B", "unit": f"cluster@{ARMB_THRESHOLD}",
                             "level": rule, "model_id": p.pid, "metric": m,
                             "n_test": len(te_idx),
                             "n_clusters": int(len(np.unique(cl_te))), **d})
            d = res["mcc"]
            print(f"    {p.pid} mcc {d['point']:.3f}  seq [{d['seq_lo']:.3f},{d['seq_hi']:.3f}]"
                  f"  clu [{d['clu_lo']:.3f},{d['clu_hi']:.3f}]  x{d['width_ratio']:.2f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / "cluster_intervals.csv", index=False)

    # ---------------- Arm A, small tasks ----------------
    for task in SMALL_TASKS:
        df = load_task(task)
        train, test = df[df.split == "train"], df[df.split == "test"]
        y_tr, y_te = train["label"].to_numpy(), test["label"].to_numpy()
        cl_te = cluster_mmseqs(test["sequence"].astype(str).tolist(),
                               min_seq_id=ARMA_CLUSTER_ID, coverage=0.8)
        print(f"\n{task}: n_test={len(test)} test clusters={len(np.unique(cl_te))}",
              flush=True)

        for p in ZOO_B:
            try:
                y_score = _fit_predict(p, train, test, y_tr, cache)
            except Exception as e:
                print(f"    [FAIL] {p.pid}: {type(e).__name__}: {e}", flush=True)
                continue
            res = bootstrap_scores(y_te, y_score, cl_te, seed=SEED)
            for m, d in res.items():
                rows.append({"arm": "A", "unit": f"cluster@{ARMA_CLUSTER_ID}",
                             "level": task, "model_id": p.pid, "metric": m,
                             "n_test": len(test),
                             "n_clusters": int(len(np.unique(cl_te))), **d})
            d = res["mcc"]
            print(f"    {p.pid} mcc {d['point']:.3f}  seq [{d['seq_lo']:.3f},{d['seq_hi']:.3f}]"
                  f"  clu [{d['clu_lo']:.3f},{d['clu_hi']:.3f}]  x{d['width_ratio']:.2f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(OUT / "cluster_intervals.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "cluster_intervals.csv", index=False)

    print("\n" + "=" * 78)
    print("Interval width: cluster bootstrap vs sequence bootstrap")
    print("=" * 78)
    for m in METRICS:
        g = res[res.metric == m]
        if g.empty:
            continue
        summ = g.groupby("arm")[["seq_width", "clu_width", "width_ratio"]].mean().round(3)
        print(f"\n{m}:")
        print(summ.to_string())
    print("\nA ratio above 1 means the sequence-level interval was too narrow by")
    print("that factor. Reported whichever way it comes out.")
    print(f"\nwritten to {OUT / 'cluster_intervals.csv'}")


if __name__ == "__main__":
    main()

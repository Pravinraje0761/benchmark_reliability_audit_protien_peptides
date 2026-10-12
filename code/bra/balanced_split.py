"""Class-balance-preserving cluster-disjoint splitting.

The problem this solves, measured on our own grid: assigning whole similarity
clusters to the held-out partition largest-first drives the test-set positive
fraction to **0.995** at a 40% identity threshold, on a corpus that is 1:1 by
construction. Antimicrobial peptides are homologous to one another and form
large clusters; negatives derived from them (fragments, shuffles) do not cluster
with them and remain singletons, so the big positive clusters land in test. For
whole-protein negatives the effect inverts (0.29), because those negatives form
real families of their own.

Threshold-dependent metrics on a 99% positive test set do not mean what they
appear to, so the artefact is worth removing rather than only reporting.

Approach
--------
Clusters stay atomic - that is the point of cluster-disjoint splitting and is
never violated here. What changes is the *order* in which clusters are taken.
Rather than largest-first, we track the remaining positive and negative quota
for the test partition and repeatedly take the cluster that best serves whichever
quota is further from being met. Ties break on seed-dependent jitter, so the
partition still varies across seeds and the seed variance component stays
meaningful.

This cannot always succeed. If every positive sits in one giant cluster, no
cluster-disjoint split can be balanced. The function therefore *reports* the
balance achieved rather than promising one.
"""
from __future__ import annotations

import numpy as np


def assign_test_clusters_balanced(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.2,
    seed: int = 0,
    target_pos_frac: float | None = None,
) -> np.ndarray:
    """Boolean mask selecting the test partition. Whole clusters only."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    n = len(labels)
    if target_pos_frac is None:
        target_pos_frac = float(labels.mean())

    # per-cluster positive/negative counts, vectorised
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    pos = np.bincount(inv, weights=labels, minlength=len(uniq)).astype(float)
    size = np.bincount(inv, minlength=len(uniq)).astype(float)
    neg = size - pos

    target_n = test_fraction * n
    need_pos = target_pos_frac * target_n
    need_neg = (1.0 - target_pos_frac) * target_n

    jitter = rng.random(len(uniq)) * 1e-6
    available = np.ones(len(uniq), dtype=bool)
    chosen: list[int] = []

    while (need_pos > 0 or need_neg > 0) and available.any():
        want_pos = need_pos >= need_neg
        counts = pos if want_pos else neg
        quota = need_pos if want_pos else need_neg
        # prefer clusters that fit inside the remaining quota
        fits = available & (counts <= max(quota, 1.0)) & (counts > 0)
        pool = fits if fits.any() else (available & (size > 0))
        if not pool.any():
            break
        idx = int(np.argmax(np.where(pool, counts + jitter, -np.inf)))
        chosen.append(idx)
        available[idx] = False
        need_pos -= pos[idx]
        need_neg -= neg[idx]

    test_clusters = np.zeros(len(uniq), dtype=bool)
    test_clusters[chosen] = True
    return test_clusters[inv]


def balance_report(labels: np.ndarray, is_test: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=float)
    tr, te = labels[~is_test], labels[is_test]
    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "test_fraction": float(len(te) / max(len(labels), 1)),
        "train_pos_frac": float(tr.mean()) if len(tr) else float("nan"),
        "test_pos_frac": float(te.mean()) if len(te) else float("nan"),
        "corpus_pos_frac": float(labels.mean()),
    }

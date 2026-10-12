"""Tests for the balance-preserving cluster-disjoint splitter.

The non-negotiable property is cluster atomicity: if the splitter ever breaks a
cluster across the partition it has silently reintroduced the leakage that
cluster-disjoint splitting exists to prevent, and a better class balance would
be worthless.
"""
import numpy as np
import pytest

from bra.balanced_split import assign_test_clusters_balanced, balance_report


def _pathological(n_pos_clusters=3, pos_per=400, n_neg=1200, seed=0):
    """The real failure mode: positives in a few huge clusters, negatives as
    singletons. Largest-first assignment sends an all-positive test set."""
    cids, labels = [], []
    cid = 0
    for _ in range(n_pos_clusters):
        cids += [cid] * pos_per
        labels += [1.0] * pos_per
        cid += 1
    for _ in range(n_neg):
        cids.append(cid); labels.append(0.0); cid += 1
    return np.array(cids), np.array(labels)


def test_clusters_are_never_split():
    cids, labels = _pathological()
    mask = assign_test_clusters_balanced(cids, labels, 0.2, seed=0)
    for c in np.unique(cids):
        sel = mask[cids == c]
        assert sel.all() or (~sel).all(), f"cluster {c} straddles the partition"


def test_balance_is_restored_on_the_pathological_case():
    cids, labels = _pathological()
    mask = assign_test_clusters_balanced(cids, labels, 0.2, seed=0)
    rep = balance_report(labels, mask)
    # the corpus here is 1200 positive / 1200 negative
    assert abs(rep["test_pos_frac"] - rep["corpus_pos_frac"]) < 0.25, rep


def test_beats_largest_first_on_the_same_data():
    """Direct comparison against the size-ordered strategy the paper audits."""
    cids, labels = _pathological()

    # largest-first, as in the original splitter
    uniq, sizes = np.unique(cids, return_counts=True)
    order = np.argsort(-sizes)
    target, filled, chosen = 0.2 * len(labels), 0, set()
    for i in order:
        if filled >= target:
            break
        chosen.add(uniq[i]); filled += sizes[i]
    naive = np.array([c in chosen for c in cids])

    bal = assign_test_clusters_balanced(cids, labels, 0.2, seed=0)
    naive_err = abs(balance_report(labels, naive)["test_pos_frac"] - 0.5)
    bal_err = abs(balance_report(labels, bal)["test_pos_frac"] - 0.5)
    assert bal_err < naive_err, (naive_err, bal_err)


def test_test_fraction_is_approximately_respected():
    cids, labels = _pathological()
    mask = assign_test_clusters_balanced(cids, labels, 0.2, seed=1)
    assert 0.10 <= balance_report(labels, mask)["test_fraction"] <= 0.35


def test_partition_varies_with_seed():
    rng = np.random.default_rng(0)
    cids = rng.integers(0, 200, size=2000)
    labels = (rng.random(2000) < 0.5).astype(float)
    a = assign_test_clusters_balanced(cids, labels, 0.2, seed=0)
    b = assign_test_clusters_balanced(cids, labels, 0.2, seed=1)
    assert not np.array_equal(a, b), "seed must change the partition"


def test_deterministic_given_seed():
    rng = np.random.default_rng(0)
    cids = rng.integers(0, 200, size=2000)
    labels = (rng.random(2000) < 0.5).astype(float)
    a = assign_test_clusters_balanced(cids, labels, 0.2, seed=3)
    b = assign_test_clusters_balanced(cids, labels, 0.2, seed=3)
    assert np.array_equal(a, b)


def test_handles_single_giant_cluster_without_crashing():
    """When balance is impossible it must report, not promise."""
    cids = np.zeros(500, dtype=int)
    labels = np.ones(500)
    mask = assign_test_clusters_balanced(cids, labels, 0.2, seed=0)
    rep = balance_report(labels, mask)
    assert rep["n_train"] + rep["n_test"] == 500
    # one cluster: it is all-or-nothing, and either way nothing is split
    assert mask.all() or (~mask).all()

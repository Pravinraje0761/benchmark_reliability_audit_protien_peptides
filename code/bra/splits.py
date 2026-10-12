"""Splitters. The clustering-threshold factor lives here.

MMseqs2 is the primary clusterer; the alignment-free k-mer splitter is the
sensitivity arm required by the short-peptide clustering bottleneck.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class LeakageError(AssertionError):
    """Raised when a split fails its own post-hoc verification."""


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

def mmseqs_available() -> bool:
    return shutil.which("mmseqs") is not None


def cluster_mmseqs(
    seqs: list[str],
    min_seq_id: float,
    coverage: float = 0.8,
    cov_mode: int = 0,
    sensitivity: float = 7.5,
) -> np.ndarray:
    """Cluster sequences, return an integer cluster id per sequence.

    `coverage` is passed explicitly and recorded in every result row. Reporting an
    identity threshold without coverage under-specifies the protocol -- for short
    peptides coverage changes cluster counts more than identity does.
    `sensitivity` is raised above the default because defaults miss short-sequence hits.
    """
    if not mmseqs_available():
        raise RuntimeError(
            "mmseqs not found. Install via bioconda, or use WSL2/Docker on Windows. "
            "For a quick local check, fall back to cluster_kmer()."
        )
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td / "in.fasta"
        fa.write_text("".join(f">s{i}\n{s}\n" for i, s in enumerate(seqs)))
        subprocess.run(
            ["mmseqs", "easy-cluster", str(fa), str(td / "out"), str(td / "tmp"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage),
             "--cov-mode", str(cov_mode), "-s", str(sensitivity)],
            check=True, capture_output=True,
        )
        rep_of = {}
        for line in (td / "out_cluster.tsv").read_text().splitlines():
            rep, member = line.split("\t")
            rep_of[int(member[1:])] = rep
        reps = {r: k for k, r in enumerate(sorted(set(rep_of.values())))}
        return np.array([reps[rep_of[i]] for i in range(len(seqs))])


def cluster_kmer(seqs: list[str], k: int = 3, threshold: float = 0.5) -> np.ndarray:
    """Alignment-free single-linkage clustering on k-mer Jaccard similarity.

    The sensitivity arm. If BRA conclusions hold under both this and MMseqs2, they
    are robust to the clustering bottleneck; if they do not, that is also a result.
    O(n^2) -- fine up to a few tens of thousands of peptides, which covers all six
    benchmarks on the AMP-binary task.
    """
    sets = [set(s[i:i + k] for i in range(max(len(s) - k + 1, 1))) for s in seqs]
    n = len(seqs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            inter = len(sets[i] & sets[j])
            if inter and inter / len(sets[i] | sets[j]) >= threshold:
                union(i, j)

    roots = {}
    return np.array([roots.setdefault(find(i), len(roots)) for i in range(n)])


def cluster_ratio(cluster_ids: np.ndarray) -> float:
    """n_clusters / n_sequences. Near 1.0 means the homology control is cosmetic."""
    return float(len(np.unique(cluster_ids)) / len(cluster_ids))


# --------------------------------------------------------------------------- #
# Fold assignment
# --------------------------------------------------------------------------- #

def assign_folds_by_cluster(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Greedy bin-packing of whole clusters into folds, largest cluster first.

    Balances fold size while keeping clusters intact. Class ratio is reported, not
    forced: silently rebalancing would make the folds unrepresentative and hide the
    class-imbalance shift that several benchmarks actually exhibit.
    """
    rng = np.random.default_rng(seed)
    uniq, sizes = np.unique(cluster_ids, return_counts=True)
    order = np.argsort(-(sizes + rng.random(len(sizes)) * 0.5))  # jittered, seed-dependent
    loads = np.zeros(n_folds, dtype=int)
    fold_of_cluster = {}
    for idx in order:
        f = int(np.argmin(loads))
        fold_of_cluster[uniq[idx]] = f
        loads[f] += sizes[idx]
    return np.array([fold_of_cluster[c] for c in cluster_ids])


def random_folds(n: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """The deliberately leaky control arm."""
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % n_folds)


def leave_source_out(sources: pd.Series) -> np.ndarray:
    """Hold out an entire originating database. Hardest, closest to deployment."""
    codes = {s: i for i, s in enumerate(sorted(sources.unique()))}
    return sources.map(codes).to_numpy()


# --------------------------------------------------------------------------- #
# Verification -- run on every cell, not once
# --------------------------------------------------------------------------- #

def measure_leakage(
    train_seqs: list[str],
    test_seqs: list[str],
    min_seq_id: float,
    coverage: float = 0.8,
    sensitivity: float = 7.5,
) -> dict:
    """Measure the residual pairwise leakage of a cluster-disjoint split.

    Why this exists, and why it is not merely a check.

    Assigning whole MMseqs2 clusters to the test partition is the standard way
    to build a homology-aware split, and it is what we do. But greedy clustering
    does **not** produce equivalence classes: two sequences can land in
    different clusters while still exceeding the identity threshold with each
    other, because each is merely nearer to its own representative.
    Cluster-disjoint therefore does not imply pairwise-disjoint, and on our
    corpus it demonstrably does not -- re-clustering a cluster-disjoint split
    found clusters spanning the partition.

    So rather than assert a guarantee the method cannot give, we measure what it
    actually delivers: an explicit search of every test sequence against the
    training set, counting those with a hit at or above the threshold. The
    number is recorded per cell and reported, because a residual leakage rate is
    a property of the splitting protocol that benchmarks currently leave unstated.

    Returns a dict with the leaked fraction and counts.
    """
    exact = set(train_seqs) & set(test_seqs)
    out = {"n_test": len(test_seqs), "n_exact_duplicates": len(exact),
           "n_test_with_hit": 0, "leak_fraction": 0.0, "measured": False}
    if exact:
        # An identical sequence on both sides is never acceptable, whatever the
        # clustering did, so this one stays fatal.
        raise LeakageError(
            f"{len(exact)} exact duplicate sequences across the split, "
            f"e.g. {next(iter(exact))[:40]}"
        )
    if min_seq_id >= 1.0 or not mmseqs_available():
        if not mmseqs_available():
            print("[warn] mmseqs missing: residual leakage NOT measured")
        return out

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        q, t = td / "test.fasta", td / "train.fasta"
        q.write_text("".join(f">q{i}\n{s}\n" for i, s in enumerate(test_seqs)))
        t.write_text("".join(f">t{i}\n{s}\n" for i, s in enumerate(train_seqs)))
        res = td / "hits.m8"
        subprocess.run(
            ["mmseqs", "easy-search", str(q), str(t), str(res), str(td / "tmp"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage),
             "--cov-mode", "0", "-s", str(sensitivity),
             "--format-output", "query,target,fident"],
            check=True, capture_output=True,
        )
        hit_queries = set()
        if res.exists():
            for line in res.read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 3 and float(parts[2]) >= min_seq_id:
                    hit_queries.add(parts[0])

    out["n_test_with_hit"] = len(hit_queries)
    out["leak_fraction"] = len(hit_queries) / max(len(test_seqs), 1)
    out["measured"] = True
    return out


def verify_no_leakage(
    train_seqs: list[str],
    test_seqs: list[str],
    min_seq_id: float,
    coverage: float = 0.8,
) -> None:
    """Assert no test sequence matches a training sequence above the threshold.

    Exact duplicates are checked unconditionally -- they are trivial, common, and
    embarrassing. The identity check needs MMseqs2 and is skipped with a loud
    warning if it is missing, because a silent skip would defeat the point.
    """
    overlap = set(train_seqs) & set(test_seqs)
    if overlap:
        raise LeakageError(
            f"{len(overlap)} exact duplicate sequences across the split, "
            f"e.g. {next(iter(overlap))[:40]!r}"
        )
    if min_seq_id >= 1.0:
        return
    if not mmseqs_available():
        print("[warn] mmseqs missing: identity-level leakage check SKIPPED")
        return

    ids = cluster_mmseqs(list(train_seqs) + list(test_seqs), min_seq_id, coverage)
    tr, te = set(ids[:len(train_seqs)]), set(ids[len(train_seqs):])
    shared = tr & te
    if shared:
        raise LeakageError(
            f"{len(shared)} clusters span train and test at identity {min_seq_id} "
            f"(coverage {coverage}). The split is leaky."
        )

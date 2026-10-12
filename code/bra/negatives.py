"""Negative-set construction. The factor Sidorczuk et al. showed dominates.

Five rules spanning the families behind the 11 methods in AMPBenchmark.
Each returns a DataFrame of negatives in the canonical peptide schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
# Kyte-Doolittle is for hydropathy; charge uses the standard ionisable set.
POSITIVE_AA, NEGATIVE_AA = set("KR"), set("DE")


def net_charge(seq: str) -> float:
    """Net charge at neutral pH, counting K/R as +1, D/E as -1, H as +0.1."""
    return sum(1.0 for a in seq if a in POSITIVE_AA) \
         - sum(1.0 for a in seq if a in NEGATIVE_AA) \
         + 0.1 * seq.count("H")


def _frame(seqs, **kw) -> pd.DataFrame:
    df = pd.DataFrame({"sequence": list(seqs)})
    df["label"] = 0.0
    for k, v in kw.items():
        df[k] = v
    return df


def random_uniprot(pool: pd.Series, n: int, seed: int = 0, **kw) -> pd.DataFrame:
    """Sample at random from a background proteome pool.

    The most common rule and the most leaky: AMPs are short and cationic, so the
    negatives differ in length *and* composition, and a two-feature model can win.
    """
    return _frame(pool.sample(n, random_state=seed, replace=len(pool) < n), **kw)


def length_matched(pool: pd.Series, positives: pd.Series, seed: int = 0, **kw) -> pd.DataFrame:
    """Match the positive length distribution, removing the length shortcut.

    The charge shortcut survives this, which is exactly why M12 (length + charge)
    is in the zoo: it separates the two shortcuts empirically.
    """
    rng = np.random.default_rng(seed)
    by_len: dict[int, list[str]] = {}
    for s in pool:
        by_len.setdefault(len(s), []).append(s)
    lengths = sorted(by_len)
    picked = []
    for target in positives.map(len):
        nearest = min(lengths, key=lambda L: abs(L - target))
        bucket = by_len[nearest]
        picked.append(bucket[int(rng.integers(len(bucket)))])
    return _frame(picked, **kw)


def composition_matched(pool: pd.Series, positives: pd.Series, seed: int = 0,
                        n_candidates: int = 200, **kw) -> pd.DataFrame:
    """Match amino-acid composition as well as length.

    Much harder, and the most honest of the synthetic rules. Risk: by matching
    composition you may select genuine but unannotated AMPs into the negative
    class, which depresses measured performance for real reasons. Say so.
    """
    rng = np.random.default_rng(seed)

    def comp(s: str) -> np.ndarray:
        v = np.array([s.count(a) for a in AA], dtype=float)
        return v / max(v.sum(), 1.0)

    pool_list = list(pool)
    pool_comp = np.vstack([comp(s) for s in pool_list])
    picked = []
    for p in positives:
        target = comp(p)
        idx = rng.choice(len(pool_list), size=min(n_candidates, len(pool_list)), replace=False)
        best = idx[np.argmin(np.abs(pool_comp[idx] - target).sum(axis=1))]
        picked.append(pool_list[best])
    return _frame(picked, **kw)


def shuffled_positives(positives: pd.Series, seed: int = 0, **kw) -> pd.DataFrame:
    """Permute each positive sequence.

    Preserves composition exactly and destroys motif order, so it isolates motif
    sensitivity. Beware: shuffled sequences are never homologous to anything, so
    this rule *interacts* with the clustering threshold. Fit the interaction.
    """
    rng = np.random.default_rng(seed)
    out = ["".join(rng.permutation(list(s))) for s in positives]
    return _frame(out, **kw)


def curated_negatives(experimental: pd.Series, n: int | None = None, seed: int = 0, **kw):
    """Peptides experimentally tested and found inactive.

    Biologically the right answer; scarce in practice, and "inactive" is
    assay- and concentration-dependent, so the label is softer than it looks.
    """
    s = experimental if n is None else experimental.sample(
        min(n, len(experimental)), random_state=seed)
    return _frame(s, **kw)


RULES = {
    "random_uniprot": random_uniprot,
    "length_matched": length_matched,
    "composition_matched": composition_matched,
    "shuffled_positives": shuffled_positives,
    "curated": curated_negatives,
}

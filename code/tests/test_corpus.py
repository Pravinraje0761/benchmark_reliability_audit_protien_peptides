"""Tests for the Arm B corpus construction.

These guard the properties the factorial depends on. If any fails, the
negative-rule factor is confounded with something else and the primary analysis
is invalid -- so they are not cosmetic.
"""
import numpy as np
import pandas as pd
import pytest

from bra import corpus as C


@pytest.fixture
def uniprot():
    """Synthetic reviewed-UniProt stand-in: long proteins, some AMP-annotated."""
    rng = np.random.default_rng(0)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    # Mix of long proteins (fragment donors for the matched rules) and short ones
    # (whole-sequence donors for the random/annotation rules, which length-filter
    # to 5-100). Real SwissProt has both; a fixture of only long proteins starves
    # the whole-sequence rules.
    for i in range(400):
        rows.append({"sequence": "".join(rng.choice(aa, 300)),
                     "keywords": "Antimicrobial;Secreted" if i % 20 == 0 else "Transport"})
    for i in range(600):
        rows.append({"sequence": "".join(rng.choice(aa, int(rng.integers(10, 95)))),
                     "keywords": "Antibiotic" if i % 25 == 0 else "Membrane"})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


@pytest.fixture
def positives():
    rng = np.random.default_rng(1)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    return ["".join(rng.choice(aa, int(rng.integers(8, 40)))) for _ in range(120)]


def test_shuffle_degenerate_detects_homopolymers():
    assert C.is_shuffle_degenerate("FFFFFFFF")
    assert C.is_shuffle_degenerate("A")
    assert not C.is_shuffle_degenerate("AF")
    assert not C.is_shuffle_degenerate("KLAKLAK")


def test_shuffled_positives_never_emits_a_positive(positives, uniprot):
    """The bug the build assertion caught: a homopolymer is its own permutation."""
    pos = positives + ["WWWWWWWW"]
    pos = [s for s in pos if not C.is_shuffle_degenerate(s)]
    neg = C.make_negatives("shuffled_positives", pos, uniprot, seed=0)
    assert not (set(neg) & set(pos))


def test_shuffled_positives_raises_on_degenerate_input(uniprot):
    """Fail loudly rather than silently emit a label contradiction."""
    with pytest.raises(RuntimeError, match="no distinct permutation"):
        C.make_negatives("shuffled_positives", ["FFFFFFFF"], uniprot, seed=0)


def test_shuffled_positives_preserves_composition(positives, uniprot):
    neg = C.make_negatives("shuffled_positives", positives, uniprot, seed=0)
    for p, n in zip(positives, neg):
        assert sorted(p) == sorted(n)


def test_length_matched_matches_the_positive_lengths(positives, uniprot):
    neg = C.make_negatives("length_matched", positives, uniprot, seed=0)
    assert [len(x) for x in neg] == [len(x) for x in positives]


def test_composition_matched_is_closer_in_composition_than_random(positives, uniprot):
    """The rule must actually do what it claims, or the factor level is a lie."""
    comp = C.make_negatives("composition_matched", positives, uniprot, seed=0)
    rand = C.make_negatives("length_matched", positives, uniprot, seed=0)

    def dist(a, b):
        return float(np.abs(C._aa_composition(a) - C._aa_composition(b)).sum())

    d_comp = np.mean([dist(p, n) for p, n in zip(positives, comp)])
    d_rand = np.mean([dist(p, n) for p, n in zip(positives, rand)])
    assert d_comp < d_rand


def test_annotation_filtered_excludes_amp_keywords(positives, uniprot):
    neg = C.make_negatives("annotation_filtered", positives, uniprot, seed=0)
    amp_annotated = set(uniprot.loc[
        uniprot["keywords"].str.contains("Antimicrobial", case=False), "sequence"])
    # negatives are drawn whole here (length filter permitting), so no drawn
    # negative may come from an AMP-annotated record
    assert not (set(neg) & amp_annotated)


def test_rules_are_deterministic_given_seed(positives, uniprot):
    a = C.make_negatives("length_matched", positives, uniprot, seed=3)
    b = C.make_negatives("length_matched", positives, uniprot, seed=3)
    c = C.make_negatives("length_matched", positives, uniprot, seed=4)
    assert a == b
    assert a != c


def test_unknown_rule_raises(positives, uniprot):
    with pytest.raises(ValueError, match="unknown rule"):
        C.make_negatives("not_a_rule", positives, uniprot, seed=0)


def test_all_negatives_are_canonical(positives, uniprot):
    for rule in C.NEG_RULES:
        neg = C.make_negatives(rule, positives, uniprot, seed=0)
        bad = [s for s in neg if not set(s) <= C.L_CANONICAL]
        assert not bad, f"{rule} produced non-canonical residues: {bad[:2]}"

"""Harmonised AMP-binary corpus for Arm B (the crossed factorial).

Positives
---------
The **published** positive set of benchmark B4 (Sidorczuk et al. 2022), taken
from `NegativeDatasets/data/positive_dataset.fa`: DBAASP records flagged active
against Gram+ or Gram-, non-standard residues removed, length >= 5, CD-HIT
reduced at 0.9. **5,190 sequences**, length 5-190, median 18.

Using their positive set rather than building our own matters: it means every
negative rule below is compared against the same positives that the published
negative-sampling study used, so the negative-rule factor is isolated.

Negatives
---------
Five rules spanning the families behind the 11 methods in B4. Definitions follow
`NegativeDatasets/functions/generate_negative_datasets.R`:

1. `random_uniprot`      - random reviewed UniProt, length-filtered 5-100.
                           The iAMP-2L family WITHOUT the annotation filter.
2. `annotation_filtered` - reviewed UniProt EXCLUDING the antimicrobial keyword
                           set ("Antimicrobial", "Antibiotic", "Fungicide",
                           "Defensin"), length 5-100, as iAMP-2L / CS-AMPPred do.
                           The faithful published rule.
3. `length_matched`      - random *fragments* cut from reviewed UniProt so the
                           negative length distribution matches the positives.
                           Removes the length shortcut. Fragment-cutting mirrors
                           `generate_cutted_sequences` in the Wang et al. rule.
4. `composition_matched` - fragments chosen to match each positive amino-acid
                           composition as well as its length. Hardest rule.
5. `shuffled_positives`  - each positive randomly permuted. Preserves composition
                           exactly, destroys motif order.

Two decisions held FIXED so that the factor under study is the negative *rule*
and nothing else:

* **Class ratio is 1:1** for every rule. Published methods use ratios from 1:1 to
  1:10; letting the ratio vary with the rule would confound rule with balance,
  and metrics like F1 and AUPRC are sensitive to balance.
* **No negative may be identical to any positive**, and non-standard residues are
  removed everywhere.

Clustering for the split factor is done downstream in `bra.splits`, not here.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

L_CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")

# iAMP-2L / CS-AMPPred exclusion keywords, verbatim from the published R source.
AMP_KEYWORDS = ("Antimicrobial", "Antibiotic", "Fungicide", "Defensin")

MIN_LEN, MAX_LEN = 5, 100
NEG_RULES = ("random_uniprot", "annotation_filtered", "length_matched",
             "composition_matched", "shuffled_positives")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def read_fasta(path: str | Path) -> list[str]:
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    return seqs


def is_shuffle_degenerate(seq: str) -> bool:
    """True when no permutation of `seq` differs from `seq` itself.

    Homopolymers ("FFFFFFFF") are the case that matters. Under the
    shuffled_positives rule such a sequence would be emitted as its own
    negative: the same string carrying both labels, a labelling contradiction
    rather than a hard example. Caught by the assertion in build_corpus.
    """
    return len(set(seq)) <= 1


def load_positives(path: str | Path, drop_degenerate: bool = True) -> list[str]:
    """B4 published positive set, re-filtered defensively.

    `drop_degenerate` removes the handful of sequences with no distinct
    permutation. They are excluded for EVERY rule, not only for
    shuffled_positives, so the positive set stays identical across rules.
    Otherwise the negative-rule factor would be confounded with a changing
    positive set, which is the confound this corpus exists to avoid.
    """
    seqs = [s.upper() for s in read_fasta(path)]
    keep = [s for s in seqs if set(s) <= L_CANONICAL and len(s) >= MIN_LEN]
    if drop_degenerate:
        degen = [s for s in keep if is_shuffle_degenerate(s)]
        if degen:
            print(f"[corpus] dropping {len(degen)} shuffle-degenerate positive(s): {degen[:5]}")
        keep = [s for s in keep if not is_shuffle_degenerate(s)]
    return keep


def load_uniprot(path: str | Path) -> pd.DataFrame:
    """Reviewed UniProt TSV: accession, length, keywords, sequence, location."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    seq_col = next(c for c in df.columns if "sequence" in c)
    kw_col = next((c for c in df.columns if "keyword" in c), None)
    out = pd.DataFrame({
        "sequence": df[seq_col].astype(str).str.upper(),
        "keywords": (df[kw_col].fillna("") if kw_col is not None else ""),
    })
    # Keep only fully standard sequences; anything else cannot be represented by
    # one-hot or AAC and would silently become a different kind of example.
    out = out.loc[out["sequence"].map(lambda s: set(s) <= L_CANONICAL)]
    # Drop duplicate SEQUENCES, not just duplicate rows: SwissProt stores the
    # same sequence under different accessions, and sampling rows without
    # replacement can therefore still draw one sequence twice -- which, after an
    # index-based split, puts identical sequences on both sides of the partition.
    # Found by the exact-duplicate assertion in measure_leakage.
    return out.drop_duplicates(subset="sequence").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Negative rules
# --------------------------------------------------------------------------- #

def _cut_fragment(seq: str, target_len: int, rng: np.random.Generator) -> str | None:
    """Random window of exactly `target_len` residues, or None if too short."""
    if len(seq) < target_len:
        return None
    start = int(rng.integers(0, len(seq) - target_len + 1))
    return seq[start:start + target_len]


def _aa_composition(seq: str) -> np.ndarray:
    v = np.zeros(20)
    idx = {a: i for i, a in enumerate(sorted(L_CANONICAL))}
    for ch in seq:
        v[idx[ch]] += 1.0
    return v / max(v.sum(), 1.0)


def _exclude_amp_annotated(up: pd.DataFrame) -> pd.DataFrame:
    pattern = "|".join(AMP_KEYWORDS)
    mask = up["keywords"].astype(str).str.contains(pattern, case=False, regex=True, na=False)
    return up.loc[~mask].reset_index(drop=True)


def make_negatives(rule: str, positives: list[str], uniprot: pd.DataFrame,
                   seed: int = 0) -> list[str]:
    """Draw len(positives) negatives under `rule`. Deterministic given `seed`."""
    if rule not in NEG_RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {NEG_RULES}")
    rng = np.random.default_rng(seed)
    n = len(positives)
    pos_set = set(positives)
    seen: set[str] = set()      # negatives must be distinct from each other too

    if rule == "shuffled_positives":
        out = []
        for s in positives:
            cand = None
            for _ in range(200):
                c = "".join(rng.permutation(list(s)))
                if c not in pos_set and c not in seen:
                    cand = c
                    break
            if cand is None:
                raise RuntimeError(
                    f"no distinct permutation for {s!r}; load_positives "
                    "should have removed it"
                )
            seen.add(cand)
            out.append(cand)
        return out

    if rule == "annotation_filtered":
        pool = _exclude_amp_annotated(uniprot)["sequence"]
        cand = pool.loc[pool.map(lambda s: MIN_LEN <= len(s) <= MAX_LEN)]
        cand = [s for s in cand if s not in pos_set]
        if len(cand) < n:
            raise RuntimeError(f"annotation_filtered pool too small: {len(cand)} < {n}")
        take = rng.choice(len(cand), size=n, replace=False)
        return [cand[i] for i in take]

    if rule == "random_uniprot":
        pool = uniprot["sequence"]
        cand = pool.loc[pool.map(lambda s: MIN_LEN <= len(s) <= MAX_LEN)]
        cand = [s for s in cand if s not in pos_set]
        if len(cand) < n:
            raise RuntimeError(f"random_uniprot pool too small: {len(cand)} < {n}")
        take = rng.choice(len(cand), size=n, replace=False)
        return [cand[i] for i in take]

    # The two matched rules cut fragments, because whole UniProt proteins cannot
    # match a 5-190 aa positive length distribution.
    pool = uniprot["sequence"].to_numpy()
    long_enough = pool[np.array([len(s) >= MIN_LEN for s in pool])]

    if rule == "length_matched":
        out = []
        for s in positives:
            tl = len(s)
            for _ in range(50):
                src = long_enough[int(rng.integers(len(long_enough)))]
                frag = _cut_fragment(src, tl, rng)
                if frag and frag not in pos_set and frag not in seen:
                    seen.add(frag)
                    out.append(frag)
                    break
            else:
                out.append(frag if frag else src[:tl])
        return out

    # composition_matched
    out = []
    n_cand = 60
    for s in positives:
        tl = len(s)
        target = _aa_composition(s)
        best, best_d = None, np.inf
        for _ in range(n_cand):
            src = long_enough[int(rng.integers(len(long_enough)))]
            frag = _cut_fragment(src, tl, rng)
            if not frag or frag in pos_set or frag in seen:
                continue
            d = float(np.abs(_aa_composition(frag) - target).sum())
            if d < best_d:
                best, best_d = frag, d
        chosen = best if best is not None else s[::-1]
        seen.add(chosen)
        out.append(chosen)
    return out


# --------------------------------------------------------------------------- #
# Corpus assembly
# --------------------------------------------------------------------------- #

def build_corpus(rule: str, positives_path: str | Path, uniprot_path: str | Path,
                 seed: int = 0) -> pd.DataFrame:
    """Canonical AMP-binary table for one negative rule."""
    pos = load_positives(positives_path)
    up = load_uniprot(uniprot_path)
    neg = make_negatives(rule, pos, up, seed=seed)

    df = pd.DataFrame({
        "sequence": list(pos) + list(neg),
        "label": [1.0] * len(pos) + [0.0] * len(neg),
    })
    df["split"] = pd.NA
    df["source"] = ["DBAASP"] * len(pos) + [f"uniprot:{rule}"] * len(neg)
    df["benchmark"] = f"armB_{rule}"
    df["task"] = "amp_binary"
    df["negative_rule"] = rule

    # A negative that is byte-identical to a positive would be a labelling
    # contradiction, not merely a hard example. Assert rather than warn.
    overlap = set(df.loc[df.label == 1, "sequence"]) & set(df.loc[df.label == 0, "sequence"])
    if overlap:
        raise AssertionError(
            f"{len(overlap)} sequences carry both labels under rule {rule!r}, "
            f"e.g. {next(iter(overlap))[:30]}"
        )

    # Every sequence must appear once. A repeated sequence survives an
    # index-based split as an identical pair straddling the partition, which is
    # leakage of the most basic kind.
    dup = df["sequence"].duplicated().sum()
    if dup:
        raise AssertionError(
            f"{dup} duplicated sequences in corpus for rule {rule!r}; "
            "negatives must be distinct from one another and from the positives"
        )

    return df.astype({"sequence": "string", "split": "string", "source": "string",
                      "benchmark": "string", "task": "string",
                      "negative_rule": "string"})

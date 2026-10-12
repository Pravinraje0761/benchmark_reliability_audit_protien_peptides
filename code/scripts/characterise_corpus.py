"""Table S2: how different are the five negative-set rules, quantitatively?

Runs on the corpora alone -- no models, no grid. The point is to establish, before
any ranking is discussed, that the five rules define measurably different problems
rather than five samples of one problem. Without this, a reader can reasonably ask
whether the rules differ at all.

    python scripts/characterise_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.armb import NEG_RULES, data_root  # noqa: E402
from bra.corpus import L_CANONICAL, _aa_composition  # noqa: E402
from bra.negatives import net_charge  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
F3 = lambda v: f"{v:.3f}"


def kl_to_positive(neg: list[str], pos_comp: np.ndarray) -> float:
    """KL divergence of the pooled negative AA distribution from the positive one."""
    c = np.zeros(20)
    idx = {a: i for i, a in enumerate(sorted(L_CANONICAL))}
    for s in neg:
        for ch in s:
            c[idx[ch]] += 1
    q = c / c.sum()
    p = pos_comp
    m = (p > 0) & (q > 0)
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def main() -> None:
    corpus = data_root() / "corpus"
    rows = []
    pos_cached = None

    for rule in NEG_RULES:
        df = pd.read_parquet(corpus / f"armB_{rule}_seed0.parquet")
        pos = df.loc[df.label == 1, "sequence"].tolist()
        neg = df.loc[df.label == 0, "sequence"].tolist()
        if pos_cached is None:
            pos_cached = pos
            pc = np.zeros(20)
            idx = {a: i for i, a in enumerate(sorted(L_CANONICAL))}
            for s in pos:
                for ch in s:
                    pc[idx[ch]] += 1
            pos_comp = pc / pc.sum()
            pos_len = np.array([len(s) for s in pos])
            pos_chg = np.array([net_charge(s) for s in pos])

        nl = np.array([len(s) for s in neg])
        nc = np.array([net_charge(s) for s in neg])

        # per-pair composition L1 distance, positives are index-aligned with
        # negatives for the matched and shuffled rules
        pair_l1 = np.mean([np.abs(_aa_composition(a) - _aa_composition(b)).sum()
                           for a, b in zip(pos[:1500], neg[:1500])])

        rows.append({
            "rule": rule,
            "n": len(df),
            "neg_len_median": int(np.median(nl)),
            "len_median_ratio": float(np.median(nl) / np.median(pos_len)),
            "neg_charge_mean": float(nc.mean()),
            "charge_gap": float(pos_chg.mean() - nc.mean()),
            "pairwise_comp_L1": float(pair_l1),
            "pooled_comp_KL": kl_to_positive(neg, pos_comp),
        })

    tab = pd.DataFrame(rows)
    print("Positives: n=%d, median length %d, mean net charge %.2f\n"
          % (len(pos_cached), int(np.median(pos_len)), float(pos_chg.mean())))
    print(tab.to_string(index=False, float_format=F3))
    tab.to_csv(OUT / "tableS2_corpus_characterisation.csv", index=False)

    print("\nReading the columns:")
    print("  len_median_ratio  1.0 means the length shortcut is removed.")
    print("  charge_gap        how much more cationic positives are; a large gap")
    print("                    means charge alone separates the classes.")
    print("  pairwise_comp_L1  0 means composition carries no signal at all.")
    print("  pooled_comp_KL    divergence of the pooled negative AA distribution.")
    print("\nTogether these establish that the five rules pose measurably different")
    print("problems, which is the premise of the factorial.")
    print(f"\nwritten to {OUT/'tableS2_corpus_characterisation.csv'}")


if __name__ == "__main__":
    main()

"""Representations. PLM embeddings are cached by sequence hash -- the single
biggest cost saver in the whole project. Build the cache before the first sweep.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
                   1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))


def _hash(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Hand-crafted
# --------------------------------------------------------------------------- #

def aac(seqs: pd.Series) -> np.ndarray:
    """Amino-acid composition, 20-d."""
    out = np.zeros((len(seqs), 20))
    for i, s in enumerate(seqs):
        for a in s:
            out[i, AA_IDX[a]] += 1
        out[i] /= max(len(s), 1)
    return out


def dpc(seqs: pd.Series) -> np.ndarray:
    """Dipeptide composition, 400-d."""
    out = np.zeros((len(seqs), 400))
    for i, s in enumerate(seqs):
        for a, b in zip(s, s[1:]):
            out[i, AA_IDX[a] * 20 + AA_IDX[b]] += 1
        out[i] /= max(len(s) - 1, 1)
    return out


def physchem(seqs: pd.Series) -> np.ndarray:
    """Charge, hydropathy, hydrophobic moment, aromaticity, aliphatic index, length."""
    from .negatives import net_charge

    rows = []
    for s in seqs:
        h = np.array([KD[a] for a in s]) if s else np.array([0.0])
        # Hydrophobic moment at the alpha-helical 100-degree periodicity.
        ang = np.deg2rad(100.0) * np.arange(len(h))
        mu = np.hypot((h * np.cos(ang)).sum(), (h * np.sin(ang)).sum()) / max(len(h), 1)
        aliphatic = (s.count("A") + 2.9 * s.count("V")
                     + 3.9 * (s.count("I") + s.count("L"))) / max(len(s), 1)
        rows.append([
            net_charge(s), h.mean(), mu,
            sum(s.count(a) for a in "FWY") / max(len(s), 1),
            aliphatic, len(s),
            net_charge(s) / max(len(s), 1),
        ])
    return np.asarray(rows, dtype=float)


def length_charge(seqs: pd.Series) -> np.ndarray:
    """M12, the deliberate null baseline. Two features, nothing else.

    Its rank is a readout of negative-set separability. If it lands mid-pack on a
    benchmark, that benchmark has a shortcut and every other number from it is suspect.
    """
    from .negatives import net_charge
    return np.array([[len(s), net_charge(s)] for s in seqs], dtype=float)


def onehot(seqs: pd.Series, max_len: int) -> np.ndarray:
    """Padded one-hot for the CNN/BiLSTM. `max_len` comes from the TRAIN fold only."""
    out = np.zeros((len(seqs), max_len, 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, a in enumerate(s[:max_len]):
            out[i, j, AA_IDX[a]] = 1.0
    return out


def kmer_tfidf(train: pd.Series, test: pd.Series, k_max: int = 3):
    """k-mer TF-IDF, fitted on train only. Returns (X_train, X_test)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, k_max), min_df=2)
    return vec.fit_transform(train).toarray(), vec.transform(test).toarray()


# --------------------------------------------------------------------------- #
# Protein language models (cached)
# --------------------------------------------------------------------------- #

class EmbeddingCache:
    """Disk cache keyed by (model, sequence hash).

    Embeddings are computed once per unique sequence across the entire project.
    With ~1,560 runs over overlapping corpora this is the difference between
    days and weeks of GPU time.
    """

    def __init__(self, root: str | Path = "data/interim/embeddings"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model: str, seq: str) -> Path:
        h = _hash(seq)
        d = self.root / model / h[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{h}.npy"

    def get_many(self, model: str, seqs: list[str]) -> tuple[np.ndarray | None, list[int]]:
        """Return (stacked embeddings or None, indices still missing).

        Existence is checked with a stat before anything is read. The first
        version of this loaded every cached array in order to discover which
        were missing and then discarded all of them, costing two full read
        passes over ~29k files per model and leaving the process I/O-bound at
        10% CPU. Now a partial cache costs only stats.
        """
        paths = [self._path(model, s) for s in seqs]
        missing = [i for i, p in enumerate(paths) if not p.exists()]
        if missing:
            return None, missing
        return np.vstack([np.load(p) for p in paths]), []

    def put(self, model: str, seq: str, vec: np.ndarray) -> None:
        np.save(self._path(model, seq), vec.astype(np.float32))


def embed_plm(seqs: pd.Series, model_name: str, cache: EmbeddingCache,
              batch_size: int = 128, device: str | None = None,
              max_tokens: int = 16384) -> np.ndarray:
    """Mean-pooled embeddings from an ESM-2 or ProtT5 checkpoint, cached by hash.

    Returns rows in the SAME ORDER as `seqs`.

    Three things make this ~20-450x faster than the naive loop, measured on this
    machine (AMD Ryzen 9, 32 threads, CPU only) over 512 peptides:

    * **Length-sorted batching.** Unsorted batches pad every member to the
      longest sequence present. With peptides spanning 5-190 residues that wasted
      most of the compute: ESM-2 8M went from 959 ms/seq to 2.1 ms/seq, and 650M
      from 2008 ms/seq to 91.6 ms/seq.
    * **`torch.inference_mode()`** rather than plain no-grad.
    * **Token-budget batching.** `max_tokens` caps batch_size x padded_length so
      long-sequence batches shrink automatically instead of exhausting memory.

    torch and transformers are imported lazily, so the classical sweep never
    pays for them.
    """
    seqs = list(seqs)
    stacked, missing = cache.get_many(model_name, seqs)
    if stacked is not None:
        return stacked

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    todo = sorted({seqs[i] for i in missing}, key=len)
    is_t5 = "t5" in model_name.lower()

    with torch.inference_mode():
        i = 0
        while i < len(todo):
            # Grow the batch while the padded token budget allows it.
            n = min(batch_size, len(todo) - i)
            while n > 1 and n * len(todo[i + n - 1]) > max_tokens:
                n //= 2
            chunk = todo[i:i + n]
            i += n

            prepared = [" ".join(x) for x in chunk] if is_t5 else chunk
            enc = tok(prepared, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(device)
            hid = mdl(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = ((hid * mask).sum(1) / mask.sum(1)).cpu().numpy()
            for sq, vec in zip(chunk, pooled):
                cache.put(model_name, sq, vec)

    stacked, still = cache.get_many(model_name, seqs)
    if still:
        raise RuntimeError(f"embedding cache incomplete for {model_name}: "
                           f"{len(still)} sequences missing")
    return stacked

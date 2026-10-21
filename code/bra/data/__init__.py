"""Benchmark loaders. One module per benchmark; all return the canonical schema.

Register a loader here and the runner picks it up. Write the QMAP loader first:
it is the best documented, and it will shape the interface for the other five.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from ..schema import validate_peptides

LOADERS: dict[str, Callable[[], pd.DataFrame]] = {}


def register(name: str):
    def deco(fn):
        LOADERS[name] = fn
        return fn
    return deco


def load(name: str) -> pd.DataFrame:
    if name not in LOADERS:
        raise KeyError(f"no loader for {name!r}; registered: {sorted(LOADERS)}")
    return validate_peptides(LOADERS[name]())


# --------------------------------------------------------------------------- #
# Stubs -- fill during milestone M1. Each must return columns:
#   sequence, label, split, source, benchmark, task
# and must reproduce the paper's stated N within 1%.
# --------------------------------------------------------------------------- #

@register("B1_genpept")
def load_b1() -> pd.DataFrame:
    """GenPept-Curated-2025. 11,000 peptides, 5,500/5,500, 10-200 aa.

    Watch for: AMP proportion varies 14.2% (10-50 aa) to 77.1% (101-150 aa).
    Record the per-length-bin balance -- it is the motivation for M12.
    """
    raise NotImplementedError("see Datasets/B1 GenPept-Curated-2025.md")


@register("B3_ppb")
def load_b3() -> pd.DataFrame:
    """PPB: 15 datasets, 161,571 sequences. Provides BOTH random and homology splits.

    Use only the subsets overlapping our tasks (amp_binary, hemolysis, mic).
    Having both split types makes B3 the internal control for the threshold factor.
    """
    raise NotImplementedError("see Datasets/B3 PPB.md")


@register("B4_ampbenchmark")
def load_b4() -> pd.DataFrame:
    """AMPBenchmark: one positive set x 11 negative-sampling methods.

    Load all 11 negative variants; they become the native levels of the
    negative_rule factor rather than a single fixed dataset.
    """
    raise NotImplementedError("see Datasets/B4 AMPBenchmark.md")


@register("B5_autopeptideml")
def load_b5() -> pd.DataFrame:
    """AutoPeptideML benchmarks. Use the Zenodo static snapshot, not HEAD,
    so the version is pinned and citable."""
    raise NotImplementedError("see Datasets/B5 AutoPeptideML.md")


@register("B6_abp_framework")
def load_b6() -> pd.DataFrame:
    """ABP reference evaluation framework.

    Verify early what is actually downloadable versus merely proposed; if the
    framework ships no data, substitute per the bench list in 01 Benchmark inventory.
    """
    raise NotImplementedError("see Datasets/B6 ABP reference framework.md")


def synthetic(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Planted-motif corpus for the harness selftest (E01). Not a benchmark."""
    import numpy as np
    rng = np.random.default_rng(seed)
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    rows = []
    for i in range(n):
        L = int(rng.integers(12, 45))
        s = "".join(rng.choice(aa, L))
        pos = i % 2 == 0
        if pos:                       # plant a cationic motif
            j = int(rng.integers(0, max(L - 5, 1)))
            s = s[:j] + "KKLLKK" + s[j + 6:]
        rows.append({"sequence": s[:L] or "AAAA", "label": float(pos),
                     "split": pd.NA, "source": "synthetic",
                     "benchmark": "SYNTH", "task": "amp_binary"})
    return pd.DataFrame(rows).astype({"sequence": "string", "split": "string",
                                      "source": "string", "benchmark": "string",
                                      "task": "string"})

# --------------------------------------------------------------------------- #
# Concrete loaders. Imported last so their @register decorators run after
# `register` is defined -- qmap_loader imports it from this module.
# --------------------------------------------------------------------------- #

from . import qmap_loader  # noqa: E402,F401  (import for side effect: registration)

"""
Shared scaffolding for the depth x activation ablations.

Every per-paper harness emits the SAME artifact schema, so the write-up for each
paper can be generated from identical inputs:

    <outdir>/runs.csv                one row per (model, depth, activation, seed)
    <outdir>/confusion_matrices.json raw integer confusion matrices, keyed
    <outdir>/summary.md              human-readable aggregate
    <outdir>/manifest.json           environment + provenance for reproducibility

Design decisions worth knowing before you run anything
------------------------------------------------------
1. EVERY configuration is run across multiple seeds. A single run per depth is
   not interpretable: the peptide audit in this same body of work measured that
   reseeding alone changes the top-ranked pipeline in 54.5% of comparisons. A
   depth sweep reported at one seed per depth would be measuring seed noise and
   calling it architecture.

2. The seed-replicate spread is reported alongside the depth effect, so the
   write-up can state whether a depth difference exceeds the noise floor rather
   than asserting it.

3. Confusion matrices are stored as raw integer counts, never as normalised
   rates. Rates can always be derived; counts cannot be recovered from rates.

4. Nothing here tunes anything. Depth and activation are the only things that
   move. Everything else -- features, splits, preprocessing, regularisation --
   is inherited from the host repository unchanged, so the comparison is
   attributable to the two factors under study.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# The sweep grid. Shared by all four harnesses so the papers stay comparable.
# --------------------------------------------------------------------------- #

DEPTHS = [3, 4, 5, 6, 7]          # number of HIDDEN layers, as requested
ACTIVATIONS = ["relu", "tanh", "logistic", "gelu"]
SEEDS = [0, 1, 2, 3, 4]

# Width policy: constant width across depth. The alternative (shrinking width to
# hold parameter count fixed) confounds depth with capacity; constant width
# confounds depth with capacity in the other direction. Neither is neutral, so
# we run constant width as the primary and expose --width-policy=pyramid as a
# sensitivity check. State which one you ran in the paper.
DEFAULT_WIDTH = 256


def hidden_sizes(depth: int, width: int = DEFAULT_WIDTH, policy: str = "constant"):
    """Hidden layer sizes for a given depth under the chosen width policy."""
    if policy == "constant":
        return tuple([width] * depth)
    if policy == "pyramid":
        # Halve each layer, floored at 32. Total parameters stay closer to flat
        # as depth grows, which is the capacity-controlled comparison.
        sizes, w = [], width
        for _ in range(depth):
            sizes.append(max(32, w))
            w //= 2
        return tuple(sizes)
    raise ValueError(f"unknown width policy {policy!r}")


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

@dataclass
class RunRecord:
    model: str
    depth: int
    activation: str
    seed: int
    width_policy: str
    hidden_sizes: str
    n_params: int
    n_train: int
    n_test: int
    fit_seconds: float
    converged: bool
    # metrics -- populated by metrics_from_confusion plus any extras
    accuracy: float = float("nan")
    balanced_accuracy: float = float("nan")
    mcc: float = float("nan")
    macro_f1: float = float("nan")
    auroc: float = float("nan")
    note: str = ""


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Derive the scalar metrics from an integer confusion matrix.

    Deriving them from the matrix rather than recomputing from predictions
    guarantees the reported matrix and the reported metrics cannot disagree --
    a failure mode that is embarrassingly common in ablation tables.
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    if n == 0:
        return {}
    correct = np.trace(cm)
    accuracy = correct / n

    # Per-class recall (sensitivity), guarded against empty classes.
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(support > 0, np.diag(cm) / support, np.nan)
        predicted = cm.sum(axis=0)
        precision = np.where(predicted > 0, np.diag(cm) / predicted, np.nan)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    balanced_accuracy = float(np.nanmean(recall))
    macro_f1 = float(np.nanmean(f1))

    # Multiclass MCC straight from the matrix (Gorodkin's formulation).
    t_k = cm.sum(axis=1)
    p_k = cm.sum(axis=0)
    c = correct
    s = n
    num = c * s - float(t_k @ p_k)
    den = np.sqrt(max(s * s - float(p_k @ p_k), 0.0)) * np.sqrt(
        max(s * s - float(t_k @ t_k), 0.0)
    )
    mcc = float(num / den) if den > 0 else float("nan")

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }


def count_params(hidden: tuple, n_in: int, n_out: int) -> int:
    """Parameter count for an MLP with the given hidden sizes (weights + biases)."""
    dims = [n_in, *hidden, n_out]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_outputs(outdir: Path, records: list[RunRecord], matrices: dict,
                  class_names: list[str], context: dict) -> None:
    """Write the four artifacts. Call once at the end of a sweep."""
    import pandas as pd

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(outdir / "runs.csv", index=False)

    (outdir / "confusion_matrices.json").write_text(
        json.dumps(
            {"class_names": class_names, "matrices": matrices},
            indent=2,
        )
    )

    (outdir / "manifest.json").write_text(json.dumps(_manifest(context), indent=2))
    (outdir / "summary.md").write_text(_summary(df, matrices, class_names, context))
    print(f"\nwrote {outdir}/runs.csv  ({len(df)} runs)")
    print(f"wrote {outdir}/confusion_matrices.json")
    print(f"wrote {outdir}/summary.md")
    print(f"wrote {outdir}/manifest.json")


def _manifest(context: dict) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "nogit"
    pkgs = {}
    for mod in ("numpy", "pandas", "sklearn", "torch", "scipy"):
        try:
            pkgs[mod] = __import__(mod).__version__
        except Exception:
            pkgs[mod] = "absent"
    return {
        "commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": pkgs,
        "grid": {"depths": DEPTHS, "activations": ACTIVATIONS, "seeds": SEEDS},
        **context,
    }


def _summary(df, matrices: dict, class_names: list[str], context: dict) -> str:
    """A readable aggregate: the depth effect against the seed noise floor."""
    lines = [
        f"# Depth x activation ablation -- {context.get('paper', 'unnamed')}",
        "",
        f"Model(s): {', '.join(sorted(df.model.unique()))}",
        # The grid as ACTUALLY RUN, not the module defaults -- a summary that
        # reports the default grid after a partial run is how a half-finished
        # sweep gets written up as a complete one.
        f"Grid: depths {sorted(int(d) for d in df.depth.unique())} "
        f"x activations {sorted(df.activation.unique())} "
        f"x seeds {sorted(int(s) for s in df.seed.unique())}",
        f"Width policy: {', '.join(sorted(df.width_policy.unique()))}",
        f"Total runs: {len(df)}",
        "",
        "## Seed noise floor",
        "",
        "Standard deviation across seed replicates, within each fixed",
        "(model, depth, activation) cell. A depth effect smaller than this is",
        "not a depth effect.",
        "",
    ]
    noise = (
        df.groupby(["model", "depth", "activation"])["mcc"]
        .std()
        .reset_index(name="mcc_sd_across_seeds")
    )
    lines += [
        f"- median within-cell SD of MCC: {noise.mcc_sd_across_seeds.median():.4f}",
        f"- max within-cell SD of MCC:    {noise.mcc_sd_across_seeds.max():.4f}",
        "",
        "## Mean MCC by depth x activation (averaged over seeds)",
        "",
    ]
    piv = df.pivot_table(index="depth", columns="activation", values="mcc", aggfunc="mean")
    lines.append(piv.round(4).to_markdown())
    lines += [
        "",
        "## Mean balanced accuracy by depth x activation",
        "",
        df.pivot_table(index="depth", columns="activation",
                       values="balanced_accuracy", aggfunc="mean").round(4).to_markdown(),
        "",
        "## Best cell",
        "",
    ]
    cell_means = df.groupby(["model", "depth", "activation"])["mcc"].mean()
    bm, bd, ba = cell_means.idxmax()
    lines += [
        f"- model `{bm}`, depth {int(bd)}, activation `{ba}` "
        f"-> mean MCC {cell_means.max():.4f}",
        "",
        "Treat 'best' as provisional until you have checked it against the noise",
        "floor above. On a grid this size the top cell is frequently the luckiest",
        "cell rather than the best architecture.",
    ]

    # Depth effect vs noise, stated numerically so the write-up does not have to
    # eyeball it.
    by_depth = df.groupby("depth")["mcc"].mean()
    spread = float(by_depth.max() - by_depth.min())
    floor = float(noise.mcc_sd_across_seeds.median())
    # Degenerate runs: a model that predicts one class for every input is not a
    # weak model, it is a failed fit, and it produces an undefined MCC. Reporting
    # it as a low score would be wrong; dropping it silently would be worse.
    degenerate = []
    for k, cm in matrices.items():
        if k.endswith("|bands"):
            continue
        arr = np.asarray(cm)
        if (arr.sum(axis=0) > 0).sum() <= 1:
            degenerate.append(k)
    if degenerate:
        lines += [
            "",
            "## Degenerate fits (single predicted class)",
            "",
            f"{len(degenerate)} of {len(matrices)} runs predicted one class for",
            "every test case. These are failed optimisations, not weak models;",
            "their MCC is undefined and appears as NaN in runs.csv.",
            "",
            "Report them as failures-to-converge rather than as low scores. If",
            "they cluster in the deep + saturating-activation cells, that is the",
            "vanishing-gradient signature and is itself the finding.",
            "",
        ] + [f"- `{k}`" for k in sorted(degenerate)[:40]]
        if len(degenerate) > 40:
            lines.append(f"- ... and {len(degenerate) - 40} more (see runs.csv)")

    lines += [
        "",
        "## Depth effect against the noise floor",
        "",
        f"- best-minus-worst depth, averaged over activations and seeds: {spread:.4f}",
        f"- median within-cell seed SD: {floor:.4f}",
        f"- ratio: {spread / floor:.2f}x" if floor > 0 else "- ratio: undefined (zero SD)",
        "",
        "A ratio below about 2 means depth did not move this task beyond what",
        "reseeding moves it on its own.",
    ]
    lines += [
        "",
        "## How to read this",
        "",
        "Compare the spread across the depth column against the seed noise floor",
        "above. If the best and worst depths differ by less than roughly two",
        "within-cell standard deviations, the honest conclusion is that depth did",
        "not matter on this task at this width -- which is a result worth",
        "reporting, not a failed experiment.",
        "",
        f"Confusion matrices for all {len(matrices)} cells are in",
        "confusion_matrices.json, keyed `model|depth|activation|seed`, as raw",
        f"integer counts over classes {class_names}.",
    ]
    return "\n".join(lines)


def key(model: str, depth: int, activation: str, seed: int) -> str:
    return f"{model}|{depth}|{activation}|{seed}"

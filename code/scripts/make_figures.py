"""Publication figures for the BRA paper. Every panel regenerates from the
single Arm B parquet plus the design-simulation CSVs -- no manual steps.

    python scripts/make_figures.py

Writes PDF (vector, for submission) and PNG (for drafts) to results/figures/.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bra.analysis import decompose, ranks  # noqa: E402
from bra.armb import data_root  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"
PRIMARY = "mcc"

# Okabe-Ito: colourblind-safe, and legible in greyscale.
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
         "#E69F00", "#56B4E9", "#F0E442", "#000000"]
RULE_SHORT = {"random_uniprot": "random\nUniProt",
              "annotation_filtered": "annotation\nfiltered",
              "length_matched": "length\nmatched",
              "composition_matched": "composition\nmatched",
              "shuffled_positives": "shuffled\npositives"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def rankmat(df, factor, metric=PRIMARY):
    lb = df[df.metric == metric].groupby([factor, "model_id"], as_index=False)["score"].mean()
    lb["rank"] = lb.groupby(factor)["score"].rank(ascending=False, method="average")
    return lb.pivot(index="model_id", columns=factor, values="rank")


# --------------------------------------------------------------------------- #


def fig0_shortcut_ladder():
    """The five negative rules as an ordered ladder of available shortcuts.

    Independent of the grid: establishes that the rules pose different problems,
    which is the premise everything else rests on.
    """
    f = TAB / "tableS2_corpus_characterisation.csv"
    if not f.exists():
        print("  (skipping fig0: tableS2 absent)")
        return
    t = pd.read_csv(f)
    order = ["random_uniprot", "annotation_filtered", "length_matched",
             "composition_matched", "shuffled_positives"]
    t = t.set_index("rule").loc[[r for r in order if r in set(t["rule"])]]
    labs = ["random{n}UniProt", "annotation{n}filtered", "length{n}matched",
            "composition{n}matched", "shuffled{n}positives"]
    labs = [l.format(n="\n") for l in labs][:len(t)]
    x = np.arange(len(t))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True)
    panels = [
        ("len_median_ratio", "negative / positive\nmedian length", OKABE[0], 1.0),
        ("charge_gap", "net-charge gap\n(positive - negative)", OKABE[1], 0.0),
        ("pairwise_comp_L1", "composition distance\n(pairwise L1)", OKABE[2], 0.0),
    ]
    for ax, (col, lab, colour, ref) in zip(axes, panels):
        ax.bar(x, t[col], color=colour, width=0.68)
        ax.axhline(ref, color="0.45", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=6, rotation=38, ha="right")
        ax.set_ylabel(lab, fontsize=7.5)
        ax.margins(y=0.18)
        for xi, v in zip(x, t[col]):
            ax.text(xi, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=6)
    fig.suptitle("Each negative-set rule removes a different shortcut", fontsize=9)
    save(fig, "fig0_shortcut_ladder")


def fig1_landscape(df):
    """Mean MCC per protocol cell: the performance landscape."""
    piv = (df[df.metric == PRIMARY].groupby(["negative_rule", "cluster_threshold"])
           ["score"].mean().unstack())
    piv = piv[["random", "0.9", "0.7", "0.4"]]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(["random", "90%", "70%", "40%"])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([RULE_SHORT.get(r, r).replace("\n", " ") for r in piv.index])
    ax.set_xlabel("clustering identity threshold")
    ax.set_title("Mean MCC across 16 pipelines")
    for i, j in itertools.product(range(piv.shape[0]), range(piv.shape[1])):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v < piv.values.mean() else "black")
    fig.colorbar(im, ax=ax, label="MCC", fraction=0.046)
    save(fig, "fig1_performance_landscape")


def fig2_tau(df, ceiling=None):
    """tau-b agreement across negative rules, with null band and ceiling."""
    kmat = rankmat(df, "negative_rule")
    pt = ranks.pairwise_tau(kmat)
    n = kmat.shape[0]
    null = ranks.permutation_null(n, 2, n_perm=3000, seed=0)
    lo, hi = np.percentile(null, [2.5, 97.5])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    vals = pt.values.copy()
    im = ax.imshow(vals, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    labs = [RULE_SHORT.get(c, c) for c in pt.columns]
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6)
    for i, j in itertools.product(range(len(labs)), range(len(labs))):
        ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(f"Kendall tau-b between negative rules (n={n} pipelines)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="tau-b")

    ax = axes[1]
    off = vals[np.triu_indices_from(vals, k=1)]
    ax.axhspan(lo, hi, color="0.88", zorder=0,
               label=f"null 95% band ({lo:+.2f}, {hi:+.2f})")
    ax.axhline(0, color="0.5", lw=0.6)
    if ceiling is not None:
        ax.axhline(ceiling, color=OKABE[2], ls="--", lw=1.2,
                   label=f"seed ceiling = {ceiling:.2f}")
    ax.scatter(np.arange(len(off)) + 1, off, color=OKABE[0], s=22, zorder=3)
    ax.axhline(off.mean(), color=OKABE[1], lw=1.2,
               label=f"mean = {off.mean():.2f}")
    ax.set_xlabel("negative-rule pair"); ax.set_ylabel("tau-b")
    ax.set_ylim(-1.05, 1.05); ax.set_xticks(range(1, len(off) + 1))
    ax.set_title("Pairwise agreement vs the achievable ceiling")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig2_tau_agreement")


def fig3_bump(df):
    """Bump chart: each pipeline rank across negative rules."""
    kmat = rankmat(df, "negative_rule")
    order = ["random_uniprot", "annotation_filtered", "length_matched",
             "composition_matched", "shuffled_positives"]
    kmat = kmat[[c for c in order if c in kmat.columns]]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    spread = (kmat.max(axis=1) - kmat.min(axis=1)).sort_values(ascending=False)
    for k, mid in enumerate(kmat.index):
        wide = mid in spread.index[:3]
        ax.plot(range(kmat.shape[1]), kmat.loc[mid].values,
                marker="o", ms=3.5, lw=2.0 if wide else 0.8,
                color=OKABE[k % len(OKABE)] if wide else "0.75",
                zorder=3 if wide else 1,
                label=mid if wide else None)
        ax.annotate(mid, (kmat.shape[1] - 1, kmat.loc[mid].values[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=5.5, va="center",
                    color="black" if wide else "0.6")
    ax.invert_yaxis()
    ax.set_xticks(range(kmat.shape[1]))
    ax.set_xticklabels([RULE_SHORT.get(c, c) for c in kmat.columns], fontsize=6)
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Pipeline rank depends on the negative-set rule")
    ax.legend(frameon=False, title="widest swing", loc="center left",
              bbox_to_anchor=(1.08, 0.5))
    save(fig, "fig3_bump_chart")


def fig4_variance(df):
    """Variance components (score vs rank) and the tau-b ablation."""
    comp = decompose.decompose_both(df)
    comp = comp.drop(index=[i for i in comp.index if i == "residual"], errors="ignore")
    comp = comp.sort_values("pct_rank_var", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(comp))
    ax.barh(y - 0.2, comp["pct_score_var"], height=0.38,
            color=OKABE[0], label="score variance")
    ax.barh(y + 0.2, comp["pct_rank_var"], height=0.38,
            color=OKABE[1], label="rank variance")
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in comp.index], fontsize=6.5)
    ax.set_xlabel("% of variance")
    ax.set_title("What moves the score, and what moves the ranking")
    ax.legend(frameon=False)

    ax = axes[1]
    ab = decompose.tau_ablation(df, PRIMARY)
    base = ab.loc[ab.pinned == "none (baseline)", "mean_tau_b"].iloc[0]
    sub = ab[ab.pinned != "none (baseline)"]
    gains = sub.groupby("pinned")["delta_vs_baseline"].mean().sort_values()
    ax.barh(range(len(gains)), gains.values,
            color=[OKABE[2] if v > 0 else OKABE[3] for v in gains.values])
    ax.set_yticks(range(len(gains)))
    ax.set_yticklabels([g.replace("_", " ") for g in gains.index], fontsize=6.5)
    ax.axvline(0, color="0.4", lw=0.7)
    ax.set_xlabel("mean change in tau-b when the factor is pinned")
    ax.set_title(f"Agreement recovered by fixing a factor\n(baseline tau-b = {base:.2f})")
    save(fig, "fig4_variance_and_ablation")


def fig5_null_baseline(df):
    """Where the 2-feature null baseline lands under each negative rule."""
    lb = df[df.metric == PRIMARY].groupby(["negative_rule", "model_id"],
                                          as_index=False)["score"].mean()
    lb["rank"] = lb.groupby("negative_rule")["score"].rank(ascending=False, method="average")
    n = lb.model_id.nunique()
    p16 = lb[lb.model_id == "P16"].set_index("negative_rule")
    order = ["random_uniprot", "annotation_filtered", "length_matched",
             "composition_matched", "shuffled_positives"]
    p16 = p16.loc[[o for o in order if o in p16.index]]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8))
    ax = axes[0]
    cols = [OKABE[1] if r <= n / 2 else OKABE[0] for r in p16["rank"]]
    ax.bar(range(len(p16)), p16["rank"], color=cols)
    ax.axhline(n / 2, color="0.4", ls="--", lw=0.9, label="median rank")
    ax.set_xticks(range(len(p16)))
    ax.set_xticklabels([RULE_SHORT.get(i, i) for i in p16.index], fontsize=6)
    ax.set_ylabel(f"rank of length+charge (of {n})")
    ax.invert_yaxis()
    ax.set_title("Null-baseline rank exposes shortcuts")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(range(len(p16)), p16["score"], color=OKABE[4])
    ax.set_xticks(range(len(p16)))
    ax.set_xticklabels([RULE_SHORT.get(i, i) for i in p16.index], fontsize=6)
    ax.set_ylabel("MCC of length+charge alone")
    ax.set_title("Two features are enough on some rules")
    save(fig, "fig5_null_baseline")


def fig6_power():
    """Design simulation: estimator behaviour and the tau-b ceiling."""
    pc = TAB / "power_curve.csv"
    ce = TAB / "power_followup_ceiling.csv"
    if not pc.exists() or not ce.exists():
        print("  (skipping fig6: power CSVs absent)")
        return
    pcd, ced = pd.read_csv(pc), pd.read_csv(ce)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax = axes[0]
    ax.plot(ced["sigma_e"], ced["ceiling_tau_b"], marker="o", ms=4, color=OKABE[0])
    ax.set_xlabel("run-to-run noise (sigma_e)")
    ax.set_ylabel("max achievable tau-b")
    ax.set_title("tau-b has a ceiling below 1.0")
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax = axes[1]
    ax.plot(pcd["population_tau_b"], pcd["power_reject_H1"],
            marker="o", ms=4, color=OKABE[1])
    ax.axhline(0.8, color="0.4", ls="--", lw=0.9, label="80% power")
    ax.set_xlabel("true mean pairwise tau-b")
    ax.set_ylabel("power to detect disagreement")
    ax.set_title("Minimum detectable disagreement")
    ax.legend(frameon=False)
    save(fig, "fig6_design_simulation")


def main():
    src = data_root() / "runs" / "all_runs_armB.parquet"
    df = pd.read_parquet(src)
    df = df[df.metric != "failed"]
    print(f"figures from {len(df)} rows")

    # seed ceiling, for the Fig 2 reference line
    sub = df[df.metric == PRIMARY]
    taus = []
    for _, g in sub.groupby(["negative_rule", "cluster_threshold"]):
        ps = (g.groupby(["seed", "model_id"], as_index=False)["score"].mean()
               .pivot(index="model_id", columns="seed", values="score")
               .rank(ascending=False, method="average"))
        taus += [ranks.tau_b(ps[a], ps[b])
                 for a, b in itertools.combinations(ps.columns, 2)]
    ceiling = float(np.mean(taus)) if taus else None

    fig0_shortcut_ladder()
    fig1_landscape(df)
    fig2_tau(df, ceiling)
    fig3_bump(df)
    fig4_variance(df)
    fig5_null_baseline(df)
    fig6_power()
    print(f"\nfigures in {FIG}")


if __name__ == "__main__":
    main()

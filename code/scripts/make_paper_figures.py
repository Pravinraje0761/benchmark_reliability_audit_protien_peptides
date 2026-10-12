"""Publication figure set for the BRA paper.

Design decisions, applied deliberately rather than by default:

* **Form before colour.** Magnitude -> sequential heatmap or bar; identity and
  change-of-rank -> bump chart; polarity (tau-b spans -1..+1) -> diverging with a
  neutral midpoint; composition of variance -> grouped horizontal bars.
* **Greyscale- and CVD-safe.** Sequential uses viridis, which is monotonic in
  lightness and therefore survives a greyscale print. Categorical uses Okabe-Ito,
  assigned in fixed order and never cycled. Diverging uses two hues with a neutral
  midpoint, never a rainbow.
* **Identity is never colour alone**: every multi-series panel carries direct
  labels or an explicit legend.
* **No dual axes anywhere.**
* Single-column width is 3.5in, double 7.2in, at 8pt base type.
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
from bra.analysis import decompose, ranks            # noqa: E402
from bra.armb import data_root                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"

# Okabe-Ito, fixed order, never cycled.
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
SEQ = "viridis"           # perceptually uniform, greyscale-monotonic
DIV = "RdBu_r"            # two hues, neutral midpoint

RULES = ["random_uniprot", "annotation_filtered", "length_matched",
         "composition_matched", "shuffled_positives"]
RLAB = {"random_uniprot": "random\nUniProt", "annotation_filtered": "annotation\nfiltered",
        "length_matched": "length\nmatched", "composition_matched": "composition\nmatched",
        "shuffled_positives": "shuffled\npositives"}
THR = ["random", "0.9", "0.7", "0.4"]
TLAB = {"random": "random", "0.9": "90%", "0.7": "70%", "0.4": "40%"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}")


def load(balanced=True):
    f = "all_runs_armB_balanced.parquet" if balanced else "all_runs_armB.parquet"
    d = pd.read_parquet(data_root() / "runs" / f)
    return d[d.metric != "failed"]


def rankmat(df, factor, metric="mcc"):
    lb = df[df.metric == metric].groupby([factor, "model_id"], as_index=False)["score"].mean()
    lb["rank"] = lb.groupby(factor)["score"].rank(ascending=False, method="average")
    return lb.pivot(index="model_id", columns=factor, values="rank")


def seed_ceiling(df, metric="mcc"):
    b = df[df.metric == metric]
    taus = []
    for _, g in b.groupby(["negative_rule", "cluster_threshold"]):
        ps = (g.groupby(["seed", "model_id"], as_index=False)["score"].mean()
                .pivot(index="model_id", columns="seed", values="score")
                .rank(ascending=False, method="average"))
        taus += [ranks.tau_b(ps[a], ps[b2]) for a, b2 in itertools.combinations(ps.columns, 2)]
    return float(np.mean(taus))


# --------------------------------------------------------------------------- #
# Fig 1  corpus characterisation: the ladder of available shortcuts
# --------------------------------------------------------------------------- #
def fig1_shortcut_ladder():
    t = pd.read_csv(TAB / "tableS2_corpus_characterisation.csv").set_index("rule")
    t = t.loc[[r for r in RULES if r in t.index]]
    x = np.arange(len(t))
    labs = [RLAB[i] for i in t.index]

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    panels = [("len_median_ratio", "negative / positive\nmedian length", 1.0),
              ("charge_gap", "net-charge gap\n(positive - negative)", 0.0),
              ("pairwise_comp_L1", "composition distance\n(pairwise L1)", 0.0)]
    for a, (col, lab, ref) in zip(ax, panels):
        a.bar(x, t[col], color=OI[0], width=0.66)
        a.axhline(ref, color="0.45", ls="--", lw=0.7, zorder=0)
        a.set_xticks(x); a.set_xticklabels(labs, fontsize=6, rotation=40, ha="right")
        a.set_ylabel(lab); a.margins(y=0.20)
        for xi, v in zip(x, t[col]):
            a.text(xi, v, f"{v:.2f}", ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=6)
    ax[1].annotate("length matching\nremoves one shortcut\nand worsens another",
                   xy=(2, t["charge_gap"].iloc[2]), xytext=(0.02, 0.72),
                   textcoords="axes fraction", fontsize=5.8,
                   arrowprops=dict(arrowstyle="->", lw=0.6, color="0.35"))
    fig.suptitle("Each negative-set rule leaves a different shortcut available",
                 fontsize=9)
    save(fig, "F1_shortcut_ladder")


# --------------------------------------------------------------------------- #
# Fig 2  performance landscape
# --------------------------------------------------------------------------- #
def fig2_landscape(df):
    piv = (df[df.metric == "mcc"].groupby(["negative_rule", "cluster_threshold"])["score"]
           .mean().unstack())
    piv = piv.loc[[r for r in RULES if r in piv.index], [c for c in THR if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    im = ax.imshow(piv.values, cmap=SEQ, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([TLAB[c] for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([RLAB[i].replace("\n", " ") for i in piv.index], fontsize=6.5)
    ax.set_xlabel("clustering identity threshold")
    for i, j in itertools.product(range(piv.shape[0]), range(piv.shape[1])):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v < 0.55 else "black")
    fig.colorbar(im, ax=ax, label="mean MCC over 16 pipelines", fraction=0.046)
    ax.set_title(f"Apparent difficulty spans {piv.values.min():.2f} to {piv.values.max():.2f} MCC",
                 fontsize=8.5)
    save(fig, "F2_performance_landscape")


# --------------------------------------------------------------------------- #
# Fig 3  bump chart: rank depends on the negative-set rule
# --------------------------------------------------------------------------- #
def fig3_bump(df):
    kmat = rankmat(df, "negative_rule")
    kmat = kmat[[c for c in RULES if c in kmat.columns]]
    spread = (kmat.max(axis=1) - kmat.min(axis=1)).sort_values(ascending=False)
    movers = list(spread.index[:4])
    winners = {c: kmat[c].idxmin() for c in kmat.columns}

    fig, ax = plt.subplots(figsize=(5.0, 3.6), constrained_layout=True)
    for mid in kmat.index:
        hi = mid in movers or mid in winners.values()
        colour = OI[movers.index(mid) % len(OI)] if mid in movers else "0.78"
        ax.plot(range(kmat.shape[1]), kmat.loc[mid], marker="o", ms=3.4,
                lw=1.9 if hi else 0.7, color=colour if hi else "0.78",
                zorder=3 if hi else 1,
                label=mid if mid in movers else None)
        ax.annotate(mid, (kmat.shape[1] - 1, kmat.loc[mid].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", fontsize=5.5,
                    va="center", color="black" if hi else "0.6")
    # mark the winner under each rule
    for j, c in enumerate(kmat.columns):
        ax.scatter([j], [kmat[c].min()], s=70, facecolors="none",
                   edgecolors="black", lw=1.0, zorder=5)
    ax.invert_yaxis()
    ax.set_xticks(range(kmat.shape[1]))
    ax.set_xticklabels([RLAB[c] for c in kmat.columns], fontsize=6.5)
    ax.set_ylabel("rank (1 = best)")
    ax.set_xlim(-0.35, kmat.shape[1] - 0.45)
    n_flip = len({winners[c] for c in kmat.columns})
    ax.set_title(f"{n_flip} different pipelines win, depending on the negative-set rule\n"
                 "(circled = winner)", fontsize=8.5)
    ax.legend(title="widest swing", loc="center left", bbox_to_anchor=(1.07, 0.5),
              fontsize=6.5, title_fontsize=6.5)
    save(fig, "F3_bump_chart")


# --------------------------------------------------------------------------- #
# Fig 4  agreement against the measured ceiling
# --------------------------------------------------------------------------- #
def fig4_agreement(df):
    kmat = rankmat(df, "negative_rule")
    kmat = kmat[[c for c in RULES if c in kmat.columns]]
    pt = ranks.pairwise_tau(kmat)
    ceil = seed_ceiling(df)
    n = kmat.shape[0]
    null = ranks.permutation_null(n, 2, n_perm=3000, seed=0)
    lo, hi = np.percentile(null, [2.5, 97.5])

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0),
                           gridspec_kw={"width_ratios": [1.15, 1]},
                           constrained_layout=True)
    a = ax[0]
    im = a.imshow(pt.values, cmap=DIV, norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    labs = [RLAB[c] for c in pt.columns]
    a.set_xticks(range(len(labs))); a.set_xticklabels(labs, fontsize=5.8, rotation=40, ha="right")
    a.set_yticks(range(len(labs))); a.set_yticklabels(labs, fontsize=5.8)
    for i, j in itertools.product(range(len(labs)), range(len(labs))):
        a.text(j, i, f"{pt.values[i,j]:.2f}", ha="center", va="center", fontsize=5.8)
    a.set_title(f"Kendall tau-b between rules (n={n} pipelines)")
    fig.colorbar(im, ax=a, fraction=0.046, label="tau-b")

    a = ax[1]
    off = pt.values[np.triu_indices_from(pt.values, k=1)]
    a.axhspan(lo, hi, color="0.90", zorder=0, label=f"null 95% band")
    a.axhline(ceil, color=OI[2], ls="--", lw=1.3, label=f"measured ceiling {ceil:.2f}")
    a.axhline(off.mean(), color=OI[1], lw=1.3, label=f"mean tau-b {off.mean():.2f}")
    a.scatter(np.arange(len(off)) + 1, off, color=OI[0], s=26, zorder=4,
              label="rule pairs")
    a.annotate("", xy=(len(off) + 0.6, ceil), xytext=(len(off) + 0.6, off.mean()),
               arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.35"))
    a.text(len(off) + 0.9, (ceil + off.mean()) / 2,
           f"gap\n{ceil-off.mean():.2f}", fontsize=6, va="center", color="0.25")
    a.set_xlabel("negative-rule pair"); a.set_ylabel("tau-b")
    a.set_ylim(-0.1, 1.05); a.set_xlim(0.3, len(off) + 2.2)
    a.set_xticks(range(1, len(off) + 1))
    a.set_title("Agreement sits below what the measurement permits")
    a.legend(loc="lower left", fontsize=6.3)
    save(fig, "F4_agreement_vs_ceiling")


# --------------------------------------------------------------------------- #
# Fig 5  the central methodological result: score variance vs rank variance
# --------------------------------------------------------------------------- #
def fig5_variance(df):
    comp = decompose.decompose_both(df[df.metric == "mcc"])
    comp = comp.drop(index=[i for i in comp.index if i == "residual"], errors="ignore")
    comp = comp.sort_values("pct_rank_var")
    pretty = {"model_id": "pipeline", "negative_rule": "negative rule",
              "cluster_threshold": "clustering threshold", "seed": "seed",
              "metric": "metric",
              "cluster_threshold x negative_rule": "threshold x rule",
              "model_id x negative_rule": "pipeline x rule",
              "model_id x cluster_threshold": "pipeline x threshold"}
    labs = [pretty.get(i, i) for i in comp.index]
    y = np.arange(len(comp))

    fig, ax = plt.subplots(figsize=(5.4, 3.2), constrained_layout=True)
    ax.barh(y - 0.20, comp["pct_score_var"], height=0.38, color=OI[0],
            label="score variance")
    ax.barh(y + 0.20, comp["pct_rank_var"], height=0.38, color=OI[1],
            label="rank variance")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7)
    ax.set_xlabel("% of variance explained")
    for yi, (sv, rv) in enumerate(zip(comp["pct_score_var"], comp["pct_rank_var"])):
        if sv > 1.5:
            ax.text(sv + 0.8, yi - 0.20, f"{sv:.0f}", va="center", fontsize=6, color=OI[0])
        if rv > 1.5:
            ax.text(rv + 0.8, yi + 0.20, f"{rv:.0f}", va="center", fontsize=6, color=OI[1])
    # Mark the protocol main effects as STRUCTURAL zeros on the rank response. Ranks are
    # assigned within a cell, so any cell-defining factor has mean rank (K+1)/2 in every
    # group and zero between-group SS identically -- for any dataset. An earlier version
    # of this figure put those zeros in the title as if they were a measured result.
    for yi, i in enumerate(comp.index):
        if i in ("negative_rule", "cluster_threshold", "seed",
                 "cluster_threshold x negative_rule"):
            ax.text(1.2, yi + 0.20, "0 by construction", va="center",
                    fontsize=5.6, color="0.35", style="italic")
    ax.set_title(f"The protocol owns the score; only interactions move the ranking\n"
                 f"negative-set rule {comp.loc['negative_rule', 'pct_score_var']:.0f}% of score "
                 f"variance; pipeline x rule carries all the reordering",
                 fontsize=8.5)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.margins(x=0.16)
    save(fig, "F5_variance_decomposition")


# --------------------------------------------------------------------------- #
# Fig 6  the two-feature null baseline tracks the shortcut ladder
# --------------------------------------------------------------------------- #
def fig6_null(df):
    b = df[df.metric == "mcc"]
    lb = b.groupby(["negative_rule", "model_id"], as_index=False)["score"].mean()
    order = [r for r in RULES if r in set(lb.negative_rule)]
    best = lb.groupby("negative_rule")["score"].max()
    p16 = lb[lb.model_id == "P16"].set_index("negative_rule")["score"]
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.plot(x, [best[r] for r in order], marker="s", ms=5, lw=1.8, color=OI[0],
            label="best of 16 pipelines")
    ax.plot(x, [p16[r] for r in order], marker="o", ms=5, lw=1.8, color=OI[1],
            label="length + net charge (2 features)")
    ax.fill_between(x, [p16[r] for r in order], [best[r] for r in order],
                    color="0.88", zorder=0)
    for xi, r in zip(x, order):
        ax.annotate(f"{best[r]-p16[r]:.2f}", (xi, (best[r] + p16[r]) / 2),
                    fontsize=6, ha="center", color="0.3")
    ax.axhline(0, color="0.45", ls="--", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([RLAB[r] for r in order], fontsize=6.5)
    ax.set_ylabel("MCC")
    ax.set_title("Two features match sophisticated models where a shortcut exists,\n"
                 "and fall to chance where it is removed", fontsize=8.5)
    ax.legend(loc="lower left", fontsize=6.8)
    save(fig, "F6_null_baseline")


# --------------------------------------------------------------------------- #
# Fig 7  three artefacts of cluster-disjoint splitting
# --------------------------------------------------------------------------- #
def fig7_artefacts(naive):
    b = naive[naive.metric == "mcc"].copy()
    b["pos_frac"] = b.n_pos_test / b.n_test
    leak = b.groupby("cluster_threshold")["leak_fraction"].mean() * 100
    ratio = b.groupby("cluster_threshold")["cluster_ratio"].mean()
    thr = [t for t in THR if t in leak.index]
    x = np.arange(len(thr))

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)

    a = ax[0]
    a.bar(x, [leak[t] for t in thr], color=OI[1], width=0.62)
    a.set_xticks(x); a.set_xticklabels([TLAB[t] for t in thr])
    a.set_ylabel("% of test sequences with an\nabove-threshold hit in train")
    a.set_title("(a) residual leakage rises\nas the threshold tightens", fontsize=7.8)
    for xi, t in zip(x, thr):
        a.text(xi, leak[t], f"{leak[t]:.1f}", ha="center", va="bottom", fontsize=6)
    a.margins(y=0.18)

    a = ax[1]
    for k, r in enumerate(RULES):
        sub = b[b.negative_rule == r].groupby("cluster_threshold")["pos_frac"].mean()
        a.plot(x, [sub.get(t, np.nan) for t in thr], marker="o", ms=4, lw=1.4,
               color=OI[k % len(OI)], label=RLAB[r].replace("\n", " "))
    a.axhline(0.5, color="0.4", ls="--", lw=0.8)
    a.set_ylim(0, 1.05)
    a.set_xticks(x); a.set_xticklabels([TLAB[t] for t in thr])
    a.set_ylabel("test-set positive fraction")
    a.set_title("(b) class balance collapses,\nand the direction depends on the rule",
                fontsize=7.8)
    a.legend(fontsize=5.6, loc="center left")

    a = ax[2]
    a.bar(x, [ratio[t] for t in thr], color=OI[2], width=0.62)
    a.set_xticks(x); a.set_xticklabels([TLAB[t] for t in thr])
    a.set_ylabel("cluster count / sequence count")
    a.set_title("(c) but the reported diagnostic\nmoves the reassuring way", fontsize=7.8)
    for xi, t in zip(x, thr):
        a.text(xi, ratio[t], f"{ratio[t]:.2f}", ha="center", va="bottom", fontsize=6)
    a.margins(y=0.18)

    fig.suptitle("Cluster-disjoint splitting does not deliver what it is assumed to",
                 fontsize=9)
    save(fig, "F7_splitting_artefacts")


# --------------------------------------------------------------------------- #
# Fig 8  correcting the balance strengthens, not weakens, the finding
# --------------------------------------------------------------------------- #
def fig8_balanced(naive, bal):
    def stats(d):
        kmat = rankmat(d, "negative_rule")
        kmat = kmat[[c for c in RULES if c in kmat.columns]]
        f = ranks.top1_flip_rate(kmat)
        tau = ranks.mean_pairwise_tau(kmat)
        ceil = seed_ceiling(d)
        b = d[d.metric == "mcc"].copy(); b["pf"] = b.n_pos_test / b.n_test
        return {"tau": tau, "ceil": ceil, "norm": tau / ceil,
                "flips": f["n_flips"], "pairs": f["n_pairs"],
                "winners": len(set(f["winners"].values())),
                "pf_min": b.pf.min(), "pf_max": b.pf.max()}
    A, B = stats(naive), stats(bal)

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    x = np.arange(2); names = ["largest-first", "balance-\npreserving"]

    a = ax[0]
    a.bar(x - 0.18, [A["pf_min"], B["pf_min"]], width=0.34, color=OI[0], label="minimum")
    a.bar(x + 0.18, [A["pf_max"], B["pf_max"]], width=0.34, color=OI[1], label="maximum")
    a.axhline(0.5, color="0.4", ls="--", lw=0.8)
    for xi, v in zip(x - 0.18, [A["pf_min"], B["pf_min"]]):
        a.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    for xi, v in zip(x + 0.18, [A["pf_max"], B["pf_max"]]):
        a.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=6.5)
    a.set_ylabel("test-set positive fraction"); a.set_ylim(0, 1.18)
    a.set_title("(a) the artefact is removed", fontsize=7.8)
    a.legend(fontsize=6, loc="upper left")

    a = ax[1]
    cols = [OI[0], OI[2]]
    a.bar(x, [A["norm"], B["norm"]], width=0.5, color=cols)
    a.axhline(0.85, color=OI[1], ls="--", lw=1.1)
    a.text(0.5, 0.86, "pre-registered threshold (0.85)", fontsize=6,
           color=OI[1], va="bottom", ha="center", transform=a.get_xaxis_transform())
    for xi, v in zip(x, [A["norm"], B["norm"]]):
        a.text(xi, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=7,
               fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(names, fontsize=6.5)
    a.set_ylabel("tau-b / measured ceiling"); a.set_ylim(0, 1.12)
    a.set_title("(b) agreement drops below\nthe threshold", fontsize=7.8)

    a = ax[2]
    a.bar(x, [A["flips"], B["flips"]], width=0.5, color=OI[2])
    for xi, v, w in zip(x, [A["flips"], B["flips"]], [A["winners"], B["winners"]]):
        a.text(xi, v, f"{v}/10\n{w} winners", ha="center", va="bottom", fontsize=6.5)
    a.set_xticks(x); a.set_xticklabels(names, fontsize=6.5)
    a.set_ylabel("rule pairs naming a different winner"); a.set_ylim(0, 12)
    a.set_title("(c) more disagreement, not less", fontsize=7.8)

    fig.suptitle("Correcting the class-balance artefact strengthens the finding", fontsize=9)
    save(fig, "F8_balanced_comparison")


# --------------------------------------------------------------------------- #
# Fig 9  design simulation
# --------------------------------------------------------------------------- #
def fig9_design():
    pc, ce = TAB / "power_curve.csv", TAB / "power_followup_ceiling.csv"
    if not (pc.exists() and ce.exists()):
        print("  (F9 skipped: power CSVs absent)"); return
    p, c = pd.read_csv(pc), pd.read_csv(ce)
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5), constrained_layout=True)
    a = ax[0]
    a.plot(c["sigma_e"], c["ceiling_tau_b"], marker="o", ms=4, lw=1.6, color=OI[0])
    a.axhline(1.0, color="0.5", ls=":", lw=0.8)
    a.text(0.02, 1.004, "naive reference (1.0)", fontsize=6, color="0.4")
    a.set_xlabel("run-to-run noise"); a.set_ylabel("maximum achievable tau-b")
    a.set_title("(a) agreement has a ceiling below 1", fontsize=7.8)
    a = ax[1]
    a.plot(p["population_tau_b"], p["power_reject_H1"], marker="o", ms=4, lw=1.6,
           color=OI[1])
    a.axhline(0.8, color="0.4", ls="--", lw=0.8)
    a.text(0.32, 0.82, "80% power", fontsize=6, color="0.35")
    a.set_xlabel("true mean pairwise tau-b"); a.set_ylabel("power to detect disagreement")
    a.set_title("(b) minimum detectable disagreement", fontsize=7.8)
    save(fig, "F9_design_simulation")


# --------------------------------------------------------------------------- #
# Fig 10  negative draw vs negative rule, against the seed ceiling
# --------------------------------------------------------------------------- #
def fig10_draw_sensitivity():
    s = pd.read_csv(TAB / "draw_sensitivity_summary.csv").iloc[0]
    per = pd.read_csv(TAB / "draw_vs_seed_vs_rule.csv").sort_values("normalised")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5),
                           gridspec_kw={"width_ratios": [1, 1.25]},
                           constrained_layout=True)

    # (a) the three sources of movement, on the normalised scale
    a = ax[0]
    names = ["negative rule", "negative draw", "seed only\n(the ceiling)"]
    vals = [s.norm_rules, s.norm_draws, 1.0]
    cols = [OI[1], OI[0], "0.55"]
    y = np.arange(3)
    a.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        a.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=7)
    a.set_yticks(y)
    a.set_yticklabels(names)
    a.set_xlim(0, 1.15)
    a.set_xlabel("rank agreement / seed ceiling")
    a.axvline(1.0, color="0.35", lw=0.7, ls="--")
    a.set_title("(a) what moves the ranking", loc="left")

    # (b) per-rule draw agreement; the matched rules are the unstable ones
    a = ax[1]
    y = np.arange(len(per))
    a.barh(y, per["normalised"], height=0.62,
           color=[OI[1] if v < 0.8 else OI[0] for v in per["normalised"]])
    for yi, v in zip(y, per["normalised"]):
        a.text(v + 0.015, yi, f"{v:.3f}", va="center", fontsize=7)
    a.set_yticks(y)
    a.set_yticklabels([RLAB[r].replace("\n", " ") for r in per["rule"]], fontsize=6.8)
    a.axvline(1.0, color="0.35", lw=0.7, ls="--")
    a.set_xlim(0, 1.18)
    a.set_xlabel("draw-to-draw agreement / seed ceiling")
    a.set_title("(b) redrawing the negatives, within a rule", loc="left")

    fig.suptitle("The rule dominates the draw about two to one -- except where "
                 "negatives are matched per positive", fontsize=8.5)
    save(fig, "F10_draw_sensitivity")


# --------------------------------------------------------------------------- #
# Fig 11  no splitter wins on all three targets
# --------------------------------------------------------------------------- #
def fig11_splitter_tradeoff():
    d = pd.read_csv(TAB / "splitter_stresstest.csv")
    order = ["largest_first", "balanced", "random", "leak_aware"]
    lab = {"largest_first": "largest-first\n(pre-registered)",
           "balanced": "quota-driven\n(ours)",
           "random": "uniformly random",
           "leak_aware": "leakage-aware"}

    fig, ax = plt.subplots(figsize=(4.8, 3.5), constrained_layout=True)
    for i, sp in enumerate(order):
        g = d[d.splitter == sp]
        ax.scatter(g["balance_error"], g["leak_fraction"], s=14, alpha=0.35,
                   color=OI[i], linewidths=0)
        ax.scatter([g["balance_error"].mean()], [g["leak_fraction"].mean()],
                   s=95, color=OI[i], edgecolor="white", linewidths=1.0, zorder=5)
        ax.annotate(lab[sp],
                    xy=(g["balance_error"].mean(), g["leak_fraction"].mean()),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=7, color=OI[i], fontweight="bold")
    ax.plot(0, 0, marker="*", ms=12, color="0.25", zorder=6)
    ax.annotate("ideal", xy=(0, 0), xytext=(11, -3), textcoords="offset points",
                fontsize=7, color="0.25")
    ax.set_xlabel("class-balance error  (0 is perfect)")
    ax.set_ylabel("residual pairwise leakage  (0 is perfect)")
    ax.set_title("Small markers are cells, large markers their means.\n"
                 "No ordering reaches the origin.", loc="left", fontsize=7.8)
    ax.set_xlim(-0.02, 0.56)
    ax.set_ylim(-0.02, 0.45)
    save(fig, "F11_splitter_tradeoff")


# --------------------------------------------------------------------------- #
# Fig 12  sequence-level intervals are too narrow, and by how much
# --------------------------------------------------------------------------- #
def fig12_cluster_intervals():
    d = pd.read_csv(TAB / "cluster_intervals.csv")
    d = d[d.metric == "mcc"]
    g = (d.groupby(["arm", "level"])
          .agg(seq=("seq_width", "mean"), clu=("clu_width", "mean"),
               ratio=("width_ratio", "mean"), n=("n_test", "first"),
               k=("n_clusters", "first"))
          .reset_index())
    g["per"] = g["n"] / g["k"]
    g = g.sort_values(["arm", "ratio"], ascending=[True, False]).reset_index(drop=True)
    labs = [(RLAB[l].replace("\n", " ") if l in RLAB else l) for l in g["level"]]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0),
                           gridspec_kw={"width_ratios": [1.3, 1]},
                           constrained_layout=True)
    a = ax[0]
    y = np.arange(len(g))
    # grey against blue rather than two blues: the lightness gap keeps the pair
    # separable in greyscale and under colour-vision deficiency
    a.barh(y - 0.19, g["seq"], height=0.36, color="0.72", label="resampling sequences")
    a.barh(y + 0.19, g["clu"], height=0.36, color=OI[0], label="resampling clusters")
    a.set_yticks(y)
    a.set_yticklabels(labs, fontsize=6.8)
    a.set_xlabel("mean 95% interval width on MCC")
    a.set_xlim(0, 0.33)
    for yi in y:
        wider = max(g["seq"].iloc[yi], g["clu"].iloc[yi])
        a.text(wider + 0.006, yi, f"x{g['ratio'].iloc[yi]:.2f}",
               va="center", fontsize=6.4, color="0.3")

    # Direct labels on the topmost pair instead of a legend box, which collided
    # with the longest bars whichever corner it was placed in.
    top = len(g) - 1
    a.text(0.155, top + 0.19, "resampling clusters", va="center",
           fontsize=6.6, color=OI[0], fontweight="bold")
    a.text(0.155, top - 0.19, "resampling sequences", va="center",
           fontsize=6.6, color="0.45", fontweight="bold")

    # Separate the arms. pandas sorts "A" before "B", so Arm A occupies the lower
    # rows and Arm B the upper ones -- an earlier version labelled these the wrong
    # way round.
    n_a = int((g["arm"] == "A").sum())
    a.axhline(n_a - 0.5, color="0.6", lw=0.7)
    a.text(0.327, (n_a - 1) / 2, "Arm A", rotation=90, va="center", ha="right",
           fontsize=7, color="0.35")
    a.text(0.327, n_a + (len(g) - n_a - 1) / 2, "Arm B", rotation=90,
           va="center", ha="right", fontsize=7, color="0.35")
    a.set_title("(a) interval width by level", loc="left")

    a = ax[1]
    for i, arm in enumerate(["B", "A"]):
        h = g[g.arm == arm]
        a.scatter(h["per"], h["ratio"], s=42, color=OI[i], zorder=4, label=f"Arm {arm}")
    a.axhline(1.0, color="0.35", lw=0.7, ls="--")
    a.set_xlabel("test sequences per cluster")
    a.set_ylabel("width ratio, cluster / sequence")
    a.legend(loc="upper left")
    a.set_title("(b) the correction tracks redundancy", loc="left")
    save(fig, "F12_cluster_intervals")


# --------------------------------------------------------------------------- #
# Fig 13  class provenance is asymmetric under two of the five rules
# --------------------------------------------------------------------------- #
def fig13_provenance():
    d = pd.read_csv(TAB / "esm_exposure_by_rule.csv")
    p = d.pivot(index="rule", columns="klass", values="pct").reindex(RULES)

    fig, ax = plt.subplots(figsize=(5.2, 2.9), constrained_layout=True)
    x = np.arange(len(p))
    ax.bar(x - 0.19, p["negatives"], width=0.38, color=OI[1], label="negatives")
    ax.bar(x + 0.19, p["positives"], width=0.38, color=OI[0], label="positives")
    for xi, v in zip(x, p["negatives"]):
        # 0.06% and 0.02% are not zero; rounding them to "0" would overstate the point
        txt = f"{v:.0f}" if v >= 1 else f"{v:.2f}"
        ax.text(xi - 0.19, v + 2.5, txt, ha="center", fontsize=6.6, color=OI[1])
    for xi, v in zip(x, p["positives"]):
        ax.text(xi + 0.19, v + 2.5, f"{v:.1f}", ha="center", fontsize=6.6, color=OI[0])
    ax.set_xticks(x)
    ax.set_xticklabels([RLAB[r] for r in p.index], fontsize=6.8)
    ax.set_ylabel("% of class verbatim in reviewed UniProt")
    ax.set_ylim(0, 114)
    ax.legend(loc="upper right")
    ax.set_title("Under the two proteome-derived rules the classes differ in\n"
                 "provenance, not only in activity", loc="left", fontsize=7.8)
    save(fig, "F13_provenance")


def main():
    naive, bal = load(balanced=False), load(balanced=True)
    print(f"largest-first {len(naive)} rows | balanced {len(bal)} rows")
    print("figures (balanced split is the primary presentation):")
    fig1_shortcut_ladder()
    fig2_landscape(bal)
    fig3_bump(bal)
    fig4_agreement(bal)
    fig5_variance(bal)
    fig6_null(bal)
    fig7_artefacts(naive)      # artefacts are a property of the naive splitter
    fig8_balanced(naive, bal)
    fig9_design()
    # Figures for the supplementary analyses. Each reads its own table rather
    # than the run records, so these redraw without re-fitting anything.
    fig10_draw_sensitivity()
    fig11_splitter_tradeoff()
    fig12_cluster_intervals()
    fig13_provenance()
    print(f"\nwritten to {FIG}")


if __name__ == "__main__":
    main()

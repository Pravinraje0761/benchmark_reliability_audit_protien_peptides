---
title: Pre-registration
tags: [project, preregistration]
date: 2026-10-04
status: draft-unlocked
---

# 04 — Pre-registration

> **Lock this file before running Arm A.** Commit it, timestamp it, optionally post to OSF.
> For a paper whose entire thesis is "evaluation practice is unreliable," running without
> a pre-registration is self-refuting — a reviewer will say so, and they will be right.

## Status

- [ ] Hypotheses + thresholds final
- [ ] Model zoo frozen (`code/configs/models.yaml` hashed)
- [ ] Benchmarks acquired and hashed
- [ ] **LOCKED** — date: ______  commit: ______

## Primary hypothesis and decision rule

**H1.** Across the AMP-binary task, mean pairwise Kendall τ-b between benchmark rankings
of the 12 fixed pipelines is **< 0.8**.

- Estimator: mean of all 10 pairwise τ-b values (5 benchmarks), Arm A, primary metric
  = each benchmark's own declared headline metric.
- Uncertainty: BCa bootstrap over sequences within each benchmark's test set, 2,000
  resamples, recomputing ranks each resample. Report the 95% CI of the mean τ-b.
- **Confirmatory if** the upper CI bound < 0.8.
- **Falsified if** the lower CI bound ≥ 0.8.
- **Inconclusive** otherwise — and that is reported as inconclusive, not spun.

## Secondary hypotheses

- **H2.** Fixing the metric to MCC across all benchmarks does not raise mean τ-b above 0.8.
- **H3.** In the Arm B variance decomposition, the negative-set variance component exceeds
  the clustering-threshold component.
- **H4.** The seed component is < 25% of the largest protocol component.
- **H5.** Mean τ-b of *representation* ranks (marginalising over learners) < mean τ-b of
  *learner* ranks (marginalising over representations).

## Pre-specified secondary outcomes

- Count of pairwise rank inversions per benchmark pair (and as a fraction of the 66
  possible pairs among 12 models).
- **Top-1 flip rate**: fraction of benchmark pairs naming a different best model. This is
  the number that goes in the abstract, because it is what a practitioner actually feels.
- Rank range per model (max rank − min rank across benchmarks).
- M12 (null baseline) rank per benchmark, as a shortcut detector.

## What would make us wrong

Stated in advance, honestly:

- **Not enough benchmarks.** 5 benchmarks → 10 pairs is thin for a CI on mean τ-b. Mitigation:
  bootstrap over sequences, not over benchmarks; report per-pair τ-b in full, never only the mean.
- **Implementation error masquerading as benchmark disagreement.** Mitigation: reproduce each
  benchmark's *own reported baseline number* to within a pre-declared tolerance (±0.03 on its
  headline metric) before that benchmark's results count. A benchmark we cannot reproduce is
  reported as a reproduction failure — itself a finding — and excluded from the τ-b pool.
- **Our pooled corpus (Arm B) is itself a 7th opinionated benchmark.** True, and we say so.
  Arm B is positioned as a *decomposition instrument*, not as the correct benchmark.
- **Low power on the regression tasks.** Pre-declared as exploratory.

## Deviations log

| Date | Deviation | Reason |
|------|-----------|--------|
| | | |

---

# Revision 1 — 2026-10-05, after [[E00 Design simulation]]

Still **UNLOCKED**. All changes below were made before any benchmark was acquired and
before any real ranking was observed, so the freeze discipline holds. The git log is the
evidence: the E00 commit precedes every data commit.

## What the simulation changed

### R1. Zoo grows 12 -> 20 pipelines

Minimum detectable disagreement (MDD) at 80% power improves from tau-b 0.575 to 0.646.
More models beats more benchmarks: estimator SD falls 46% going 8 to 20 models versus 21%
going 3 to 6 benchmarks, and a model costs one fit on cached features while a benchmark
costs acquisition, a loader and a reproduction gate. See [[02 Model zoo]] for M13-M20.

### R2. The primary hypothesis now states its sensitivity

**H1 (revised).** Across the AMP-binary task, mean pairwise Kendall tau-b between benchmark
rankings of the 20 fixed pipelines is < 0.8.

- Decision rule unchanged: **confirmatory if the upper 95% bound < 0.8**, falsified if the
  lower bound >= 0.8, inconclusive otherwise.
- **Added, and binding:** with 20 pipelines and 5 benchmarks the study has 80% power only
  when the true mean tau-b is **<= 0.646**. Between 0.65 and 0.80 the study is
  underpowered and an inconclusive result there will be reported as inconclusive, with the
  MDD quoted. This is stated now so it cannot later be presented as a near-miss.
- Null reference for the mean statistic: 0.002 +/- 0.069, 95% [-0.115, +0.152] at 20
  models x 5 benchmarks. The earlier worry about tau-b noise applied to a *single* pair
  (SD 0.22); for the mean over 10 pairs it is 0.069. Per-pair values keep the wide band.

### R3. Co-primary outcome: normalised agreement

tau-b cannot reach 1.0 even between two seeds of the *same* benchmark under the *same*
protocol, because run noise reorders near-tied models. Simulated ceiling: 0.955 at 5% run
noise, 0.910 at 10%. Judging observed tau-b against 1.0 therefore overstates disagreement.

**Co-primary.**

    normalised agreement = mean between-benchmark tau-b / mean within-benchmark seed ceiling

- The ceiling is measured, not assumed: both arms run 5 seeds, so tau-b between seed pairs
  within a benchmark gives it at zero extra cost. Implemented as
  `ranks.ceiling_from_seeds` and `ranks.normalised_agreement` (4 tests).
- Interpretation: near 1.0 means the benchmarks agree about as well as a benchmark agrees
  with itself, i.e. interchangeable. Near 0 means protocol, not noise, drives the ranking.
- **Threshold: normalised agreement < 0.85 supports H1.** Chosen before data.
- The raw tau-b and the ceiling are both reported alongside it. No single number is allowed
  to stand on its own.

### R4. Top-1 flip rate is demoted to descriptive

It tracks tau-b sensitively (0.07 flips at tau 0.95 rising to 0.60 at tau 0.50) but its
across-study SD is about 0.32 with only 10 benchmark pairs. It stays in the abstract
because it is what a practitioner feels; it carries **no** confirmatory weight.

### R5. Deferred

Re-estimate run noise from the real seed variance once Arm A completes, then re-run E00
with that value. The ceiling reported in the paper must be the measured one. If real seed
noise turns out larger than 10%, the achievable ceiling drops below 0.91 and the
normalised-agreement threshold in R3 may need restating — which would itself be a finding
worth reporting, not a problem to hide.

## Deviations log

| Date | Deviation | Reason |
|------|-----------|--------|
| 2026-10-05 | Zoo 12 -> 20 pipelines | [[E00 Design simulation]]: MDD 0.575 -> 0.646; models are the cheap axis |
| 2026-10-05 | Added normalised agreement as co-primary, threshold 0.85 | tau-b has a noise ceiling below 1.0; raw tau-b overstates disagreement |
| 2026-10-05 | H1 now quotes its MDD explicitly | Study is underpowered in the 0.65-0.80 band; disclosed in advance |
| 2026-10-05 | Top-1 flip rate demoted to descriptive | Across-study SD about 0.32 at 10 pairs |

---

# Revision 2 - 2026-10-05, deviations forced by measured compute cost

All three were found by direct measurement during execution, before any result was analysed,
and none changes what is being compared - only how affordably it is computed.

| Date | Deviation | Measurement that forced it |
|------|-----------|----------------------------|
| 2026-10-05 | `svm_rbf` is now an **approximate** RBF SVM: Nystroem kernel map (512 components) + LinearSVC with sigmoid calibration, in place of exact `SVC(probability=True)` | Exact SVC did not finish one cell in **9 minutes** on 8,299 training points, because Platt scaling refits the kernel machine five times. Over 100 cells that is 16+ hours for a single pipeline. The approximation runs in **11.2 s** and returns calibrated probabilities on the same 0.5 threshold as every other pipeline. sklearn 1.9 independently deprecates `SVC(probability=True)` in favour of `CalibratedClassifierCV`. |
| 2026-10-05 | Worker threads capped at **8** (`n_jobs=8`, `BRA_N_JOBS` overridable) instead of `-1`; XGBoost uses `tree_method="hist"` | XGBoost on a **7-feature** problem took **165 s** with `n_jobs=-1` on a 32-thread machine - coordination overhead, not computation. With the cap it takes **1.7 s**, a 97x difference. Per-cell cost for the eight embedding-free pipelines fell from 201 s to 28 s. |
| 2026-10-05 | End-to-end CNN/BiLSTM and ProtT5-XL excluded; representation axis covered by four ESM-2 scales | Their cost cannot be amortised by feature caching, and no GPU is available. Recorded in Revision 1 context; restated here with the others. |

**Why these are not results-contingent.** Each was decided from a timing measurement on a
single cell of one corpus, with no metric comparison across pipelines and no ranking
computed. The grid had not been run. Nothing here was chosen because it improved an outcome.

**What a reviewer should check.** The Nystroem substitution is the only one that changes a
model rather than its scheduling. If it matters, the exact SVC can be run on a reduced grid
to confirm the approximation preserves the pipeline rank; we note this as available on
request rather than claiming it has been done.

---

# Revision 3 — 2026-10-08, deviations found during manuscript audit

Written **after** the analysis was complete. Everything below is disclosed because it was
missing from the logs above, not because it changes a result. Revisions 1 and 2 were
written before the relevant data existed; this one was not, and is labelled accordingly.

## D1. The unit of comparison in H1 changed — the consequential deviation

H1 as written above compares **five independently published AMP benchmarks**. That
comparison was never executed. The published resources do not share a task definition, a
label convention, or a redistributable positive set, and the pre-specified reproduction
gate (reproduce each benchmark's own reported baseline to ±0.03 before its results count)
cannot be applied to resources whose headline numbers are not computed on a common target.

What was executed instead: Arm B compares **five negative-set rules** on one harmonised
corpus, and Arm A compares **ten biological tasks** from one curated collection. The
estimator, the 0.8 / 0.85 thresholds and the MDD statement carried over unchanged, but the
hypothesis is tested over a different population of evaluation levels than the one named.

Consequence for the paper: H1's status is **pre-specified, not pre-registered**, and the
original cross-benchmark question remains open. This is now stated in the manuscript
(Methods, "The analysis plan, and every deviation from it") and listed as deviation 1 there.

## D2. The pre-registration was never locked, and there is no git log

The LOCKED checkbox above was never ticked and no date or commit was recorded. Revision 1
asserts "the git log is the evidence" — **the project was not under version control**, so
that evidence does not exist. The only ordering evidence is file modification time:

| Artefact | mtime |
|---|---|
| this plan, last revision before execution | 2026-10-05 20:40 |
| Arm B results (pre-registered splitter) | 2026-10-06 13:49 |
| Arm B results (balance-preserving) | 2026-10-07 02:36 |
| Arm A results | 2026-10-08 01:43 |

mtimes are mutable and are weak evidence. The manuscript now says so explicitly rather than
claiming a guarantee the record cannot support. A repository initialised today can
timestamp everything from today forward; it cannot backdate what came before, and we do
not present it as if it could.

## D3. H5 and the reproduction gate were not reported

H5 (representation ranks less stable than learner ranks) is computed by the harness but is
not a claim in the manuscript. The ±0.03 reproduction gate became inapplicable with D1.
Both are recorded here so the plan and the paper can be diffed without surprises.

## D4. Post hoc additions, already labelled as such in the manuscript

- The balance-preserving splitter and the entire second grid (built after the null).
- Residual pairwise leakage measurement (replaced an assertion that failed on real data).
- The exact-vs-Nyström SVM validation that Revision 2 listed as "available on request".
- The matched top-1 flip-rate ceiling, added after a referee-style audit of our own
  reasoning: we had been quoting a flip rate against an implicit floor of zero while
  insisting elsewhere that τ-b be read against a measured ceiling.

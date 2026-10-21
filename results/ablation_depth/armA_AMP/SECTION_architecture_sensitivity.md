# §X Architecture sensitivity: depth and activation

*Supplementary section for the benchmark reliability audit. Numbers from
`runs.csv`, `confusion_matrices.json` and `summary.md` in this directory;
120 fits, all converged.*

---

## X.1 What varies, and what does not

This sweep varies **one thing**: the hidden-layer depth and activation function
of the MLP learner, on the antimicrobial (AMP) task of Arm A under its native
homology-aware split, with mean-pooled ESM-2 35M embeddings as the
representation. Everything else is held at the values the main grid used —
width 256 per layer, `max_iter=500`, early stopping on, the same
`StandardScaler` front end, the same features, the same split, the same five
seeds.

**This is supplementary to a pre-registered grid whose hyperparameters were
frozen before corpus construction, and it does not revise it.** That freeze is
what prevents protocol preference from being confounded with search effort,
which is the paper's central methodological commitment. The main results are
unchanged by anything below, and these numbers must not be merged into them.

Grid: depths 1, 3, 4, 5, 6, 7 × activations {ReLU, tanh, logistic, identity} ×
5 seeds = **120 fits**, on 10,266 training and 2,566 test sequences. Width is
held constant at 256 units per layer. Constant width confounds depth with
capacity — a 7-layer net here has 4.2× the parameters of the published
1-layer configuration — so the capacity-controlled pyramid policy is available
as a sensitivity check and is not the primary reported here.

## X.2 The reproducibility floor for this comparison

Across seed replicates at fixed depth and activation, MCC varied by
**0.0156** (median within-cell SD; maximum 0.0242). Every difference below is
stated against that floor.

This subsection is not optional in this paper specifically. Our own headline is
that reseeding alone changes the top-ranked pipeline in 54.5% of comparisons. A
depth sweep reporting one run per cell, in the same manuscript, would be
refuted by its own argument.

## X.3 Depth

**Table X.1** — mean MCC (SD across 5 seeds) by depth and activation.

| Depth | ReLU | tanh | logistic | identity |
|---|---|---|---|---|
| **1 (published)** | **0.3291** (0.0204) | 0.3362 (0.0165) | 0.3264 (0.0081) | 0.2479 (0.0225) |
| 3 | 0.3218 (0.0213) | 0.3164 (0.0131) | 0.3126 (0.0106) | 0.2514 (0.0093) |
| 4 | 0.3244 (0.0130) | 0.3193 (0.0242) | 0.3199 (0.0078) | 0.2819 (0.0239) |
| 5 | 0.3231 (0.0207) | 0.3235 (0.0123) | 0.3187 (0.0185) | 0.2589 (0.0147) |
| 6 | 0.3300 (0.0162) | 0.3263 (0.0065) | 0.3087 (0.0198) | 0.2538 (0.0149) |
| 7 | 0.3387 (0.0198) | 0.3125 (0.0206) | 0.3219 (0.0132) | 0.2685 (0.0151) |

Best-minus-worst across depth, averaged over activations and seeds: **0.0108**,
which is **0.69×** the seed noise floor.

**Depth did not move this task.** The spread across six depths is smaller than
the spread across five reseedings of the same depth. The nominal best cell
(depth 7, ReLU, 0.3387) beats the published configuration (depth 1, ReLU,
0.3291) by 0.0096 — again below the 0.0156 floor, and on a grid of 24
depth × activation cells the top cell is more often the luckiest than the best.
We report this as a **null result on depth**, in those words.

What makes it unsurprising is *not* that the representation is linearly
separable — §X.4 shows it is not. It is that one hidden layer of 256 units
already supplies whatever non-linear capacity this task can use. Adding layers
multiplies parameters (123,650 → 518,402) without adding usable capacity
against 10,266 training examples.

## X.4 Activation

**Table X.2** — mean MCC by activation, marginalised over depth.

| Activation | Mean MCC | Δ vs ReLU | SD across seeds |
|---|---|---|---|
| ReLU | 0.3278 | — | 0.0180 |
| tanh | 0.3224 | −0.0055 | 0.0170 |
| logistic | 0.3180 | −0.0098 | 0.0139 |
| **identity** | **0.2604** | **−0.0674** | 0.0198 |

The three non-linear activations are **indistinguishable from one another**:
ReLU leads tanh by 0.0055 and logistic by 0.0098, both inside the 0.0156 noise
floor. Choosing among them is not a decision this experiment can inform.

The `identity` row is the control, and it is the informative one. A stack of
identity-activated layers collapses to a single linear map regardless of depth,
so its gap to the best non-linear row measures how much non-linearity the task
needed. That gap is **0.0674, or 4.3× the noise floor** — comfortably real.

The two results together are the point: **non-linearity matters and depth does
not.** The decision surface this task needs cannot be reached by a linear map
on ESM-2 35M features, but one hidden layer reaches it, and six more add
nothing. There is no vanishing-gradient signature either — the saturating
activations (tanh, logistic) do not degrade with depth relative to ReLU, which
is what one would expect to see if depth were failing for optimisation reasons
rather than simply being unnecessary.

## X.5 Confusion-matrix structure

**Table X.3** — confusion matrices summed over 5 seeds, raw counts
(12,830 predictions per configuration).

| | pred. negative | pred. positive | sensitivity | specificity |
|---|---|---|---|---|
| **Published (depth 1, ReLU)** | | | 0.5573 | 0.7639 |
| actual negative | 4,904 | 1,516 | | |
| actual positive | 2,838 | 3,572 | | |
| **Best cell (depth 7, ReLU)** | | | 0.5895 | 0.7445 |
| actual negative | 4,780 | 1,640 | | |
| actual positive | 2,631 | 3,779 | | |
| **Linear control (depth 1, identity)** | | | 0.4413 | 0.7900 |
| actual negative | 5,072 | 1,348 | | |
| actual positive | 3,581 | 2,829 | | |

The accuracy difference is already in Table X.1; what matters here is whether
the errors **relocate**. They do. Going from the published configuration to the
nominal best cell trades **207 false negatives for 124 false positives**, a net
reduction of 83 errors out of 12,830 predictions (0.6%). Sensitivity rises
0.5573 → 0.5895 while specificity falls 0.7639 → 0.7445.

So the deeper network is not uniformly better — it is slightly less
conservative. For antimicrobial screening, where a missed active peptide costs
more than a false lead, that trade runs in the useful direction; but at 83 net
errors it is well inside what reseeding moves, and no decision should rest on
it.

The linear control shows the same axis more starkly: it is the most
conservative of the three (specificity 0.7900, the highest) and pays for it
with sensitivity 0.4413, missing 3,581 of 6,410 actives. Its deficit is
concentrated almost entirely in false negatives, which is the failure mode that
matters for this task.

## X.6 Why the published configuration is retained

**Parsimony under a null.** No depth exceeded the seed noise floor, so the
shallowest configuration is kept: it has **123,650 parameters against 518,402**
at depth 7, a 4.2× reduction, with no accuracy cost this experiment can detect
(Δ MCC 0.0096, 0.62× the noise floor). The published `(256,)` ReLU MLP is
retained on efficiency grounds, and the choice is now evidenced rather than
inherited.

This is a case where "best" would have been the wrong thing to report. Depth 7
with ReLU does score highest; saying so without the noise floor beside it would
have turned a 0.0096 difference into an architectural recommendation, which is
precisely the error this paper exists to document.

## X.7 Limitations of this sweep

- **One task, one representation.** AMP with ESM-2 35M embeddings. The paper's
  own Arm A result is that rankings transfer poorly across tasks, so this null
  should not be assumed to hold on the other nine.
- **Constant width confounds depth with capacity.** Depth and parameter count
  move together here; the pyramid policy separates them and is available but
  was not run as primary.
- **Learning rate, dropout and patience were held fixed across depths.** Deeper
  networks are more sensitive to all three, so a per-depth tuned sweep could
  differ — and would answer a different question than this one, which is
  whether depth helps *at the paper's own settings*.
- **Supplementary to a pre-registered grid.** It does not revise the main
  results and is not merged into them.
- **One optimisation outlier.** Depth 6 / identity / seed 0 required 2,645 s
  against a 9 s median for its depth. It converged and scored in line with the
  other identity cells; we report it because a configuration that is 300×
  slower to fit is a fact about that configuration, not noise to drop.

---

### Reporting rules applied

Confusion matrices are raw integer counts, rates derived from them. Every
scalar carries its across-seed SD. All 120 runs converged; none were dropped.
"Best" is never claimed from a single seed. Differences below the noise floor
are reported as null results, in those words.

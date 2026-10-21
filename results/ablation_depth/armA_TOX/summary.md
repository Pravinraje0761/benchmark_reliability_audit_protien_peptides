# Depth x activation ablation -- Benchmark reliability audit (peptide property prediction)

Model(s): mlp:esm2_35M:armA:TOX
Grid: depths [1, 3, 4, 5, 6, 7] x activations ['identity', 'logistic', 'relu', 'tanh'] x seeds [0, 1, 2, 3, 4]
Width policy: constant
Total runs: 120

## Seed noise floor

Standard deviation across seed replicates, within each fixed
(model, depth, activation) cell. A depth effect smaller than this is
not a depth effect.

- median within-cell SD of MCC: 0.0286
- max within-cell SD of MCC:    0.0516

## Mean MCC by depth x activation (averaged over seeds)

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.3282 |     0.3599 | 0.3662 | 0.3717 |
|       3 |     0.3223 |     0.3352 | 0.3601 | 0.3607 |
|       4 |     0.3257 |     0.3263 | 0.356  | 0.3476 |
|       5 |     0.3177 |     0.3124 | 0.3486 | 0.3422 |
|       6 |     0.3226 |     0.3098 | 0.3492 | 0.3364 |
|       7 |     0.3221 |     0.2883 | 0.3172 | 0.3172 |

## Mean balanced accuracy by depth x activation

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.6634 |     0.6764 | 0.6787 | 0.6811 |
|       3 |     0.6606 |     0.6616 | 0.673  | 0.6732 |
|       4 |     0.6607 |     0.6586 | 0.6725 | 0.6667 |
|       5 |     0.6574 |     0.6535 | 0.6651 | 0.6652 |
|       6 |     0.6611 |     0.6538 | 0.6683 | 0.6613 |
|       7 |     0.6598 |     0.6145 | 0.6533 | 0.6515 |

## Best cell

- model `mlp:esm2_35M:armA:TOX`, depth 1, activation `tanh` -> mean MCC 0.3717

Treat 'best' as provisional until you have checked it against the noise
floor above. On a grid this size the top cell is frequently the luckiest
cell rather than the best architecture.

## Degenerate fits (single predicted class)

1 of 120 runs predicted one class for
every test case. These are failed optimisations, not weak models;
their MCC is undefined and appears as NaN in runs.csv.

Report them as failures-to-converge rather than as low scores. If
they cluster in the deep + saturating-activation cells, that is the
vanishing-gradient signature and is itself the finding.

- `mlp:esm2_35M:armA:TOX|7|logistic|3`

## Depth effect against the noise floor

- best-minus-worst depth, averaged over activations and seeds: 0.0441
- median within-cell seed SD: 0.0286
- ratio: 1.54x

A ratio below about 2 means depth did not move this task beyond what
reseeding moves it on its own.

## How to read this

Compare the spread across the depth column against the seed noise floor
above. If the best and worst depths differ by less than roughly two
within-cell standard deviations, the honest conclusion is that depth did
not matter on this task at this width -- which is a result worth
reporting, not a failed experiment.

Confusion matrices for all 120 cells are in
confusion_matrices.json, keyed `model|depth|activation|seed`, as raw
integer counts over classes ['negative', 'positive'].
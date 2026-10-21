# Depth x activation ablation -- Benchmark reliability audit (peptide property prediction)

Model(s): mlp:esm2_35M:armA:AMP
Grid: depths [1, 3, 4, 5, 6, 7] x activations ['identity', 'logistic', 'relu', 'tanh'] x seeds [0, 1, 2, 3, 4]
Width policy: constant
Total runs: 120

## Seed noise floor

Standard deviation across seed replicates, within each fixed
(model, depth, activation) cell. A depth effect smaller than this is
not a depth effect.

- median within-cell SD of MCC: 0.0156
- max within-cell SD of MCC:    0.0242

## Mean MCC by depth x activation (averaged over seeds)

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.2479 |     0.3264 | 0.3291 | 0.3362 |
|       3 |     0.2514 |     0.3126 | 0.3218 | 0.3164 |
|       4 |     0.2819 |     0.3199 | 0.3244 | 0.3193 |
|       5 |     0.2589 |     0.3187 | 0.3231 | 0.3235 |
|       6 |     0.2538 |     0.3087 | 0.33   | 0.3263 |
|       7 |     0.2685 |     0.3219 | 0.3387 | 0.3125 |

## Mean balanced accuracy by depth x activation

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.6157 |     0.6584 | 0.6606 | 0.6649 |
|       3 |     0.6169 |     0.655  | 0.6586 | 0.655  |
|       4 |     0.631  |     0.6587 | 0.6603 | 0.6571 |
|       5 |     0.6215 |     0.6578 | 0.6596 | 0.6584 |
|       6 |     0.6189 |     0.6519 | 0.6636 | 0.6599 |
|       7 |     0.6273 |     0.6592 | 0.667  | 0.6513 |

## Best cell

- model `mlp:esm2_35M:armA:AMP`, depth 7, activation `relu` -> mean MCC 0.3387

Treat 'best' as provisional until you have checked it against the noise
floor above. On a grid this size the top cell is frequently the luckiest
cell rather than the best architecture.

## Depth effect against the noise floor

- best-minus-worst depth, averaged over activations and seeds: 0.0108
- median within-cell seed SD: 0.0156
- ratio: 0.69x

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
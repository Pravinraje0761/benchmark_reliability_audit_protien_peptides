# Depth x activation ablation -- Benchmark reliability audit (peptide property prediction)

Model(s): mlp:esm2_35M:armA:AMP
Grid: depths [1, 3, 4, 5, 6, 7] x activations ['identity', 'logistic', 'relu', 'tanh'] x seeds [0, 1, 2, 3, 4]
Width policy: pyramid
Total runs: 120

## Seed noise floor

Standard deviation across seed replicates, within each fixed
(model, depth, activation) cell. A depth effect smaller than this is
not a depth effect.

- median within-cell SD of MCC: 0.0135
- max within-cell SD of MCC:    0.0296

## Mean MCC by depth x activation (averaged over seeds)

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.2479 |     0.3264 | 0.3291 | 0.3362 |
|       3 |     0.2756 |     0.3181 | 0.3332 | 0.3214 |
|       4 |     0.2641 |     0.3075 | 0.3364 | 0.3225 |
|       5 |     0.264  |     0.3087 | 0.3294 | 0.3155 |
|       6 |     0.2563 |     0.3199 | 0.3183 | 0.3241 |
|       7 |     0.2584 |     0.3129 | 0.3128 | 0.3143 |

## Mean balanced accuracy by depth x activation

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.6157 |     0.6584 | 0.6606 | 0.6649 |
|       3 |     0.6293 |     0.6562 | 0.6639 | 0.6561 |
|       4 |     0.624  |     0.6523 | 0.6666 | 0.6578 |
|       5 |     0.6251 |     0.6532 | 0.6631 | 0.6546 |
|       6 |     0.6189 |     0.6578 | 0.6569 | 0.6596 |
|       7 |     0.62   |     0.6543 | 0.6544 | 0.6535 |

## Best cell

- model `mlp:esm2_35M:armA:AMP`, depth 4, activation `relu` -> mean MCC 0.3364

Treat 'best' as provisional until you have checked it against the noise
floor above. On a grid this size the top cell is frequently the luckiest
cell rather than the best architecture.

## Depth effect against the noise floor

- best-minus-worst depth, averaged over activations and seeds: 0.0124
- median within-cell seed SD: 0.0135
- ratio: 0.92x

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
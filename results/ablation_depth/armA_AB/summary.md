# Depth x activation ablation -- Benchmark reliability audit (peptide property prediction)

Model(s): mlp:esm2_35M:armA:AB
Grid: depths [1, 3, 4, 5, 6, 7] x activations ['identity', 'logistic', 'relu', 'tanh'] x seeds [0, 1, 2, 3, 4]
Width policy: constant
Total runs: 120

## Seed noise floor

Standard deviation across seed replicates, within each fixed
(model, depth, activation) cell. A depth effect smaller than this is
not a depth effect.

- median within-cell SD of MCC: 0.0134
- max within-cell SD of MCC:    0.0259

## Mean MCC by depth x activation (averaged over seeds)

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.4255 |     0.4664 | 0.4443 | 0.452  |
|       3 |     0.4286 |     0.4278 | 0.4118 | 0.4187 |
|       4 |     0.4102 |     0.4288 | 0.4135 | 0.4325 |
|       5 |     0.4271 |     0.4329 | 0.4273 | 0.4306 |
|       6 |     0.4305 |     0.4312 | 0.419  | 0.4396 |
|       7 |     0.4174 |     0.4393 | 0.4052 | 0.426  |

## Mean balanced accuracy by depth x activation

|   depth |   identity |   logistic |   relu |   tanh |
|--------:|-----------:|-----------:|-------:|-------:|
|       1 |     0.7054 |     0.7301 | 0.7176 | 0.7219 |
|       3 |     0.7055 |     0.7106 | 0.7031 | 0.7054 |
|       4 |     0.6966 |     0.7134 | 0.7037 | 0.7126 |
|       5 |     0.7061 |     0.7147 | 0.7097 | 0.7117 |
|       6 |     0.7053 |     0.713  | 0.7067 | 0.7169 |
|       7 |     0.699  |     0.7181 | 0.6989 | 0.7088 |

## Best cell

- model `mlp:esm2_35M:armA:AB`, depth 1, activation `logistic` -> mean MCC 0.4664

Treat 'best' as provisional until you have checked it against the noise
floor above. On a grid this size the top cell is frequently the luckiest
cell rather than the best architecture.

## Depth effect against the noise floor

- best-minus-worst depth, averaged over activations and seeds: 0.0258
- median within-cell seed SD: 0.0134
- ratio: 1.92x

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
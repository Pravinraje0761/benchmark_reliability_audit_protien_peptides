# `bra`: Benchmark Reliability Audit harness

The package, scripts and tests behind the paper. See the [top-level README](../README.md)
for the study and its results; this page is about running the code.

## Install

```bash
conda create -n bra python=3.11 -y && conda activate bra
pip install -r requirements.txt
conda install -c bioconda mmseqs2 -y     # MMseqs2 must be on PATH (WSL2 or Docker on Windows)
```

Use **Python 3.11 or 3.12**; torch and transformers wheels are not reliably available for
3.14. The protein-language-model pipelines need `torch`, `transformers` and `sentencepiece`
as well (commented out in `requirements.txt`).

## Verify before anything else

```bash
pytest                          # runs from code/; statistics are tested with no benchmark data
python -m bra.cli selftest      # end-to-end on synthetic data
```

`selftest` must show planted-signal AUROC > 0.9 **and** shuffled-label AUROC ≈ 0.50. The
second check is the one that catches real bugs.

## Reproduce everything

```bash
bash reproduce.sh               # resumable; see the top-level README for runtime
```

Raw inputs and caches go to `$BRA_DATA` (default `~/bra-data`), outside the repository.

## Layout

```
bra/
  schema.py           canonical peptide and results tables, with validation
  corpus.py           harmonised antibacterial corpus for Arm B
  negatives.py        the five negative-set rules
  splits.py           MMseqs2 and alignment-free clustering, fold assignment, leakage verification
  balanced_split.py   class-balance-preserving cluster-disjoint splitter
  features.py         AAC / DPC / physicochemical / one-hot / k-mer features, PLM embeddings cached by sequence hash
  models.py           the frozen pipeline zoo
  nn.py               CNN / BiLSTM definitions (excluded from the reported grids; see the deviations log)
  metrics.py          classification and regression metrics, computed from stored predictions
  runner.py           builds a cell and hands identical folds to every pipeline
  arma.py, armb.py    Arm A (tasks) and Arm B (protocol factorial)
  cli.py              selftest / run-arm-a / run-arm-b / analyse
  analysis/
    ranks.py          Kendall tau-b, null band, RBO, footrule, inversions, top-1 flip, bootstrap
    decompose.py      variance components on scores and ranks
    inference.py      uncertainty and hypothesis tests for rank agreement
    power.py          pre-specified design simulation
configs/
  benchmarks.yaml     native protocol per Arm A benchmark
  grid.yaml           the Arm B factorial
scripts/              one entry point per stage, in the order reproduce.sh runs them
tests/                unit tests, including regression guards for two claims withdrawn in revision
```

## Design rules the code enforces

1. **Every pipeline sees identical folds** within a cell, asserted on index and label hashes.
2. **Metrics are computed downstream from stored predictions**, so every metric provably sees
   the same predictions.
3. **Leakage is verified after every split**, not assumed once.
4. **Data-derived parameters come from the training fold only** (for example `max_len`).
5. **PLM embeddings are cached by sequence hash** across the whole project.
6. **A failed pipeline writes NaN rows and is logged** rather than killing the sweep, and it
   stays visible in the results table.

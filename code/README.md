# `bra` — Benchmark Reliability Audit harness

Code for the paper scaffolded in `../Projects/BRA/`. The research design lives in those
notes; this README is about running the thing.

## Install

```
conda create -n bra python=3.11 -y && conda activate bra
pip install -r requirements.txt
conda install -c bioconda mmseqs2 -y     # WSL2 or Docker on Windows
```

**Python 3.11 or 3.12.** The machine currently has 3.14, where torch/transformers
wheels are not reliably available. Do not fight this; make the 3.11 env.

## Verify before anything else

```
pytest tests -q                 # stats are testable with no benchmark data
python -m bra.cli selftest      # end-to-end on synthetic data (E01 gate)
```

`selftest` must show planted-signal AUROC > 0.9 **and** shuffled-label AUROC ≈ 0.50.
The second check is the one that catches real bugs.

## Layout

```
bra/
  schema.py        canonical peptide + results tables, with validation
  data/            one loader per benchmark (stubs -> fill in milestone M1)
  splits.py        MMseqs2 + alignment-free clustering, fold assignment, leakage verification
  negatives.py     the five negative-sampling rules
  features.py      AAC/DPC/physchem/one-hot/k-mer + cached PLM embeddings
  models.py        the frozen 12-pipeline zoo
  nn.py            CNN / BiLSTM (M05, M06) — torch imported lazily
  metrics.py       classification + regression metrics, computed from stored predictions
  runner.py        builds a Cell, shares folds across all 12 pipelines, asserts fairness
  cli.py           selftest / run-arm-a / run-arm-b / analyse
  analysis/
    ranks.py       tau-b, null band, RBO, footrule, inversions, top-1 flip, bootstrap
    decompose.py   variance components (score AND rank), tau-b ablation, H5 test
configs/
  models.yaml      frozen zoo — hash this before Arm A
  benchmarks.yaml  native protocol per benchmark (fill in M1)
  grid.yaml        the Arm B factorial
```

## Design rules this code enforces

1. **All 12 pipelines see identical folds** within a cell. Asserted on index and label hashes.
2. **Metrics are computed downstream from stored predictions.** Metric is therefore a free
   factor, and every metric provably sees the same predictions.
3. **Leakage is verified after every split**, not assumed once.
4. **Data-derived parameters come from the train fold only** (e.g. `max_len`).
5. **PLM embeddings are cached by sequence hash** across the whole project. Build this
   cache before the first sweep; it is the single largest cost saver.
6. **A failed pipeline writes NaN rows and is logged** rather than killing the sweep —
   but it stays visible in the results table.

## Writing a benchmark loader

Register it and return the canonical schema:

```python
@register("B2_qmap")
def load_b2() -> pd.DataFrame:
    ...
    return df   # sequence, label, split, source, benchmark, task
```

`validate_peptides` will reject non-canonical residues and warn on duplicates.
Start with QMAP (`pip install qmap-benchmark`) — best documented, and it will shape
the interface for the other five.

## What is not implemented yet

- All six benchmark loaders (deliberate — they need the real downloads; see `data/__init__.py`)
- `nn_train.fit_predict_torch` (the torch training loop for M05/M06)
- Arm B pooled-corpus construction (`cmd_run_arm_b`)
- Plotting functions for Figures 1–5

Everything in `analysis/` is complete and tested.

## Known environment issue on this machine (2026-10-04)

`pytest tests` passes 16/16 on the host Python 3.14 user-site install, but **scikit-learn
cannot load**:

```
ImportError: DLL load failed while importing _base:
An Application Control policy has blocked this file.
```

Windows Smart App Control is blocking unsigned compiled extensions in
`%APPDATA%\Roaming\Python\...`. Use **WSL2** — you need it for MMseqs2 regardless. See
`../Experiments/E01 Environment and harness selftest.md` for the full log and alternatives.

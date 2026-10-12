# Benchmark reliability audit: peptide property prediction

Code, manuscript and results for a factorial audit of evaluation design in
peptide property prediction — asking not whether leakage-controlled benchmarks
report the right *level* of performance, but whether they agree on the *ranking*
of methods.

Two crossed designs hold a fixed set of model pipelines constant:

- **Arm B** fixes the task and varies the protocol: 4 clustering thresholds × 5
  negative-set rules × 5 seeds, on one harmonised antibacterial corpus.
- **Arm A** fixes the protocol and varies the biological task: 10 curated
  peptide bioactivity datasets under their native homology-aware partitions.

**5,490 model fits in total** — 4,400 in the two main grids, 1,090 across six
supplementary analyses.

## Headline results

| | |
|---|---|
| Primary hypothesis (protocols rank methods differently) | **not supported** — normalised agreement 0.881 against a pre-specified threshold of 0.85 |
| Negative-set rule's share of MCC variance | **71.0%**, against 14.8% for the pipeline |
| Rank agreement across 10 tasks vs 5 rules | +0.218, *p* = 0.024 generalising over levels |
| Winner changed by reseeding alone | **54.5%** of comparisons |
| Residual pairwise leakage as the threshold *tightens* | 1.4% → **21.6%** |
| Test class balance on a 1:1 corpus | up to **99.5%** positive |
| Released datasets with cross-split exact duplicates | **11 of 16** |

Scope: these are conclusions about *five negative-set rules on one corpus* and
*ten tasks from one curator*. They are not evidence that independently built
benchmarks disagree in general — that comparison is not performed here and
remains open.

## Layout

```
code/
  bra/              the package: corpus construction, splitters, features,
                    metrics, and the analysis modules (ranks, decompose,
                    inference, power)
  scripts/          one entry point per stage; see reproduce.sh for the order
  tests/            65 tests, including regression guards for two claims that
                    were wrong in earlier drafts
  reproduce.sh      runs every stage end to end, resumable
paper/
  main.tex/.pdf          the manuscript (34 pp), self-contained, no BibTeX needed
  supplementary.tex/.pdf supplementary material (5 pp)
  figures/               the 13 figures the documents include
  preregistration.md     dated analysis plan, all revisions, full deviations log
results/
  tables/           every number in the paper, as CSV
  figures/          the same figures as PDF and PNG
```

## Reproducing

```bash
cd code
pip install -r requirements.txt     # needs MMseqs2 on PATH
bash reproduce.sh
```

`reproduce.sh` retrieves inputs, builds the corpora, precomputes embeddings,
runs both grids and all supplementary analyses, then regenerates every figure
and table. It is idempotent and resumable: the embedding stage caches by
sequence hash, the grid stage writes one file per cell, and the negative-draw
sensitivity caches per cell too.

Reference hardware: AMD Ryzen 9 8940HX, 32 threads, **no GPU**. The pipeline
families that cannot be amortised by feature caching (end-to-end CNN/BiLSTM,
ProtT5-XL) are excluded for that reason and logged as deviations.

## What is not here

- **The Arm B corpus is not redistributed.** Its positives derive from DBAASP
  via the AMPBenchmark positive set and its negatives from UniProt, and those
  terms govern redistribution. The construction scripts are released with
  SHA-256 hashes of both inputs, which reconstructs the corpus from the same
  source files. `reproduce.sh` fails loudly on a hash mismatch rather than
  silently proceeding with different negatives.
- **The Arm A datasets** are redistributed by their original authors and are not
  duplicated here.
- **Cached embeddings and raw downloads**, which are large and regenerable.

## Corrections

Claims withdrawn or narrowed during revision are documented rather than quietly
removed, in the manuscript and in `paper/preregistration.md`:

- the rank-variance main effects are zero **by construction**, not by
  measurement — an algebraic identity an earlier draft reported as a finding;
- the clustering threshold is benign for the *ordering* but not for the
  *winner*, which only holds after averaging over seeds and rules;
- the provenance audit is an overlap check against the **current** UniProt
  release and bounds ESM-2 pretraining exposure in neither direction;
- the recommendation to adopt the balance-preserving splitter is **withdrawn** —
  a stress test against three alternatives shows uniformly random cluster order
  halves residual leakage at negligible cost in balance.

## Status

The manuscript is not yet submitted and the data deposit is pending; the
availability section carries a placeholder DOI. No licence is declared yet —
add one before reuse.

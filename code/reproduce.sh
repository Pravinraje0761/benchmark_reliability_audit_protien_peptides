#!/usr/bin/env bash
# Full reproduction of the BRA paper from a clean checkout.
#
#   bash reproduce.sh
#
# Stages are idempotent and resumable: the embedding cache is keyed by sequence
# hash and the grid writes one parquet per cell, so an interrupted run continues
# where it stopped rather than starting over.
#
# Wall clock on the reference machine (AMD Ryzen 9 8940HX, 32 threads, CPU only,
# no GPU): roughly 1 h for embeddings and 2-3 h for the 1,600 fits.
set -euo pipefail

: "${BRA_DATA:=$HOME/bra-data}"
export BRA_DATA
export HF_HOME="$BRA_DATA/hf"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$BRA_DATA/raw"

say() { printf "\n\033[1m=== %s ===\033[0m\n" "$1"; }

say "0. environment"
python --version
python -c "import numpy,pandas,scipy,sklearn,xgboost;print('core deps ok')"
command -v mmseqs >/dev/null || { echo "mmseqs2 not found (conda install -c bioconda mmseqs2)"; exit 1; }
mmseqs version

say "1. tests (must pass before anything is computed)"
python -m pytest -q

say "2. inputs"
cd "$BRA_DATA/raw"
[ -d NegativeDatasets ] || git clone --depth 1 https://github.com/BioGenies/NegativeDatasets.git
mkdir -p uniprot
if ! gzip -t uniprot/sprot.tsv.gz 2>/dev/null; then
  echo "retrieving reviewed UniProt (about 127 MB)"
  curl -sS -L --retry 3 -o uniprot/sprot.tsv.gz \
    "https://rest.uniprot.org/uniprotkb/stream?query=reviewed:true&format=tsv&compressed=true&fields=accession,length,keyword,sequence"
  gzip -t uniprot/sprot.tsv.gz || { echo "download truncated"; exit 1; }
fi
sha256sum uniprot/sprot.tsv.gz NegativeDatasets/data/positive_dataset.fa | tee "$BRA_DATA/input_hashes.txt"
cd - >/dev/null

say "3. design simulation (pre-specified; independent of the data)"
python scripts/run_power_analysis.py
python scripts/run_power_followup.py

say "4. corpus"
python - <<'INNER'
import os
from pathlib import Path
from bra.corpus import build_corpus, NEG_RULES
root = Path(os.environ["BRA_DATA"])
P = root/"raw/NegativeDatasets/data/positive_dataset.fa"
U = root/"raw/uniprot/sprot.tsv.gz"
out = root/"corpus"; out.mkdir(parents=True, exist_ok=True)
for rule in NEG_RULES:
    for seed in range(5):
        f = out/f"armB_{rule}_seed{seed}.parquet"
        if f.exists():
            continue
        build_corpus(rule, P, U, seed=seed).to_parquet(f)
    print("  ", rule, "ok", flush=True)
INNER

say "5. embeddings (cached by sequence hash)"
python scripts/run_arm_b.py --stage embed

say "6. the factorial (resumable, one parquet per cell)"
python scripts/run_arm_b.py --stage grid
python scripts/run_arm_b.py --stage merge

say "7. verification (no number is reported from output that fails this)"
python scripts/verify_results.py

say "8. corpus characterisation and residual leakage (model-independent)"
python scripts/characterise_corpus.py | tee ../results/tables/corpus_report.txt

say "9. analysis, sensitivity and figures"
python scripts/analyse_arm_b.py       | tee ../results/tables/analysis_report.txt
python scripts/check_svm_approximation.py

# Supplementary analyses. Each writes its own table under ../results/tables so the
# deposit matches what the paper reports. run_sensitivity_draw caches one file per
# cell under sensitivity_draw_cells/, so an interrupted run resumes rather than
# starting over -- it is the longest of the three.
say "9a. negative-draw sensitivity (300 fits, resumable per cell)"
python scripts/run_sensitivity_draw.py

say "9b. splitter stress test (no model fits; 60 splits + leakage searches)"
python scripts/run_splitter_stresstest.py

say "9c. cluster-aware score intervals (200 fits, 2x2000 bootstrap each)"
python scripts/run_cluster_intervals.py

say "9d. equal tuning budget (560 fits: 6 configs per learner + refit)"
python scripts/run_equal_tuning.py

python scripts/make_figures.py

say "10. manuscript: substitute numbers, assemble, gate"
# Numbers are substituted mechanically from the CSVs rather than transcribed by
# hand, so the prose cannot drift from the artefacts it reports.
# These two scripts operate on the author's Markdown manuscript drafts in
# ../Projects/BRA/manuscript/, which are not part of this repository. The
# released manuscript is ../paper/main.tex; every number it reports is in
# ../results/tables/. Skip cleanly when the drafts are absent.
if [ -d ../Projects/BRA/manuscript ]; then
  python scripts/fill_results.py --apply
  python scripts/assemble_manuscript.py
  MANUSCRIPT_NOTE="manuscript -> ../Projects/BRA/manuscript/MANUSCRIPT.md"
else
  echo "Manuscript drafts not present; skipping (released manuscript: ../paper/main.tex)."
  MANUSCRIPT_NOTE="manuscript -> ../paper/main.tex (compile with pdflatex, twice)"
fi

say "done"
echo "tables     -> ../results/tables"
echo "figures    -> ../results/figures"
echo "$MANUSCRIPT_NOTE"
echo
echo "The submission gate above lists anything still outstanding."

# Manuscript package

Two self-contained LaTeX documents, each compiling on its own. Neither uses
`\input`, and the bibliography is embedded, so **BibTeX is not required**.

```
main.tex            the paper              -> 34 pages
supplementary.tex   supplementary material ->  5 pages
figures/            13 vector PDFs, shared by both documents
tables/             every number in either document, as CSV
preregistration.md  the dated analysis plan, all revisions, full deviations log
```

## Compile

```
pdflatex main && pdflatex main
pdflatex supplementary && pdflatex supplementary
```

Twice each, for cross-references. Both were last compiled with Tectonic 0.15.0
and produce **zero warnings** — no overfull or underfull boxes, no undefined
references. On Overleaf, upload the folder and set the main document.

## What is where

The main text carries the argument and the evidence it rests on. The supplement
holds detail a reader consults rather than reads:

| | content |
|---|---|
| Figure S1 | splitter pseudocode, as implemented |
| Figure S2 | pre-specified design simulation |
| Table S1 | hand-checkable worked example of the splitter |
| Table S2 | per-cell balance and leakage, all 20 cells, both splitters |
| Table S3 | all 16 released datasets: sizes, duplicates, both-label counts |
| Table S4 | cluster-aware interval widths by level |
| Table S5 | inventory of every computation in the deposit |

References to supplementary floats are written as literal text ("Supplementary
Table S2") because LaTeX cannot resolve `\ref` across documents. **If you reorder
the supplement, update those numbers by hand** — the file order currently matches
the numbering, which is what keeps them in step.

## Figures

All 13 are vector PDF, built form-first: viridis for magnitude (monotonic in
lightness, so it survives greyscale printing), Okabe-Ito in fixed order for
categorical identity, a diverging map with a neutral midpoint for tau-b. No dual
axes anywhere. F10–F13 cover the supplementary analyses: negative-draw
sensitivity, the splitter trade-off, cluster-aware intervals, and class
provenance. The equal-tuning analysis (Section "Would an equal tuning budget
change the answer?") is reported as a table rather than a figure, since its
result is that nothing moves.

## Before submitting

1. Fill the author block in `main.tex` (currently a placeholder).
2. Replace `INSERT_AT_DEPOSIT` in the availability section with the archive URL
   or DOI. It is set in red so it cannot be missed, and the surrounding text
   currently reads as an undertaking rather than a completed deposit.
3. Trim the abstract to the target journal's limit (~425 words at present).
4. Uncomment the `lineno` package line in `main.tex` if line numbers are wanted.
5. Decide whether more should move to the supplement. Next candidates, with rough
   savings: the SVM validation subsection (~0.75 page), the computational notes
   (~0.5), Figure F8 (~0.5). I would keep the status table and the deviations
   list in the main text — they are what the pre-registration claim rests on.

## Provenance

Every number in both documents is produced by a script in the code release and
written to `tables/`; nothing is transcribed by hand. `reproduce.sh` runs all
stages in order, including the four supplementary analyses.

Claims withdrawn or corrected during revision are documented rather than quietly
removed: the rank-variance main effects (zero by construction, not by
measurement), the clustering threshold's effect on winner identity (benign only
after averaging over seeds and rules), the pretraining-exposure claim (now an
overlap audit against the current UniProt release, which bounds exposure in
neither direction), and the recommendation to adopt the balance-preserving
splitter (withdrawn — the stress test shows uniformly random cluster order
dominates it on leakage at negligible cost in balance).

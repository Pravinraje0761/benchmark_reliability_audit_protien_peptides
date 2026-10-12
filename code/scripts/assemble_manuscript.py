"""Assemble the manuscript sections into one document and gate it.

    python scripts/assemble_manuscript.py            # assemble + check
    python scripts/assemble_manuscript.py --strict   # non-zero exit if gated

Two jobs:
  1. Concatenate the numbered sections in reading order into a single file.
  2. Refuse to call it submission-ready while any placeholder, pending claim or
     unverified reference remains. A bracketed [T] that survives to submission
     is the failure mode this exists to prevent.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / "Projects" / "BRA" / "manuscript"
OUT = MS / "MANUSCRIPT.md"

ORDER = [
    ("Abstract", "00 Abstract and claims.md"),
    ("Introduction", "02 Introduction.md"),
    ("Methods", "01 Methods.md"),
    ("Results", "04 Results.md"),
    ("Discussion", "03 Discussion.md"),
    ("References", "05 References.md"),
    ("Submission notes", "06 Submission package.md"),
]

# A bracketed single capital, or a bracketed instruction, is a fill-in slot.
# [V] and [TOOL] are reference-status markers, not placeholders, and are
# excluded here; [TOOL] is checked separately because it must be resolved too.
PLACEHOLDER = re.compile(r"\[(?!V\]|TOOL\])(?:[A-Z]|rule|benchmark|repository|"
                         r"Fill[^\]]*|Condition[^\]]*|Full-grid[^\]]*)\]")


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            return text[end + 3:].lstrip()
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    parts, missing = [], []
    for label, fn in ORDER:
        f = MS / fn
        if not f.exists():
            missing.append(fn)
            continue
        parts.append(f"<!-- ===== {label} ({fn}) ===== -->\n\n"
                     + strip_frontmatter(f.read_text()))
    doc = "\n\n---\n\n".join(parts)
    OUT.write_text(doc)

    words = len(doc.split())
    print(f"assembled {len(parts)} sections -> {OUT}")
    print(f"  {len(doc.splitlines())} lines, ~{words:,} words")
    if missing:
        print(f"  MISSING SECTIONS: {missing}")

    print("\n--- submission gate ---")
    problems = []

    ph = sorted(set(PLACEHOLDER.findall(doc)))
    if ph:
        problems.append(f"{len(ph)} unfilled placeholder token(s): {ph[:12]}")

    pending = [l.strip() for l in doc.splitlines()
               if l.strip().startswith("| C") and "pending" in l.lower()]
    if pending:
        problems.append(f"{len(pending)} claim(s) still pending evidence")
        for l in pending[:8]:
            print("   pending:", l.split("|")[2].strip()[:72])

    if "[TOOL]" in doc:
        n = doc.count("[TOOL]")
        problems.append(f"{n} reference(s) still marked [TOOL] (publisher detail needed)")

    unchecked = doc.count("- [ ]")
    if unchecked:
        problems.append(f"{unchecked} unchecked checklist item(s)")

    if problems:
        print("NOT SUBMISSION-READY:")
        for p in problems:
            print(f"  - {p}")
        return 1 if args.strict else 0
    print("GATE PASSED: no placeholders, no pending claims, no [TOOL] refs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

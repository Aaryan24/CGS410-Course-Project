#!/usr/bin/env python3
"""
Compute the causal-attention ceiling per UD relation.

For each gold UD edge (dependent → head), a causal (left-to-right) model
can only recover it if the head is to the LEFT of the dependent, i.e.
head_idx < dependent_idx.

This script reads the UD CoNLL-U file directly and reports, per relation:
  - total edges
  - causally recoverable edges (head < dependent)
  - ceiling (fraction recoverable)

No model inference is needed. Runs in <1 second.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import csv
import sys

# --- adapt this import to reuse your existing parser ---
sys.path.insert(0, str(Path(__file__).parent))
import gpt2_ud_utils as base


def compute_causal_ceiling(
    split: str = "test",
    min_words: int = 3,
    max_words: int = 100,
    out_name: str | None = None,
):
    ud_paths = base.ensure_ud_files(Path(__file__).parent / ".cache")
    sentences = base.select_sentences(
        split=split, ud_paths=ud_paths,
        min_words=min_words, max_words=max_words, limit=0,
    )

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "recoverable": 0})
    overall_total = 0
    overall_recoverable = 0

    for example in sentences:
        for token in example.tokens:
            if token.head == 0:  # root — skip (same as ud_edges)
                continue
            rel = token.deprel
            counts[rel]["total"] += 1
            overall_total += 1
            if token.head < token.idx:
                counts[rel]["recoverable"] += 1
                overall_recoverable += 1

    # Sort by total edges descending
    rows = []
    for rel in sorted(counts, key=lambda r: counts[r]["total"], reverse=True):
        total = counts[rel]["total"]
        recoverable = counts[rel]["recoverable"]
        ceiling = recoverable / total if total > 0 else 0.0
        rows.append({
            "relation": rel,
            "total_edges": total,
            "recoverable_edges": recoverable,
            "causal_ceiling": round(ceiling, 4),
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"Causal-Attention Ceiling — UD English-EWT ({split} split)")
    print(f"{'='*60}")
    print(f"Sentences: {len(sentences)}")
    print(f"Total edges (excl. root): {overall_total}")
    print(f"Causally recoverable: {overall_recoverable} ({overall_recoverable/overall_total*100:.1f}%)")
    print(f"{'='*60}\n")

    print(f"{'Relation':<20} {'Total':>8} {'Recov.':>8} {'Ceiling':>8}")
    print("-" * 48)
    for row in rows:
        print(f"{row['relation']:<20} {row['total_edges']:>8} {row['recoverable_edges']:>8} {row['causal_ceiling']:>8.1%}")

    # Also save CSV
    output_name = out_name or f"causal_ceiling_{split}.csv"
    out_path = Path(__file__).parent / "results" / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relation", "total_edges", "recoverable_edges", "causal_ceiling"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the causal recoverability ceiling for UD edges.")
    parser.add_argument("split", nargs="?", default="test", choices=["train", "dev", "test", "all"])
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()
    compute_causal_ceiling(
        split=args.split,
        min_words=args.min_words,
        max_words=args.max_words,
        out_name=args.out_name,
    )

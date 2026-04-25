#!/usr/bin/env python3
"""
Batch runner for multilingual mBERT dependency-head metrics.

This script intentionally runs only the implementation stage used for the final
study: dependency-head metric extraction over selected UD treebanks. It does
not build reports, PDFs, or LaTeX files.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = Path("multilingual_language_config.json")
DEFAULT_LANGUAGES = "hi_hdtb,ur_udtb,fi_tdt,id_gsd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multilingual mBERT dependency-head metrics.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--languages", default=DEFAULT_LANGUAGES, help="Comma-separated config keys, or 'all'.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--results-dir", type=Path, default=Path("results/mbert_dependency_head"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/mbert_dependency_head_ud"))
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0, help="0 means all filtered sentences.")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--relation-min-count", type=int, default=50)
    parser.add_argument("--attribution-sentences", type=int, default=200)
    parser.add_argument("--attribution-tokens-per-sentence", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, dict]:
    payload = json.loads((ROOT / config_path).read_text(encoding="utf-8"))
    return payload["languages"]


def select_language_keys(all_languages: dict[str, dict], text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(all_languages)
    keys = [part.strip() for part in text.split(",") if part.strip()]
    missing = [key for key in keys if key not in all_languages]
    if missing:
        raise KeyError(f"Unknown language keys: {missing}. Available: {sorted(all_languages)}")
    return keys


def expected_metric_outputs(results_dir: Path, run_name: str) -> list[Path]:
    return [
        results_dir / f"{run_name}_head_characterization.csv",
        results_dir / f"{run_name}_relation_characterization.csv",
        results_dir / f"{run_name}_relation_baselines.csv",
        results_dir / f"{run_name}_language_summary.json",
    ]


def has_final_metrics(results_dir: Path, run_name: str) -> bool:
    outputs = expected_metric_outputs(results_dir, run_name)
    if not all(path.exists() for path in outputs):
        return False
    summary = json.loads(outputs[-1].read_text(encoding="utf-8"))
    return bool(summary.get("final"))


def metrics_command(args: argparse.Namespace, cfg: dict) -> list[str]:
    return [
        sys.executable,
        "mbert_dependency_head_analysis.py",
        "--language",
        cfg["language"],
        "--treebank",
        cfg["treebank"],
        "--run-name",
        cfg["run_name"],
        "--results-dir",
        str(args.results_dir),
        "--cache-dir",
        str(args.cache_dir),
        "--split",
        args.split,
        "--limit",
        str(args.limit),
        "--min-words",
        str(args.min_words),
        "--max-words",
        str(args.max_words),
        "--layers",
        "all",
        "--heads",
        "all",
        "--relation-min-count",
        str(args.relation_min_count),
        "--exclude-upos",
        "PUNCT",
        "--exclude-relations",
        "punct",
        "--attribution-sentences",
        str(args.attribution_sentences),
        "--attribution-tokens-per-sentence",
        str(args.attribution_tokens_per_sentence),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--device",
        args.device,
        "--ud-files-json",
        json.dumps(cfg["ud_files"]),
    ]


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("\n$ " + shlex.join(command), flush=True)
    if dry_run:
        return
    start = time.time()
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"Finished in {(time.time() - start) / 60.0:.1f} min", flush=True)


def main() -> None:
    args = parse_args()
    configs = load_config(args.config)
    language_keys = select_language_keys(configs, args.languages)
    print(f"Selected languages: {', '.join(language_keys)}", flush=True)

    for key in language_keys:
        cfg = configs[key]
        print(f"\n=== {cfg['language']} / {cfg['treebank']} ===", flush=True)
        if has_final_metrics(args.results_dir, cfg["run_name"]) and not args.force:
            print(f"Metrics already final for {cfg['run_name']}; skipping. Use --force to rerun.", flush=True)
        else:
            run_command(metrics_command(args, cfg), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

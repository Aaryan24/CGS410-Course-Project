#!/usr/bin/env python3
"""
GPT-2 category ablation/sufficiency runner for the bridge study.

This reuses the existing GPT-2 phase-1 outputs and phase-2 utility functions,
but rebuilds retained-head subsets using the same bridge categories used for
the mBERT study:

  syntactic_relation, pure_syntactic_relation, structural_graph, positional,
  syntax_position_hybrid, syntax_composite

The tasks are:

  ablation    : remove selected category heads from the full model
  sufficiency : keep only selected category heads, mirroring the GPT-2
                retained-head study

Both tasks evaluate syntax recovery plus next-token prediction on held-out UD
EWT, and compare each category against layer-matched random controls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

BRIDGE_DIR = Path(__file__).resolve().parent
ROOT = BRIDGE_DIR
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gpt2_ud_utils as base  # noqa: E402
import gpt2_phase2_sufficiency as phase2  # noqa: E402


OUTPUT_DIR = BRIDGE_DIR / "outputs" / "gpt2_category_bridge"
DEFAULT_PHASE1_PREFIX = (
    ROOT
    / "results"
    / "gpt2_medium_phase1_h100_dev_checkpointed"
)
DEFAULT_ABLATION_BUDGETS = (4, 16, 32)
DEFAULT_SUFFICIENCY_BUDGETS = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 288)
CATEGORY_FAMILIES = (
    "syntactic_relation",
    "pure_syntactic_relation",
    "structural_graph",
    "positional",
    "syntax_position_hybrid",
    "syntax_composite",
)
SECONDARY_FAMILIES = {"syntax_position_hybrid", "syntax_composite"}


def mode_default_budgets(mode: str) -> tuple[int, ...]:
    if mode == "ablation":
        return DEFAULT_ABLATION_BUDGETS
    if mode == "sufficiency":
        return DEFAULT_SUFFICIENCY_BUDGETS
    raise ValueError(mode)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-2 bridge category ablation/sufficiency.")
    parser.add_argument("--mode", choices=("ablation", "sufficiency"), default="sufficiency")
    parser.add_argument("--model-name", default="gpt2-medium")
    parser.add_argument("--phase1-prefix", type=Path, default=DEFAULT_PHASE1_PREFIX)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/ud_english_ewt")
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--eval-split", choices=("train", "dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--budgets", default="")
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--include-uniform-random", action="store_true")
    parser.add_argument("--positional-threshold", type=float, default=0.80)
    parser.add_argument("--top-k-values", default="1")
    parser.add_argument("--compute-tree-distance", action="store_true")
    parser.add_argument("--prediction-only", action="store_true")
    parser.add_argument("--prediction-batch-size", type=int, default=24)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No integer values found in {text!r}")
    return values


def output_paths(args: argparse.Namespace) -> dict[str, Path]:
    prefix = args.results_dir / args.run_name
    return {
        "ranking": prefix.with_name(prefix.name + "_ranking.csv"),
        "subset": prefix.with_name(prefix.name + "_subset_definitions.csv"),
        "sentence": prefix.with_name(prefix.name + "_sentence.csv"),
        "relation_rows": prefix.with_name(prefix.name + "_relation_rows.csv"),
        "relation_summary": prefix.with_name(prefix.name + "_relation_summary.csv"),
        "summary": prefix.with_name(prefix.name + "_summary.csv"),
        "random_contrasts": prefix.with_name(prefix.name + "_random_contrasts.csv"),
        "progress": prefix.with_name(prefix.name + "_progress.json"),
    }


def normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def add_bridge_scores(ranking_df: pd.DataFrame, positional_threshold: float) -> pd.DataFrame:
    df = ranking_df.copy()
    if "head_index" not in df.columns and "head" in df.columns:
        df["head_index"] = df["head"]
    df["head_label"] = df.apply(lambda row: f"L{int(row.layer)}-H{int(row.head_index)}", axis=1)
    df["positional_score"] = normalize(df.get("positional_share", pd.Series(0.0, index=df.index)))
    rare_source = "rare_word_rate" if "rare_word_rate" in df.columns else "rare_top2_rate"
    df["rare_lexical_score"] = normalize(df.get(rare_source, pd.Series(0.0, index=df.index)))
    df["near_positional"] = df.get("positional_share", pd.Series(0.0, index=df.index)).fillna(0.0) >= positional_threshold
    df["syntactic_flag"] = df.get("is_syntactic_head", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["pure_syntax_flag"] = df["syntactic_flag"] & ~df["near_positional"]
    df["syntax_position_score"] = 0.50 * normalize(df["syntactic_score"]) + 0.50 * df["positional_score"]
    df["syntax_composite_score"] = (
        0.45 * normalize(df["structural_score"])
        + 0.35 * normalize(df["syntactic_score"])
        + 0.20 * normalize(df.get("confidence_mean", pd.Series(0.0, index=df.index)))
    )
    df["attention_only_utility_score"] = (
        0.35 * normalize(df["structural_score"])
        + 0.35 * normalize(df["syntactic_score"])
        + 0.15 * df["positional_score"]
        + 0.15 * df["rare_lexical_score"]
    )
    return df


def coords(df: pd.DataFrame, budget: int) -> list[tuple[int, int]]:
    return [(int(row.layer), int(row.head_index)) for row in df.head(budget).itertuples(index=False)]


def choose_category_heads(ranking_df: pd.DataFrame, budget: int) -> dict[str, list[tuple[int, int]]]:
    pure = ranking_df[ranking_df["pure_syntax_flag"]].sort_values(
        ["syntactic_score", "structural_score", "head_label"],
        ascending=[False, False, True],
    )
    if len(pure) < budget:
        pure = ranking_df[~ranking_df["near_positional"]].sort_values(
            ["syntactic_score", "structural_score", "head_label"],
            ascending=[False, False, True],
        )
    return {
        "syntactic_relation": coords(
            ranking_df.sort_values(["syntactic_score", "structural_score"], ascending=False), budget
        ),
        "pure_syntactic_relation": coords(pure, budget),
        "structural_graph": coords(
            ranking_df.sort_values(["structural_score", "syntactic_score"], ascending=False), budget
        ),
        "positional": coords(
            ranking_df.sort_values(["positional_score", "confidence_mean"], ascending=False), budget
        ),
        "syntax_position_hybrid": coords(
            ranking_df.sort_values(["syntax_position_score", "syntactic_score"], ascending=False), budget
        ),
        "syntax_composite": coords(
            ranking_df.sort_values(["syntax_composite_score", "structural_score"], ascending=False), budget
        ),
        "rare_lexical": coords(
            ranking_df.sort_values(["rare_lexical_score", "confidence_mean"], ascending=False), budget
        ),
        "low_utility": coords(
            ranking_df.sort_values(["attention_only_utility_score", "confidence_mean"], ascending=[True, True]),
            budget,
        ),
    }


def layer_matched_random(target_heads: list[tuple[int, int]], seed: int, num_heads: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    by_layer: dict[int, int] = defaultdict(int)
    target_set = set(target_heads)
    for layer, _head in target_heads:
        by_layer[layer] += 1
    sampled: list[tuple[int, int]] = []
    for layer, count in sorted(by_layer.items()):
        non_target_pool = [(layer, head) for head in range(num_heads) if (layer, head) not in target_set]
        if len(non_target_pool) >= count:
            sampled.extend(rng.sample(non_target_pool, count))
            continue

        # For high budgets an exact-size, exact-layer, fully disjoint control may
        # be impossible. Preserve the layer histogram and minimize overlap.
        sampled.extend(non_target_pool)
        target_pool = [(layer, head) for head in range(num_heads) if (layer, head) in target_set]
        sampled.extend(rng.sample(target_pool, count - len(non_target_pool)))
    return sorted(sampled)


def build_subset_definitions(
    ranking_df: pd.DataFrame,
    all_heads: list[tuple[int, int]],
    budgets: list[int],
    random_seeds: list[int],
    num_heads: int,
    mode: str,
    include_uniform_random: bool,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    all_head_set = set(all_heads)

    def active_for(target_heads: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if mode == "sufficiency":
            return sorted(target_heads)
        if mode == "ablation":
            return sorted(all_head_set - set(target_heads))
        raise ValueError(mode)

    def condition_name(family: str, budget: int, seed: int = -1) -> str:
        if seed >= 0:
            return f"layer_random_for_{family}_k{budget}_seed{seed}"
        if mode == "ablation":
            return f"minus_{family}_k{budget}"
        return f"{family}_k{budget}"

    definitions: list[dict[str, object]] = [
        {
            "subset_name": "full_model",
            "family": "full",
            "mode": mode,
            "budget": 0,
            "subset_size": len(all_heads),
            "active_head_count": len(all_heads),
            "target_head_count": 0,
            "seed": -1,
            "target_heads": [],
            "selected_heads": sorted(all_heads),
        }
    ]
    for budget in budgets:
        category_heads = choose_category_heads(ranking_df, budget)
        for family, heads in category_heads.items():
            if family not in CATEGORY_FAMILIES:
                continue
            active_heads = active_for(heads)
            definitions.append(
                {
                    "subset_name": condition_name(family, budget),
                    "family": family,
                    "mode": mode,
                    "budget": budget,
                    "subset_size": len(heads),
                    "active_head_count": len(active_heads),
                    "target_head_count": len(heads),
                    "seed": -1,
                    "control_overlap_count": 0,
                    "control_overlap_fraction": 0.0,
                    "target_heads": sorted(heads),
                    "selected_heads": active_heads,
                }
            )
            for seed in random_seeds:
                family_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(family))
                sampled = layer_matched_random(heads, seed + 10_000 + family_offset + budget * 100, num_heads)
                overlap_count = len(set(sampled) & set(heads))
                active_sampled = active_for(sampled)
                definitions.append(
                    {
                        "subset_name": condition_name(family, budget, seed),
                        "family": f"layer_random_for_{family}",
                        "mode": mode,
                        "budget": budget,
                        "subset_size": len(sampled),
                        "active_head_count": len(active_sampled),
                        "target_head_count": len(sampled),
                        "seed": seed,
                        "control_overlap_count": overlap_count,
                        "control_overlap_fraction": overlap_count / len(sampled) if sampled else 0.0,
                        "target_heads": sorted(sampled),
                        "selected_heads": active_sampled,
                    }
                )
        if include_uniform_random:
            for seed in random_seeds:
                rng = random.Random(seed + budget * 1000)
                selected = sorted(rng.sample(all_heads, budget))
                active_selected = active_for(selected)
                definitions.append(
                    {
                        "subset_name": f"uniform_random_k{budget}_seed{seed}",
                        "family": "uniform_random",
                        "mode": mode,
                        "budget": budget,
                        "subset_size": budget,
                        "active_head_count": len(active_selected),
                        "target_head_count": budget,
                        "seed": seed,
                        "control_overlap_count": 0,
                        "control_overlap_fraction": 0.0,
                        "target_heads": selected,
                        "selected_heads": active_selected,
                    }
                )

    subset_df = pd.DataFrame(
        [
            {
                "subset_name": item["subset_name"],
                "family": item["family"],
                "mode": item["mode"],
                "budget": item["budget"],
                "subset_size": item["subset_size"],
                "active_head_count": item["active_head_count"],
                "target_head_count": item["target_head_count"],
                "seed": item["seed"],
                "control_overlap_count": item.get("control_overlap_count", 0),
                "control_overlap_fraction": item.get("control_overlap_fraction", 0.0),
                "target_heads": ";".join(f"{layer}:{head}" for layer, head in item["target_heads"]),
                "heads": ";".join(f"{layer}:{head}" for layer, head in item["selected_heads"]),
            }
            for item in definitions
        ]
    )
    return definitions, subset_df


def bootstrap_ci(values: np.ndarray, seed: int = 13, iters: int = 1000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = [
        float(rng.choice(values, size=values.size, replace=True).mean())
        for _ in range(iters)
    ]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def build_random_contrasts(sentence_df: pd.DataFrame, subset_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if sentence_df.empty:
        return pd.DataFrame()
    meta_columns = [
        "subset_name",
        "family",
        "mode",
        "budget",
        "active_head_count",
        "target_head_count",
        "seed",
        "control_overlap_count",
        "control_overlap_fraction",
    ]
    meta = subset_df[[col for col in meta_columns if col in subset_df.columns]].drop_duplicates("subset_name")
    work = sentence_df.drop(
        columns=[col for col in ["family", "mode", "budget", "active_head_count", "target_head_count", "seed"] if col in sentence_df.columns],
        errors="ignore",
    ).merge(meta, on="subset_name", how="left")
    work["nll_per_token"] = work["sentence_nll"] / work["predicted_token_count"].replace(0, np.nan)
    full = work[work["subset_name"] == "full_model"][["sent_id", "nll_per_token", "sentence_accuracy"]].rename(
        columns={"nll_per_token": "full_nll_per_token", "sentence_accuracy": "full_accuracy"}
    )
    work = work.merge(full, on="sent_id", how="inner")
    work["delta_nll_per_token"] = work["nll_per_token"] - work["full_nll_per_token"]
    work["delta_accuracy"] = work["sentence_accuracy"] - work["full_accuracy"]

    rows: list[dict[str, object]] = []
    for family in CATEGORY_FAMILIES:
        target = work[work["family"] == family].copy()
        control_family = f"layer_random_for_{family}"
        control = work[work["family"] == control_family].copy()
        if target.empty or control.empty:
            continue
        control_mean = (
            control.groupby(["sent_id", "budget"], sort=False)
            .agg(
                control_delta_nll_per_token=("delta_nll_per_token", "mean"),
                control_delta_accuracy=("delta_accuracy", "mean"),
                control_nll_per_token=("nll_per_token", "mean"),
                control_accuracy=("sentence_accuracy", "mean"),
                control_mean_overlap_fraction=("control_overlap_fraction", "mean")
                if "control_overlap_fraction" in control.columns
                else ("delta_nll_per_token", lambda _series: float("nan")),
            )
            .reset_index()
        )
        merged = target.merge(control_mean, on=["sent_id", "budget"], how="inner")
        if merged.empty:
            continue
        if mode == "sufficiency":
            merged["nll_contrast"] = merged["control_delta_nll_per_token"] - merged["delta_nll_per_token"]
            claim_type = "prediction_sufficient_vs_layer_random"
        else:
            merged["nll_contrast"] = merged["delta_nll_per_token"] - merged["control_delta_nll_per_token"]
            claim_type = "prediction_necessary_vs_layer_random"
        merged["accuracy_contrast"] = merged["delta_accuracy"] - merged["control_delta_accuracy"]
        for budget, budget_df in merged.groupby("budget", sort=True):
            nll_values = budget_df["nll_contrast"].dropna().to_numpy(dtype=np.float64)
            acc_values = budget_df["accuracy_contrast"].dropna().to_numpy(dtype=np.float64)
            nll_ci_low, nll_ci_high = bootstrap_ci(nll_values, seed=17 + int(budget))
            acc_ci_low, acc_ci_high = bootstrap_ci(acc_values, seed=29 + int(budget))
            first = budget_df.iloc[0]
            rows.append(
                {
                    "mode": mode,
                    "family": family,
                    "condition": str(first["subset_name"]),
                    "budget": int(budget),
                    "target_mean_delta_nll_per_token": float(budget_df["delta_nll_per_token"].mean()),
                    "control_mean_delta_nll_per_token": float(budget_df["control_delta_nll_per_token"].mean()),
                    "effect_vs_control_nll": float(budget_df["nll_contrast"].mean()),
                    "effect_nll_ci_low": nll_ci_low,
                    "effect_nll_ci_high": nll_ci_high,
                    "target_mean_accuracy": float(budget_df["sentence_accuracy"].mean()),
                    "control_mean_accuracy": float(budget_df["control_accuracy"].mean()),
                    "control_overlap_fraction": float(budget_df["control_mean_overlap_fraction"].mean())
                    if "control_mean_overlap_fraction" in budget_df.columns
                    else float("nan"),
                    "effect_vs_control_accuracy": float(budget_df["accuracy_contrast"].mean()),
                    "effect_accuracy_ci_low": acc_ci_low,
                    "effect_accuracy_ci_high": acc_ci_high,
                    "sentence_count": int(budget_df["sent_id"].nunique()),
                    "claim_type": claim_type,
                    "claim_supported": bool(nll_ci_low > 0.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_prediction_results(sentence_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_name, subset in sentence_df.groupby("subset_name", sort=True):
        first = subset.iloc[0]
        predicted_tokens = float(subset["predicted_token_count"].sum())
        rows.append(
            {
                "subset_name": subset_name,
                "family": str(first["family"]),
                "subset_size": int(first["subset_size"]),
                "seed": int(first["seed"]),
                "top_k": int(first["top_k"]),
                "num_sentences": int(len(subset)),
                "num_words": int(subset["num_words"].sum()),
                "total_ud_edges": int(subset["ud_edge_count"].sum()),
                "total_ai_edges": int(subset["ai_edge_count"].sum()),
                "total_matched_edges": int(subset["matched_edge_count"].sum()),
                "corpus_overlap_precision": float("nan"),
                "corpus_overlap_recall": float("nan"),
                "corpus_overlap_f1": float("nan"),
                "mean_sentence_overlap_precision": float("nan"),
                "mean_sentence_overlap_recall": float("nan"),
                "mean_sentence_overlap_f1": float("nan"),
                "mean_ud_distance": float(subset["mean_ud_distance"].mean()),
                "mean_ai_distance": float("nan"),
                "mean_relation_match_rate": float("nan"),
                "mean_tree_distance_spearman": float("nan"),
                "mean_next_token_nll": float(subset["sentence_nll"].mean()),
                "mean_next_token_nll_per_token": float(subset["sentence_nll"].sum() / predicted_tokens)
                if predicted_tokens
                else float("nan"),
                "mean_sentence_accuracy": float(subset["sentence_accuracy"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["family", "subset_size", "seed", "top_k"]).reset_index(drop=True)
    return phase2.add_recovery_metrics(summary)


def write_outputs(
    *,
    args: argparse.Namespace,
    ranking_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    sentence_rows: list[dict[str, float | int | str]],
    relation_rows: list[dict[str, float | int | str]],
    processed_sentences: int,
    total_selected_sentences: int,
    final: bool,
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(args)
    ranking_df.to_csv(paths["ranking"], index=False)
    subset_df.to_csv(paths["subset"], index=False)
    sentence_df = pd.DataFrame(sentence_rows)
    relation_df = pd.DataFrame(relation_rows)
    if not sentence_df.empty:
        sentence_df.to_csv(paths["sentence"], index=False)
    if not relation_df.empty:
        relation_df.to_csv(paths["relation_rows"], index=False)
    relation_summary = pd.DataFrame()
    if not relation_df.empty:
        relation_summary = (
            relation_df.groupby(
                ["subset_name", "family", "mode", "budget", "subset_size", "active_head_count", "seed", "top_k", "relation"],
                sort=True,
            )
            .agg(
                num_sentences_with_relation=("sent_id", "nunique"),
                ud_edge_count=("ud_edge_count", "sum"),
                match_count=("match_count", "sum"),
                mean_ud_distance=("ud_distance", "mean"),
                mean_ai_distance=("predicted_distance", "mean"),
            )
            .reset_index()
        )
        relation_summary["match_rate"] = relation_summary["match_count"] / relation_summary["ud_edge_count"]
        relation_summary.to_csv(paths["relation_summary"], index=False)
    if not sentence_df.empty:
        if not relation_summary.empty:
            summary = phase2.summarize_subset_results(sentence_df, relation_summary)
            summary = phase2.add_recovery_metrics(summary)
        else:
            summary = summarize_prediction_results(sentence_df)
        meta_cols = [
            "subset_name",
            "mode",
            "budget",
            "active_head_count",
            "target_head_count",
        ]
        summary = summary.drop(columns=[col for col in meta_cols if col in summary.columns and col != "subset_name"], errors="ignore")
        summary = summary.merge(subset_df[meta_cols].drop_duplicates("subset_name"), on="subset_name", how="left")
        summary.to_csv(paths["summary"], index=False)
        contrast = build_random_contrasts(sentence_df, subset_df, args.mode)
        if not contrast.empty:
            contrast.to_csv(paths["random_contrasts"], index=False)
    progress = {
        "run_name": args.run_name,
        "mode": args.mode,
        "processed_sentences": processed_sentences,
        "total_selected_sentences": total_selected_sentences,
        "final": final,
    }
    paths["progress"].write_text(json.dumps(progress, indent=2), encoding="utf-8")


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(format(value, floatfmt))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def add_random_baseline(ax, contrast: pd.DataFrame) -> None:
    if contrast.empty:
        return
    random_by_budget = (
        contrast.groupby("budget")["control_mean_delta_nll_per_token"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .sort_values("budget")
    )
    ax.fill_between(
        random_by_budget["budget"],
        random_by_budget["min"],
        random_by_budget["max"],
        color="#999999",
        alpha=0.16,
        label="layer-random range",
    )
    ax.plot(
        random_by_budget["budget"],
        random_by_budget["mean"],
        color="#222222",
        linestyle="--",
        linewidth=2.4,
        label="layer-random mean",
    )


def save_mode_figures(results_dir: Path, mode: str, summary: pd.DataFrame, contrast: pd.DataFrame) -> dict[str, Path]:
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if summary.empty:
        return paths
    plot = summary[summary["family"].isin(CATEGORY_FAMILIES)].copy()
    if not plot.empty:
        full_nll = float(summary.loc[summary["subset_name"] == "full_model", "mean_next_token_nll_per_token"].iloc[0])
        plot["delta_nll_vs_full"] = plot["mean_next_token_nll_per_token"] - full_nll
        fig, ax = plt.subplots(figsize=(15.5, 7.0), dpi=180)
        add_random_baseline(ax, contrast)
        for family, family_df in plot.groupby("family", sort=True):
            family_df = family_df.sort_values("budget")
            ax.plot(
                family_df["budget"],
                family_df["delta_nll_vs_full"],
                marker="o",
                markersize=6.0,
                linewidth=2.2,
                label=family,
            )
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_title(f"GPT-2 {mode}: next-token NLL delta vs full")
        ax.set_xlabel("Budget k")
        ax.set_ylabel("Delta NLL/token")
        ax.grid(True, axis="y", alpha=0.18)
        ax.legend(ncol=4, fontsize=8.5, frameon=True)
        fig.tight_layout()
        path = figure_dir / f"gpt2_{mode}_nll_delta_by_budget.png"
        fig.savefig(path)
        plt.close(fig)
        paths["nll_delta"] = path
    if not contrast.empty:
        fig, ax = plt.subplots(figsize=(17.0, 7.5), dpi=180)
        labels: list[str] = []
        values: list[float] = []
        colors: list[str] = []
        for row in contrast.sort_values(["budget", "family"]).itertuples(index=False):
            labels.append(f"{row.family}\nk={int(row.budget)}")
            values.append(float(row.effect_vs_control_nll))
            colors.append("#d95f02" if bool(row.claim_supported) else "#7570b3")
        x = np.arange(len(labels))
        ax.bar(x, values, color=colors, alpha=0.86)
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=68, ha="right", fontsize=8)
        ax.set_title(f"GPT-2 {mode}: effect vs layer-matched random")
        if mode == "sufficiency":
            ax.set_ylabel("Layer-random delta NLL/token - target delta NLL/token")
        else:
            ax.set_ylabel("Target delta NLL/token - layer-random delta NLL/token")
        fig.tight_layout()
        path = figure_dir / f"gpt2_{mode}_effect_vs_layer_random.png"
        fig.savefig(path)
        plt.close(fig)
        paths["effect"] = path
    return paths


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if not args.run_name:
        args.run_name = f"gpt2_medium_bridge_category_{args.mode}"
    if not args.budgets:
        args.budgets = ",".join(str(k) for k in mode_default_budgets(args.mode))
    budgets = sorted(set(parse_int_list(args.budgets)))
    random_seeds = parse_int_list(args.random_seeds)
    top_k_values = parse_int_list(args.top_k_values)
    if len(top_k_values) != 1:
        raise ValueError("Bridge GPT-2 category study currently supports exactly one --top-k-values value.")
    phase1_prefix = args.phase1_prefix.expanduser().resolve()

    if args.force:
        for path in output_paths(args).values():
            path.unlink(missing_ok=True)

    ranking_df, all_heads, _available = phase2.build_rankings(phase1_prefix)
    ranking_df = add_bridge_scores(ranking_df, args.positional_threshold)
    num_heads = int(max(head for _layer, head in all_heads) + 1)
    num_layers_from_rankings = int(max(layer for layer, _head in all_heads) + 1)
    subset_definitions, subset_df = build_subset_definitions(
        ranking_df=ranking_df,
        all_heads=all_heads,
        budgets=budgets,
        random_seeds=random_seeds,
        num_heads=num_heads,
        mode=args.mode,
        include_uniform_random=args.include_uniform_random,
    )

    if args.prepare_only:
        write_outputs(
            args=args,
            ranking_df=ranking_df,
            subset_df=subset_df,
            sentence_rows=[],
            relation_rows=[],
            processed_sentences=0,
            total_selected_sentences=0,
            final=False,
        )
        log("Prepared GPT-2 category subset definitions only.")
        return

    device = base.choose_device(args.device)
    dtype = phase2.resolve_torch_dtype(args.torch_dtype, device)
    log(
        f"GPT-2 category {args.mode} selected device: {device}, dtype={dtype}, "
        f"budgets={budgets}, subsets={len(subset_definitions)}"
    )
    tokenizer, model = phase2.load_model_and_tokenizer(args.model_name, device, dtype)
    if model.config.n_layer != num_layers_from_rankings or model.config.n_head != num_heads:
        raise RuntimeError(
            "Phase-1 rankings are incompatible with the requested model: "
            f"rankings={num_layers_from_rankings}x{num_heads}, "
            f"model={model.config.n_layer}x{model.config.n_head}"
        )

    ud_paths = base.ensure_ud_files(args.cache_dir)
    selected_sentences = base.select_sentences(
        split=args.eval_split,
        ud_paths=ud_paths,
        min_words=args.min_words,
        max_words=args.max_words,
        limit=args.limit,
    )
    if not selected_sentences:
        raise RuntimeError("No GPT-2 evaluation sentences selected.")

    num_layers = int(model.config.n_layer)
    masks = {
        item["subset_name"]: base.make_head_mask(
            num_layers=num_layers,
            num_heads=num_heads,
            active_heads=item["selected_heads"],
            device=device,
        )
        for item in subset_definitions
    }

    if args.prediction_only:
        run_prediction_only_batched(
            args=args,
            ranking_df=ranking_df,
            subset_df=subset_df,
            subset_definitions=subset_definitions,
            selected_sentences=selected_sentences,
            tokenizer=tokenizer,
            model=model,
            masks=masks,
            device=device,
            top_k_values=top_k_values,
        )
        return

    sentence_rows: list[dict[str, float | int | str]] = []
    relation_rows: list[dict[str, float | int | str]] = []
    start_idx = 1
    paths = output_paths(args)
    has_resume_relation_rows = args.prediction_only or paths["relation_rows"].exists()
    if not args.no_resume and paths["progress"].exists() and paths["sentence"].exists() and has_resume_relation_rows:
        progress = json.loads(paths["progress"].read_text(encoding="utf-8"))
        if progress.get("run_name") == args.run_name and progress.get("mode") == args.mode and not progress.get("final", False):
            processed = int(progress.get("processed_sentences", 0))
            sentence_rows = pd.read_csv(paths["sentence"]).to_dict("records")
            if paths["relation_rows"].exists():
                relation_rows = pd.read_csv(paths["relation_rows"]).to_dict("records")
            start_idx = processed + 1
            log(f"Resuming {args.mode} from sentence {start_idx}/{len(selected_sentences)}")
    log(f"Evaluating GPT-2 category {args.mode} on {len(selected_sentences)} sentences")

    for sent_idx, example in enumerate(selected_sentences[start_idx - 1 :], start=start_idx):
        try:
            encoding = base.encode_words(tokenizer, example.words)
        except ValueError as exc:
            log(f"Skipping {example.sent_id}: {exc}")
            continue
        word_ids = encoding.word_ids(batch_index=0)
        assert word_ids is not None
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        gold_edges = base.ud_edges(example)
        ud_distance = base.edge_distance(gold_edges)
        if args.compute_tree_distance:
            gold_matrix = base.shortest_path_matrix(gold_edges, example.length)
            gold_values = base.upper_triangle_values(gold_matrix)
        else:
            gold_values = np.array([], dtype=np.float64)

        for item in subset_definitions:
            subset_name = str(item["subset_name"])
            selected_heads = list(item["selected_heads"])
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    head_mask=masks[subset_name],
                    output_attentions=not args.prediction_only,
                    return_dict=True,
                )
            sentence_nll, correct_token_count, predicted_token_count = base.next_token_metrics_from_logits(
                outputs.logits[0],
                input_ids=input_ids,
            )
            if args.prediction_only:
                for top_k in top_k_values:
                    sentence_rows.append(
                        {
                            "sent_id": example.sent_id,
                            "split": example.split,
                            "subset_name": subset_name,
                            "family": str(item["family"]),
                            "mode": str(item["mode"]),
                            "budget": int(item["budget"]),
                            "subset_size": int(item["subset_size"]),
                            "active_head_count": int(item["active_head_count"]),
                            "target_head_count": int(item["target_head_count"]),
                            "seed": int(item["seed"]),
                            "top_k": int(top_k),
                            "num_words": example.length,
                            "ud_edge_count": len(gold_edges),
                            "ai_edge_count": 0,
                            "matched_edge_count": 0,
                            "sentence_precision": float("nan"),
                            "sentence_recall": float("nan"),
                            "sentence_f1": float("nan"),
                            "mean_ud_distance": ud_distance,
                            "mean_ai_distance": float("nan"),
                            "tree_distance_spearman": float("nan"),
                            "sentence_nll": sentence_nll,
                            "predicted_token_count": predicted_token_count,
                            "sentence_accuracy": correct_token_count / predicted_token_count if predicted_token_count else float("nan"),
                        }
                    )
                continue

            combined_attention = phase2.subset_attention_matrix(
                attentions=outputs.attentions,
                word_ids=word_ids,
                num_words=example.length,
                selected_heads=selected_heads,
            )
            for top_k in top_k_values:
                ai_edges = base.build_attention_edges_topk(combined_attention, top_k)
                matched, gold_total, predicted_total = base.overlap_counts(gold_edges, ai_edges)
                precision, recall, f1 = base.precision_recall_f1(matched, gold_total, predicted_total)
                selected_edge_set = base.edge_set(ai_edges)
                primary_targets = base.primary_targets_from_topk(combined_attention, top_k)
                if args.compute_tree_distance:
                    ai_matrix = base.shortest_path_matrix(ai_edges, example.length)
                    tree_spearman = phase2.safe_spearman(gold_values, base.upper_triangle_values(ai_matrix))
                else:
                    tree_spearman = float("nan")
                sentence_rows.append(
                    {
                        "sent_id": example.sent_id,
                        "split": example.split,
                        "subset_name": subset_name,
                        "family": str(item["family"]),
                        "mode": str(item["mode"]),
                        "budget": int(item["budget"]),
                        "subset_size": int(item["subset_size"]),
                        "active_head_count": int(item["active_head_count"]),
                        "target_head_count": int(item["target_head_count"]),
                        "seed": int(item["seed"]),
                        "top_k": int(top_k),
                        "num_words": example.length,
                        "ud_edge_count": gold_total,
                        "ai_edge_count": predicted_total,
                        "matched_edge_count": matched,
                        "sentence_precision": precision,
                        "sentence_recall": recall,
                        "sentence_f1": f1,
                        "mean_ud_distance": ud_distance,
                        "mean_ai_distance": base.edge_distance(ai_edges),
                        "tree_distance_spearman": tree_spearman,
                        "sentence_nll": sentence_nll,
                        "predicted_token_count": predicted_token_count,
                        "sentence_accuracy": correct_token_count / predicted_token_count if predicted_token_count else float("nan"),
                    }
                )
                for row in base.relation_match_summary(example, primary_targets, selected_edge_set):
                    relation_rows.append(
                        {
                            "sent_id": example.sent_id,
                            "split": example.split,
                            "subset_name": subset_name,
                            "family": str(item["family"]),
                            "mode": str(item["mode"]),
                            "budget": int(item["budget"]),
                            "subset_size": int(item["subset_size"]),
                            "active_head_count": int(item["active_head_count"]),
                            "target_head_count": int(item["target_head_count"]),
                            "seed": int(item["seed"]),
                            "top_k": int(top_k),
                            **row,
                        }
                    )

        if sent_idx % 10 == 0 or sent_idx == len(selected_sentences):
            log(f"GPT-2 category {args.mode} processed {sent_idx}/{len(selected_sentences)}")
        if args.checkpoint_every > 0 and sent_idx % args.checkpoint_every == 0:
            write_outputs(
                args=args,
                ranking_df=ranking_df,
                subset_df=subset_df,
                sentence_rows=sentence_rows,
                relation_rows=relation_rows,
                processed_sentences=sent_idx,
                total_selected_sentences=len(selected_sentences),
                final=False,
            )

    write_outputs(
        args=args,
        ranking_df=ranking_df,
        subset_df=subset_df,
        sentence_rows=sentence_rows,
        relation_rows=relation_rows,
        processed_sentences=len(selected_sentences),
        total_selected_sentences=len(selected_sentences),
        final=True,
    )
    log(f"GPT-2 category {args.mode} outputs written to {args.results_dir}")


if __name__ == "__main__":
    main()

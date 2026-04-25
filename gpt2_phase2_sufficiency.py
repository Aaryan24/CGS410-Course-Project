#!/usr/bin/env python3
"""
Phase 2 retained-head sufficiency evaluation for the GPT-2 track.

This script consumes Phase 1 outputs, builds several retained-head subsets,
and evaluates whether those subsets preserve:

- dependency-grammar signals against UD
- relation-level recovery
- next-token predictive quality

Unlike Phase 1, the evaluation here operates on retained multi-head subsets
rather than individual heads.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gpt2_ud_utils as base


DEFAULT_MODEL_NAME = "gpt2-medium"
DEFAULT_FAMILIES = "full,top_structural,top_syntactic,top_hybrid,layer_balanced,random"
DEFAULT_SUBSET_SIZES = "4,8,12,16,24,32,48,64,96,128,192,256"
DEFAULT_RANDOM_SEEDS = "0,1,2,3,4"


def resolve_torch_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def parse_text_list(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"No values found in '{text}'")
    return values


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    result = spearmanr(x, y).statistic
    return float(result) if result is not None else float("nan")


def load_model_and_tokenizer(model_name: str, device: torch.device, torch_dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, add_prefix_space=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        dtype=torch_dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def normalize_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
    minimum = float(valid.min())
    maximum = float(valid.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(series), dtype=np.float64), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def build_rankings(phase1_prefix: Path) -> tuple[pd.DataFrame, list[tuple[int, int]], dict[str, str | None]]:
    summary_path = phase1_prefix.parent / f"{phase1_prefix.name}_summary.csv"
    head_path = phase1_prefix.parent / f"{phase1_prefix.name}_head_characterization.csv"
    relation_path = phase1_prefix.parent / f"{phase1_prefix.name}_relation_characterization.csv"
    if not summary_path.exists() or not head_path.exists() or not relation_path.exists():
        raise FileNotFoundError("Phase 1 outputs are incomplete or missing.")

    summary_df = pd.read_csv(summary_path)
    head_df = pd.read_csv(head_path)
    relation_df = pd.read_csv(relation_path)

    structural_df = (
        summary_df[
            (summary_df["analysis_mode"] == "head")
            & (summary_df["top_k"] == 1)
        ][
            [
                "layer",
                "head_index",
                "mean_sentence_overlap_f1",
                "mean_relation_match_rate",
                "mean_tree_distance_spearman",
                "mean_ai_distance",
            ]
        ]
        .rename(
            columns={
                "mean_sentence_overlap_f1": "structural_f1",
                "mean_relation_match_rate": "structural_relation_match_rate",
                "mean_tree_distance_spearman": "structural_tree_spearman",
                "mean_ai_distance": "structural_mean_ai_distance",
            }
        )
    )

    relation_scores = (
        relation_df.assign(
            positive_margin=relation_df["accuracy_margin"].clip(lower=0.0),
            syntactic_hit=relation_df["is_syntactic_relation"].fillna(False).astype(int),
        )
        .groupby(["layer", "head_index"], sort=True)
        .agg(
            syntactic_margin_sum=("positive_margin", "sum"),
            syntactic_relation_count_from_rel=("syntactic_hit", "sum"),
            max_relation_margin=("accuracy_margin", "max"),
        )
        .reset_index()
    )

    ranking_df = (
        head_df.merge(structural_df, on=["layer", "head_index"], how="left")
        .merge(relation_scores, on=["layer", "head_index"], how="left")
        .fillna(
            {
                "structural_f1": 0.0,
                "structural_relation_match_rate": 0.0,
                "structural_tree_spearman": 0.0,
                "structural_mean_ai_distance": float("nan"),
                "syntactic_margin_sum": 0.0,
                "syntactic_relation_count_from_rel": 0,
                "max_relation_margin": 0.0,
            }
        )
    )

    ranking_df["structural_score"] = normalize_series(ranking_df["structural_f1"])
    ranking_df["syntactic_score"] = (
        0.6 * normalize_series(ranking_df["syntactic_margin_sum"])
        + 0.4 * normalize_series(ranking_df["syntactic_relation_count"])
    )
    ranking_df["hybrid_score"] = (
        0.45 * normalize_series(ranking_df["structural_f1"])
        + 0.20 * normalize_series(ranking_df["structural_tree_spearman"])
        + 0.20 * normalize_series(ranking_df["syntactic_margin_sum"])
        + 0.15 * normalize_series(ranking_df["syntactic_relation_count"])
    )

    importance_available = ranking_df["importance_delta_nll_mean"].nunique(dropna=True) > 1
    if importance_available:
        ranking_df["importance_score"] = normalize_series(ranking_df["importance_delta_nll_mean"])
    else:
        ranking_df["importance_score"] = 0.0

    ranking_df = ranking_df.sort_values(["hybrid_score", "structural_f1"], ascending=[False, False]).reset_index(drop=True)
    all_heads = [
        (int(row.layer), int(row.head_index))
        for row in ranking_df.itertuples(index=False)
    ]
    available_families = {
        "top_structural": "structural_score",
        "top_syntactic": "syntactic_score",
        "top_hybrid": "hybrid_score",
        "top_importance": "importance_score" if importance_available else None,
    }
    return ranking_df, all_heads, available_families


def select_top_heads(ranking_df: pd.DataFrame, score_column: str, subset_size: int) -> list[tuple[int, int]]:
    subset = ranking_df.sort_values(
        [score_column, "structural_f1", "syntactic_margin_sum", "layer", "head_index"],
        ascending=[False, False, False, True, True],
    ).head(subset_size)
    return [(int(row.layer), int(row.head_index)) for row in subset.itertuples(index=False)]


def select_layer_balanced_heads(ranking_df: pd.DataFrame, score_column: str, subset_size: int) -> list[tuple[int, int]]:
    per_layer: dict[int, list[tuple[int, int]]] = {}
    layer_priority = []
    for layer, layer_df in ranking_df.groupby("layer", sort=True):
        ordered = [
            (int(row.layer), int(row.head_index))
            for row in layer_df.sort_values(
                [score_column, "structural_f1", "syntactic_margin_sum", "head_index"],
                ascending=[False, False, False, True],
            ).itertuples(index=False)
        ]
        per_layer[int(layer)] = ordered
        top_score = float(layer_df[score_column].max())
        layer_priority.append((top_score, int(layer)))

    selected: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    ordered_layers = [layer for _, layer in sorted(layer_priority, reverse=True)]

    while len(selected) < subset_size:
        progressed = False
        for layer in ordered_layers:
            choices = per_layer[layer]
            while choices and choices[0] in used:
                choices.pop(0)
            if not choices:
                continue
            head = choices.pop(0)
            selected.append(head)
            used.add(head)
            progressed = True
            if len(selected) >= subset_size:
                break
        if not progressed:
            break
    return selected


def build_subset_definitions(
    ranking_df: pd.DataFrame,
    all_heads: list[tuple[int, int]],
    available_families: dict[str, str | None],
    families: list[str],
    subset_sizes: list[int],
    random_seeds: list[int],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    definitions: list[dict[str, object]] = []

    definitions.append(
        {
            "subset_name": "full_model",
            "family": "full",
            "subset_size": len(all_heads),
            "seed": -1,
            "score_column": "full_model",
            "selected_heads": list(all_heads),
        }
    )

    for family in families:
        if family == "full":
            continue
        if family == "random":
            for subset_size in subset_sizes:
                for seed in random_seeds:
                    rng = random.Random(seed)
                    selected = sorted(rng.sample(all_heads, subset_size))
                    definitions.append(
                        {
                            "subset_name": f"random_k{subset_size}_seed{seed}",
                            "family": "random",
                            "subset_size": subset_size,
                            "seed": seed,
                            "score_column": "random",
                            "selected_heads": selected,
                        }
                    )
            continue

        if family == "layer_balanced":
            for subset_size in subset_sizes:
                selected = select_layer_balanced_heads(ranking_df, "hybrid_score", subset_size)
                definitions.append(
                    {
                        "subset_name": f"layer_balanced_k{subset_size}",
                        "family": "layer_balanced",
                        "subset_size": subset_size,
                        "seed": -1,
                        "score_column": "hybrid_score",
                        "selected_heads": selected,
                    }
                )
            continue

        score_column = available_families.get(family)
        if score_column is None:
            continue
        for subset_size in subset_sizes:
            selected = select_top_heads(ranking_df, score_column, subset_size)
            definitions.append(
                {
                    "subset_name": f"{family}_k{subset_size}",
                    "family": family,
                    "subset_size": subset_size,
                    "seed": -1,
                    "score_column": score_column,
                    "selected_heads": selected,
                }
            )

    subset_df = pd.DataFrame(
        [
            {
                "subset_name": item["subset_name"],
                "family": item["family"],
                "subset_size": item["subset_size"],
                "seed": item["seed"],
                "score_column": item["score_column"],
                "heads": ";".join(f"{layer}:{head}" for layer, head in item["selected_heads"]),
            }
            for item in definitions
        ]
    )
    return definitions, subset_df


def subset_attention_matrix(
    attentions: tuple[torch.Tensor, ...],
    word_ids: list[int | None],
    num_words: int,
    selected_heads: list[tuple[int, int]],
) -> np.ndarray:
    selected_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer_idx, head_idx in selected_heads:
        selected_by_layer[layer_idx].append(head_idx)

    collected: list[np.ndarray] = []
    for layer_idx, layer_heads in selected_by_layer.items():
        layer_attention = attentions[layer_idx][0].detach().to(torch.float32).cpu().numpy()
        views = base.aggregate_word_attention_views(
            layer_attention=layer_attention,
            word_ids=word_ids,
            num_words=num_words,
            head_mode="all",
            selected_heads=sorted(layer_heads),
        )
        for _, _, word_attention in views:
            collected.append(word_attention)

    if not collected:
        raise RuntimeError("No attention views were available for the selected subset.")
    return np.mean(np.stack(collected, axis=0), axis=0)


def summarize_subset_results(sentence_df: pd.DataFrame, relation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subset_name, subset_df in sentence_df.groupby("subset_name", sort=True):
        relation_subset = relation_df[relation_df["subset_name"] == subset_name]
        first = subset_df.iloc[0]
        ai_edge_total = int(subset_df["ai_edge_count"].sum())
        matched_total = int(subset_df["matched_edge_count"].sum())
        ud_edge_total = int(subset_df["ud_edge_count"].sum())
        precision = matched_total / ai_edge_total if ai_edge_total else 0.0
        recall = matched_total / ud_edge_total if ud_edge_total else 0.0
        f1 = 0.0 if precision + recall == 0.0 else (2.0 * precision * recall / (precision + recall))
        rows.append(
            {
                "subset_name": subset_name,
                "family": str(first["family"]),
                "subset_size": int(first["subset_size"]),
                "seed": int(first["seed"]),
                "top_k": int(first["top_k"]),
                "num_sentences": int(len(subset_df)),
                "num_words": int(subset_df["num_words"].sum()),
                "total_ud_edges": ud_edge_total,
                "total_ai_edges": ai_edge_total,
                "total_matched_edges": matched_total,
                "corpus_overlap_precision": precision,
                "corpus_overlap_recall": recall,
                "corpus_overlap_f1": f1,
                "mean_sentence_overlap_precision": float(subset_df["sentence_precision"].mean()),
                "mean_sentence_overlap_recall": float(subset_df["sentence_recall"].mean()),
                "mean_sentence_overlap_f1": float(subset_df["sentence_f1"].mean()),
                "mean_ud_distance": float(subset_df["mean_ud_distance"].mean()),
                "mean_ai_distance": float(subset_df["mean_ai_distance"].mean()),
                "mean_relation_match_rate": float(relation_subset["match_rate"].mean()) if not relation_subset.empty else float("nan"),
                "mean_tree_distance_spearman": float(subset_df["tree_distance_spearman"].mean()),
                "mean_next_token_nll": float(subset_df["sentence_nll"].mean()),
                "mean_next_token_nll_per_token": float(
                    subset_df["sentence_nll"].sum() / subset_df["predicted_token_count"].sum()
                    if subset_df["predicted_token_count"].sum()
                    else float("nan")
                ),
                "mean_sentence_accuracy": float(subset_df["sentence_accuracy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "subset_size", "seed", "top_k"]).reset_index(drop=True)


def add_recovery_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    full_rows = summary_df[summary_df["subset_name"] == "full_model"]
    if full_rows.empty:
        return summary_df

    full = full_rows.iloc[0]
    full_accuracy = float(full["mean_sentence_accuracy"])
    full_nll = float(full["mean_next_token_nll_per_token"])
    full_syntax_f1 = float(full["mean_sentence_overlap_f1"])
    full_tree_spearman = float(full["mean_tree_distance_spearman"])
    full_corpus_f1 = float(full["corpus_overlap_f1"])

    result = summary_df.copy()
    result["accuracy_gap_vs_full"] = full_accuracy - result["mean_sentence_accuracy"]
    result["accuracy_retention_vs_full"] = np.where(
        full_accuracy == 0.0,
        np.nan,
        result["mean_sentence_accuracy"] / full_accuracy,
    )
    result["nll_gap_vs_full"] = result["mean_next_token_nll_per_token"] - full_nll
    result["nll_ratio_vs_full"] = np.where(
        full_nll == 0.0,
        np.nan,
        result["mean_next_token_nll_per_token"] / full_nll,
    )
    result["syntax_f1_delta_vs_full"] = result["mean_sentence_overlap_f1"] - full_syntax_f1
    result["corpus_f1_delta_vs_full"] = result["corpus_overlap_f1"] - full_corpus_f1
    result["tree_spearman_delta_vs_full"] = result["mean_tree_distance_spearman"] - full_tree_spearman
    result["accuracy_retention_pct"] = result["accuracy_retention_vs_full"] * 100.0
    return result


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
    ranking_path = args.results_dir / f"{args.run_name}_ranking.csv"
    subset_path = args.results_dir / f"{args.run_name}_subset_definitions.csv"
    sentence_path = args.results_dir / f"{args.run_name}_sentence.csv"
    relation_path = args.results_dir / f"{args.run_name}_relation_summary.csv"
    summary_path = args.results_dir / f"{args.run_name}_summary.csv"
    progress_path = args.results_dir / f"{args.run_name}_progress.json"

    ranking_df.to_csv(ranking_path, index=False)
    subset_df.to_csv(subset_path, index=False)

    sentence_df = pd.DataFrame(sentence_rows)
    relation_df = pd.DataFrame(relation_rows)
    if not sentence_df.empty:
        sentence_df.to_csv(sentence_path, index=False)
    if not relation_df.empty:
        relation_summary = (
            relation_df.groupby(
                ["subset_name", "family", "subset_size", "seed", "top_k", "relation"],
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
        relation_summary.to_csv(relation_path, index=False)
    else:
        relation_summary = pd.DataFrame()
    if not sentence_df.empty and not relation_summary.empty:
        summary_df = summarize_subset_results(sentence_df, relation_summary)
        summary_df = add_recovery_metrics(summary_df)
        summary_df.to_csv(summary_path, index=False)

    progress_path.write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "processed_sentences": processed_sentences,
                "total_selected_sentences": total_selected_sentences,
                "final": final,
            },
            indent=2,
        )
    )

    if final:
        base.log(f"Wrote ranking to {ranking_path}")
        base.log(f"Wrote subset definitions to {subset_path}")
        base.log(f"Wrote sentence-level metrics to {sentence_path}")
        base.log(f"Wrote relation summary to {relation_path}")
        base.log(f"Wrote summary to {summary_path}")
    else:
        base.log(
            f"Checkpoint saved after {processed_sentences}/{total_selected_sentences} sentences "
            f"to {args.results_dir}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retained-head sufficiency evaluation for GPT-2.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--phase1-prefix", type=Path, required=True, help="Path prefix for Phase 1 outputs without suffix.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ud_english_ewt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="gpt2_medium_phase2_sufficiency")
    parser.add_argument("--eval-split", choices=("train", "dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--top-k-values", default="1")
    parser.add_argument("--subset-sizes", default=DEFAULT_SUBSET_SIZES)
    parser.add_argument("--families", default=DEFAULT_FAMILIES)
    parser.add_argument("--random-seeds", default=DEFAULT_RANDOM_SEEDS)
    parser.add_argument("--compute-tree-distance", action="store_true")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write partial outputs every N evaluation sentences. Use 0 to disable checkpointing.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = base.parse_int_list(args.top_k_values)
    subset_sizes = sorted(set(base.parse_int_list(args.subset_sizes)))
    random_seeds = base.parse_int_list(args.random_seeds)
    families = parse_text_list(args.families)

    phase1_prefix = args.phase1_prefix.expanduser().resolve()
    ranking_df, all_heads, available_families = build_rankings(phase1_prefix)
    subset_definitions, subset_df = build_subset_definitions(
        ranking_df=ranking_df,
        all_heads=all_heads,
        available_families=available_families,
        families=families,
        subset_sizes=subset_sizes,
        random_seeds=random_seeds,
    )

    ud_paths = base.ensure_ud_files(args.cache_dir)
    device_request = "cpu" if args.prepare_only and args.device == "auto" else args.device
    device = base.choose_device(device_request)
    torch_dtype = resolve_torch_dtype(args.torch_dtype, device)

    base.log(
        f"Selected device: {device} (dtype={torch_dtype}) "
        f"for eval_split={args.eval_split} with {len(subset_definitions)} subsets."
    )
    if args.wait_for_gpu and device.type == "mps":
        base.wait_for_gpu_if_needed(args.poll_interval)

    tokenizer, model = load_model_and_tokenizer(args.model_name, device, torch_dtype)

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
        base.log("Preparation complete. Exiting without inference.")
        return

    selected_sentences = base.select_sentences(
        split=args.eval_split,
        ud_paths=ud_paths,
        min_words=args.min_words,
        max_words=args.max_words,
        limit=args.limit,
    )
    if not selected_sentences:
        raise RuntimeError("No evaluation sentences remained after filtering.")

    num_layers = int(model.config.n_layer)
    num_heads = int(model.config.n_head)
    max_layer = max(layer for layer, _ in all_heads)
    max_head = max(head for _, head in all_heads)
    if max_layer >= num_layers or max_head >= num_heads:
        raise RuntimeError(
            "Phase 1 rankings are incompatible with the requested model. "
            f"Phase 1 needs at least {max_layer + 1} layers and {max_head + 1} heads, "
            f"but {args.model_name} exposes {num_layers} layers and {num_heads} heads."
        )
    subset_masks = {
        item["subset_name"]: base.make_head_mask(
            num_layers=num_layers,
            num_heads=num_heads,
            active_heads=item["selected_heads"],
            device=device,
        )
        for item in subset_definitions
    }

    sentence_rows: list[dict[str, float | int | str]] = []
    relation_rows: list[dict[str, float | int | str]] = []

    base.log(
        f"Evaluating {len(selected_sentences)} sentences across {len(subset_definitions)} subsets "
        f"for model={args.model_name}"
    )

    for sent_idx, example in enumerate(selected_sentences, start=1):
        try:
            encoding = base.encode_words(tokenizer, example.words)
        except ValueError as exc:
            base.log(f"Skipping {example.sent_id}: {exc}")
            continue

        word_ids = encoding.word_ids(batch_index=0)
        assert word_ids is not None
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        gold_edges = base.ud_edges(example)
        ud_distance = base.edge_distance(gold_edges)
        gold_matrix = None
        if args.compute_tree_distance:
            gold_matrix = base.shortest_path_matrix(gold_edges, example.length)
            gold_values = base.upper_triangle_values(gold_matrix)
        else:
            gold_values = np.array([], dtype=np.float64)

        for item in subset_definitions:
            subset_name = str(item["subset_name"])
            family = str(item["family"])
            subset_size = int(item["subset_size"])
            seed = int(item["seed"])
            selected_heads = list(item["selected_heads"])
            head_mask = subset_masks[subset_name]

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    head_mask=head_mask,
                    output_attentions=True,
                    return_dict=True,
                )

            sentence_nll, correct_token_count, predicted_token_count = base.next_token_metrics_from_logits(
                outputs.logits[0],
                input_ids=input_ids,
            )
            combined_attention = subset_attention_matrix(
                attentions=outputs.attentions,
                word_ids=word_ids,
                num_words=example.length,
                selected_heads=selected_heads,
            )

            for top_k in top_k_values:
                ai_edges = base.build_attention_edges_topk(combined_attention, top_k)
                matched, gold_total, predicted_total = base.overlap_counts(gold_edges, ai_edges)
                precision, recall, f1 = base.precision_recall_f1(matched, gold_total, predicted_total)
                ai_distance = base.edge_distance(ai_edges)
                selected_edge_set = base.edge_set(ai_edges)
                primary_targets = base.primary_targets_from_topk(combined_attention, top_k)

                tree_spearman = float("nan")
                if args.compute_tree_distance and gold_matrix is not None:
                    ai_matrix = base.shortest_path_matrix(ai_edges, example.length)
                    ai_values = base.upper_triangle_values(ai_matrix)
                    tree_spearman = safe_spearman(gold_values, ai_values)

                sentence_rows.append(
                    {
                        "sent_id": example.sent_id,
                        "split": example.split,
                        "subset_name": subset_name,
                        "family": family,
                        "subset_size": subset_size,
                        "seed": seed,
                        "top_k": top_k,
                        "num_words": example.length,
                        "ud_edge_count": gold_total,
                        "ai_edge_count": predicted_total,
                        "matched_edge_count": matched,
                        "sentence_precision": precision,
                        "sentence_recall": recall,
                        "sentence_f1": f1,
                        "mean_ud_distance": ud_distance,
                        "mean_ai_distance": ai_distance,
                        "tree_distance_spearman": tree_spearman,
                        "sentence_nll": sentence_nll,
                        "predicted_token_count": predicted_token_count,
                        "sentence_accuracy": (
                            correct_token_count / predicted_token_count if predicted_token_count else float("nan")
                        ),
                    }
                )

                for row in base.relation_match_summary(example, primary_targets, selected_edge_set):
                    relation_rows.append(
                        {
                            "sent_id": example.sent_id,
                            "split": example.split,
                            "subset_name": subset_name,
                            "family": family,
                            "subset_size": subset_size,
                            "seed": seed,
                            "top_k": top_k,
                            **row,
                        }
                    )

        if sent_idx % 10 == 0 or sent_idx == len(selected_sentences):
            base.log(f"Processed {sent_idx}/{len(selected_sentences)} evaluation sentences")
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


if __name__ == "__main__":
    main()

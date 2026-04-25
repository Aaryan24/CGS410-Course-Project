#!/usr/bin/env python3
"""
First-pass GPT-2 head analysis for the causal-LM track.

This script mirrors the earlier mBERT infrastructure where useful, but narrows
the initial scope to:

- English UD sentence loading
- word-level GPT-2 attention aggregation
- per-head / layer-mean attention graphs
- global overlap with UD edges
- relation-level match rates
- optional tree-distance correlation

It intentionally does not yet implement the later sufficiency experiments. The
goal of this first slice is to give the GPT-2 stream a working structural
analysis pipeline that can feed those later experiments.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gpt2_ud_utils as base


DEFAULT_MODEL_NAME = "gpt2-medium"
POSITIONAL_SHARE_THRESHOLD = 0.9
SYNTACTIC_MARGIN_THRESHOLD = 0.10
RARE_WORD_RATE_THRESHOLD = 0.5


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a first-pass GPT-2 UD head analysis.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ud_english_ewt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="gpt2_medium_head_analysis")
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--layers", default="all")
    parser.add_argument(
        "--head-mode",
        choices=("mean", "all", "both"),
        default="all",
        help="Evaluate layer means, individual heads, or both.",
    )
    parser.add_argument("--heads", default="all")
    parser.add_argument("--top-k-values", default="1")
    parser.add_argument(
        "--compute-tree-distance",
        action="store_true",
        help="Compute sentence-level graph-distance Spearman correlation. Slower but useful.",
    )
    parser.add_argument(
        "--importance-limit",
        type=int,
        default=50,
        help="Number of sentences to use for per-head NLL ablation importance. Use 0 to reuse all selected sentences.",
    )
    parser.add_argument(
        "--relation-min-count",
        type=int,
        default=20,
        help="Minimum opportunity count before a relation can qualify as syntactic for a head.",
    )
    parser.add_argument(
        "--positional-threshold",
        type=float,
        default=POSITIONAL_SHARE_THRESHOLD,
        help="Minimum share on one relative position to call a head positional.",
    )
    parser.add_argument(
        "--syntactic-margin-threshold",
        type=float,
        default=SYNTACTIC_MARGIN_THRESHOLD,
        help="Minimum accuracy lift over the positional baseline to call a head syntactic for a relation.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=float,
        default=RARE_WORD_RATE_THRESHOLD,
        help="Minimum rare-target hit rate to flag a head as a rare-word head.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Write partial CSV outputs every N processed sentences. Use 0 to disable checkpointing.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model loading dtype. 'auto' uses bfloat16 on CUDA and float32 elsewhere.",
    )
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def config_name(layer_idx: int, top_k: int, head_label: str) -> str:
    if head_label == "mean":
        return f"layer{layer_idx}_top{top_k}"
    return f"layer{layer_idx}_{head_label}_top{top_k}"


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


def empty_head_stats() -> dict[str, object]:
    return {
        "eligible_positions": 0,
        "confidence_sum": 0.0,
        "offset_counts": {},
        "rare_hits": 0,
        "rare_eligible": 0,
        "relation_match": {},
        "importance_delta_nll_sum": 0.0,
        "importance_delta_nll_count": 0,
    }


def merge_count(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


def summarize(
    sentence_df: pd.DataFrame,
    relation_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for config, config_df in sentence_df.groupby("config_name", sort=True):
        relation_subset = relation_df[relation_df["config_name"] == config]
        first_row = config_df.iloc[0]
        rows.append(
            {
                "config_name": config,
                "analysis_mode": str(first_row["analysis_mode"]),
                "layer": int(first_row["layer"]),
                "head_label": str(first_row["head_label"]),
                "head_index": int(first_row["head_index"]),
                "top_k": int(first_row["top_k"]),
                "num_sentences": int(len(config_df)),
                "num_words": int(config_df["num_words"].sum()),
                "total_ud_edges": int(config_df["ud_edge_count"].sum()),
                "total_ai_edges": int(config_df["ai_edge_count"].sum()),
                "total_matched_edges": int(config_df["matched_edge_count"].sum()),
                "corpus_overlap_precision": float(config_df["matched_edge_count"].sum() / config_df["ai_edge_count"].sum()),
                "corpus_overlap_recall": float(config_df["matched_edge_count"].sum() / config_df["ud_edge_count"].sum()),
                "corpus_overlap_f1": float(
                    0.0
                    if (
                        config_df["matched_edge_count"].sum() == 0
                        or (config_df["ai_edge_count"].sum() + config_df["ud_edge_count"].sum()) == 0
                    )
                    else (
                        2.0
                        * (config_df["matched_edge_count"].sum() / config_df["ai_edge_count"].sum())
                        * (config_df["matched_edge_count"].sum() / config_df["ud_edge_count"].sum())
                        / (
                            (config_df["matched_edge_count"].sum() / config_df["ai_edge_count"].sum())
                            + (config_df["matched_edge_count"].sum() / config_df["ud_edge_count"].sum())
                        )
                    )
                ),
                "mean_sentence_overlap_precision": float(config_df["sentence_precision"].mean()),
                "mean_sentence_overlap_recall": float(config_df["sentence_recall"].mean()),
                "mean_sentence_overlap_f1": float(config_df["sentence_f1"].mean()),
                "mean_ud_distance": float(config_df["mean_ud_distance"].mean()),
                "mean_ai_distance": float(config_df["mean_ai_distance"].mean()),
                "mean_relation_match_rate": float(relation_subset["match_rate"].mean()) if not relation_subset.empty else float("nan"),
                "mean_tree_distance_spearman": float(config_df["tree_distance_spearman"].mean()),
                "mean_next_token_nll": float(config_df["sentence_nll"].mean()),
                "mean_next_token_nll_per_token": float(
                    (config_df["sentence_nll"].sum() / config_df["predicted_token_count"].sum())
                    if config_df["predicted_token_count"].sum()
                    else float("nan")
                ),
                "mean_sentence_accuracy": float(config_df["sentence_accuracy"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis_mode", "layer", "head_index", "top_k"]).reset_index(drop=True)


def build_head_characterization(
    head_stats: dict[tuple[int, int], dict[str, object]],
    relation_baselines: dict[str, dict[str, float | int]],
    relation_min_count: int,
    positional_threshold: float,
    syntactic_margin_threshold: float,
    rare_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    head_rows: list[dict[str, float | int | str]] = []
    relation_rows: list[dict[str, float | int | str]] = []

    for (layer_idx, head_idx), stats in sorted(head_stats.items()):
        eligible_positions = int(stats["eligible_positions"])
        confidence_mean = float(stats["confidence_sum"] / eligible_positions) if eligible_positions else float("nan")

        offset_counts = {int(key): int(value) for key, value in dict(stats["offset_counts"]).items()}
        if offset_counts:
            best_offset, best_offset_count = max(offset_counts.items(), key=lambda item: item[1])
            positional_share = best_offset_count / eligible_positions
        else:
            best_offset = 0
            positional_share = float("nan")
        is_positional = bool(eligible_positions and positional_share >= positional_threshold)

        rare_eligible = int(stats["rare_eligible"])
        rare_hits = int(stats["rare_hits"])
        rare_word_rate = rare_hits / rare_eligible if rare_eligible else float("nan")
        is_rare_word_head = bool(rare_eligible and rare_word_rate >= rare_threshold)

        relation_match = dict(stats["relation_match"])
        syntactic_relations: list[str] = []
        best_relation = ""
        best_relation_accuracy = float("nan")
        best_relation_margin = float("nan")

        for relation, baseline_payload in relation_baselines.items():
            opportunity_count = int(baseline_payload["opportunity_count"])
            if opportunity_count == 0:
                continue
            match_count = int(relation_match.get(relation, 0))
            accuracy = match_count / opportunity_count
            baseline_accuracy = float(baseline_payload["baseline_accuracy"])
            margin = accuracy - baseline_accuracy
            is_syntactic = bool(opportunity_count >= relation_min_count and margin >= syntactic_margin_threshold)
            if is_syntactic:
                syntactic_relations.append(relation)
            relation_rows.append(
                {
                    "layer": layer_idx,
                    "head_index": head_idx,
                    "relation": relation,
                    "opportunity_count": opportunity_count,
                    "match_count": match_count,
                    "accuracy": accuracy,
                    "baseline_accuracy": baseline_accuracy,
                    "accuracy_margin": margin,
                    "baseline_offset": int(baseline_payload["baseline_offset"]),
                    "is_syntactic_relation": is_syntactic,
                }
            )
            if math.isnan(best_relation_accuracy) or accuracy > best_relation_accuracy:
                best_relation = relation
                best_relation_accuracy = accuracy
                best_relation_margin = margin

        head_rows.append(
            {
                "layer": layer_idx,
                "head_index": head_idx,
                "confidence_mean": confidence_mean,
                "eligible_positions": eligible_positions,
                "most_common_offset": best_offset,
                "positional_share": positional_share,
                "is_positional": is_positional,
                "rare_word_rate": rare_word_rate,
                "rare_hits": rare_hits,
                "rare_eligible": rare_eligible,
                "is_rare_word_head": is_rare_word_head,
                "best_relation": best_relation,
                "best_relation_accuracy": best_relation_accuracy,
                "best_relation_margin": best_relation_margin,
                "syntactic_relations": ",".join(syntactic_relations),
                "is_syntactic_head": bool(syntactic_relations),
                "syntactic_relation_count": len(syntactic_relations),
                "importance_delta_nll_mean": (
                    float(stats["importance_delta_nll_sum"] / stats["importance_delta_nll_count"])
                    if int(stats["importance_delta_nll_count"])
                    else float("nan")
                ),
                "importance_sentence_count": int(stats["importance_delta_nll_count"]),
            }
        )

    head_df = pd.DataFrame(head_rows)
    if not head_df.empty:
        ranked = head_df["importance_delta_nll_mean"].rank(method="dense", ascending=False)
        head_df["importance_rank"] = ranked.astype("Int64")
        max_rank = float(ranked.max()) if ranked.notna().any() else float("nan")
        head_df["importance_percentile"] = (
            1.0 - ((ranked - 1.0) / max_rank) if max_rank and not math.isnan(max_rank) else float("nan")
        )
        head_df = head_df.sort_values(["importance_rank", "layer", "head_index"], na_position="last").reset_index(drop=True)

    relation_df = pd.DataFrame(relation_rows)
    if not relation_df.empty:
        relation_df = relation_df.sort_values(["layer", "head_index", "relation"]).reset_index(drop=True)
    return head_df, relation_df


def write_outputs(
    *,
    args: argparse.Namespace,
    sentence_rows: list[dict[str, float | int | str]],
    relation_rows: list[dict[str, float | int | str]],
    head_stats: dict[tuple[int, int], dict[str, object]],
    relation_baseline_counts: dict[str, dict[int, int]],
    processed_sentences: int,
    total_selected_sentences: int,
    final: bool,
) -> None:
    sentence_df = pd.DataFrame(sentence_rows)
    relation_df = pd.DataFrame(relation_rows)
    if sentence_df.empty or relation_df.empty:
        return

    relation_summary = (
        relation_df.groupby(
            ["config_name", "analysis_mode", "layer", "head_label", "head_index", "top_k", "relation"],
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

    summary_df = summarize(sentence_df, relation_summary)
    relation_baselines: dict[str, dict[str, float | int]] = {}
    for relation, offset_counts in relation_baseline_counts.items():
        opportunity_count = int(sum(offset_counts.values()))
        baseline_offset, baseline_count = max(offset_counts.items(), key=lambda item: item[1])
        relation_baselines[relation] = {
            "opportunity_count": opportunity_count,
            "baseline_offset": int(baseline_offset),
            "baseline_accuracy": float(baseline_count / opportunity_count) if opportunity_count else float("nan"),
        }

    head_characterization_df, relation_characterization_df = build_head_characterization(
        head_stats=head_stats,
        relation_baselines=relation_baselines,
        relation_min_count=args.relation_min_count,
        positional_threshold=args.positional_threshold,
        syntactic_margin_threshold=args.syntactic_margin_threshold,
        rare_threshold=args.rare_threshold,
    )
    relation_baseline_df = (
        pd.DataFrame(
            [
                {
                    "relation": relation,
                    **payload,
                }
                for relation, payload in sorted(relation_baselines.items())
            ]
        )
        if relation_baselines
        else pd.DataFrame(columns=["relation", "opportunity_count", "baseline_offset", "baseline_accuracy"])
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / f"{args.run_name}_summary.csv"
    sentence_path = args.results_dir / f"{args.run_name}_sentence.csv"
    relation_path = args.results_dir / f"{args.run_name}_relation_summary.csv"
    head_characterization_path = args.results_dir / f"{args.run_name}_head_characterization.csv"
    relation_characterization_path = args.results_dir / f"{args.run_name}_relation_characterization.csv"
    relation_baseline_path = args.results_dir / f"{args.run_name}_relation_baselines.csv"
    progress_path = args.results_dir / f"{args.run_name}_progress.json"

    summary_df.to_csv(summary_path, index=False)
    sentence_df.to_csv(sentence_path, index=False)
    relation_summary.to_csv(relation_path, index=False)
    head_characterization_df.to_csv(head_characterization_path, index=False)
    relation_characterization_df.to_csv(relation_characterization_path, index=False)
    relation_baseline_df.to_csv(relation_baseline_path, index=False)
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
        base.log(f"Wrote summary to {summary_path}")
        base.log(f"Wrote sentence-level metrics to {sentence_path}")
        base.log(f"Wrote relation summary to {relation_path}")
        base.log(f"Wrote head characterization to {head_characterization_path}")
        base.log(f"Wrote relation characterization to {relation_characterization_path}")
        base.log(f"Wrote relation baselines to {relation_baseline_path}")
    else:
        base.log(
            f"Checkpoint saved after {processed_sentences}/{total_selected_sentences} sentences "
            f"to {args.results_dir}"
        )


def main() -> None:
    args = parse_args()
    top_k_values = base.parse_int_list(args.top_k_values)
    characterization_top_k = 1 if 1 in top_k_values else top_k_values[0]
    ud_paths = base.ensure_ud_files(args.cache_dir)
    device_request = "cpu" if args.prepare_only and args.device == "auto" else args.device
    device = base.choose_device(device_request)
    torch_dtype = resolve_torch_dtype(args.torch_dtype, device)

    base.log(f"Selected device: {device} (dtype={torch_dtype})")
    if args.wait_for_gpu and device.type == "mps":
        base.wait_for_gpu_if_needed(args.poll_interval)

    tokenizer, model = load_model_and_tokenizer(args.model_name, device, torch_dtype)

    if args.prepare_only:
        base.log("Preparation complete. Exiting without inference.")
        return

    selected_sentences = base.select_sentences(
        split=args.split,
        ud_paths=ud_paths,
        min_words=args.min_words,
        max_words=args.max_words,
        limit=args.limit,
    )
    if not selected_sentences:
        raise RuntimeError("No sentences remained after filtering.")
    train_examples = base.parse_conllu(ud_paths["train"], "train")
    token_frequency = base.build_token_frequency(train_examples)

    base.log(
        f"Processing {len(selected_sentences)} sentences from split={args.split} "
        f"for model={args.model_name}, layers={args.layers}, head_mode={args.head_mode}, heads={args.heads}"
    )

    sentence_rows: list[dict[str, float | int | str]] = []
    relation_rows: list[dict[str, float | int | str]] = []
    head_stats: dict[tuple[int, int], dict[str, object]] = {}
    relation_baseline_counts: dict[str, dict[int, int]] = {}

    resolved_layers: list[int] | None = None
    resolved_heads: list[int] = []
    ablation_masks: dict[tuple[int, int], torch.Tensor] = {}
    importance_limit = args.importance_limit if args.importance_limit > 0 else len(selected_sentences)

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
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                output_attentions=True,
                return_dict=True,
            )

        if resolved_layers is None:
            resolved_layers = base.resolve_index_selection(args.layers, len(outputs.attentions), "layer")
            if args.head_mode in ("all", "both"):
                resolved_heads = base.resolve_index_selection(args.heads, outputs.attentions[0].shape[1], "head")
                num_layers = len(outputs.attentions)
                num_heads = outputs.attentions[0].shape[1]
                for candidate_layer in resolved_layers:
                    for candidate_head in resolved_heads:
                        active_heads = [
                            (layer_idx, head_idx)
                            for layer_idx in range(num_layers)
                            for head_idx in range(num_heads)
                            if not (layer_idx == candidate_layer and head_idx == candidate_head)
                        ]
                        ablation_masks[(candidate_layer, candidate_head)] = base.make_head_mask(
                            num_layers=num_layers,
                            num_heads=num_heads,
                            active_heads=active_heads,
                            device=device,
                        )
            base.log(
                f"Resolved layers={resolved_layers}"
                + (
                    f" and heads={resolved_heads}"
                    if args.head_mode in ("all", "both")
                    else " with mean head aggregation"
                )
            )

        sentence_nll, correct_token_count, predicted_token_count = base.next_token_metrics_from_logits(
            outputs.logits[0],
            input_ids=input_ids,
        )

        gold_edges = base.ud_edges(example)
        ud_distance = base.edge_distance(gold_edges)
        rare_targets = base.rare_previous_targets(example, token_frequency)
        accessible_rows = base.build_accessible_relation_rows(example)
        for opportunity in accessible_rows:
            relation = str(opportunity["relation"])
            offset = int(opportunity["relative_offset"])
            if relation not in relation_baseline_counts:
                relation_baseline_counts[relation] = {}
            relation_baseline_counts[relation][offset] = relation_baseline_counts[relation].get(offset, 0) + 1
        gold_matrix = None
        if args.compute_tree_distance:
            gold_matrix = base.shortest_path_matrix(gold_edges, example.length)
            gold_values = base.upper_triangle_values(gold_matrix)
        else:
            gold_values = np.array([], dtype=np.float64)

        for layer_idx in resolved_layers:
            layer_attention = outputs.attentions[layer_idx][0].detach().to(torch.float32).cpu().numpy()
            attention_views = base.aggregate_word_attention_views(
                layer_attention=layer_attention,
                word_ids=word_ids,
                num_words=example.length,
                head_mode=args.head_mode,
                selected_heads=resolved_heads,
            )

            for top_k in top_k_values:
                for head_label, head_index, word_attention in attention_views:
                    current_config = config_name(layer_idx, top_k, head_label)
                    ai_edges = base.build_attention_edges_topk(word_attention, top_k)
                    matched, gold_total, predicted_total = base.overlap_counts(gold_edges, ai_edges)
                    precision, recall, f1 = base.precision_recall_f1(matched, gold_total, predicted_total)
                    ai_distance = base.edge_distance(ai_edges)
                    selected_edge_set = base.edge_set(ai_edges)
                    primary_targets = base.primary_targets_from_topk(word_attention, top_k)

                    tree_spearman = float("nan")
                    if args.compute_tree_distance and gold_matrix is not None:
                        ai_matrix = base.shortest_path_matrix(ai_edges, example.length)
                        ai_values = base.upper_triangle_values(ai_matrix)
                        tree_spearman = safe_spearman(gold_values, ai_values)

                    sentence_rows.append(
                        {
                            "sent_id": example.sent_id,
                            "split": example.split,
                            "num_words": example.length,
                            "config_name": current_config,
                            "analysis_mode": "layer" if head_label == "mean" else "head",
                            "layer": layer_idx,
                            "head_label": head_label,
                            "head_index": head_index,
                            "top_k": top_k,
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
                                "config_name": current_config,
                                "analysis_mode": "layer" if head_label == "mean" else "head",
                                "layer": layer_idx,
                                "head_label": head_label,
                                "head_index": head_index,
                                "top_k": top_k,
                                **row,
                            }
                        )

                    if head_label == "mean" or top_k != characterization_top_k:
                        continue

                    head_key = (layer_idx, head_index)
                    stats = head_stats.setdefault(head_key, empty_head_stats())
                    top_targets, top_weights, relative_offsets = base.top1_targets_and_weights(word_attention)

                    stats["eligible_positions"] = int(stats["eligible_positions"]) + len(top_targets)
                    stats["confidence_sum"] = float(stats["confidence_sum"]) + float(sum(top_weights.values()))

                    offset_counts = dict(stats["offset_counts"])
                    for relative_offset in relative_offsets.values():
                        offset_counts[relative_offset] = offset_counts.get(relative_offset, 0) + 1
                    stats["offset_counts"] = offset_counts

                    rare_eligible = int(stats["rare_eligible"])
                    rare_hits = int(stats["rare_hits"])
                    for source_idx, rare_candidates in rare_targets.items():
                        if not rare_candidates:
                            continue
                        rare_eligible += 1
                        predicted_head = top_targets.get(source_idx)
                        if predicted_head in rare_candidates:
                            rare_hits += 1
                    stats["rare_eligible"] = rare_eligible
                    stats["rare_hits"] = rare_hits

                    relation_match = dict(stats["relation_match"])
                    for opportunity in accessible_rows:
                        relation = str(opportunity["relation"])
                        source_idx = int(opportunity["source_index"])
                        target_idx = int(opportunity["target_index"])
                        if top_targets.get(source_idx) == target_idx:
                            merge_count(relation_match, relation)
                    stats["relation_match"] = relation_match

                    if sent_idx <= importance_limit:
                        ablated_nll, _ = base.sentence_next_token_nll(
                            model=model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            device=device,
                            head_mask=ablation_masks[head_key],
                        )
                        stats["importance_delta_nll_sum"] = float(stats["importance_delta_nll_sum"]) + (
                            ablated_nll - sentence_nll
                        )
                        stats["importance_delta_nll_count"] = int(stats["importance_delta_nll_count"]) + 1

        if sent_idx % 25 == 0 or sent_idx == len(selected_sentences):
            base.log(f"Processed {sent_idx}/{len(selected_sentences)} sentences")
        if args.checkpoint_every > 0 and sent_idx % args.checkpoint_every == 0:
            write_outputs(
                args=args,
                sentence_rows=sentence_rows,
                relation_rows=relation_rows,
                head_stats=head_stats,
                relation_baseline_counts=relation_baseline_counts,
                processed_sentences=sent_idx,
                total_selected_sentences=len(selected_sentences),
                final=False,
            )

    sentence_df = pd.DataFrame(sentence_rows)
    relation_df = pd.DataFrame(relation_rows)
    if sentence_df.empty or relation_df.empty:
        raise RuntimeError("No rows were generated. Check sentence filtering or model outputs.")
    write_outputs(
        args=args,
        sentence_rows=sentence_rows,
        relation_rows=relation_rows,
        head_stats=head_stats,
        relation_baseline_counts=relation_baseline_counts,
        processed_sentences=len(selected_sentences),
        total_selected_sentences=len(selected_sentences),
        final=True,
    )


if __name__ == "__main__":
    main()

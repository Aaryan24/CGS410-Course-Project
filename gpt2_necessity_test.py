#!/usr/bin/env python3
"""
Necessity-side head ablation for the GPT-2 dependency grammar project.

Phase 2 asked: if we keep only selected heads, how much survives?
This script asks the complement question: if we remove selected heads from the
full model, how much syntax and prediction is lost?
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch

import gpt2_ud_utils as base
import gpt2_phase2_sufficiency as phase2


def build_necessity_definitions(
    ranking_df: pd.DataFrame,
    all_heads: list[tuple[int, int]],
    remove_sizes: list[int],
    random_seeds: list[int],
    remove_families: list[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    score_columns = {
        "structural": "structural_score",
        "syntactic": "syntactic_score",
        "hybrid": "hybrid_score",
        "importance": "importance_score",
    }
    unknown = sorted(set(remove_families) - set(score_columns))
    if unknown:
        raise ValueError(f"Unknown removal families: {', '.join(unknown)}")

    all_head_set = set(all_heads)
    definitions: list[dict[str, object]] = [
        {
            "subset_name": "full_model",
            "family": "full",
            "removed_size": 0,
            "seed": -1,
            "score_column": "full_model",
            "removed_heads": [],
            "active_heads": sorted(all_head_set),
        }
    ]

    for size in remove_sizes:
        for family in remove_families:
            score_column = score_columns[family]
            removed = phase2.select_top_heads(ranking_df, score_column, size)
            active = sorted(all_head_set - set(removed))
            definitions.append(
                {
                    "subset_name": f"minus_top_{family}_k{size}",
                    "family": f"minus_top_{family}",
                    "removed_size": size,
                    "seed": -1,
                    "score_column": score_column,
                    "removed_heads": removed,
                    "active_heads": active,
                }
            )

        for seed in random_seeds:
            rng = random.Random(seed)
            random_removed = sorted(rng.sample(all_heads, size))
            random_active = sorted(all_head_set - set(random_removed))
            definitions.append(
                {
                    "subset_name": f"minus_random_k{size}_seed{seed}",
                    "family": "minus_random",
                    "removed_size": size,
                    "seed": seed,
                    "score_column": "random",
                    "removed_heads": random_removed,
                    "active_heads": random_active,
                }
            )

    definition_df = pd.DataFrame(
        [
            {
                "subset_name": item["subset_name"],
                "family": item["family"],
                "removed_size": item["removed_size"],
                "active_size": len(item["active_heads"]),
                "seed": item["seed"],
                "score_column": item["score_column"],
                "removed_heads": ";".join(f"{layer}:{head}" for layer, head in item["removed_heads"]),
                "active_heads": ";".join(f"{layer}:{head}" for layer, head in item["active_heads"]),
            }
            for item in definitions
        ]
    )
    return definitions, definition_df


def mps_memory_mb(device: torch.device) -> tuple[float, float]:
    if device.type != "mps" or not hasattr(torch, "mps"):
        return float("nan"), float("nan")
    allocated = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
    driver = (
        float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)
        if hasattr(torch.mps, "driver_allocated_memory")
        else float("nan")
    )
    return allocated, driver


def summarize(sentence_df: pd.DataFrame, relation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for subset_name, subset_df in sentence_df.groupby("subset_name", sort=True):
        relation_subset = relation_df[relation_df["subset_name"] == subset_name]
        first = subset_df.iloc[0]
        ai_edge_total = int(subset_df["ai_edge_count"].sum())
        matched_total = int(subset_df["matched_edge_count"].sum())
        ud_edge_total = int(subset_df["ud_edge_count"].sum())
        precision = matched_total / ai_edge_total if ai_edge_total else 0.0
        recall = matched_total / ud_edge_total if ud_edge_total else 0.0
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        rows.append(
            {
                "subset_name": subset_name,
                "family": str(first["family"]),
                "removed_size": int(first["removed_size"]),
                "active_size": int(first["active_size"]),
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
                "mean_sentence_overlap_f1": float(subset_df["sentence_f1"].mean()),
                "mean_ud_distance": float(subset_df["mean_ud_distance"].mean()),
                "mean_ai_distance": float(subset_df["mean_ai_distance"].mean()),
                "mean_relation_match_rate": (
                    float(relation_subset["match_rate"].mean()) if not relation_subset.empty else float("nan")
                ),
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
    summary = pd.DataFrame(rows).sort_values(["family", "removed_size", "seed", "top_k"]).reset_index(drop=True)
    full = summary[summary["subset_name"] == "full_model"].iloc[0]
    summary["syntax_f1_delta_vs_full"] = summary["mean_sentence_overlap_f1"] - float(full["mean_sentence_overlap_f1"])
    summary["tree_spearman_delta_vs_full"] = (
        summary["mean_tree_distance_spearman"] - float(full["mean_tree_distance_spearman"])
    )
    summary["accuracy_delta_vs_full"] = summary["mean_sentence_accuracy"] - float(full["mean_sentence_accuracy"])
    summary["nll_delta_vs_full"] = summary["mean_next_token_nll_per_token"] - float(
        full["mean_next_token_nll_per_token"]
    )
    return summary


def bootstrap_ci(diff: np.ndarray, reps: int = 2000, seed: int = 410) -> tuple[float, float]:
    if diff.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for rep_idx in range(reps):
        sample_idx = rng.integers(0, diff.size, diff.size)
        means[rep_idx] = float(np.mean(diff[sample_idx]))
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_stats_vs_full(sentence_df: pd.DataFrame) -> pd.DataFrame:
    if sentence_df.empty or "full_model" not in set(sentence_df["subset_name"]):
        return pd.DataFrame()
    work = sentence_df.copy()
    work["nll_per_token"] = work["sentence_nll"] / work["predicted_token_count"]
    full = work[work["subset_name"] == "full_model"].set_index("sent_id")
    metrics = {
        "sentence_f1": "syntax F1",
        "tree_distance_spearman": "tree Spearman",
        "sentence_accuracy": "accuracy",
        "nll_per_token": "NLL/token",
    }
    rows: list[dict[str, float | int | str]] = []
    for subset_name, subset_df in work.groupby("subset_name", sort=True):
        if subset_name == "full_model":
            continue
        current = subset_df.set_index("sent_id").loc[full.index]
        for column, label in metrics.items():
            diff = (current[column] - full[column]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            ci_low, ci_high = bootstrap_ci(diff)
            try:
                wilcoxon_p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                wilcoxon_p = float("nan")
            rows.append(
                {
                    "subset_name": subset_name,
                    "metric": label,
                    "mean_delta_vs_full": float(np.mean(diff)) if diff.size else float("nan"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": wilcoxon_p,
                    "n": int(diff.size),
                }
            )
    return pd.DataFrame(rows)


def paired_family_comparison_stats(sentence_df: pd.DataFrame) -> pd.DataFrame:
    if sentence_df.empty:
        return pd.DataFrame()
    work = sentence_df.copy()
    work["nll_per_token"] = work["sentence_nll"] / work["predicted_token_count"]
    rows: list[dict[str, float | int | str]] = []
    metrics = {"sentence_accuracy": "accuracy", "nll_per_token": "NLL/token"}
    sizes = sorted(set(work["removed_size"]) - {0})
    for size in sizes:
        hybrid_name = f"minus_top_hybrid_k{size}"
        structural_name = f"minus_top_structural_k{size}"
        if hybrid_name not in set(work["subset_name"]) or structural_name not in set(work["subset_name"]):
            continue
        hybrid = work[work["subset_name"] == hybrid_name].set_index("sent_id")
        structural = work[work["subset_name"] == structural_name].set_index("sent_id")
        common_ids = hybrid.index.intersection(structural.index)
        for column, label in metrics.items():
            diff = (
                hybrid.loc[common_ids, column] - structural.loc[common_ids, column]
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            ci_low, ci_high = bootstrap_ci(diff)
            try:
                wilcoxon_p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                wilcoxon_p = float("nan")
            rows.append(
                {
                    "comparison": f"{hybrid_name} - {structural_name}",
                    "metric": label,
                    "mean_difference": float(np.mean(diff)) if diff.size else float("nan"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": wilcoxon_p,
                    "n": int(diff.size),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    args: argparse.Namespace,
    ranking_df: pd.DataFrame,
    definition_df: pd.DataFrame,
    sentence_rows: list[dict[str, float | int | str]],
    relation_rows: list[dict[str, float | int | str]],
    processed_sentences: int,
    total_selected_sentences: int,
    final: bool,
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_dir / args.run_name
    ranking_df.to_csv(f"{prefix}_ranking.csv", index=False)
    definition_df.to_csv(f"{prefix}_ablation_definitions.csv", index=False)

    sentence_df = pd.DataFrame(sentence_rows)
    relation_df = pd.DataFrame(relation_rows)
    if not sentence_df.empty:
        sentence_df.to_csv(f"{prefix}_sentence.csv", index=False)
    if not relation_df.empty:
        relation_summary = (
            relation_df.groupby(
                ["subset_name", "family", "removed_size", "active_size", "seed", "top_k", "relation"],
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
        relation_summary.to_csv(f"{prefix}_relation_summary.csv", index=False)
    else:
        relation_summary = pd.DataFrame()
    if not sentence_df.empty and not relation_summary.empty:
        summarize(sentence_df, relation_summary).to_csv(f"{prefix}_summary.csv", index=False)
    if not sentence_df.empty:
        paired_stats_vs_full(sentence_df).to_csv(f"{prefix}_paired_stats.csv", index=False)
        family_stats = paired_family_comparison_stats(sentence_df)
        if not family_stats.empty:
            family_stats.to_csv(f"{prefix}_family_comparison_stats.csv", index=False)

    Path(f"{prefix}_progress.json").write_text(
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
    label = "Wrote final outputs" if final else "Checkpoint saved"
    base.log(f"{label} after {processed_sentences}/{total_selected_sentences} sentences to {args.results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-2 full-minus-heads necessity ablations.")
    parser.add_argument("--model-name", default="gpt2-medium")
    parser.add_argument("--phase1-prefix", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ud_english_ewt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="gpt2_medium_necessity_test_mps")
    parser.add_argument("--eval-split", choices=("train", "dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--top-k-values", default="1")
    parser.add_argument("--remove-sizes", default="4,8,16,32")
    parser.add_argument(
        "--remove-families",
        default="structural",
        help="Comma-separated ranking families to remove: structural, syntactic, hybrid, importance.",
    )
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--empty-mps-cache-every", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--compute-tree-distance", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = base.parse_int_list(args.top_k_values)
    remove_sizes = sorted(set(base.parse_int_list(args.remove_sizes)))
    remove_families = phase2.parse_text_list(args.remove_families)
    random_seeds = base.parse_int_list(args.random_seeds)

    phase1_prefix = args.phase1_prefix.expanduser().resolve()
    ranking_df, all_heads, _ = phase2.build_rankings(phase1_prefix)
    definitions, definition_df = build_necessity_definitions(
        ranking_df=ranking_df,
        all_heads=all_heads,
        remove_sizes=remove_sizes,
        random_seeds=random_seeds,
        remove_families=remove_families,
    )

    ud_paths = base.ensure_ud_files(args.cache_dir)
    device = base.choose_device(args.device)
    dtype = phase2.resolve_torch_dtype(args.torch_dtype, device)
    base.log(f"Selected device: {device} (dtype={dtype}) for {len(definitions)} necessity ablations.")
    if args.wait_for_gpu and device.type == "mps":
        base.wait_for_gpu_if_needed(args.poll_interval)

    tokenizer, model = phase2.load_model_and_tokenizer(args.model_name, device, dtype)
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
    masks = {
        item["subset_name"]: base.make_head_mask(
            num_layers=num_layers,
            num_heads=num_heads,
            active_heads=item["active_heads"],
            device=device,
        )
        for item in definitions
    }

    sentence_rows: list[dict[str, float | int | str]] = []
    relation_rows: list[dict[str, float | int | str]] = []
    base.log(f"Evaluating {len(selected_sentences)} sentences on split={args.eval_split}.")

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
        if args.compute_tree_distance:
            gold_matrix = base.shortest_path_matrix(gold_edges, example.length)
            gold_values = base.upper_triangle_values(gold_matrix)
        else:
            gold_matrix = None
            gold_values = np.array([], dtype=np.float64)

        for item in definitions:
            subset_name = str(item["subset_name"])
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    head_mask=masks[subset_name],
                    output_attentions=True,
                    return_dict=True,
                )
            sentence_nll, correct_token_count, predicted_token_count = base.next_token_metrics_from_logits(
                outputs.logits[0],
                input_ids=input_ids,
            )
            combined_attention = phase2.subset_attention_matrix(
                attentions=outputs.attentions,
                word_ids=word_ids,
                num_words=example.length,
                selected_heads=item["active_heads"],
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
                    tree_spearman = phase2.safe_spearman(gold_values, base.upper_triangle_values(ai_matrix))

                sentence_rows.append(
                    {
                        "sent_id": example.sent_id,
                        "split": example.split,
                        "subset_name": subset_name,
                        "family": item["family"],
                        "removed_size": int(item["removed_size"]),
                        "active_size": len(item["active_heads"]),
                        "seed": int(item["seed"]),
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
                            "family": item["family"],
                            "removed_size": int(item["removed_size"]),
                            "active_size": len(item["active_heads"]),
                            "seed": int(item["seed"]),
                            "top_k": top_k,
                            **row,
                        }
                    )

            del outputs

        if device.type == "mps" and args.empty_mps_cache_every > 0 and sent_idx % args.empty_mps_cache_every == 0:
            torch.mps.empty_cache()

        if sent_idx % 10 == 0 or sent_idx == len(selected_sentences):
            allocated, driver = mps_memory_mb(device)
            memory_note = "" if np.isnan(allocated) else f", mps_allocated={allocated:.0f}MB, mps_driver={driver:.0f}MB"
            base.log(f"Processed {sent_idx}/{len(selected_sentences)} sentences{memory_note}")
        if args.checkpoint_every > 0 and sent_idx % args.checkpoint_every == 0:
            write_outputs(
                args=args,
                ranking_df=ranking_df,
                definition_df=definition_df,
                sentence_rows=sentence_rows,
                relation_rows=relation_rows,
                processed_sentences=sent_idx,
                total_selected_sentences=len(selected_sentences),
                final=False,
            )

    write_outputs(
        args=args,
        ranking_df=ranking_df,
        definition_df=definition_df,
        sentence_rows=sentence_rows,
        relation_rows=relation_rows,
        processed_sentences=len(selected_sentences),
        total_selected_sentences=len(selected_sentences),
        final=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sharper syntax-side necessity ablation for the GPT-2 dependency grammar project.

The earlier necessity experiment measured syntax with an average over all
remaining heads. That deliberately diluted a 4-16 head removal inside a
384-head model. This script instead evaluates fixed syntax extractors:

- full_selected: selected syntax heads in the unablated full model
- target_ablated_selected: the same selected heads after they are ablated
- target_ablated_backup: next-best same-family heads after selected heads are ablated
- random_ablated_selected: selected syntax heads after random same-size heads are ablated

This separates the linguistic syntax-signal question from the predictive
next-token ablation question.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch

import gpt2_phase2_sufficiency as phase2
import gpt2_ud_utils as base


SCORE_COLUMNS = {
    "structural": "structural_score",
    "hybrid": "hybrid_score",
    "syntactic": "syntactic_score",
}


def serialize_heads(heads: list[tuple[int, int]]) -> str:
    return ";".join(f"{layer}:{head}" for layer, head in heads)


def select_backup_heads(
    ranking_df: pd.DataFrame,
    score_column: str,
    size: int,
    excluded: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    ordered = ranking_df.sort_values(
        [score_column, "structural_f1", "syntactic_margin_sum", "layer", "head_index"],
        ascending=[False, False, False, True, True],
    )
    selected: list[tuple[int, int]] = []
    for row in ordered.itertuples(index=False):
        head = (int(row.layer), int(row.head_index))
        if head in excluded:
            continue
        selected.append(head)
        if len(selected) >= size:
            break
    return selected


def build_design(
    ranking_df: pd.DataFrame,
    all_heads: list[tuple[int, int]],
    families: list[str],
    sizes: list[int],
    random_seed: int,
    core_only: bool,
) -> tuple[dict[str, torch.Tensor | None], list[dict[str, object]], pd.DataFrame]:
    all_head_set = set(all_heads)
    rng = random.Random(random_seed)
    random_removed_by_size = {
        size: sorted(rng.sample(all_heads, size))
        for size in sizes
    }

    condition_rows: list[dict[str, object]] = []
    ablation_to_removed: dict[str, list[tuple[int, int]]] = {"full_model": []}

    for size in sizes:
        random_removed = random_removed_by_size[size]
        ablation_to_removed[f"ablate_random_k{size}_seed{random_seed}"] = random_removed

        for family in families:
            score_column = SCORE_COLUMNS[family]
            selected = phase2.select_top_heads(ranking_df, score_column, size)
            backup = select_backup_heads(ranking_df, score_column, size, set(selected))
            ablation_name = f"ablate_top_{family}_k{size}"
            random_ablation_name = f"ablate_random_k{size}_seed{random_seed}"
            if not core_only:
                ablation_to_removed[ablation_name] = selected

            specs = [
                {
                    "condition_name": f"full_extract_top_{family}_k{size}",
                    "condition_type": "full_selected",
                    "ablation_name": "full_model",
                    "extractor_heads": selected,
                    "synthetic_zero_attention": False,
                },
                {
                    "condition_name": f"target_ablated_extract_top_{family}_k{size}",
                    "condition_type": "target_ablated_selected",
                    "ablation_name": f"synthetic_{ablation_name}" if core_only else ablation_name,
                    "extractor_heads": selected,
                    "synthetic_zero_attention": core_only,
                },
                {
                    "condition_name": f"random_ablated_extract_top_{family}_k{size}",
                    "condition_type": "random_ablated_selected",
                    "ablation_name": random_ablation_name,
                    "extractor_heads": selected,
                    "synthetic_zero_attention": False,
                },
            ]
            if not core_only:
                specs.insert(
                    1,
                    {
                        "condition_name": f"full_extract_backup_{family}_k{size}",
                        "condition_type": "full_backup",
                        "ablation_name": "full_model",
                        "extractor_heads": backup,
                        "synthetic_zero_attention": False,
                    },
                )
                specs.insert(
                    3,
                    {
                        "condition_name": f"target_ablated_extract_backup_{family}_k{size}",
                        "condition_type": "target_ablated_backup",
                        "ablation_name": ablation_name,
                        "extractor_heads": backup,
                        "synthetic_zero_attention": False,
                    },
                )

            for spec in specs:
                removed = selected if spec["synthetic_zero_attention"] else ablation_to_removed[str(spec["ablation_name"])]
                condition_rows.append(
                    {
                        "condition_name": spec["condition_name"],
                        "condition_type": spec["condition_type"],
                        "family": family,
                        "size": size,
                        "score_column": score_column,
                        "ablation_name": spec["ablation_name"],
                        "removed_heads": removed,
                        "extractor_heads": spec["extractor_heads"],
                        "synthetic_zero_attention": bool(spec["synthetic_zero_attention"]),
                        "baseline_condition": f"full_extract_top_{family}_k{size}",
                        "active_heads": sorted(all_head_set - set(removed)),
                    }
                )

    design_df = pd.DataFrame(
        [
            {
                "condition_name": row["condition_name"],
                "condition_type": row["condition_type"],
                "family": row["family"],
                "size": row["size"],
                "score_column": row["score_column"],
                "ablation_name": row["ablation_name"],
                "synthetic_zero_attention": row["synthetic_zero_attention"],
                "baseline_condition": row["baseline_condition"],
                "removed_heads": serialize_heads(row["removed_heads"]),
                "extractor_heads": serialize_heads(row["extractor_heads"]),
                "active_heads": serialize_heads(row["active_heads"]),
            }
            for row in condition_rows
        ]
    )
    return ablation_to_removed, condition_rows, design_df


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


def summarize(sentence_df: pd.DataFrame, relation_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition_name, condition_df in sentence_df.groupby("condition_name", sort=True):
        first = condition_df.iloc[0]
        relation_subset = relation_summary[relation_summary["condition_name"] == condition_name]
        matched_total = int(condition_df["matched_edge_count"].sum())
        gold_total = int(condition_df["ud_edge_count"].sum())
        predicted_total = int(condition_df["ai_edge_count"].sum())
        precision, recall, f1 = base.precision_recall_f1(matched_total, gold_total, predicted_total)
        rows.append(
            {
                "condition_name": condition_name,
                "condition_type": str(first["condition_type"]),
                "family": str(first["family"]),
                "size": int(first["size"]),
                "ablation_name": str(first["ablation_name"]),
                "baseline_condition": str(first["baseline_condition"]),
                "num_sentences": int(len(condition_df)),
                "num_words": int(condition_df["num_words"].sum()),
                "total_ud_edges": gold_total,
                "total_ai_edges": predicted_total,
                "total_matched_edges": matched_total,
                "corpus_overlap_precision": precision,
                "corpus_overlap_recall": recall,
                "corpus_overlap_f1": f1,
                "mean_sentence_overlap_f1": float(condition_df["sentence_f1"].mean()),
                "mean_tree_distance_spearman": float(condition_df["tree_distance_spearman"].mean()),
                "mean_ai_distance": float(condition_df["mean_ai_distance"].mean()),
                "mean_relation_match_rate": (
                    float(relation_subset["match_rate"].mean()) if not relation_subset.empty else float("nan")
                ),
                "mean_extractor_attention_mass": float(condition_df["extractor_attention_mass"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    baseline_lookup = summary.set_index("condition_name")
    for metric in [
        "mean_sentence_overlap_f1",
        "mean_tree_distance_spearman",
        "mean_relation_match_rate",
        "mean_extractor_attention_mass",
    ]:
        summary[f"{metric}_delta_vs_baseline"] = summary.apply(
            lambda row: float(row[metric]) - float(baseline_lookup.loc[row["baseline_condition"], metric])
            if row["baseline_condition"] in baseline_lookup.index
            else float("nan"),
            axis=1,
        )
    return summary.sort_values(["family", "size", "condition_type"]).reset_index(drop=True)


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


def paired_stats(sentence_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = {
        "sentence_f1": "syntax F1",
        "tree_distance_spearman": "tree Spearman",
        "extractor_attention_mass": "extractor attention mass",
    }
    for condition_name, condition_df in sentence_df.groupby("condition_name", sort=True):
        first = condition_df.iloc[0]
        baseline_name = str(first["baseline_condition"])
        if condition_name == baseline_name:
            continue
        baseline = sentence_df[sentence_df["condition_name"] == baseline_name].set_index("sent_id")
        current = condition_df.set_index("sent_id")
        common_ids = current.index.intersection(baseline.index)
        for column, label in metrics.items():
            diff = (
                current.loc[common_ids, column] - baseline.loc[common_ids, column]
            ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            ci_low, ci_high = bootstrap_ci(diff)
            try:
                wilcoxon_p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                wilcoxon_p = float("nan")
            rows.append(
                {
                    "condition_name": condition_name,
                    "condition_type": str(first["condition_type"]),
                    "family": str(first["family"]),
                    "size": int(first["size"]),
                    "baseline_condition": baseline_name,
                    "metric": label,
                    "mean_delta_vs_baseline": float(np.mean(diff)) if diff.size else float("nan"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "wilcoxon_p": wilcoxon_p,
                    "n": int(diff.size),
                }
            )
    return pd.DataFrame(rows)


def drop_contrast_stats(sentence_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = {
        "sentence_f1": "syntax F1",
        "tree_distance_spearman": "tree Spearman",
    }
    for (family, size), group_df in sentence_df.groupby(["family", "size"], sort=True):
        target_name = f"target_ablated_extract_top_{family}_k{size}"
        random_name = f"random_ablated_extract_top_{family}_k{size}"
        baseline_name = f"full_extract_top_{family}_k{size}"
        names = set(group_df["condition_name"])
        if not {target_name, random_name, baseline_name}.issubset(names):
            continue
        baseline = sentence_df[sentence_df["condition_name"] == baseline_name].set_index("sent_id")
        target = sentence_df[sentence_df["condition_name"] == target_name].set_index("sent_id")
        random_control = sentence_df[sentence_df["condition_name"] == random_name].set_index("sent_id")
        common_ids = baseline.index.intersection(target.index).intersection(random_control.index)
        for column, label in metrics.items():
            target_drop = baseline.loc[common_ids, column] - target.loc[common_ids, column]
            random_drop = baseline.loc[common_ids, column] - random_control.loc[common_ids, column]
            diff = (target_drop - random_drop).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            ci_low, ci_high = bootstrap_ci(diff)
            try:
                wilcoxon_p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except Exception:
                wilcoxon_p = float("nan")
            rows.append(
                {
                    "family": family,
                    "size": int(size),
                    "metric": label,
                    "target_drop_minus_random_drop": float(np.mean(diff)) if diff.size else float("nan"),
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
    design_df: pd.DataFrame,
    sentence_rows: list[dict[str, object]],
    relation_rows: list[dict[str, object]],
    processed_sentences: int,
    total_selected_sentences: int,
    final: bool,
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_dir / args.run_name
    ranking_df.to_csv(f"{prefix}_ranking.csv", index=False)
    design_df.to_csv(f"{prefix}_design.csv", index=False)
    sentence_df = pd.DataFrame(sentence_rows)
    relation_df = pd.DataFrame(relation_rows)
    if not sentence_df.empty:
        sentence_df.to_csv(f"{prefix}_sentence.csv", index=False)
        paired_stats(sentence_df).to_csv(f"{prefix}_paired_stats.csv", index=False)
        drop_contrast_stats(sentence_df).to_csv(f"{prefix}_drop_contrast_stats.csv", index=False)
    if not relation_df.empty:
        relation_summary = (
            relation_df.groupby(
                ["condition_name", "condition_type", "family", "size", "relation"],
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
    parser = argparse.ArgumentParser(description="Run sharp GPT-2 syntax necessity ablations.")
    parser.add_argument("--model-name", default="gpt2-medium")
    parser.add_argument("--phase1-prefix", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ud_english_ewt"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="gpt2_medium_syntax_necessity_mps")
    parser.add_argument("--eval-split", choices=("train", "dev", "test", "all"), default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--top-k-values", default="1")
    parser.add_argument("--families", default="structural,hybrid")
    parser.add_argument("--sizes", default="4,16")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--empty-mps-cache-every", type=int, default=5)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run the sharp core contrast only: full selected, random ablated selected, and synthetic target-ablated selected.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--torch-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--compute-tree-distance", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_k_values = base.parse_int_list(args.top_k_values)
    families = phase2.parse_text_list(args.families)
    unknown = sorted(set(families) - set(SCORE_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(unknown)}")
    sizes = sorted(set(base.parse_int_list(args.sizes)))

    phase1_prefix = args.phase1_prefix.expanduser().resolve()
    ranking_df, all_heads, _ = phase2.build_rankings(phase1_prefix)
    ablation_to_removed, condition_rows, design_df = build_design(
        ranking_df=ranking_df,
        all_heads=all_heads,
        families=families,
        sizes=sizes,
        random_seed=args.random_seed,
        core_only=args.core_only,
    )

    ud_paths = base.ensure_ud_files(args.cache_dir)
    device = base.choose_device(args.device)
    dtype = phase2.resolve_torch_dtype(args.torch_dtype, device)
    base.log(
        f"Selected device: {device} (dtype={dtype}) for "
        f"{len(ablation_to_removed)} ablation masks and {len(condition_rows)} extractor conditions."
    )
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
    all_head_set = set(all_heads)
    masks = {
        ablation_name: base.make_head_mask(
            num_layers=num_layers,
            num_heads=num_heads,
            active_heads=sorted(all_head_set - set(removed_heads)),
            device=device,
        )
        for ablation_name, removed_heads in ablation_to_removed.items()
    }
    conditions_by_ablation: dict[str, list[dict[str, object]]] = defaultdict(list)
    synthetic_conditions: list[dict[str, object]] = []
    for row in condition_rows:
        if bool(row.get("synthetic_zero_attention", False)):
            synthetic_conditions.append(row)
        else:
            conditions_by_ablation[str(row["ablation_name"])].append(row)

    sentence_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
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

        for ablation_name, mask in masks.items():
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    head_mask=mask,
                    output_attentions=True,
                    return_dict=True,
                )

            for condition in conditions_by_ablation[ablation_name]:
                extractor_heads = list(condition["extractor_heads"])
                combined_attention = phase2.subset_attention_matrix(
                    attentions=outputs.attentions,
                    word_ids=word_ids,
                    num_words=example.length,
                    selected_heads=extractor_heads,
                )
                extractor_attention_mass = float(np.sum(combined_attention))

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
                        tree_spearman = phase2.safe_spearman(
                            gold_values,
                            base.upper_triangle_values(ai_matrix),
                        )

                    sentence_rows.append(
                        {
                            "sent_id": example.sent_id,
                            "split": example.split,
                            "condition_name": condition["condition_name"],
                            "condition_type": condition["condition_type"],
                            "family": condition["family"],
                            "size": int(condition["size"]),
                            "score_column": condition["score_column"],
                            "ablation_name": ablation_name,
                            "baseline_condition": condition["baseline_condition"],
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
                            "extractor_attention_mass": extractor_attention_mass,
                        }
                    )
                    for row in base.relation_match_summary(example, primary_targets, selected_edge_set):
                        relation_rows.append(
                            {
                                "sent_id": example.sent_id,
                                "split": example.split,
                                "condition_name": condition["condition_name"],
                                "condition_type": condition["condition_type"],
                                "family": condition["family"],
                                "size": int(condition["size"]),
                                "ablation_name": ablation_name,
                                "top_k": top_k,
                                **row,
                            }
                        )

            del outputs

        for condition in synthetic_conditions:
            combined_attention = np.zeros((example.length, example.length), dtype=np.float64)
            extractor_attention_mass = 0.0
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
                    tree_spearman = phase2.safe_spearman(
                        gold_values,
                        base.upper_triangle_values(ai_matrix),
                    )

                sentence_rows.append(
                    {
                        "sent_id": example.sent_id,
                        "split": example.split,
                        "condition_name": condition["condition_name"],
                        "condition_type": condition["condition_type"],
                        "family": condition["family"],
                        "size": int(condition["size"]),
                        "score_column": condition["score_column"],
                        "ablation_name": condition["ablation_name"],
                        "baseline_condition": condition["baseline_condition"],
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
                        "extractor_attention_mass": extractor_attention_mass,
                    }
                )
                for row in base.relation_match_summary(example, primary_targets, selected_edge_set):
                    relation_rows.append(
                        {
                            "sent_id": example.sent_id,
                            "split": example.split,
                            "condition_name": condition["condition_name"],
                            "condition_type": condition["condition_type"],
                            "family": condition["family"],
                            "size": int(condition["size"]),
                            "ablation_name": condition["ablation_name"],
                            "top_k": top_k,
                            **row,
                        }
                    )

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
                design_df=design_df,
                sentence_rows=sentence_rows,
                relation_rows=relation_rows,
                processed_sentences=sent_idx,
                total_selected_sentences=len(selected_sentences),
                final=False,
            )

    write_outputs(
        args=args,
        ranking_df=ranking_df,
        design_df=design_df,
        sentence_rows=sentence_rows,
        relation_rows=relation_rows,
        processed_sentences=len(selected_sentences),
        total_selected_sentences=len(selected_sentences),
        final=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
English bridge category study for mBERT.

This runner creates a prediction-focused, category-standardized extension of
the mBERT project. It is intentionally English-only so that we can compare it
cleanly with the existing GPT-2 English UD EWT study.

Stages:
  taxonomy        Build all-144-head category rankings from dev-only attention
                  metrics. No prediction-derived columns are used.
  attention_eval  Evaluate selected category subsets on held-out test attention
                  graph recovery.
  mlm_ablation    Remove category heads from the full mBERT model and measure
                  relation-targeted MLM loss.
  mlm_sufficiency Keep only category heads and measure relation-targeted MLM
                  loss, mirroring the GPT-2 retained-head sufficiency study.

The MLM stages are checkpointed by condition and can be resumed safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

BRIDGE_DIR = Path(__file__).resolve().parent
ROOT = BRIDGE_DIR
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mbert_dependency_head_analysis as core  # noqa: E402


OUTPUT_DIR = BRIDGE_DIR / "outputs" / "mbert_english"
FIGURE_DIR = OUTPUT_DIR / "figures"
CONFIG_PATH = BRIDGE_DIR / "multilingual_language_config.json"
RESULTS_DIR = BRIDGE_DIR / "results" / "mbert_dependency_head"
RUN_NAME = "en_ewt_bridge_categories"
ENGLISH_SPEC = {
    "key": "en_ewt",
    "language": "English",
    "run_name": "en_ewt_mbert_dependency_head_dev",
    "config_key": "en_ewt",
    "split": "dev",
    "limit": 0,
}
DEFAULT_BUDGETS = (4, 16)
EXPANDED_SUFFICIENCY_BUDGETS = (4, 16, 32, 48, 72, 96, 108)
REPORT_BUDGETS = EXPANDED_SUFFICIENCY_BUDGETS
DEFAULT_TARGET_RELATIONS = (
    "nsubj,obj,iobj,obl,nmod,amod,advmod,case,det,aux,cop,mark,"
    "compound,conj,xcomp,acl"
)


def model_ref() -> str:
    return os.environ.get("MBERT_MODEL_PATH", core.MODEL_NAME)
CATEGORY_FAMILIES = (
    "syntactic_relation",
    "pure_syntactic_relation",
    "structural_graph",
    "positional",
    "syntax_position_hybrid",
    "syntax_composite",
)
PRIMARY_FAMILIES = {
    "syntactic_relation",
    "pure_syntactic_relation",
    "structural_graph",
    "positional",
}


@dataclass
class RelationMaskedExample:
    example_id: str
    sent_id: str
    word_index: int
    word: str
    upos: str
    relation: str
    relation_family: str
    dependency_distance: int
    wordpiece_count: int
    input_ids: list[int]
    attention_mask: list[int]
    target_pos: int
    target_id: int


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run English mBERT bridge category study.")
    parser.add_argument(
        "--stage",
        choices=("all", "taxonomy", "attention_eval", "mlm_ablation", "mlm_sufficiency"),
        default="all",
    )
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--budgets", default=",".join(str(k) for k in DEFAULT_BUDGETS))
    parser.add_argument("--seed", type=int, default=410)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--structural-limit", type=int, default=0, help="Debug only. 0 means full dev split.")
    parser.add_argument("--attention-limit", type=int, default=0, help="0 means full held-out test split.")
    parser.add_argument("--mlm-sentences", type=int, default=0, help="0 means full held-out test split.")
    parser.add_argument("--mlm-tokens-per-sentence", type=int, default=6)
    parser.add_argument("--target-relations", default=DEFAULT_TARGET_RELATIONS)
    parser.add_argument("--uniform-random-seeds", type=int, default=5)
    parser.add_argument("--layer-random-seeds", type=int, default=5)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--positional-threshold", type=float, default=0.80)
    parser.add_argument("--relation-min-count", type=int, default=50)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=0,
        help="Debug/smoke option. 0 means score all conditions.",
    )
    return parser.parse_args()


def parse_budgets(text: str) -> list[int]:
    budgets = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not budgets:
        raise ValueError("--budgets must contain at least one integer.")
    if any(budget <= 0 for budget in budgets):
        raise ValueError("--budgets must be positive.")
    return budgets


def relation_family(relation: str) -> str:
    return relation.split(":", 1)[0]


def normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    if math.isclose(minimum, maximum):
        return pd.Series(np.zeros(len(numeric), dtype=np.float64), index=series.index)
    return (numeric - minimum) / (maximum - minimum)


def all_heads() -> list[tuple[int, int]]:
    return [(layer, head) for layer in range(12) for head in range(12)]


def label_from_head(head: tuple[int, int]) -> str:
    return f"L{head[0]}-H{head[1]}"


def head_from_label(label: str) -> tuple[int, int]:
    layer, head = label.replace("L", "").split("-H", 1)
    return int(layer), int(head)


def format_heads(heads: Iterable[tuple[int, int]]) -> str:
    return ", ".join(label_from_head(head) for head in heads)


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_config() -> dict[str, dict]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["languages"]


def load_heads(run_name: str) -> pd.DataFrame:
    heads = pd.read_csv(RESULTS_DIR / f"{run_name}_head_characterization.csv")
    heads["head_label"] = heads.apply(lambda row: f"L{int(row.layer)}-H{int(row.head_index)}", axis=1)
    return heads


def load_relations(run_name: str) -> pd.DataFrame:
    relations = pd.read_csv(RESULTS_DIR / f"{run_name}_relation_characterization.csv")
    relations["head_label"] = relations.apply(lambda row: f"L{int(row.layer)}-H{int(row.head_index)}", axis=1)
    return relations


def english_spec_for_split(split: str, limit: int = 0) -> dict:
    spec = dict(ENGLISH_SPEC)
    spec["split"] = split
    spec["limit"] = limit
    return spec


def selected_sentences(split: str, limit: int = 0) -> list[core.SentenceExample]:
    spec = english_spec_for_split(split, limit)
    cfg = load_config()[spec["config_key"]]
    args = argparse.Namespace(ud_files_json=json.dumps(cfg["ud_files"]), ud_train="", ud_dev="", ud_test="")
    cache_dir = BRIDGE_DIR / ".cache" / "mbert_dependency_head_ud" / cfg["treebank"].replace("/", "_")
    ud_paths = core.ensure_ud_files(core.resolve_ud_sources(args), cache_dir)
    return core.select_sentences(ud_paths, spec["split"], min_words=5, max_words=30, limit=int(spec["limit"]))


def valid_token_indices(example: core.SentenceExample, excluded_upos: set[str]) -> set[int]:
    return {token.idx for token in example.tokens if token.upos not in excluded_upos}


def gold_edges_for_example(example: core.SentenceExample, excluded_upos: set[str]) -> list[tuple[int, int, float]]:
    valid = valid_token_indices(example, excluded_upos)
    rows: list[tuple[int, int, float]] = []
    for token in example.tokens:
        if token.head == 0:
            continue
        if token.idx in valid and token.head in valid:
            rows.append((token.idx, token.head, 1.0))
    return rows


def edge_set(edges: list[tuple[int, int, float]]) -> set[tuple[int, int]]:
    return {(dep, head) for dep, head, _ in edges}


def precision_recall_f1(matched: int, gold_total: int, predicted_total: int) -> tuple[float, float, float]:
    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / gold_total if gold_total else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def top1_edges_bidirectional(
    word_attention: np.ndarray,
    eligible_sources: list[int],
    eligible_targets: set[int],
) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    num_words = word_attention.shape[0]
    for source_idx in eligible_sources:
        source_zero = source_idx - 1
        candidates = [
            target for target in eligible_targets
            if target != source_idx and 1 <= target <= num_words
        ]
        if not candidates:
            continue
        candidate_zero = np.asarray([target - 1 for target in candidates], dtype=np.int64)
        scores = word_attention[source_zero, candidate_zero]
        best_pos = int(np.argmax(scores))
        target_idx = int(candidates[best_pos])
        edges.append((source_idx, target_idx, float(scores[best_pos])))
    return edges


def compute_mbert_structural_scores(
    *,
    device: torch.device,
    force: bool,
    limit: int = 0,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if limit == 0 else f"_limit{limit}"
    out_path = OUTPUT_DIR / f"mbert_english_dev_structural_scores{suffix}.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    sentences = selected_sentences("dev", limit=limit)
    excluded_upos = {"PUNCT"}
    tokenizer = AutoTokenizer.from_pretrained(model_ref(), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_ref(), attn_implementation="eager")
    model.to(device)
    model.eval()

    stats = {
        head: {
            "matched": 0,
            "gold": 0,
            "predicted": 0,
            "sentence_f1_sum": 0.0,
            "sentence_count": 0,
            "predicted_distance_sum": 0.0,
            "predicted_distance_count": 0,
        }
        for head in all_heads()
    }

    start = time.time()
    log(f"Structural scoring on English dev: {len(sentences)} sentences")
    for sent_idx, example in enumerate(sentences, start=1):
        try:
            encoding = core.encode_words(tokenizer, example.words)
        except ValueError:
            continue
        word_ids = encoding.word_ids(batch_index=0)
        if word_ids is None:
            continue
        gold_edges = gold_edges_for_example(example, excluded_upos)
        gold = edge_set(gold_edges)
        eligible = sorted(valid_token_indices(example, excluded_upos))
        eligible_set = set(eligible)
        if not gold or len(eligible) < 2:
            continue

        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                output_attentions=True,
                return_dict=True,
            )
        matrices = core.word_attention_by_head(outputs.attentions, word_ids, example.length)
        for head, matrix in matrices.items():
            predicted = top1_edges_bidirectional(matrix, eligible, eligible_set)
            predicted_set = edge_set(predicted)
            matched = len(gold & predicted_set)
            precision, recall, f1 = precision_recall_f1(matched, len(gold), len(predicted_set))
            item = stats[head]
            item["matched"] += matched
            item["gold"] += len(gold)
            item["predicted"] += len(predicted_set)
            item["sentence_f1_sum"] += f1
            item["sentence_count"] += 1
            item["predicted_distance_sum"] += sum(abs(dep - head_idx) for dep, head_idx, _ in predicted)
            item["predicted_distance_count"] += len(predicted)

        del outputs
        if sent_idx % 50 == 0 or sent_idx == len(sentences):
            elapsed = (time.time() - start) / 60.0
            log(f"  structural dev processed {sent_idx}/{len(sentences)} sentences in {elapsed:.1f} min")
            clear_device_cache(device)

    rows: list[dict[str, object]] = []
    for (layer, head), item in stats.items():
        precision, recall, corpus_f1 = precision_recall_f1(
            int(item["matched"]), int(item["gold"]), int(item["predicted"])
        )
        rows.append(
            {
                "layer": layer,
                "head_index": head,
                "head_label": label_from_head((layer, head)),
                "structural_matched_edges": int(item["matched"]),
                "structural_gold_edges": int(item["gold"]),
                "structural_predicted_edges": int(item["predicted"]),
                "structural_precision": precision,
                "structural_recall": recall,
                "structural_corpus_f1": corpus_f1,
                "structural_mean_sentence_f1": (
                    item["sentence_f1_sum"] / item["sentence_count"] if item["sentence_count"] else 0.0
                ),
                "structural_mean_predicted_distance": (
                    item["predicted_distance_sum"] / item["predicted_distance_count"]
                    if item["predicted_distance_count"]
                    else float("nan")
                ),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log(f"Wrote structural scores to {out_path}")
    del model
    clear_device_cache(device)
    return df


def build_taxonomy(args: argparse.Namespace, budgets: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    structural = compute_mbert_structural_scores(
        device=core.choose_device(args.device),
        force=args.force,
        limit=args.structural_limit,
    )

    heads = load_heads(ENGLISH_SPEC["run_name"]).copy()
    relations = load_relations(ENGLISH_SPEC["run_name"]).copy()
    relations["positive_margin"] = relations["accuracy_margin"].clip(lower=0.0)
    relations["syntactic_hit"] = (
        (relations["opportunity_count"] >= args.relation_min_count)
        & (relations["accuracy_margin"] >= args.margin_threshold)
    ).astype(int)
    relation_scores = (
        relations.groupby(["layer", "head_index"], sort=True)
        .agg(
            syntactic_margin_sum=("positive_margin", "sum"),
            syntactic_relation_count_from_rel=("syntactic_hit", "sum"),
            best_positive_margin=("accuracy_margin", "max"),
        )
        .reset_index()
    )

    taxonomy = (
        heads.merge(structural, on=["layer", "head_index", "head_label"], how="left")
        .merge(relation_scores, on=["layer", "head_index"], how="left")
        .fillna(
            {
                "structural_corpus_f1": 0.0,
                "structural_mean_sentence_f1": 0.0,
                "syntactic_margin_sum": 0.0,
                "syntactic_relation_count_from_rel": 0,
                "best_positive_margin": 0.0,
            }
        )
    )
    taxonomy["near_positional"] = taxonomy["positional_share"] >= args.positional_threshold
    taxonomy["strict_positional"] = taxonomy["is_positional"].fillna(False).astype(bool)
    taxonomy["syntactic_flag"] = taxonomy["is_syntactic_head"].fillna(False).astype(bool)
    taxonomy["pure_syntax_flag"] = taxonomy["syntactic_flag"] & ~taxonomy["near_positional"]
    taxonomy["syntax_position_flag"] = taxonomy["syntactic_flag"] & taxonomy["near_positional"]
    taxonomy["rare_flag"] = taxonomy["is_rare_word_head"].fillna(False).astype(bool)
    taxonomy["structural_score"] = normalize(taxonomy["structural_mean_sentence_f1"])
    taxonomy["syntactic_score"] = (
        0.60 * normalize(taxonomy["syntactic_margin_sum"])
        + 0.40 * normalize(taxonomy["syntactic_relation_count_from_rel"])
    )
    taxonomy["positional_score"] = normalize(taxonomy["positional_share"])
    taxonomy["rare_lexical_score"] = normalize(taxonomy["rare_top2_rate"])
    taxonomy["syntax_position_score"] = (
        0.50 * taxonomy["syntactic_score"] + 0.50 * taxonomy["positional_score"]
    )
    taxonomy["syntax_composite_score"] = (
        0.45 * taxonomy["structural_score"]
        + 0.35 * taxonomy["syntactic_score"]
        + 0.20 * normalize(taxonomy["confidence_mean"])
    )
    taxonomy["attention_only_utility_score"] = (
        0.35 * taxonomy["structural_score"]
        + 0.35 * taxonomy["syntactic_score"]
        + 0.15 * taxonomy["positional_score"]
        + 0.15 * taxonomy["rare_lexical_score"]
    )

    taxonomy = taxonomy.sort_values(["layer", "head_index"]).reset_index(drop=True)
    prediction_derived_cols = [
        col for col in taxonomy.columns
        if col.startswith("mlm_attr_") or col in {"mlm_attr_count", "mlm_attr_gold_abs_rank", "mlm_attr_top1_abs_rank"}
    ]
    taxonomy.drop(columns=prediction_derived_cols, errors="ignore").to_csv(
        OUTPUT_DIR / "mbert_english_all144_head_taxonomy.csv",
        index=False,
    )

    definitions: list[dict[str, object]] = []
    definitions.append(
        {
            "condition": "full",
            "family": "full",
            "test_tier": "baseline",
            "budget": 0,
            "seed": -1,
            "head_count": 144,
            "heads": format_heads(all_heads()),
            "selection_rule": "all heads active",
        }
    )

    rng = random.Random(args.seed + 999)
    for budget in budgets:
        category_heads = choose_category_heads(taxonomy, budget)
        for family, heads_for_family in category_heads.items():
            if family not in CATEGORY_FAMILIES:
                continue
            definitions.append(
                {
                    "condition": f"{family}_k{budget}",
                    "family": family,
                    "test_tier": "primary" if family in PRIMARY_FAMILIES else "secondary",
                    "budget": budget,
                    "seed": -1,
                    "head_count": len(heads_for_family),
                    "heads": format_heads(heads_for_family),
                    "selection_rule": category_selection_rule(family),
                }
            )

            for seed_idx in range(args.layer_random_seeds):
                family_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(family))
                seed_value = args.seed + 10_000 + budget * 100 + seed_idx + family_offset
                sampled = layer_matched_random(heads_for_family, seed_value)
                definitions.append(
                    {
                        "condition": f"layer_random_for_{family}_k{budget}_seed{seed_idx}",
                        "family": f"layer_random_for_{family}",
                        "test_tier": "control",
                        "budget": budget,
                        "seed": seed_idx,
                        "head_count": len(sampled),
                        "heads": format_heads(sampled),
                        "selection_rule": f"layer-matched random control for {family}",
                    }
                )

        for seed_idx in range(args.uniform_random_seeds):
            sampled = rng.sample(all_heads(), budget)
            definitions.append(
                {
                    "condition": f"uniform_random_k{budget}_seed{seed_idx}",
                    "family": "uniform_random",
                    "test_tier": "control",
                    "budget": budget,
                    "seed": seed_idx,
                    "head_count": len(sampled),
                    "heads": format_heads(sampled),
                    "selection_rule": "uniform random heads across all layers",
                }
            )

    definitions_df = pd.DataFrame(definitions)
    definitions_df.to_csv(OUTPUT_DIR / "mbert_english_category_definitions.csv", index=False)
    log(f"Wrote taxonomy and definitions to {OUTPUT_DIR}")
    return taxonomy, definitions_df


def build_definitions_from_saved_taxonomy(args: argparse.Namespace, budgets: list[int]) -> pd.DataFrame:
    taxonomy_path = OUTPUT_DIR / "mbert_english_all144_head_taxonomy.csv"
    if not taxonomy_path.exists():
        raise FileNotFoundError(taxonomy_path)

    taxonomy = pd.read_csv(taxonomy_path)
    definitions: list[dict[str, object]] = [
        {
            "condition": "full",
            "family": "full",
            "test_tier": "baseline",
            "budget": 0,
            "seed": -1,
            "head_count": 144,
            "heads": format_heads(all_heads()),
            "selection_rule": "all heads active",
        }
    ]

    rng = random.Random(args.seed + 999)
    for budget in budgets:
        category_heads = choose_category_heads(taxonomy, budget)
        for family, heads_for_family in category_heads.items():
            if family not in CATEGORY_FAMILIES:
                continue
            definitions.append(
                {
                    "condition": f"{family}_k{budget}",
                    "family": family,
                    "test_tier": "primary" if family in PRIMARY_FAMILIES else "secondary",
                    "budget": budget,
                    "seed": -1,
                    "head_count": len(heads_for_family),
                    "heads": format_heads(heads_for_family),
                    "selection_rule": category_selection_rule(family),
                }
            )

            for seed_idx in range(args.layer_random_seeds):
                family_offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(family))
                seed_value = args.seed + 10_000 + budget * 100 + seed_idx + family_offset
                sampled = layer_matched_random(heads_for_family, seed_value)
                definitions.append(
                    {
                        "condition": f"layer_random_for_{family}_k{budget}_seed{seed_idx}",
                        "family": f"layer_random_for_{family}",
                        "test_tier": "control",
                        "budget": budget,
                        "seed": seed_idx,
                        "head_count": len(sampled),
                        "heads": format_heads(sampled),
                        "selection_rule": f"layer-matched random control for {family}",
                    }
                )

        for seed_idx in range(args.uniform_random_seeds):
            sampled = rng.sample(all_heads(), budget)
            definitions.append(
                {
                    "condition": f"uniform_random_k{budget}_seed{seed_idx}",
                    "family": "uniform_random",
                    "test_tier": "control",
                    "budget": budget,
                    "seed": seed_idx,
                    "head_count": len(sampled),
                    "heads": format_heads(sampled),
                    "selection_rule": "uniform random heads across all layers",
                }
            )

    definitions_df = pd.DataFrame(definitions)
    definitions_df.to_csv(OUTPUT_DIR / "mbert_english_category_definitions.csv", index=False)
    log(f"Rebuilt category definitions from saved taxonomy at {taxonomy_path}")
    return definitions_df


def manifests_are_compatible_for_resume(
    *,
    previous: dict[str, object],
    current: dict[str, object],
    output_prefix: str,
) -> bool:
    if output_prefix != "mlm_sufficiency":
        return False
    previous_budgets = set(previous.get("budgets", []))
    current_budgets = set(current.get("budgets", []))
    if not previous_budgets.issubset(current_budgets):
        return False
    for key, value in current.items():
        if key == "budgets":
            continue
        if previous.get(key) != value:
            return False
    return True


def category_selection_rule(family: str) -> str:
    return {
        "syntactic_relation": "top heads by dev relation-margin syntactic score",
        "pure_syntactic_relation": "top syntactic heads after excluding near-positional heads",
        "structural_graph": "top heads by dev unlabeled UD graph F1",
        "positional": "top heads by fixed-offset positional share",
        "syntax_position_hybrid": "top heads by combined syntactic and positional score",
        "syntax_composite": "top heads by structural + syntactic + confidence composite score",
        "rare_lexical": "top heads by rare-token attention score",
        "low_utility": "bottom heads by attention-only utility score",
    }.get(family, family)


def choose_category_heads(taxonomy: pd.DataFrame, budget: int) -> dict[str, list[tuple[int, int]]]:
    def coords(df: pd.DataFrame) -> list[tuple[int, int]]:
        return [(int(row.layer), int(row.head_index)) for row in df.head(budget).itertuples(index=False)]

    ordered: dict[str, pd.DataFrame] = {}
    ordered["syntactic_relation"] = taxonomy.sort_values(
        ["syntactic_score", "syntactic_margin_sum", "confidence_mean"],
        ascending=False,
    )
    pure_pool = taxonomy[taxonomy["pure_syntax_flag"]].sort_values(
        ["syntactic_score", "syntactic_margin_sum", "confidence_mean"],
        ascending=False,
    )
    if len(pure_pool) < budget:
        pure_pool = taxonomy[~taxonomy["near_positional"]].sort_values(
            ["syntactic_score", "syntactic_margin_sum", "confidence_mean"],
            ascending=False,
        )
    ordered["pure_syntactic_relation"] = pure_pool
    ordered["structural_graph"] = taxonomy.sort_values(
        ["structural_mean_sentence_f1", "structural_corpus_f1", "confidence_mean"],
        ascending=False,
    )
    ordered["positional"] = taxonomy.sort_values(["positional_share", "confidence_mean"], ascending=False)
    ordered["syntax_position_hybrid"] = taxonomy.sort_values(
        ["syntax_position_score", "syntactic_score", "positional_score"],
        ascending=False,
    )
    ordered["syntax_composite"] = taxonomy.sort_values(
        ["syntax_composite_score", "structural_score", "syntactic_score"],
        ascending=False,
    )
    ordered["rare_lexical"] = taxonomy.sort_values(["rare_top2_rate", "rare_top1_rate"], ascending=False)
    ordered["low_utility"] = taxonomy.sort_values(
        ["attention_only_utility_score", "confidence_mean"],
        ascending=[True, True],
    )
    return {family: coords(df) for family, df in ordered.items()}


def layer_matched_random(target_heads: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    by_layer: dict[int, int] = defaultdict(int)
    target_set = set(target_heads)
    for layer, _head in target_heads:
        by_layer[layer] += 1
    sampled: list[tuple[int, int]] = []
    for layer, count in sorted(by_layer.items()):
        pool = [(layer, head) for head in range(12) if (layer, head) not in target_set]
        if len(pool) < count:
            pool = [(layer, head) for head in range(12)]
        sampled.extend(rng.sample(pool, count))
    return sorted(sampled)


def parse_heads_cell(value: object) -> list[tuple[int, int]]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [head_from_label(part.strip()) for part in text.split(",") if part.strip()]


def make_head_mask(
    device: torch.device,
    heads: list[tuple[int, int]],
    mode: str,
) -> torch.Tensor | None:
    if mode == "full":
        return None
    if mode == "ablate":
        mask = torch.ones((12, 12), dtype=torch.float32, device=device)
        for layer, head in heads:
            mask[layer, head] = 0.0
        return mask
    if mode == "sufficiency":
        mask = torch.zeros((12, 12), dtype=torch.float32, device=device)
        for layer, head in heads:
            mask[layer, head] = 1.0
        return mask
    raise ValueError(f"Unknown mask mode: {mode}")


def batch_iter(items: list[RelationMaskedExample], batch_size: int) -> Iterable[list[RelationMaskedExample]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def pad_batch(
    examples: list[RelationMaskedExample],
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(example.input_ids) for example in examples)
    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    target_positions: list[int] = []
    target_ids: list[int] = []
    for example in examples:
        pad_len = max_len - len(example.input_ids)
        input_rows.append(example.input_ids + [pad_id] * pad_len)
        attention_rows.append(example.attention_mask + [0] * pad_len)
        target_positions.append(example.target_pos)
        target_ids.append(example.target_id)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
        torch.tensor(target_positions, dtype=torch.long, device=device),
        torch.tensor(target_ids, dtype=torch.long, device=device),
    )


def select_relation_masked_examples(
    *,
    tokenizer,
    sentence_count: int,
    tokens_per_sentence: int,
    target_families: set[str],
    seed: int,
) -> list[RelationMaskedExample]:
    sentences = selected_sentences("test", limit=0)
    rng = random.Random(seed)
    if sentence_count > 0 and len(sentences) > sentence_count:
        sentences = rng.sample(sentences, sentence_count)
    examples: list[RelationMaskedExample] = []
    excluded_upos = {"PUNCT"}

    for sent_number, example in enumerate(sentences):
        by_idx = {token.idx: token for token in example.tokens}
        candidates = [
            token for token in example.tokens
            if token.head != 0
            and token.upos not in excluded_upos
            and relation_family(token.deprel) in target_families
        ]
        rng.shuffle(candidates)
        candidates = candidates[:tokens_per_sentence]
        if not candidates:
            continue
        try:
            encoding = core.encode_words(tokenizer, example.words)
        except ValueError:
            continue
        word_ids = encoding.word_ids(batch_index=0)
        if word_ids is None:
            continue
        input_ids_base = encoding["input_ids"][0].tolist()
        attention_base = encoding["attention_mask"][0].tolist()
        for token in candidates:
            piece_positions = [idx for idx, word_id in enumerate(word_ids) if word_id == token.idx - 1]
            if len(piece_positions) != 1:
                # Primary MLM analysis avoids WordPiece leakage: if a word splits into
                # multiple pieces, masking only the first piece leaves the rest visible.
                continue
            target_pos = piece_positions[0]
            target_id = int(input_ids_base[target_pos])
            input_ids = list(input_ids_base)
            input_ids[target_pos] = tokenizer.mask_token_id
            examples.append(
                RelationMaskedExample(
                    example_id=f"en_test:{sent_number}:{token.idx}:{token.deprel}",
                    sent_id=example.sent_id,
                    word_index=token.idx,
                    word=token.form,
                    upos=token.upos,
                    relation=token.deprel,
                    relation_family=relation_family(token.deprel),
                    dependency_distance=abs(token.idx - token.head),
                    wordpiece_count=len(piece_positions),
                    input_ids=input_ids,
                    attention_mask=list(attention_base),
                    target_pos=target_pos,
                    target_id=target_id,
                )
            )
    return examples


def score_mlm_condition(
    *,
    model,
    tokenizer,
    device: torch.device,
    examples: list[RelationMaskedExample],
    condition_row: pd.Series,
    mode: str,
    batch_size: int,
) -> pd.DataFrame:
    heads = parse_heads_cell(condition_row["heads"])
    condition = str(condition_row["condition"])
    family = str(condition_row["family"])
    budget = int(condition_row["budget"])
    seed = int(condition_row["seed"])
    test_tier = str(condition_row["test_tier"])
    head_mask = make_head_mask(device, heads, "full" if condition == "full" else mode)
    rows: list[dict[str, object]] = []

    for batch in batch_iter(examples, batch_size):
        input_ids, attention_mask, target_positions, target_ids = pad_batch(batch, tokenizer.pad_token_id, device)
        with torch.no_grad():
            encoder_outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                return_dict=True,
            )
            batch_indices = torch.arange(len(batch), device=device)
            masked_hidden = encoder_outputs.last_hidden_state[batch_indices, target_positions]
            logits = model.cls(masked_hidden)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            nll = -log_probs[batch_indices, target_ids]
            predictions = torch.argmax(logits, dim=-1)
        nll_cpu = nll.detach().cpu().numpy()
        pred_cpu = predictions.detach().cpu().numpy()
        target_cpu = target_ids.detach().cpu().numpy()
        for idx, example in enumerate(batch):
            rows.append(
                {
                    "example_id": example.example_id,
                    "sent_id": example.sent_id,
                    "word_index": example.word_index,
                    "word": example.word,
                    "upos": example.upos,
                    "relation": example.relation,
                    "relation_family": example.relation_family,
                    "dependency_distance": example.dependency_distance,
                    "wordpiece_count": example.wordpiece_count,
                    "condition": condition,
                    "family": family,
                    "test_tier": test_tier,
                    "budget": budget,
                    "seed": seed,
                    "head_count": int(condition_row["head_count"]),
                    "target_id": int(target_cpu[idx]),
                    "predicted_id": int(pred_cpu[idx]),
                    "nll": float(nll_cpu[idx]),
                    "accuracy": float(pred_cpu[idx] == target_cpu[idx]),
                }
            )
        del encoder_outputs, masked_hidden, logits, log_probs, nll, predictions, input_ids, attention_mask
    clear_device_cache(device)
    return pd.DataFrame(rows)


def run_mlm_stage(args: argparse.Namespace, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode not in {"ablate", "sufficiency"}:
        raise ValueError(mode)
    output_prefix = "mlm_ablation" if mode == "ablate" else "mlm_sufficiency"
    token_path = OUTPUT_DIR / f"{output_prefix}_token_scores.csv"
    condition_dir = OUTPUT_DIR / f"{output_prefix}_condition_scores"
    manifest_path = OUTPUT_DIR / f"{output_prefix}_manifest.json"
    manifest = {
        "mode": mode,
        "budgets": parse_budgets(args.budgets),
        "seed": args.seed,
        "mlm_sentences": args.mlm_sentences,
        "mlm_tokens_per_sentence": args.mlm_tokens_per_sentence,
        "target_relations": sorted(part.strip() for part in args.target_relations.split(",") if part.strip()),
        "layer_random_seeds": args.layer_random_seeds,
        "uniform_random_seeds": args.uniform_random_seeds,
        "batch_size": args.batch_size,
        "max_conditions": args.max_conditions,
    }
    if args.force and condition_dir.exists():
        shutil.rmtree(condition_dir)
    if manifest_path.exists() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest and not manifests_are_compatible_for_resume(
            previous=previous,
            current=manifest,
            output_prefix=output_prefix,
        ):
            raise RuntimeError(
                f"Existing {output_prefix} outputs were produced with different settings. "
                f"Use --force or remove {condition_dir}."
            )
    condition_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    definitions_path = OUTPUT_DIR / "mbert_english_category_definitions.csv"
    if not definitions_path.exists():
        taxonomy_path = OUTPUT_DIR / "mbert_english_all144_head_taxonomy.csv"
        if taxonomy_path.exists():
            build_definitions_from_saved_taxonomy(args, parse_budgets(args.budgets))
        else:
            build_taxonomy(args, parse_budgets(args.budgets))
    definitions = pd.read_csv(definitions_path)
    requested_budgets = set(parse_budgets(args.budgets))
    available_budgets = set(definitions.loc[definitions["budget"] != 0, "budget"].astype(int))
    if not requested_budgets.issubset(available_budgets):
        definitions = build_definitions_from_saved_taxonomy(args, parse_budgets(args.budgets))
    definitions = definitions.sort_values(["budget", "test_tier", "family", "seed", "condition"]).reset_index(drop=True)

    if args.max_conditions > 0:
        full = definitions[definitions["condition"] == "full"]
        rest = definitions[definitions["condition"] != "full"].head(args.max_conditions)
        definitions = pd.concat([full, rest], ignore_index=True).drop_duplicates("condition")

    device = core.choose_device(args.device)
    log(f"{output_prefix}: selected device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_ref(), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_ref(), attn_implementation="eager")
    model.to(device)
    model.eval()
    target_families = {part.strip() for part in args.target_relations.split(",") if part.strip()}
    examples = select_relation_masked_examples(
        tokenizer=tokenizer,
        sentence_count=args.mlm_sentences,
        tokens_per_sentence=args.mlm_tokens_per_sentence,
        target_families=target_families,
        seed=args.seed,
    )
    meta_path = OUTPUT_DIR / f"{output_prefix}_example_metadata.csv"
    pd.DataFrame([example.__dict__ for example in examples]).drop(
        columns=["input_ids", "attention_mask"], errors="ignore"
    ).to_csv(meta_path, index=False)
    log(
        f"{output_prefix}: {len(examples)} masked targets across "
        f"{len({example.sent_id for example in examples})} sentences"
    )

    expected_part_paths: list[Path] = []
    for idx, row in definitions.iterrows():
        condition = str(row["condition"])
        part_path = condition_dir / f"{condition}.csv"
        expected_part_paths.append(part_path)
        if part_path.exists() and not args.force:
            log(f"{output_prefix}: skip existing {condition}")
            continue
        start = time.time()
        log(
            f"{output_prefix}: {idx + 1}/{len(definitions)} {condition} "
            f"({row['head_count']} heads)"
        )
        scored = score_mlm_condition(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=examples,
            condition_row=row,
            mode=mode,
            batch_size=args.batch_size,
        )
        scored.to_csv(part_path, index=False)
        log(f"{output_prefix}: wrote {part_path.name} in {(time.time() - start) / 60.0:.1f} min")

    all_parts = [pd.read_csv(path) for path in expected_part_paths if path.exists()]
    if not all_parts:
        raise RuntimeError(f"No condition outputs found in {condition_dir}")
    token_df = pd.concat(all_parts, ignore_index=True)
    token_df.to_csv(token_path, index=False)
    summary = summarize_mlm_token_scores(token_df, output_prefix, args.bootstrap_iters)
    del model
    clear_device_cache(device)
    return token_df, summary


def summarize_mlm_token_scores(token_df: pd.DataFrame, output_prefix: str, bootstrap_iters: int) -> pd.DataFrame:
    full = token_df[token_df["condition"] == "full"][
        ["example_id", "sent_id", "nll", "accuracy"]
    ].rename(columns={"nll": "full_nll", "accuracy": "full_accuracy"})
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(410)
    for (condition, family, budget, seed, test_tier), condition_df in token_df.groupby(
        ["condition", "family", "budget", "seed", "test_tier"], sort=True
    ):
        merged = condition_df.merge(full, on=["example_id", "sent_id"], how="inner")
        if condition == "full":
            delta = np.zeros(len(merged), dtype=np.float64)
        else:
            delta = merged["nll"].to_numpy() - merged["full_nll"].to_numpy()
        sentence_delta = (
            pd.DataFrame({"sent_id": merged["sent_id"], "delta": delta})
            .groupby("sent_id")["delta"]
            .mean()
            .to_numpy()
        )
        if len(sentence_delta) > 1 and condition != "full":
            boot = [
                float(rng.choice(sentence_delta, size=len(sentence_delta), replace=True).mean())
                for _ in range(bootstrap_iters)
            ]
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        else:
            ci_low = ci_high = float(np.mean(sentence_delta)) if len(sentence_delta) else 0.0
        rows.append(
            {
                "condition": condition,
                "family": family,
                "test_tier": test_tier,
                "budget": int(budget),
                "seed": int(seed),
                "token_count": int(len(merged)),
                "sentence_count": int(merged["sent_id"].nunique()),
                "mean_nll": float(merged["nll"].mean()),
                "accuracy": float(merged["accuracy"].mean()),
                "mean_delta_nll_vs_full": float(delta.mean()) if len(delta) else float("nan"),
                "delta_nll_ci_low": float(ci_low),
                "delta_nll_ci_high": float(ci_high),
                "fraction_tokens_hurt": float(np.mean(delta > 0.0)) if len(delta) else float("nan"),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / f"{output_prefix}_summary.csv", index=False)
    relation = summarize_by_group(token_df, full, output_prefix, group_cols=["relation_family"])
    relation.to_csv(OUTPUT_DIR / f"{output_prefix}_by_relation_family.csv", index=False)
    pos = summarize_by_group(token_df, full, output_prefix, group_cols=["upos"])
    pos.to_csv(OUTPUT_DIR / f"{output_prefix}_by_upos.csv", index=False)
    contrast = build_random_contrasts(token_df, output_prefix, bootstrap_iters)
    contrast.to_csv(OUTPUT_DIR / f"{output_prefix}_random_contrasts.csv", index=False)
    return summary


def summarize_by_group(
    token_df: pd.DataFrame,
    full: pd.DataFrame,
    output_prefix: str,
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, condition_df in token_df[token_df["condition"] != "full"].groupby(
        ["condition", "family", "budget", "seed", *group_cols], sort=True
    ):
        condition, family, budget, seed, *group_values = keys
        merged = condition_df.merge(full, on=["example_id", "sent_id"], how="inner")
        delta = merged["nll"].to_numpy() - merged["full_nll"].to_numpy()
        row = {
            "condition": condition,
            "family": family,
            "budget": int(budget),
            "seed": int(seed),
            "token_count": int(len(merged)),
            "mean_delta_nll_vs_full": float(delta.mean()) if len(delta) else float("nan"),
            "accuracy": float(merged["accuracy"].mean()) if len(merged) else float("nan"),
        }
        for col, value in zip(group_cols, group_values):
            row[col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def build_random_contrasts(token_df: pd.DataFrame, output_prefix: str, bootstrap_iters: int) -> pd.DataFrame:
    full = token_df[token_df["condition"] == "full"][["example_id", "sent_id", "nll"]].rename(
        columns={"nll": "full_nll"}
    )
    work = token_df[token_df["condition"] != "full"].merge(full, on=["example_id", "sent_id"], how="inner")
    work["delta_nll"] = work["nll"] - work["full_nll"]
    rng = np.random.default_rng(911)
    rows: list[dict[str, object]] = []

    targets = work[work["test_tier"].isin(["primary", "secondary"])]
    for (condition, family, budget), target_df in targets.groupby(["condition", "family", "budget"], sort=True):
        for control_kind, control_family in [
            ("layer_random", f"layer_random_for_{family}"),
            ("uniform_random", "uniform_random"),
        ]:
            control_df = work[(work["family"] == control_family) & (work["budget"] == budget)]
            if control_df.empty:
                continue
            control_mean = (
                control_df.groupby("example_id", sort=False)["delta_nll"]
                .mean()
                .rename("control_delta")
                .reset_index()
            )
            target_delta = target_df[["example_id", "sent_id", "delta_nll"]].merge(control_mean, on="example_id", how="inner")
            if target_delta.empty:
                continue
            if output_prefix == "mlm_sufficiency":
                # In sufficiency, lower retained-subset delta than random is better.
                target_delta["contrast"] = target_delta["control_delta"] - target_delta["delta_nll"]
                claim_type = "prediction_sufficient_vs_control"
            else:
                # In ablation, higher targeted delta than random means removing the
                # family hurts more than removing comparable random heads.
                target_delta["contrast"] = target_delta["delta_nll"] - target_delta["control_delta"]
                claim_type = "prediction_necessary_vs_control"
            sentence_contrast = target_delta.groupby("sent_id")["contrast"].mean().to_numpy()
            if len(sentence_contrast) > 1:
                boot = [
                    float(rng.choice(sentence_contrast, size=len(sentence_contrast), replace=True).mean())
                    for _ in range(bootstrap_iters)
                ]
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            else:
                ci_low = ci_high = float(sentence_contrast.mean()) if len(sentence_contrast) else float("nan")
            rows.append(
                {
                    "condition": condition,
                    "family": family,
                    "budget": int(budget),
                    "control_kind": control_kind,
                    "target_mean_delta_nll": float(target_delta["delta_nll"].mean()),
                    "control_mean_delta_nll": float(target_delta["control_delta"].mean()),
                    "effect_vs_control": float(target_delta["contrast"].mean()),
                    "effect_ci_low": float(ci_low),
                    "effect_ci_high": float(ci_high),
                    "sentence_count": int(target_delta["sent_id"].nunique()),
                    "token_count": int(len(target_delta)),
                    "claim_type": claim_type,
                    "claim_supported": bool(ci_low > 0.0),
                }
            )
    return pd.DataFrame(rows)


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
        positions, membership, src_counts = core.word_membership_matrix(word_ids, num_words)
        if not positions:
            continue
        active_attention = layer_attention[:, positions][:, :, positions]
        to_word = np.einsum("hts,sw->htw", active_attention, membership, optimize=True)
        by_head = np.einsum("tv,htw->hvw", membership, to_word, optimize=True)
        by_head = by_head / src_counts[np.newaxis, :, np.newaxis]
        for head_idx in sorted(layer_heads):
            collected.append(by_head[head_idx])
    if not collected:
        raise RuntimeError("No attention matrices were available for selected subset.")
    return np.mean(np.stack(collected, axis=0), axis=0)


def run_attention_eval(args: argparse.Namespace) -> pd.DataFrame:
    definitions_path = OUTPUT_DIR / "mbert_english_category_definitions.csv"
    if not definitions_path.exists():
        build_taxonomy(args, parse_budgets(args.budgets))
    definitions = pd.read_csv(definitions_path)
    definitions = definitions[definitions["family"].isin(set(CATEGORY_FAMILIES) | {"full"})].copy()
    definitions = definitions.sort_values(["budget", "test_tier", "family"]).drop_duplicates("condition")

    device = core.choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref(), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_ref(), attn_implementation="eager")
    model.to(device)
    model.eval()
    sentences = selected_sentences("test", limit=args.attention_limit)
    excluded_upos = {"PUNCT"}
    rows: list[dict[str, object]] = []
    log(f"Attention eval on English test: {len(sentences)} sentences, {len(definitions)} subsets")

    start = time.time()
    for sent_idx, example in enumerate(sentences, start=1):
        try:
            encoding = core.encode_words(tokenizer, example.words)
        except ValueError:
            continue
        word_ids = encoding.word_ids(batch_index=0)
        if word_ids is None:
            continue
        gold_edges = gold_edges_for_example(example, excluded_upos)
        gold = edge_set(gold_edges)
        eligible = sorted(valid_token_indices(example, excluded_upos))
        eligible_set = set(eligible)
        if not gold or len(eligible) < 2:
            continue
        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                output_attentions=True,
                return_dict=True,
            )

        for row in definitions.itertuples(index=False):
            heads = all_heads() if row.condition == "full" else parse_heads_cell(row.heads)
            combined = subset_attention_matrix(outputs.attentions, word_ids, example.length, heads)
            predicted = top1_edges_bidirectional(combined, eligible, eligible_set)
            matched = len(gold & edge_set(predicted))
            precision, recall, f1 = precision_recall_f1(matched, len(gold), len(predicted))
            rows.append(
                {
                    "sent_id": example.sent_id,
                    "condition": row.condition,
                    "family": row.family,
                    "test_tier": row.test_tier,
                    "budget": int(row.budget),
                    "head_count": int(row.head_count),
                    "gold_edges": len(gold),
                    "predicted_edges": len(predicted),
                    "matched_edges": matched,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

        del outputs
        if sent_idx % 50 == 0 or sent_idx == len(sentences):
            log(f"  attention eval processed {sent_idx}/{len(sentences)} in {(time.time() - start) / 60.0:.1f} min")
            clear_device_cache(device)

    sentence_df = pd.DataFrame(rows)
    sentence_df.to_csv(OUTPUT_DIR / "mbert_english_attention_eval_sentence.csv", index=False)
    summary = (
        sentence_df.groupby(["condition", "family", "test_tier", "budget", "head_count"], sort=True)
        .agg(
            sentence_count=("sent_id", "nunique"),
            total_gold_edges=("gold_edges", "sum"),
            total_predicted_edges=("predicted_edges", "sum"),
            total_matched_edges=("matched_edges", "sum"),
            mean_sentence_f1=("f1", "mean"),
        )
        .reset_index()
    )
    summary["corpus_precision"] = summary["total_matched_edges"] / summary["total_predicted_edges"]
    summary["corpus_recall"] = summary["total_matched_edges"] / summary["total_gold_edges"]
    summary["corpus_f1"] = np.where(
        summary["corpus_precision"] + summary["corpus_recall"] == 0.0,
        0.0,
        2.0 * summary["corpus_precision"] * summary["corpus_recall"]
        / (summary["corpus_precision"] + summary["corpus_recall"]),
    )
    summary.to_csv(OUTPUT_DIR / "mbert_english_attention_eval_summary.csv", index=False)
    del model
    clear_device_cache(device)
    return summary


def save_figures() -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    taxonomy_path = OUTPUT_DIR / "mbert_english_all144_head_taxonomy.csv"
    if taxonomy_path.exists():
        taxonomy = pd.read_csv(taxonomy_path)
        heat = np.zeros((12, 12), dtype=float)
        for row in taxonomy.itertuples(index=False):
            heat[int(row.layer), int(row.head_index)] = float(row.syntax_composite_score)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
        im = ax.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=max(heat.max(), 1e-9))
        ax.set_title("mBERT English syntax-composite score by head")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(12))
        ax.set_yticks(range(12))
        fig.colorbar(im, ax=ax, label="score")
        fig.tight_layout()
        path = FIGURE_DIR / "mbert_head_taxonomy_map.png"
        fig.savefig(path)
        plt.close(fig)
        paths["taxonomy_map"] = path

    for prefix in ("mlm_ablation", "mlm_sufficiency"):
        summary_path = OUTPUT_DIR / f"{prefix}_summary.csv"
        contrast_path = OUTPUT_DIR / f"{prefix}_random_contrasts.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            plot = summary[
                summary["family"].isin(CATEGORY_FAMILIES)
                & summary["budget"].isin(REPORT_BUDGETS)
            ].copy()
            if not plot.empty:
                fig, ax = plt.subplots(figsize=(15.5, 7.0), dpi=180)
                if contrast_path.exists():
                    contrast_for_baseline = pd.read_csv(contrast_path)
                    random_baseline = contrast_for_baseline[
                        (contrast_for_baseline["control_kind"] == "layer_random")
                        & contrast_for_baseline["family"].isin(CATEGORY_FAMILIES)
                    ].copy()
                    if not random_baseline.empty:
                        random_by_budget = (
                            random_baseline.groupby("budget")["control_mean_delta_nll"]
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
                            linewidth=2.6,
                            label="layer-random mean",
                        )
                for family, family_df in plot.groupby("family", sort=True):
                    family_df = family_df.sort_values("budget")
                    ax.plot(
                        family_df["budget"],
                        family_df["mean_delta_nll_vs_full"],
                        marker="o",
                        markersize=6.0,
                        linewidth=2.2,
                        label=family,
                    )
                ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--")
                ax.set_title(f"{prefix.replace('_', ' ')}: delta NLL vs full")
                ax.set_xlabel("Budget k")
                ax.set_ylabel("Delta NLL")
                ax.grid(True, axis="y", alpha=0.18)
                ax.legend(ncol=4, fontsize=8.5, frameon=True)
                fig.tight_layout()
                path = FIGURE_DIR / f"{prefix}_delta_by_budget.png"
                fig.savefig(path)
                plt.close(fig)
                paths[f"{prefix}_delta"] = path
        if contrast_path.exists():
            contrast = pd.read_csv(contrast_path)
            plot = contrast[
                (contrast["control_kind"] == "layer_random")
                & contrast["family"].isin(CATEGORY_FAMILIES)
            ].copy()
            if not plot.empty:
                fig, ax = plt.subplots(figsize=(17, 7.5), dpi=180)
                labels: list[str] = []
                values: list[float] = []
                lows: list[float] = []
                highs: list[float] = []
                colors_list: list[str] = []
                for row in plot.sort_values(["budget", "family"]).itertuples(index=False):
                    labels.append(f"{row.family}\nk={int(row.budget)}")
                    values.append(float(row.effect_vs_control))
                    lows.append(max(0.0, float(row.effect_vs_control - row.effect_ci_low)))
                    highs.append(max(0.0, float(row.effect_ci_high - row.effect_vs_control)))
                    colors_list.append("#d95f02" if row.claim_supported else "#7570b3")
                x = np.arange(len(labels))
                ax.bar(x, values, yerr=[lows, highs], color=colors_list, alpha=0.85, capsize=2)
                ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=68, ha="right", fontsize=8)
                ax.set_title(f"{prefix.replace('_', ' ')}: effect vs layer-matched random")
                if prefix == "mlm_sufficiency":
                    ax.set_ylabel("Layer-random delta NLL - target delta NLL")
                else:
                    ax.set_ylabel("Target delta NLL - layer-random delta NLL")
                fig.tight_layout()
                path = FIGURE_DIR / f"{prefix}_effect_vs_layer_random.png"
                fig.savefig(path)
                plt.close(fig)
                paths[f"{prefix}_contrast"] = path
    return paths


def rel(path: Path) -> str:
    return path.relative_to(OUTPUT_DIR).as_posix()


def df_to_markdown(df: pd.DataFrame, *, floatfmt: str | None = None) -> str:
    if df.empty:
        return ""
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]) and floatfmt is not None:
            table[col] = table[col].map(lambda value: format(value, floatfmt) if pd.notna(value) else "")
        else:
            table[col] = table[col].map(lambda value: "" if pd.isna(value) else str(value))
        table[col] = table[col].str.replace("|", "\\|", regex=False)

    headers = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "taxonomy"}:
        if args.force or not (OUTPUT_DIR / "mbert_english_category_definitions.csv").exists():
            build_taxonomy(args, budgets)
        else:
            log("Taxonomy outputs already exist; use --force to rebuild.")
    if args.stage in {"all", "attention_eval"}:
        if args.force or not (OUTPUT_DIR / "mbert_english_attention_eval_summary.csv").exists():
            run_attention_eval(args)
        else:
            log("Attention eval outputs already exist; use --force to rebuild.")
    if args.stage in {"all", "mlm_ablation"}:
        run_mlm_stage(args, mode="ablate")
    if args.stage in {"all", "mlm_sufficiency"}:
        run_mlm_stage(args, mode="sufficiency")


if __name__ == "__main__":
    main()

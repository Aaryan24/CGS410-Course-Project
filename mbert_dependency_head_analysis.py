#!/usr/bin/env python3
"""
dependency-head mBERT head characterization for UD treebanks.

This implements the transferable metrics from the prior attention-head analysis literature through section 5.3:

- head confidence
- positional heads
- syntactic heads against relation-specific positional baselines
- rare-word heads
- optional LRP-style attribution via gradient * activation at head outputs

The default data source is UD English-EWT. Other languages can be supplied with
split URLs or local .conllu paths.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "bert-base-multilingual-cased"
DEFAULT_UD_VERSION = "r2.17"
DEFAULT_UD_BASE_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/"
    f"{DEFAULT_UD_VERSION}"
)
DEFAULT_UD_FILES = {
    "train": f"{DEFAULT_UD_BASE_URL}/en_ewt-ud-train.conllu",
    "dev": f"{DEFAULT_UD_BASE_URL}/en_ewt-ud-dev.conllu",
    "test": f"{DEFAULT_UD_BASE_URL}/en_ewt-ud-test.conllu",
}

POSITIONAL_SHARE_THRESHOLD = 0.90
SYNTACTIC_MARGIN_THRESHOLD = 0.10
RARE_TOP2_THRESHOLD = 0.50


@dataclass
class UDToken:
    idx: int
    form: str
    head: int
    deprel: str
    upos: str


@dataclass
class SentenceExample:
    sent_id: str
    split: str
    tokens: list[UDToken]

    @property
    def words(self) -> list[str]:
        return [token.form for token in self.tokens]

    @property
    def length(self) -> int:
        return len(self.tokens)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dependency-head mBERT head metrics on a UD treebank.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--language", default="English")
    parser.add_argument("--treebank", default="UD_English-EWT")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/mbert_dependency_head_ud"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/mbert_dependency_head"))
    parser.add_argument("--run-name", default="en_ewt_mbert_dependency_head_dev")
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 for all filtered sentences.")
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=30)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--ud-files-json", default=os.environ.get("CGS410_UD_FILES_JSON", ""))
    parser.add_argument("--ud-train", default=os.environ.get("CGS410_UD_TRAIN_URL", ""))
    parser.add_argument("--ud-dev", default=os.environ.get("CGS410_UD_DEV_URL", ""))
    parser.add_argument("--ud-test", default=os.environ.get("CGS410_UD_TEST_URL", ""))
    parser.add_argument("--relation-min-count", type=int, default=50)
    parser.add_argument("--positional-threshold", type=float, default=POSITIONAL_SHARE_THRESHOLD)
    parser.add_argument("--syntactic-margin-threshold", type=float, default=SYNTACTIC_MARGIN_THRESHOLD)
    parser.add_argument("--rare-threshold", type=float, default=RARE_TOP2_THRESHOLD)
    parser.add_argument("--exclude-upos", default="PUNCT", help="Comma-separated UPOS labels to exclude from head metrics.")
    parser.add_argument("--exclude-relations", default="", help="Comma-separated dependency relations to exclude from headline labels.")
    parser.add_argument("--attribution-sentences", type=int, default=0, help="Number of selected sentences for attribution. 0 disables.")
    parser.add_argument("--attribution-tokens-per-sentence", type=int, default=2)
    parser.add_argument("--attribution-seed", type=int, default=13)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def parse_text_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",") if part.strip()}


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise ValueError(f"No integer values found in {text!r}")
    return values


def resolve_index(idx: int, size: int, label: str) -> int:
    resolved = idx if idx >= 0 else size + idx
    if resolved < 0 or resolved >= size:
        raise IndexError(f"Invalid {label} index {idx} for size {size}.")
    return resolved


def resolve_selection(text: str, size: int, label: str) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(size))
    seen: set[int] = set()
    selected: list[int] = []
    for idx in parse_int_list(text):
        resolved = resolve_index(idx, size, label)
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append(resolved)
    return selected


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_ud_sources(args: argparse.Namespace) -> dict[str, str]:
    if args.ud_files_json:
        mapping = json.loads(args.ud_files_json)
        missing = {"train", "dev", "test"} - set(mapping)
        if missing:
            raise ValueError(f"--ud-files-json missing splits: {sorted(missing)}")
        return {split: str(mapping[split]) for split in ("train", "dev", "test")}
    if args.ud_train or args.ud_dev or args.ud_test:
        if not (args.ud_train and args.ud_dev and args.ud_test):
            raise ValueError("Provide all of --ud-train, --ud-dev, and --ud-test together.")
        return {"train": args.ud_train, "dev": args.ud_dev, "test": args.ud_test}
    return dict(DEFAULT_UD_FILES)


def local_name_for_source(source: str, split: str) -> str:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        return Path(parsed.path).name or f"{split}.conllu"
    return Path(source).name or f"{split}.conllu"


def ensure_ud_files(sources: dict[str, str], cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, source in sources.items():
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            local_path = cache_dir / local_name_for_source(source, split)
            if not local_path.exists():
                log(f"Downloading {split} split from {source}")
                urlretrieve(source, local_path)
            paths[split] = local_path
        else:
            path = Path(source).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"UD {split} path does not exist: {path}")
            paths[split] = path
    return paths


def parse_conllu(path: Path, split: str) -> list[SentenceExample]:
    sentences: list[SentenceExample] = []
    sent_id: str | None = None
    tokens: list[UDToken] = []

    def flush() -> None:
        nonlocal sent_id, tokens
        if tokens:
            sentences.append(SentenceExample(sent_id=sent_id or f"{split}_{len(sentences)+1}", split=split, tokens=tokens))
        sent_id = None
        tokens = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("#"):
                if line.startswith("# sent_id = "):
                    sent_id = line.split("=", 1)[1].strip()
                continue
            parts = line.split("\t")
            if len(parts) != 10:
                continue
            idx_text, form, _lemma, upos, _xpos, _feats, head_text, deprel = parts[:8]
            if "-" in idx_text or "." in idx_text:
                continue
            if not idx_text.isdigit() or not head_text.isdigit():
                continue
            tokens.append(
                UDToken(
                    idx=int(idx_text),
                    form=form,
                    head=int(head_text),
                    deprel=deprel,
                    upos=upos,
                )
            )
    flush()
    return sentences


def select_sentences(
    ud_paths: dict[str, Path],
    split: str,
    min_words: int,
    max_words: int,
    limit: int,
) -> list[SentenceExample]:
    splits = ["train", "dev", "test"] if split == "all" else [split]
    selected: list[SentenceExample] = []
    for split_name in splits:
        for example in parse_conllu(ud_paths[split_name], split_name):
            if not (min_words <= example.length <= max_words):
                continue
            selected.append(example)
            if limit > 0 and len(selected) >= limit:
                return selected
    return selected


def token_frequency_key(text: str) -> str:
    return text.lower()


def build_token_frequency(sentences: list[SentenceExample], excluded_upos: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in sentences:
        for token in example.tokens:
            if token.upos in excluded_upos:
                continue
            counts[token_frequency_key(token.form)] += 1
    return counts


def encode_words(tokenizer, words: list[str]):
    encoding = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=False)
    if encoding["input_ids"].shape[1] > tokenizer.model_max_length:
        raise ValueError(
            f"Sentence requires {encoding['input_ids'].shape[1]} wordpieces, "
            f"exceeding max length {tokenizer.model_max_length}."
        )
    return encoding


def word_membership_matrix(word_ids: list[int | None], num_words: int) -> tuple[list[int], np.ndarray, np.ndarray]:
    positions = [idx for idx, word_id in enumerate(word_ids) if word_id is not None]
    membership = np.zeros((len(positions), num_words), dtype=np.float64)
    for row_idx, token_idx in enumerate(positions):
        word_idx = word_ids[token_idx]
        if word_idx is not None:
            membership[row_idx, word_idx] = 1.0
    counts = membership.sum(axis=0)
    counts[counts == 0.0] = 1.0
    return positions, membership, counts


def word_attention_by_head(outputs_attentions, word_ids: list[int | None], num_words: int) -> dict[tuple[int, int], np.ndarray]:
    positions, membership, src_counts = word_membership_matrix(word_ids, num_words)
    matrices: dict[tuple[int, int], np.ndarray] = {}
    if not positions:
        return matrices
    for layer_idx, layer_tensor in enumerate(outputs_attentions):
        layer_attention = layer_tensor[0].detach().to(torch.float32).cpu().numpy()
        active_attention = layer_attention[:, positions][:, :, positions]
        to_word = np.einsum("hts,sw->htw", active_attention, membership, optimize=True)
        by_head = np.einsum("tv,htw->hvw", membership, to_word, optimize=True)
        by_head = by_head / src_counts[np.newaxis, :, np.newaxis]
        for head_idx in range(by_head.shape[0]):
            matrices[(layer_idx, head_idx)] = by_head[head_idx]
    return matrices


def eligible_word_indices(example: SentenceExample, excluded_upos: set[str]) -> list[int]:
    return [token.idx for token in example.tokens if token.upos not in excluded_upos]


def top1_targets_and_weights(
    word_attention: np.ndarray,
    eligible_sources: list[int],
    eligible_targets: set[int],
) -> tuple[dict[int, int], dict[int, float], dict[int, int]]:
    targets: dict[int, int] = {}
    weights: dict[int, float] = {}
    offsets: dict[int, int] = {}
    num_words = word_attention.shape[0]
    for source_idx in eligible_sources:
        source_zero = source_idx - 1
        candidates = [target for target in eligible_targets if target != source_idx and 1 <= target <= num_words]
        if not candidates:
            continue
        candidate_zero = np.asarray([target - 1 for target in candidates], dtype=np.int64)
        candidate_scores = word_attention[source_zero, candidate_zero]
        best_pos = int(np.argmax(candidate_scores))
        target_idx = int(candidates[best_pos])
        targets[source_idx] = target_idx
        weights[source_idx] = float(candidate_scores[best_pos])
        offsets[source_idx] = target_idx - source_idx
    return targets, weights, offsets


def rare_targets_for_sentence(
    example: SentenceExample,
    frequency: Counter[str],
    excluded_upos: set[str],
) -> dict[int, dict[str, set[int]]]:
    candidate_tokens = [token for token in example.tokens if token.upos not in excluded_upos]
    if not candidate_tokens:
        return {}
    scored = sorted(
        ((frequency[token_frequency_key(token.form)], token.idx) for token in candidate_tokens),
        key=lambda item: (item[0], item[1]),
    )
    rare_top1 = {scored[0][1]}
    rare_top2 = {idx for _freq, idx in scored[: min(2, len(scored))]}
    result: dict[int, dict[str, set[int]]] = {}
    for token in candidate_tokens:
        top1 = set(rare_top1)
        top2 = set(rare_top2)
        top1.discard(token.idx)
        top2.discard(token.idx)
        result[token.idx] = {"top1": top1, "top2": top2}
    return result


def relation_opportunities(example: SentenceExample, excluded_upos: set[str], excluded_relations: set[str]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    by_idx = {token.idx: token for token in example.tokens}
    for token in example.tokens:
        if token.head == 0 or token.deprel in excluded_relations:
            continue
        head = by_idx.get(token.head)
        if head is None:
            continue
        if token.upos in excluded_upos or head.upos in excluded_upos:
            continue
        dep_idx = token.idx
        head_idx = token.head
        rows.append(
            {
                "relation": token.deprel,
                "direction": "dep_to_head",
                "relation_direction": f"{token.deprel}:dep_to_head",
                "source_index": dep_idx,
                "target_index": head_idx,
                "relative_offset": head_idx - dep_idx,
                "distance": abs(head_idx - dep_idx),
            }
        )
        rows.append(
            {
                "relation": token.deprel,
                "direction": "head_to_dep",
                "relation_direction": f"{token.deprel}:head_to_dep",
                "source_index": head_idx,
                "target_index": dep_idx,
                "relative_offset": dep_idx - head_idx,
                "distance": abs(head_idx - dep_idx),
            }
        )
    return rows


def empty_head_stats() -> dict[str, object]:
    return {
        "eligible_positions": 0,
        "confidence_sum": 0.0,
        "offset_counts": {},
        "rare_top1_hits": 0,
        "rare_top2_hits": 0,
        "rare_eligible": 0,
        "relation_match": {},
        "attr_gold_signed_sum": 0.0,
        "attr_gold_abs_sum": 0.0,
        "attr_top1_signed_sum": 0.0,
        "attr_top1_abs_sum": 0.0,
        "attr_count": 0,
    }


def add_count(mapping: dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


def run_attribution_for_sentence(
    *,
    model,
    tokenizer,
    device: torch.device,
    example: SentenceExample,
    encoding,
    eligible_sources: list[int],
    max_tokens: int,
    rng: random.Random,
) -> tuple[dict[tuple[int, int], dict[str, float]], int]:
    if max_tokens <= 0 or not eligible_sources:
        return {}, 0

    word_ids = encoding.word_ids(batch_index=0)
    assert word_ids is not None
    word_to_piece_positions: dict[int, list[int]] = defaultdict(list)
    for token_pos, word_id in enumerate(word_ids):
        if word_id is not None:
            word_to_piece_positions[word_id + 1].append(token_pos)

    candidate_words = [idx for idx in eligible_sources if word_to_piece_positions.get(idx)]
    rng.shuffle(candidate_words)
    candidate_words = candidate_words[:max_tokens]

    sums: dict[tuple[int, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    attribution_count = 0
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    for word_idx in candidate_words:
        piece_positions = word_to_piece_positions[word_idx]
        if not piece_positions:
            continue
        target_pos = piece_positions[0]
        input_ids = encoding["input_ids"].clone()
        attention_mask = encoding["attention_mask"].clone()
        gold_id = int(input_ids[0, target_pos].item())
        input_ids[0, target_pos] = tokenizer.mask_token_id

        captured: list[torch.Tensor] = []
        hooks = []

        def make_hook():
            def hook(_module, _inputs, output):
                context = output[0] if isinstance(output, tuple) else output
                context.retain_grad()
                captured.append(context)
            return hook

        for layer in model.bert.encoder.layer:
            hooks.append(layer.attention.self.register_forward_hook(make_hook()))

        try:
            model.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_dict=True,
            )
            gold_logit = outputs.logits[0, target_pos, gold_id]
            top1_id = int(outputs.logits[0, target_pos].argmax().item())
            top1_logit = outputs.logits[0, target_pos, top1_id]

            gold_logit.backward(retain_graph=True)
            for layer_idx, context in enumerate(captured):
                if context.grad is None:
                    continue
                attr = (context.detach() * context.grad.detach()).reshape(1, context.shape[1], num_heads, head_dim)
                signed = attr.sum(dim=(0, 1, 3)).to(torch.float32).cpu().numpy()
                abs_values = attr.abs().sum(dim=(0, 1, 3)).to(torch.float32).cpu().numpy()
                for head_idx in range(num_heads):
                    sums[(layer_idx, head_idx)]["gold_signed"] += float(signed[head_idx])
                    sums[(layer_idx, head_idx)]["gold_abs"] += float(abs_values[head_idx])

            model.zero_grad(set_to_none=True)
            for context in captured:
                context.grad = None
            top1_logit.backward()
            for layer_idx, context in enumerate(captured):
                if context.grad is None:
                    continue
                attr = (context.detach() * context.grad.detach()).reshape(1, context.shape[1], num_heads, head_dim)
                signed = attr.sum(dim=(0, 1, 3)).to(torch.float32).cpu().numpy()
                abs_values = attr.abs().sum(dim=(0, 1, 3)).to(torch.float32).cpu().numpy()
                for head_idx in range(num_heads):
                    sums[(layer_idx, head_idx)]["top1_signed"] += float(signed[head_idx])
                    sums[(layer_idx, head_idx)]["top1_abs"] += float(abs_values[head_idx])
            attribution_count += 1
        finally:
            for hook in hooks:
                hook.remove()
            model.zero_grad(set_to_none=True)
            for context in captured:
                context.grad = None
            captured.clear()
            try:
                del outputs
            except UnboundLocalError:
                pass
            try:
                del gold_logit
            except UnboundLocalError:
                pass
            try:
                del top1_logit
            except UnboundLocalError:
                pass
            try:
                del input_ids
            except UnboundLocalError:
                pass
            try:
                del attention_mask
            except UnboundLocalError:
                pass
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    return sums, attribution_count


def build_relation_baselines(relation_baseline_counts: dict[str, dict[int, int]]) -> dict[str, dict[str, float | int | str]]:
    baselines: dict[str, dict[str, float | int | str]] = {}
    for relation_direction, offset_counts in relation_baseline_counts.items():
        total = int(sum(offset_counts.values()))
        if total == 0:
            continue
        baseline_offset, baseline_count = max(offset_counts.items(), key=lambda item: item[1])
        relation, direction = relation_direction.rsplit(":", 1)
        baselines[relation_direction] = {
            "relation": relation,
            "direction": direction,
            "opportunity_count": total,
            "baseline_offset": int(baseline_offset),
            "baseline_accuracy": float(baseline_count / total),
        }
    return baselines


def finalize_outputs(
    *,
    args: argparse.Namespace,
    head_stats: dict[tuple[int, int], dict[str, object]],
    relation_baseline_counts: dict[str, dict[int, int]],
    processed_sentences: int,
    selected_sentences: int,
    final: bool,
) -> None:
    baselines = build_relation_baselines(relation_baseline_counts)
    head_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []

    for (layer_idx, head_idx), stats in sorted(head_stats.items()):
        eligible = int(stats["eligible_positions"])
        confidence = float(stats["confidence_sum"] / eligible) if eligible else float("nan")
        offset_counts = {int(k): int(v) for k, v in dict(stats["offset_counts"]).items()}
        if offset_counts:
            best_offset, best_offset_count = max(offset_counts.items(), key=lambda item: item[1])
            positional_share = best_offset_count / eligible if eligible else float("nan")
        else:
            best_offset = 0
            positional_share = float("nan")
        is_positional = bool(eligible and positional_share >= args.positional_threshold)

        rare_eligible = int(stats["rare_eligible"])
        rare_top1_rate = int(stats["rare_top1_hits"]) / rare_eligible if rare_eligible else float("nan")
        rare_top2_rate = int(stats["rare_top2_hits"]) / rare_eligible if rare_eligible else float("nan")
        is_rare = bool(rare_eligible and rare_top2_rate >= args.rare_threshold)

        relation_match = dict(stats["relation_match"])
        syntactic_relations: list[str] = []
        best_relation_direction = ""
        best_relation = ""
        best_direction = ""
        best_accuracy = float("nan")
        best_baseline = float("nan")
        best_margin = float("nan")

        for relation_direction, baseline in sorted(baselines.items()):
            opportunity_count = int(baseline["opportunity_count"])
            match_count = int(relation_match.get(relation_direction, 0))
            accuracy = match_count / opportunity_count if opportunity_count else float("nan")
            baseline_accuracy = float(baseline["baseline_accuracy"])
            margin = accuracy - baseline_accuracy
            is_syntactic = bool(
                opportunity_count >= args.relation_min_count and margin >= args.syntactic_margin_threshold
            )
            if is_syntactic:
                syntactic_relations.append(relation_direction)
            relation_rows.append(
                {
                    "language": args.language,
                    "treebank": args.treebank,
                    "split": args.split,
                    "layer": layer_idx,
                    "head_index": head_idx,
                    "relation": baseline["relation"],
                    "direction": baseline["direction"],
                    "relation_direction": relation_direction,
                    "opportunity_count": opportunity_count,
                    "match_count": match_count,
                    "accuracy": accuracy,
                    "baseline_offset": int(baseline["baseline_offset"]),
                    "baseline_accuracy": baseline_accuracy,
                    "accuracy_margin": margin,
                    "is_syntactic_relation": is_syntactic,
                }
            )
            if math.isnan(best_accuracy) or accuracy > best_accuracy:
                best_relation_direction = relation_direction
                best_relation = str(baseline["relation"])
                best_direction = str(baseline["direction"])
                best_accuracy = accuracy
                best_baseline = baseline_accuracy
                best_margin = margin

        attr_count = int(stats["attr_count"])
        head_rows.append(
            {
                "language": args.language,
                "treebank": args.treebank,
                "split": args.split,
                "layer": layer_idx,
                "head_index": head_idx,
                "confidence_mean": confidence,
                "eligible_positions": eligible,
                "most_common_offset": best_offset,
                "positional_share": positional_share,
                "is_positional": is_positional,
                "rare_top1_rate": rare_top1_rate,
                "rare_top2_rate": rare_top2_rate,
                "rare_eligible": rare_eligible,
                "is_rare_word_head": is_rare,
                "best_relation": best_relation,
                "best_direction": best_direction,
                "best_relation_direction": best_relation_direction,
                "best_relation_accuracy": best_accuracy,
                "best_relation_baseline": best_baseline,
                "best_relation_margin": best_margin,
                "syntactic_relations": ",".join(syntactic_relations),
                "is_syntactic_head": bool(syntactic_relations),
                "syntactic_relation_count": len(syntactic_relations),
                "mlm_attr_gold_signed_mean": float(stats["attr_gold_signed_sum"] / attr_count) if attr_count else float("nan"),
                "mlm_attr_gold_abs_mean": float(stats["attr_gold_abs_sum"] / attr_count) if attr_count else float("nan"),
                "mlm_attr_top1_signed_mean": float(stats["attr_top1_signed_sum"] / attr_count) if attr_count else float("nan"),
                "mlm_attr_top1_abs_mean": float(stats["attr_top1_abs_sum"] / attr_count) if attr_count else float("nan"),
                "mlm_attr_count": attr_count,
            }
        )

    head_df = pd.DataFrame(head_rows)
    if not head_df.empty:
        for col in ("mlm_attr_gold_abs_mean", "mlm_attr_top1_abs_mean"):
            rank_col = col.replace("_mean", "_rank")
            head_df[rank_col] = head_df[col].rank(method="dense", ascending=False).astype("Int64")
        head_df = head_df.sort_values(["layer", "head_index"]).reset_index(drop=True)

    relation_df = pd.DataFrame(relation_rows)
    relation_baseline_df = pd.DataFrame(
        [{"language": args.language, "treebank": args.treebank, "split": args.split, "relation_direction": key, **value} for key, value in sorted(baselines.items())]
    )
    summary = {
        "language": args.language,
        "treebank": args.treebank,
        "split": args.split,
        "run_name": args.run_name,
        "processed_sentences": processed_sentences,
        "selected_sentences": selected_sentences,
        "num_heads": int(len(head_df)),
        "num_positional_heads": int(head_df["is_positional"].sum()) if not head_df.empty else 0,
        "num_syntactic_heads": int(head_df["is_syntactic_head"].sum()) if not head_df.empty else 0,
        "num_rare_word_heads": int(head_df["is_rare_word_head"].sum()) if not head_df.empty else 0,
        "attribution_enabled": bool(args.attribution_sentences > 0),
        "attribution_sentences": args.attribution_sentences,
        "attribution_tokens_per_sentence": args.attribution_tokens_per_sentence,
        "final": final,
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    head_path = args.results_dir / f"{args.run_name}_head_characterization.csv"
    relation_path = args.results_dir / f"{args.run_name}_relation_characterization.csv"
    baseline_path = args.results_dir / f"{args.run_name}_relation_baselines.csv"
    summary_path = args.results_dir / f"{args.run_name}_language_summary.json"
    head_df.to_csv(head_path, index=False)
    relation_df.to_csv(relation_path, index=False)
    relation_baseline_df.to_csv(baseline_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if final:
        log(f"Wrote head characterization to {head_path}")
        log(f"Wrote relation characterization to {relation_path}")
        log(f"Wrote relation baselines to {baseline_path}")
        log(f"Wrote language summary to {summary_path}")
    else:
        log(f"Checkpointed {processed_sentences}/{selected_sentences} sentences to {args.results_dir}")


def main() -> None:
    args = parse_args()
    excluded_upos = parse_text_set(args.exclude_upos)
    excluded_relations = parse_text_set(args.exclude_relations)
    ud_paths = ensure_ud_files(resolve_ud_sources(args), args.cache_dir / args.treebank.replace("/", "_"))

    device = choose_device(args.device)
    log(f"Selected device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, attn_implementation="eager")
    model.to(device)
    model.eval()

    selected = select_sentences(ud_paths, args.split, args.min_words, args.max_words, args.limit)
    if not selected:
        raise RuntimeError("No sentences selected after filtering.")
    train_examples = parse_conllu(ud_paths["train"], "train")
    frequency = build_token_frequency(train_examples, excluded_upos)
    rng = random.Random(args.attribution_seed)

    log(
        f"Processing {len(selected)} {args.language}/{args.treebank} sentences from split={args.split}; "
        f"attribution_sentences={args.attribution_sentences}"
    )

    head_stats: dict[tuple[int, int], dict[str, object]] = {}
    relation_baseline_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    resolved_layers: list[int] | None = None
    resolved_heads: list[int] | None = None

    for sent_idx, example in enumerate(selected, start=1):
        try:
            encoding = encode_words(tokenizer, example.words)
        except ValueError as exc:
            log(f"Skipping {example.sent_id}: {exc}")
            continue

        word_ids = encoding.word_ids(batch_index=0)
        assert word_ids is not None
        with torch.no_grad():
            outputs = model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device),
                output_attentions=True,
                return_dict=True,
            )

        if resolved_layers is None or resolved_heads is None:
            resolved_layers = resolve_selection(args.layers, len(outputs.attentions), "layer")
            resolved_heads = resolve_selection(args.heads, outputs.attentions[0].shape[1], "head")
            for layer_idx in resolved_layers:
                for head_idx in resolved_heads:
                    head_stats.setdefault((layer_idx, head_idx), empty_head_stats())
            log(f"Resolved layers={resolved_layers}; heads={resolved_heads}")

        eligible_sources = eligible_word_indices(example, excluded_upos)
        eligible_targets = set(eligible_sources)
        opportunities = relation_opportunities(example, excluded_upos, excluded_relations)
        rare_targets = rare_targets_for_sentence(example, frequency, excluded_upos)
        for opportunity in opportunities:
            relation_direction = str(opportunity["relation_direction"])
            offset = int(opportunity["relative_offset"])
            relation_baseline_counts[relation_direction][offset] += 1

        matrices = word_attention_by_head(outputs.attentions, word_ids, example.length)
        for layer_idx in resolved_layers:
            for head_idx in resolved_heads:
                word_attention = matrices[(layer_idx, head_idx)]
                top_targets, top_weights, offsets = top1_targets_and_weights(
                    word_attention=word_attention,
                    eligible_sources=eligible_sources,
                    eligible_targets=eligible_targets,
                )
                stats = head_stats[(layer_idx, head_idx)]
                stats["eligible_positions"] = int(stats["eligible_positions"]) + len(top_targets)
                stats["confidence_sum"] = float(stats["confidence_sum"]) + float(sum(top_weights.values()))

                offset_counts = dict(stats["offset_counts"])
                for offset in offsets.values():
                    offset_counts[offset] = offset_counts.get(offset, 0) + 1
                stats["offset_counts"] = offset_counts

                rare_eligible = int(stats["rare_eligible"])
                rare_top1_hits = int(stats["rare_top1_hits"])
                rare_top2_hits = int(stats["rare_top2_hits"])
                for source_idx, candidates in rare_targets.items():
                    if source_idx not in top_targets:
                        continue
                    if not candidates["top2"]:
                        continue
                    rare_eligible += 1
                    target = top_targets[source_idx]
                    if target in candidates["top1"]:
                        rare_top1_hits += 1
                    if target in candidates["top2"]:
                        rare_top2_hits += 1
                stats["rare_eligible"] = rare_eligible
                stats["rare_top1_hits"] = rare_top1_hits
                stats["rare_top2_hits"] = rare_top2_hits

                relation_match = dict(stats["relation_match"])
                for opportunity in opportunities:
                    source = int(opportunity["source_index"])
                    target = int(opportunity["target_index"])
                    if top_targets.get(source) == target:
                        add_count(relation_match, str(opportunity["relation_direction"]))
                stats["relation_match"] = relation_match

        if args.attribution_sentences > 0 and sent_idx <= args.attribution_sentences:
            attribution_sums, attr_count = run_attribution_for_sentence(
                model=model,
                tokenizer=tokenizer,
                device=device,
                example=example,
                encoding=encoding,
                eligible_sources=eligible_sources,
                max_tokens=args.attribution_tokens_per_sentence,
                rng=rng,
            )
            for (layer_idx, head_idx), values in attribution_sums.items():
                if (layer_idx, head_idx) not in head_stats:
                    continue
                stats = head_stats[(layer_idx, head_idx)]
                stats["attr_gold_signed_sum"] = float(stats["attr_gold_signed_sum"]) + float(values.get("gold_signed", 0.0))
                stats["attr_gold_abs_sum"] = float(stats["attr_gold_abs_sum"]) + float(values.get("gold_abs", 0.0))
                stats["attr_top1_signed_sum"] = float(stats["attr_top1_signed_sum"]) + float(values.get("top1_signed", 0.0))
                stats["attr_top1_abs_sum"] = float(stats["attr_top1_abs_sum"]) + float(values.get("top1_abs", 0.0))
                stats["attr_count"] = int(stats["attr_count"]) + attr_count

        if sent_idx % 25 == 0 or sent_idx == len(selected):
            log(f"Processed {sent_idx}/{len(selected)} sentences")
        if args.checkpoint_every > 0 and sent_idx % args.checkpoint_every == 0:
            finalize_outputs(
                args=args,
                head_stats=head_stats,
                relation_baseline_counts=relation_baseline_counts,
                processed_sentences=sent_idx,
                selected_sentences=len(selected),
                final=False,
            )
        del outputs, matrices
        if sent_idx % 10 == 0:
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    finalize_outputs(
        args=args,
        head_stats=head_stats,
        relation_baseline_counts=relation_baseline_counts,
        processed_sentences=len(selected),
        selected_sentences=len(selected),
        final=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Utilities for GPT-2 + UD structural analyses.

This module keeps the English UD loading, token/word alignment, attention
aggregation, graph construction, and a few evaluation helpers in one place so
the GPT-2 track can stay separate from the earlier mBERT experiments.
"""

from __future__ import annotations

import os
import json
import subprocess
import time
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np
import torch


DEFAULT_UD_VERSION = "r2.17"
DEFAULT_UD_BASE_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/"
    f"{DEFAULT_UD_VERSION}"
)


def load_ud_files() -> dict[str, str]:
    """
    Resolve the UD split URLs.

    The default remains English EWT, but teammates can override this without
    code changes by exporting CGS410_UD_FILES_JSON or split-specific URLs.
    """

    raw_mapping = os.environ.get("CGS410_UD_FILES_JSON")
    if raw_mapping:
        mapping = json.loads(raw_mapping)
        if not isinstance(mapping, dict):
            raise ValueError("CGS410_UD_FILES_JSON must decode to a JSON object.")
        required = {"train", "dev", "test"}
        missing = sorted(required - set(mapping))
        if missing:
            raise ValueError(f"CGS410_UD_FILES_JSON is missing splits: {', '.join(missing)}")
        return {split: str(mapping[split]) for split in ("train", "dev", "test")}

    base_url = os.environ.get("CGS410_UD_BASE_URL", DEFAULT_UD_BASE_URL).rstrip("/")
    return {
        "train": os.environ.get("CGS410_UD_TRAIN_URL", f"{base_url}/en_ewt-ud-train.conllu"),
        "dev": os.environ.get("CGS410_UD_DEV_URL", f"{base_url}/en_ewt-ud-dev.conllu"),
        "test": os.environ.get("CGS410_UD_TEST_URL", f"{base_url}/en_ewt-ud-test.conllu"),
    }


UD_FILES = load_ud_files()


@dataclass
class UDToken:
    idx: int
    form: str
    head: int
    deprel: str


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


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError(f"No integers found in '{text}'")
    return values


def unique_preserve_order(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def resolve_index(idx: int, size: int, label: str) -> int:
    resolved = idx if idx >= 0 else size + idx
    if resolved < 0 or resolved >= size:
        raise IndexError(f"Invalid {label} index {idx} for size {size}.")
    return resolved


def resolve_index_selection(text: str, size: int, label: str) -> list[int]:
    cleaned = text.strip().lower()
    if cleaned == "all":
        return list(range(size))
    return unique_preserve_order(resolve_index(value, size, label) for value in parse_int_list(text))


def ensure_ud_files(cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths: dict[str, Path] = {}
    for split, url in UD_FILES.items():
        local_path = cache_dir / Path(url).name
        local_paths[split] = local_path
        if local_path.exists():
            continue
        log(f"Downloading {split} split from {url}")
        urlretrieve(url, local_path)
    return local_paths


def parse_conllu(path: Path, split: str) -> list[SentenceExample]:
    sentences: list[SentenceExample] = []
    sent_id: str | None = None
    tokens: list[UDToken] = []

    def flush() -> None:
        nonlocal sent_id, tokens
        if tokens:
            label = sent_id or f"{split}_sent_{len(sentences) + 1}"
            sentences.append(SentenceExample(sent_id=label, split=split, tokens=tokens))
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

            idx = parts[0]
            if "-" in idx or "." in idx:
                continue

            head = parts[6]
            if not idx.isdigit() or not head.isdigit():
                continue

            tokens.append(
                UDToken(
                    idx=int(idx),
                    form=parts[1],
                    head=int(head),
                    deprel=parts[7],
                )
            )

    flush()
    return sentences


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_other_gpu_processes() -> list[str]:
    current_pid = os.getpid()
    result = subprocess.run(
        ["ps", "-Ao", "pid,command"],
        capture_output=True,
        text=True,
        check=True,
    )
    suspects: list[str] = []
    for line in result.stdout.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid == current_pid:
            continue
        command_lower = command.lower()
        if any(
            marker in command_lower
            for marker in ("--device mps", "device=mps", " device mps", " mlx", "ollama")
        ):
            suspects.append(line)
    return suspects


def wait_for_gpu_if_needed(interval_seconds: int) -> None:
    while True:
        suspects = find_other_gpu_processes()
        if not suspects:
            return
        log("MPS appears busy. Waiting before starting model inference.")
        for suspect in suspects:
            log(f"  busy: {suspect}")
        time.sleep(interval_seconds)


def filter_sentences(
    sentences: Iterable[SentenceExample],
    min_words: int,
    max_words: int,
    limit: int,
) -> list[SentenceExample]:
    selected: list[SentenceExample] = []
    for example in sentences:
        if not (min_words <= example.length <= max_words):
            continue
        selected.append(example)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def select_sentences(
    split: str,
    ud_paths: dict[str, Path],
    min_words: int,
    max_words: int,
    limit: int,
) -> list[SentenceExample]:
    split_order = ["train", "dev", "test"] if split == "all" else [split]
    remaining = limit
    selected: list[SentenceExample] = []

    for split_name in split_order:
        examples = parse_conllu(ud_paths[split_name], split_name)
        split_limit = remaining if remaining > 0 else 0
        filtered = filter_sentences(examples, min_words, max_words, split_limit)
        selected.extend(filtered)
        if remaining > 0:
            remaining -= len(filtered)
            if remaining <= 0:
                break
    return selected


def encode_words(tokenizer, words: list[str]):
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=False,
    )
    if encoding["input_ids"].shape[1] > tokenizer.model_max_length:
        raise ValueError(
            f"Sentence requires {encoding['input_ids'].shape[1]} tokens, "
            f"which exceeds model_max_length={tokenizer.model_max_length}."
        )
    return encoding


def wordpiece_positions(word_ids: list[int | None]) -> list[int]:
    return [idx for idx, word_id in enumerate(word_ids) if word_id is not None]


def word_membership_matrix(
    word_ids: list[int | None],
    num_words: int,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    positions = wordpiece_positions(word_ids)
    membership = np.zeros((len(positions), num_words), dtype=np.float64)
    for row_idx, token_idx in enumerate(positions):
        word_idx = word_ids[token_idx]
        if word_idx is None:
            continue
        membership[row_idx, word_idx] = 1.0
    counts = membership.sum(axis=0)
    counts[counts == 0.0] = 1.0
    return positions, membership, counts


def resolve_layer_idx(layer: int, num_layers: int) -> int:
    layer_idx = layer if layer >= 0 else num_layers + layer
    if layer_idx < 0 or layer_idx >= num_layers:
        raise IndexError(f"Invalid layer index {layer} for {num_layers} layers.")
    return layer_idx


def aggregate_word_attention_views(
    layer_attention: np.ndarray,
    word_ids: list[int | None],
    num_words: int,
    head_mode: str,
    selected_heads: list[int],
) -> list[tuple[str, int, np.ndarray]]:
    positions, membership, src_counts = word_membership_matrix(word_ids, num_words)
    if not positions:
        return []

    active_attention = layer_attention[:, positions][:, :, positions]
    to_word = np.einsum("hts,sw->htw", active_attention, membership, optimize=True)
    by_head = np.einsum("tv,htw->hvw", membership, to_word, optimize=True)
    by_head = by_head / src_counts[np.newaxis, :, np.newaxis]

    views: list[tuple[str, int, np.ndarray]] = []
    if head_mode in ("mean", "both"):
        views.append(("mean", -1, by_head.mean(axis=0)))
    if head_mode in ("all", "both"):
        for head_idx in selected_heads:
            views.append((f"head{head_idx}", head_idx, by_head[head_idx]))
    return views


def top1_targets_and_weights(
    word_attention: np.ndarray,
) -> tuple[dict[int, int], dict[int, float], dict[int, int]]:
    primary_targets: dict[int, int] = {}
    top_weights: dict[int, float] = {}
    relative_offsets: dict[int, int] = {}

    num_words = word_attention.shape[0]
    for dep_idx in range(num_words):
        if dep_idx == 0:
            continue
        row = word_attention[dep_idx, :dep_idx]
        if row.size == 0:
            continue
        target_idx = int(np.argmax(row))
        primary_targets[dep_idx + 1] = target_idx + 1
        top_weights[dep_idx + 1] = float(row[target_idx])
        relative_offsets[dep_idx + 1] = target_idx - dep_idx
    return primary_targets, top_weights, relative_offsets


def select_topk_targets(word_attention: np.ndarray, top_k: int) -> list[list[int]]:
    num_words = word_attention.shape[0]
    targets_by_word: list[list[int]] = []
    for dep_idx in range(num_words):
        valid_k = min(top_k, dep_idx)
        if valid_k == 0:
            targets_by_word.append([])
            continue
        row = word_attention[dep_idx, :dep_idx]
        target_indices = np.argsort(row)[-valid_k:][::-1]
        ordered_targets = [int(head_idx) for head_idx in target_indices if np.isfinite(row[head_idx])]
        targets_by_word.append(ordered_targets)
    return targets_by_word


def build_attention_edges_topk(
    word_attention: np.ndarray,
    top_k: int,
) -> list[tuple[int, int, float]]:
    targets_by_word = select_topk_targets(word_attention, top_k)
    edges: list[tuple[int, int, float]] = []
    for dep_idx, target_indices in enumerate(targets_by_word):
        for head_idx in target_indices:
            edges.append((dep_idx + 1, head_idx + 1, float(word_attention[dep_idx, head_idx])))
    return edges


def primary_targets_from_topk(word_attention: np.ndarray, top_k: int) -> dict[int, int]:
    targets_by_word = select_topk_targets(word_attention, top_k)
    primary: dict[int, int] = {}
    for dep_idx, targets in enumerate(targets_by_word, start=1):
        if targets:
            primary[dep_idx] = targets[0] + 1
    return primary


def ud_edges(example: SentenceExample) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    for token in example.tokens:
        if token.head == 0:
            continue
        edges.append((token.idx, token.head, 1.0))
    return edges


def edge_set(edges: list[tuple[int, int, float]]) -> set[tuple[int, int]]:
    return {(dep_idx, head_idx) for dep_idx, head_idx, _ in edges}


def overlap_counts(
    gold_edges: list[tuple[int, int, float]],
    predicted_edges: list[tuple[int, int, float]],
) -> tuple[int, int, int]:
    gold_set = edge_set(gold_edges)
    predicted_set = edge_set(predicted_edges)
    matched = len(gold_set & predicted_set)
    return matched, len(gold_set), len(predicted_set)


def precision_recall_f1(matched: int, gold_total: int, predicted_total: int) -> tuple[float, float, float]:
    precision = matched / predicted_total if predicted_total else 0.0
    recall = matched / gold_total if gold_total else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def edge_distance(edges: list[tuple[int, int, float]]) -> float:
    if not edges:
        return 0.0
    return float(np.mean([abs(dep_idx - head_idx) for dep_idx, head_idx, _ in edges]))


def relation_match_summary(
    example: SentenceExample,
    primary_targets: dict[int, int],
    selected_edges: set[tuple[int, int]],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for token in example.tokens:
        if token.head == 0:
            continue
        predicted_head = primary_targets.get(token.idx)
        rows.append(
            {
                "relation": token.deprel,
                "ud_edge_count": 1,
                "match_count": int((token.idx, token.head) in selected_edges),
                "ud_distance": abs(token.idx - token.head),
                "predicted_distance": abs(token.idx - predicted_head) if predicted_head is not None else float("nan"),
            }
        )
    return rows


def token_frequency_key(token_text: str) -> str:
    return token_text.lower()


def build_token_frequency(sentences: Iterable[SentenceExample]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for example in sentences:
        for token in example.tokens:
            counts[token_frequency_key(token.form)] += 1
    return counts


def rare_previous_targets(
    example: SentenceExample,
    frequency: Counter[str],
) -> dict[int, set[int]]:
    rare_targets: dict[int, set[int]] = {}
    if example.length <= 1:
        return rare_targets

    words = example.words
    for idx in range(2, example.length + 1):
        previous = words[: idx - 1]
        if not previous:
            continue
        scored = [frequency[token_frequency_key(word)] for word in previous]
        minimum = min(scored)
        rare_targets[idx] = {pos + 1 for pos, score in enumerate(scored) if score == minimum}
    return rare_targets


def build_accessible_relation_rows(example: SentenceExample) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for token in example.tokens:
        if token.head == 0:
            continue
        later = max(token.idx, token.head)
        earlier = min(token.idx, token.head)
        rows.append(
            {
                "relation": token.deprel,
                "source_index": later,
                "target_index": earlier,
                "relative_offset": earlier - later,
                "distance": later - earlier,
            }
        )
    return rows


def build_relation_lookup(example: SentenceExample) -> dict[int, dict[str, set[int]]]:
    lookup: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in build_accessible_relation_rows(example):
        lookup[int(row["source_index"])][str(row["relation"])].add(int(row["target_index"]))
    return {source: dict(relations) for source, relations in lookup.items()}


def adjacency_from_edges(edges: list[tuple[int, int, float]], num_words: int) -> list[set[int]]:
    adjacency = [set() for _ in range(num_words)]
    for dep_idx, head_idx, _ in edges:
        dep = dep_idx - 1
        head = head_idx - 1
        if dep == head:
            continue
        adjacency[dep].add(head)
        adjacency[head].add(dep)
    return adjacency


def shortest_path_matrix(edges: list[tuple[int, int, float]], num_words: int) -> np.ndarray:
    adjacency = adjacency_from_edges(edges, num_words)
    unreachable = float(num_words + 1)
    matrix = np.full((num_words, num_words), unreachable, dtype=np.float64)
    for start in range(num_words):
        matrix[start, start] = 0.0
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            node, dist = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                matrix[start, neighbor] = float(dist + 1)
                queue.append((neighbor, dist + 1))
    return matrix


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.array([], dtype=np.float64)
    indices = np.triu_indices(matrix.shape[0], k=1)
    return matrix[indices]


def sentence_next_token_nll(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    head_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    if input_ids.shape[1] < 2:
        return 0.0, 0

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            head_mask=head_mask,
            return_dict=True,
        )
        logits = outputs.logits[0].float()

    target_ids = input_ids[0, 1:].to(device)
    predictor_logits = logits[:-1]
    log_probs = torch.log_softmax(predictor_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    nll = -float(token_log_probs.sum().item())
    return nll, int(target_ids.numel())


def next_token_metrics_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> tuple[float, int, int]:
    if input_ids.shape[1] < 2:
        return 0.0, 0, 0

    predictor_logits = logits[:-1].float()
    target_ids = input_ids[0, 1:].to(predictor_logits.device)
    log_probs = torch.log_softmax(predictor_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    nll = -float(token_log_probs.sum().item())
    predictions = predictor_logits.argmax(dim=-1)
    correct = int((predictions == target_ids).sum().item())
    return nll, correct, int(target_ids.numel())


def sentence_next_token_accuracy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    head_mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    if input_ids.shape[1] < 2:
        return 0, 0

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            head_mask=head_mask,
            return_dict=True,
        )
        logits = outputs.logits[0]

    target_ids = input_ids[0, 1:].to(device)
    predictor_logits = logits[:-1]
    predictions = predictor_logits.argmax(dim=-1)
    correct = int((predictions == target_ids).sum().item())
    return correct, int(target_ids.numel())


def make_head_mask(
    num_layers: int,
    num_heads: int,
    active_heads: Iterable[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((num_layers, num_heads), dtype=torch.float32, device=device)
    for layer_idx, head_idx in active_heads:
        mask[layer_idx, head_idx] = 1.0
    return mask

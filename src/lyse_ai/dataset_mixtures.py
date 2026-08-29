from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from datasets import load_dataset


SPEAKER_RE = re.compile(r"^\[(.+?)\:]\s*(.*)$")
VALID_ROLES = {"system", "user", "assistant"}


def normalize_text(text: str) -> str:
    lines = []
    for raw_line in (text or "").replace("\u00a0", " ").splitlines():
        line = " ".join(raw_line.strip().split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def normalize_messages(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = normalize_text(msg.get("content", ""))
        if role not in VALID_ROLES or not content:
            continue
        if normalized and normalized[-1]["role"] == role:
            normalized[-1]["content"] += "\n" + content
        else:
            normalized.append({"role": role, "content": content})
    return normalized


def is_valid_sft_conversation(messages: Sequence[Dict[str, str]]) -> bool:
    if len(messages) < 2:
        return False
    if not any(msg.get("role") == "assistant" for msg in messages):
        return False
    if messages[0].get("role") not in {"system", "user"}:
        return False
    return True


def _iter_local_jsonl_rows(pattern: str) -> Iterator[Tuple[str, Dict]]:
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield path, json.loads(line)


def _iter_local_txt_lines(pattern: str) -> Iterator[Tuple[str, str]]:
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield path, line


def _extract_first_text_field(row: Dict, preferred_key: str = "text") -> str:
    if preferred_key in row and isinstance(row[preferred_key], str):
        return row[preferred_key]
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _yield_limited_texts(items: Iterable[str], max_docs: int) -> Iterator[str]:
    count = 0
    for text in items:
        if max_docs and count >= max_docs:
            break
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        yield cleaned
        count += 1


def iter_text_source(source: Dict) -> Iterator[str]:
    source_type = source.get("type", "").strip().lower()
    max_docs = int(source.get("max_docs", 0) or 0)

    if source_type == "fineweb_edu":
        dataset = source.get("dataset", "HuggingFaceFW/fineweb-edu")
        split = source.get("split", "train")
        allowed_languages = {str(x) for x in source.get("allowed_languages", []) if str(x).strip()}
        ds = load_dataset(dataset, split=split, streaming=True)
        emitted = 0
        for row in ds:
            language = str(row.get("language", "")).strip().lower()
            if allowed_languages and language not in allowed_languages:
                continue
            text = normalize_text(row.get("text", ""))
            if not text:
                continue
            yield text
            emitted += 1
            if max_docs and emitted >= max_docs:
                break
        return

    if source_type == "culturax":
        dataset = source.get("dataset", "uonlp/CulturaX")
        language = source.get("language", "fr")
        split = source.get("split", "train")
        ds = load_dataset(dataset, language, split=split, streaming=True)
        emitted = 0
        for row in ds:
            text = normalize_text(row.get("text", ""))
            if not text:
                continue
            yield text
            emitted += 1
            if max_docs and emitted >= max_docs:
                break
        return

    if source_type == "hf_text":
        dataset = source["dataset"]
        dataset_config = source.get("dataset_config") or None
        split = source.get("split", "train")
        text_key = source.get("text_key", "text")
        ds = load_dataset(dataset, dataset_config, split=split, streaming=True)
        yield from _yield_limited_texts((row.get(text_key, "") for row in ds), max_docs)
        return

    if source_type == "local_jsonl_text":
        text_key = source.get("text_key", "text")
        rows = (row.get(text_key, "") for _, row in _iter_local_jsonl_rows(source["glob"]))
        yield from _yield_limited_texts(rows, max_docs)
        return

    if source_type == "local_txt":
        lines = (line for _, line in _iter_local_txt_lines(source["glob"]))
        yield from _yield_limited_texts(lines, max_docs)
        return

    raise ValueError(f"Type de source texte non supporté: {source_type}")


def iter_named_text_sources(sources: Sequence[Dict]) -> Iterator[Tuple[str, str]]:
    for idx, source in enumerate(sources):
        name = source.get("name") or source.get("type") or f"source_{idx}"
        try:
            for text in iter_text_source(source):
                yield str(name), text
        except Exception as exc:
            print(f"[WARN] Source texte ignorée: {name} ({exc})")


def _row_to_messages(row: Dict) -> List[Dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return normalize_messages(row["messages"])
    prompt = normalize_text(row.get("prompt", ""))
    response = normalize_text(row.get("response", ""))
    if prompt and response:
        return normalize_messages(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
        )
    return []


def iter_ultrachat_messages(source: Dict) -> Iterator[List[Dict[str, str]]]:
    dataset = source.get("dataset", "HuggingFaceH4/ultrachat_200k")
    split = source.get("split", "train_sft")
    max_conversations = int(source.get("max_conversations", 0) or 0)
    ds = load_dataset(dataset, split=split, streaming=True)
    emitted = 0
    for row in ds:
        messages = normalize_messages(row.get("messages", []))
        if not is_valid_sft_conversation(messages):
            continue
        yield messages
        emitted += 1
        if max_conversations and emitted >= max_conversations:
            break


def _choose_children(child_ids: Sequence[str], rows_by_id: Dict[str, Dict], max_children: int) -> List[str]:
    def sort_key(child_id: str):
        row = rows_by_id[child_id]
        rank = row.get("rank")
        rank_missing = rank is None
        rank_value = int(rank) if isinstance(rank, int) else (int(rank) if isinstance(rank, str) and rank.isdigit() else 999999)
        return (rank_missing, rank_value, str(row.get("created_date", "")), child_id)

    return sorted(child_ids, key=sort_key)[:max_children]


def iter_oasst1_messages(source: Dict) -> Iterator[List[Dict[str, str]]]:
    dataset = source.get("dataset", "OpenAssistant/oasst1")
    split = source.get("split", "train")
    max_conversations = int(source.get("max_conversations", 0) or 0)
    allowed_languages = {str(x).strip().lower() for x in source.get("allowed_languages", []) if str(x).strip()}
    max_children = int(source.get("max_children", 2) or 2)
    ds = load_dataset(dataset, split=split)

    rows_by_id: Dict[str, Dict] = {}
    children: Dict[str | None, List[str]] = defaultdict(list)
    for row in ds:
        if row.get("deleted"):
            continue
        message_id = row.get("message_id")
        if not message_id:
            continue
        rows_by_id[message_id] = dict(row)
        children[row.get("parent_id")].append(message_id)

    def role_of(row: Dict) -> str:
        raw_role = str(row.get("role", "")).strip().lower()
        if raw_role == "assistant":
            return "assistant"
        if raw_role == "prompter":
            return "user"
        return ""

    roots = list(children.get(None, []))
    emitted = 0
    stack: List[Tuple[str, List[Dict[str, str]]]] = [(root_id, []) for root_id in reversed(roots)]

    while stack:
        message_id, path = stack.pop()
        row = rows_by_id.get(message_id)
        if row is None:
            continue
        lang = str(row.get("lang", "")).strip().lower()
        if allowed_languages and lang and lang not in allowed_languages:
            continue
        role = role_of(row)
        content = normalize_text(row.get("text", ""))
        if not role or not content:
            continue

        new_path = normalize_messages(path + [{"role": role, "content": content}])
        child_ids = [cid for cid in children.get(message_id, []) if cid in rows_by_id]
        if allowed_languages:
            child_ids = [cid for cid in child_ids if not rows_by_id[cid].get("lang") or str(rows_by_id[cid].get("lang", "")).strip().lower() in allowed_languages]
        child_ids = _choose_children(child_ids, rows_by_id, max_children=max_children)

        if not child_ids:
            if is_valid_sft_conversation(new_path):
                yield new_path
                emitted += 1
                if max_conversations and emitted >= max_conversations:
                    return
            continue

        for child_id in reversed(child_ids):
            stack.append((child_id, new_path))


def parse_claire_block(block: str) -> List[Dict[str, str]]:
    speaker_map: Dict[str, str] = {}
    messages: List[Dict[str, str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_RE.match(line)
        if not match:
            continue
        speaker = match.group(1).strip()
        content = normalize_text(match.group(2))
        if not content:
            continue
        if speaker not in speaker_map:
            if not speaker_map:
                speaker_map[speaker] = "user"
            elif len(speaker_map) == 1:
                speaker_map[speaker] = "assistant"
            else:
                continue
        role = speaker_map.get(speaker, "")
        if not role:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})
    return normalize_messages(messages)


def _iter_claire_blocks_from_text(raw_text: str) -> Iterator[List[Dict[str, str]]]:
    current_lines: List[str] = []
    for raw_line in (raw_text or "").splitlines():
        line = raw_line.rstrip()
        if line.strip():
            current_lines.append(line)
            continue
        if current_lines:
            messages = parse_claire_block("\n".join(current_lines))
            if is_valid_sft_conversation(messages):
                yield messages
            current_lines = []
    if current_lines:
        messages = parse_claire_block("\n".join(current_lines))
        if is_valid_sft_conversation(messages):
            yield messages


def iter_claire_txt_messages(source: Dict) -> Iterator[List[Dict[str, str]]]:
    max_conversations = int(source.get("max_conversations", 0) or 0)
    emitted = 0
    for path in sorted(glob.glob(source["glob"], recursive=True)):
        raw_text = Path(path).read_text(encoding="utf-8")
        for messages in _iter_claire_blocks_from_text(raw_text):
            yield messages
            emitted += 1
            if max_conversations and emitted >= max_conversations:
                return


def iter_claire_hf_messages(source: Dict) -> Iterator[List[Dict[str, str]]]:
    dataset = source.get("dataset", "OpenLLM-France/Claire-Dialogue-French-0.1")
    split = source.get("split", "train")
    sample_by = source.get("sample_by", "paragraph")
    text_key = source.get("text_key", "text")
    max_conversations = int(source.get("max_conversations", 0) or 0)
    ds = load_dataset(dataset, split=split, streaming=True, sample_by=sample_by)
    emitted = 0
    for row in ds:
        raw_text = _extract_first_text_field(row, preferred_key=text_key)
        for messages in _iter_claire_blocks_from_text(raw_text):
            yield messages
            emitted += 1
            if max_conversations and emitted >= max_conversations:
                return


def iter_local_messages_jsonl(source: Dict) -> Iterator[List[Dict[str, str]]]:
    max_conversations = int(source.get("max_conversations", 0) or 0)
    emitted = 0
    for _, row in _iter_local_jsonl_rows(source["glob"]):
        messages = _row_to_messages(row)
        if not is_valid_sft_conversation(messages):
            continue
        yield messages
        emitted += 1
        if max_conversations and emitted >= max_conversations:
            break


def _pick_first_non_empty_string(row: Dict, keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key, "")
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    return ""


def iter_hf_prompt_response_messages(source: Dict) -> Iterator[List[Dict[str, str]]]:
    dataset = source["dataset"]
    dataset_config = source.get("dataset_config") or None
    split = source.get("split", "train")
    max_conversations = int(source.get("max_conversations", 0) or 0)
    prompt_keys = source.get("prompt_keys", ["prompt", "instruction", "question"])
    context_keys = source.get("context_keys", ["context", "input"])
    response_keys = source.get("response_keys", ["response", "answer", "output", "completion"])
    system_prompt = normalize_text(source.get("system_prompt", ""))

    ds = load_dataset(dataset, dataset_config, split=split, streaming=True)
    emitted = 0
    for row in ds:
        prompt = _pick_first_non_empty_string(row, prompt_keys)
        context = _pick_first_non_empty_string(row, context_keys)
        response = _pick_first_non_empty_string(row, response_keys)
        if context:
            prompt = f"{prompt}\n\nContexte:\n{context}".strip() if prompt else context
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if prompt:
            messages.append({"role": "user", "content": prompt})
        if response:
            messages.append({"role": "assistant", "content": response})
        messages = normalize_messages(messages)
        if not is_valid_sft_conversation(messages):
            continue
        yield messages
        emitted += 1
        if max_conversations and emitted >= max_conversations:
            break


def iter_conversation_source(source: Dict) -> Iterator[List[Dict[str, str]]]:
    source_type = source.get("type", "").strip().lower()
    if source_type == "ultrachat":
        yield from iter_ultrachat_messages(source)
        return
    if source_type == "oasst1":
        yield from iter_oasst1_messages(source)
        return
    if source_type == "claire_txt":
        yield from iter_claire_txt_messages(source)
        return
    if source_type == "claire_hf":
        yield from iter_claire_hf_messages(source)
        return
    if source_type == "local_messages_jsonl":
        yield from iter_local_messages_jsonl(source)
        return
    if source_type == "hf_prompt_response":
        yield from iter_hf_prompt_response_messages(source)
        return
    raise ValueError(f"Type de source conversationnelle non supporté: {source_type}")


def iter_named_conversation_sources(sources: Sequence[Dict]) -> Iterator[Tuple[str, List[Dict[str, str]]]]:
    for idx, source in enumerate(sources):
        name = source.get("name") or source.get("type") or f"source_{idx}"
        try:
            for messages in iter_conversation_source(source):
                yield str(name), messages
        except Exception as exc:
            print(f"[WARN] Source SFT ignorée: {name} ({exc})")

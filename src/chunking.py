# ------------------------------------------------------------
# Chunking & Entry Segmentation
#
# This module converts the parsed diary text into retrieval-ready
# chunks used by the RAG system.
#
# Design decisions:
# - The diary is first segmented into chronological entries using
#   date-based headers (e.g., "18th Day of Blossomtide 1855").
# - Each entry is then split into overlapping chunks to improve
#   retrieval granularity while preserving narrative context.
#
# Chunking parameters were chosen after inspecting the diary:
# - target chunk size ≈ 260 words
# - overlap ≈ 40 words
#
# These values balance:
# - embedding quality
# - retrieval precision
# - context continuity across chunk boundaries.
#
# The resulting chunks are later embedded and stored in the
# vector database for semantic search.
# ------------------------------------------------------------

import json
import re

from src.utils import clean_text


ENTRY_BOUNDARY_PATTERN = re.compile(
    r"(?m)^(?P<header>\d{1,2}(?:st|nd|rd|th) Day of [A-Za-z]+ \d{4}(?: - .+)?)$"
)

DATE_ONLY_HEADER_PATTERN = re.compile(
    r"^\d{1,2}(?:st|nd|rd|th) Day of [A-Za-z]+ \d{4}$"
)


def combine_raw_pages(pages):
    parts = []
    for page in pages:
        raw_text = page.get("raw_text", "").strip()
        if raw_text:
            parts.append(raw_text)
    return "\n".join(parts)


def build_title_from_body(body: str, max_words: int = 7) -> str:
    if not body:
        return "Untitled Entry"

    stop_words = {
        "a", "an", "and", "as", "at", "but", "by", "for", "from",
        "in", "into", "of", "on", "or", "since", "the", "to", "with",
    }

    words = body.split()
    title_words = words[:max_words]

    while title_words and title_words[-1].strip(".,:;!?-—").lower() in stop_words:
        title_words.pop()

    title = " ".join(title_words).strip(" .,:;!?-—")

    if not title:
        return "Untitled Entry"

    return title


def enrich_header_if_date_only(header: str, body: str) -> str:
    if DATE_ONLY_HEADER_PATTERN.match(header.strip()):
        generated_title = build_title_from_body(body)
        return f"{header} - {generated_title}"
    return header


def segment_entries_from_raw_text(full_raw_text):
    matches = list(ENTRY_BOUNDARY_PATTERN.finditer(full_raw_text))
    entries = []

    for i, match in enumerate(matches):
        start_index = match.start()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(full_raw_text)

        raw_entry_text = full_raw_text[start_index:end_index].strip()
        raw_lines = [line.strip() for line in raw_entry_text.splitlines() if line.strip()]

        if not raw_lines:
            continue

        header = raw_lines[0]
        body_raw = "\n".join(raw_lines[1:]).strip()

        body = clean_text(body_raw)
        header = enrich_header_if_date_only(header, body)
        full_text = f"{header} {body}".strip() if body else header

        entries.append(
            {
                "entry_id": i + 1,
                "header": header,
                "body": body,
                "text": full_text,
                "word_count": len(full_text.split()),
            }
        )

    return entries


def save_entries_to_json(entries, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def build_entry_body_word_stats(entries):
    stats = []

    for entry in entries:
        body = entry.get("body", "").strip()
        body_word_count = len(body.split()) if body else 0

        stats.append(
            {
                "entry_id": entry["entry_id"],
                "header": entry["header"],
                "body_word_count": body_word_count,
            }
        )

    return stats

def split_text_by_words(text, target_words=260, overlap_words=40):
    words = text.split()

    if len(words) <= target_words:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + target_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append(chunk_text)

        if end == len(words):
            break

        start = max(end - overlap_words, start + 1)

    return chunks


def build_retrieval_chunks(
    entries,
    target_words=260,
    overlap_words=40,
    min_chunk_words=100,
):
    chunks = []

    for entry in entries:
        entry_id = entry["entry_id"]
        header = entry["header"]
        body = entry.get("body", "").strip()

        if not body:
            chunk_texts = [header]
        elif len(body.split()) <= target_words:
            chunk_texts = [f"{header} {body}".strip()]
        else:
            body_chunks = split_text_by_words(
                body,
                target_words=target_words,
                overlap_words=overlap_words,
            )
            chunk_texts = [f"{header} {chunk}".strip() for chunk in body_chunks]

        for idx, chunk_text in enumerate(chunk_texts, start=1):
            word_count = len(chunk_text.split())

            if word_count < min_chunk_words and len(chunk_texts) > 1:
                continue

            chunks.append(
                {
                    "chunk_id": f"entry_{entry_id}_chunk_{idx}",
                    "entry_id": entry_id,
                    "header": header,
                    "chunk_index": idx,
                    "text": chunk_text,
                    "word_count": word_count,
                }
            )

    return chunks


def save_chunks_to_json(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
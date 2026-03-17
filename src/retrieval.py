# ------------------------------------------------------------
# Retrieval Utilities
#
# This module implements the retrieval logic used in the RAG
# pipeline. It provides:
#
# - lexical search based on token overlap
# - loading of retrieval chunks
# - hybrid merging of dense and lexical retrieval results
#
# Hybrid ranking improves recall by combining semantic vector
# search with keyword-based matching.
# ------------------------------------------------------------

import json
import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def load_chunks(chunks_path: str):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = set(tokenize(text))

    if not query_tokens:
        return 0.0

    overlap = sum(1 for tok in query_tokens if tok in text_tokens)
    return overlap / len(query_tokens)


def lexical_search(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    scored = []

    for chunk in chunks:
        score = lexical_score(query, chunk["text"])
        if score > 0:
            scored.append(
                {
                    "source": "lexical",
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "entry_id": chunk["entry_id"],
                    "header": chunk["header"],
                    "chunk_index": chunk["chunk_index"],
                    "word_count": chunk["word_count"],
                    "text": chunk["text"],
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

def merge_results(dense_results: list[dict], lexical_results: list[dict], top_k: int = 10):
    merged = {}

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        dense_score = 1 / rank
        if chunk_id not in merged:
            merged[chunk_id] = {
                **item,
                "dense_rank": rank,
                "lexical_rank": None,
                "hybrid_score": 0.0,
            }
        merged[chunk_id]["hybrid_score"] += dense_score

    for rank, item in enumerate(lexical_results, start=1):
        chunk_id = item["chunk_id"]
        lexical_score = 1 / rank
        if chunk_id not in merged:
            merged[chunk_id] = {
                **item,
                "dense_rank": None,
                "lexical_rank": rank,
                "hybrid_score": 0.0,
            }
        merged[chunk_id]["lexical_rank"] = rank
        merged[chunk_id]["hybrid_score"] += lexical_score

    results = list(merged.values())
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]
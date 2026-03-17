# ------------------------------------------------------------
# Reranking Utilities
#
# This module applies a cross-encoder reranker to improve the
# relevance ordering of retrieved chunks.
#
# After the hybrid retrieval step produces candidate passages,
# the reranker scores each (query, passage) pair and selects
# the most relevant results.
# ------------------------------------------------------------

from sentence_transformers import CrossEncoder


def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)


def rerank_results(query: str, candidates: list[dict], reranker, top_k: int = 5) -> list[dict]:
    if not candidates:
        return []

    pairs = [(query, item["text"]) for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for item, score in zip(candidates, scores):
        reranked_item = dict(item)
        reranked_item["reranker_score"] = float(score)
        reranked.append(reranked_item)

    reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]
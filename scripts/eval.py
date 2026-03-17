# ------------------------------------------------------------
# Evaluation Pipeline
#
# This script evaluates the RAG system using the provided
# question–answer pairs.
#
# It retrieves relevant chunks, extracts the most relevant
# sentence, and compares it with the gold answer using a
# token-overlap rule.
#
# The script also generates confidence thresholds used by the
# API to decide when the system should abstain.
# ------------------------------------------------------------

import json
import re
import math

from src.config import (
    QUESTIONS_PATH,
    ANSWERS_PATH,
    EVAL_RESULTS_PATH,
    EVAL_SUMMARY_PATH,
    CONFIDENCE_CONFIG_PATH,
    MILVUS_DB_PATH,
    MILVUS_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNKS_PATH,
    RERANKER_MODEL_NAME,
)
from src.embeddings import load_embedding_model, embed_texts
from src.vector_store import (
    create_milvus_client,
    search_similar_chunks,
    format_search_results,
)
from src.retrieval import load_chunks, lexical_search, merge_results
from src.reranker import load_reranker, rerank_results


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "as", "and", "or", "but", "that",
    "this", "these", "those", "it", "its", "into", "about", "over", "under",
    "during", "after", "before", "between", "among", "their", "his", "her",
    "them", "they", "he", "she", "you", "your", "our", "we"
}

NOISE_TOKENS = {
    "day", "called", "known", "form", "forms", "led", "lies",
    "current", "popular", "significant"
}


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return [t for t in raw_tokens if len(t) > 1 and t not in NOISE_TOKENS]


def answer_tokens(gold_answer: str) -> list[str]:
    tokens = tokenize(gold_answer)
    return [t for t in tokens if t not in STOPWORDS]


def select_best_answer_candidate(question: str, retrieved_chunks: list[dict]) -> str:
    if not retrieved_chunks:
        return ""

    question_tokens = set(tokenize(question))
    best_chunk = None
    best_score = -1

    for chunk in retrieved_chunks:
        chunk_tokens = set(tokenize(chunk["text"]))
        overlap_score = len(question_tokens.intersection(chunk_tokens))

        if overlap_score > best_score:
            best_score = overlap_score
            best_chunk = chunk

    return best_chunk["text"] if best_chunk else retrieved_chunks[0]["text"]


def extract_best_answer_sentence(question: str, answer_candidate: str) -> str:
    if not answer_candidate:
        return ""

    question_tokens = set(tokenize(question))
    sentences = split_into_sentences(answer_candidate)

    best_sentence = ""
    best_score = -1

    for sentence in sentences:
        sentence_tokens = set(tokenize(sentence))
        overlap_score = len(question_tokens.intersection(sentence_tokens))

        if overlap_score > best_score:
            best_score = overlap_score
            best_sentence = sentence

    return best_sentence if best_sentence else answer_candidate


def sentence_token_match(sentence: str, gold_answer: str) -> dict:
    gold_toks = answer_tokens(gold_answer)
    sent_toks = set(tokenize(sentence))

    matched = [tok for tok in gold_toks if tok in sent_toks]
    ratio = len(matched) / len(gold_toks) if gold_toks else 0.0

    return {
        "gold_tokens": gold_toks,
        "matched_tokens": matched,
        "match_count": len(matched),
        "match_ratio": round(ratio, 4),
    }


def best_sentence_match_in_text(text: str, gold_answer: str) -> dict:
    sentences = split_into_sentences(text)

    best = {
        "sentence": "",
        "gold_tokens": [],
        "matched_tokens": [],
        "match_count": 0,
        "match_ratio": 0.0,
    }

    for sentence in sentences:
        score = sentence_token_match(sentence, gold_answer)
        if (
            score["match_ratio"] > best["match_ratio"]
            or (
                score["match_ratio"] == best["match_ratio"]
                and score["match_count"] > best["match_count"]
            )
        ):
            best = {
                "sentence": sentence,
                **score,
            }

    return best


def is_sentence_level_match(match_info: dict) -> bool:
    return match_info["match_count"] >= 2 and match_info["match_ratio"] >= 0.6


def best_retrieved_sentence_match(retrieved_chunks: list[dict], gold_answer: str) -> dict:
    best = {
        "retrieval_rank": None,
        "chunk_id": None,
        "header": None,
        "sentence": "",
        "gold_tokens": [],
        "matched_tokens": [],
        "match_count": 0,
        "match_ratio": 0.0,
    }

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        sentence_best = best_sentence_match_in_text(chunk["text"], gold_answer)

        if (
            sentence_best["match_ratio"] > best["match_ratio"]
            or (
                sentence_best["match_ratio"] == best["match_ratio"]
                and sentence_best["match_count"] > best["match_count"]
            )
        ):
            best = {
                "retrieval_rank": rank,
                "chunk_id": chunk["chunk_id"],
                "header": chunk["header"],
                **sentence_best,
            }

    return best


def retrieve_with_hybrid_rerank(
    question: str,
    embedding_model,
    milvus_client,
    all_chunks,
    reranker_model,
    dense_top_k: int = 10,
    lexical_top_k: int = 10,
    hybrid_top_k: int = 10,
    final_top_k: int = 3,
) -> list[dict]:
    query_embedding = embed_texts(embedding_model, [question])[0]

    dense_raw = search_similar_chunks(
        milvus_client,
        MILVUS_COLLECTION_NAME,
        query_embedding,
        top_k=dense_top_k,
    )
    dense_results = format_search_results(dense_raw)

    lexical_results = lexical_search(
        question,
        all_chunks,
        top_k=lexical_top_k,
    )

    hybrid_results = merge_results(
        dense_results,
        lexical_results,
        top_k=hybrid_top_k,
    )

    reranked_results = rerank_results(
        question,
        hybrid_results,
        reranker_model,
        top_k=final_top_k,
    )

    return reranked_results


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    values = sorted(values)
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return values[lower]

    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main():
    questions = load_lines(QUESTIONS_PATH)
    answers = load_lines(ANSWERS_PATH)

    if len(questions) != len(answers):
        raise ValueError(
            f"Question/answer count mismatch: {len(questions)} questions vs {len(answers)} answers"
        )

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    reranker_model = load_reranker(RERANKER_MODEL_NAME)
    milvus_client = create_milvus_client(MILVUS_DB_PATH)
    all_chunks = load_chunks(CHUNKS_PATH)

    results = []

    top1_sentence_match_count = 0
    topk_sentence_match_count = 0
    candidate_sentence_match_count = 0
    short_answer_sentence_match_count = 0
    success_count = 0

    successful_candidate_ratios = []
    successful_short_answer_ratios = []

    for idx, (question, gold_answer) in enumerate(zip(questions, answers), start=1):
        retrieved_chunks = retrieve_with_hybrid_rerank(
            question=question,
            embedding_model=embedding_model,
            milvus_client=milvus_client,
            all_chunks=all_chunks,
            reranker_model=reranker_model,
            dense_top_k=10,
            lexical_top_k=10,
            hybrid_top_k=10,
            final_top_k=3,
        )

        answer_candidate = select_best_answer_candidate(question, retrieved_chunks)
        predicted_answer = extract_best_answer_sentence(question, answer_candidate)

        topk_best = best_retrieved_sentence_match(retrieved_chunks, gold_answer)
        candidate_best = best_sentence_match_in_text(answer_candidate, gold_answer)
        short_best = best_sentence_match_in_text(predicted_answer, gold_answer)

        topk_sentence_match = is_sentence_level_match(topk_best)
        candidate_sentence_match = is_sentence_level_match(candidate_best)
        short_answer_sentence_match = is_sentence_level_match(short_best)

        if topk_sentence_match and topk_best["retrieval_rank"] == 1:
            top1_sentence_match_count += 1
        if topk_sentence_match:
            topk_sentence_match_count += 1
        if candidate_sentence_match:
            candidate_sentence_match_count += 1
        if short_answer_sentence_match:
            short_answer_sentence_match_count += 1

        success = candidate_sentence_match or topk_sentence_match
        if success:
            success_count += 1
            successful_candidate_ratios.append(candidate_best["match_ratio"])
            successful_short_answer_ratios.append(short_best["match_ratio"])

        results.append(
            {
                "question_id": idx,
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_candidate": answer_candidate,
                "topk_best_sentence": topk_best["sentence"],
                "topk_best_match_count": topk_best["match_count"],
                "topk_best_match_ratio": topk_best["match_ratio"],
                "topk_best_matched_tokens": topk_best["matched_tokens"],
                "topk_best_rank": topk_best["retrieval_rank"],
                "topk_sentence_match": topk_sentence_match,
                "candidate_best_sentence": candidate_best["sentence"],
                "candidate_match_count": candidate_best["match_count"],
                "candidate_match_ratio": candidate_best["match_ratio"],
                "candidate_matched_tokens": candidate_best["matched_tokens"],
                "candidate_sentence_match": candidate_sentence_match,
                "short_answer_best_sentence": short_best["sentence"],
                "short_answer_match_count": short_best["match_count"],
                "short_answer_match_ratio": short_best["match_ratio"],
                "short_answer_matched_tokens": short_best["matched_tokens"],
                "short_answer_sentence_match": short_answer_sentence_match,
                "success": success,
                "retrieved_headers": [chunk["header"] for chunk in retrieved_chunks],
                "retrieved_chunk_ids": [chunk["chunk_id"] for chunk in retrieved_chunks],
                "retrieved_scores": [
                    {
                        "hybrid_score": chunk.get("hybrid_score"),
                        "reranker_score": chunk.get("reranker_score"),
                    }
                    for chunk in retrieved_chunks
                ],
            }
        )

    total_questions = len(questions)

    summary = {
        "total_questions": total_questions,
        "top1_sentence_match_count": top1_sentence_match_count,
        "top1_sentence_match_rate": round(top1_sentence_match_count / total_questions, 4),
        "topk_sentence_match_count": topk_sentence_match_count,
        "topk_sentence_match_rate": round(topk_sentence_match_count / total_questions, 4),
        "candidate_sentence_match_count": candidate_sentence_match_count,
        "candidate_sentence_match_rate": round(candidate_sentence_match_count / total_questions, 4),
        "short_answer_sentence_match_count": short_answer_sentence_match_count,
        "short_answer_sentence_match_rate": round(short_answer_sentence_match_count / total_questions, 4),
        "success_count": success_count,
        "success_rate": round(success_count / total_questions, 4),
        "success_definition": "candidate_sentence_match OR topk_sentence_match",
        "sentence_match_rule": "same sentence, token overlap >= 0.6 and at least 2 matched tokens",
        "retrieval_pipeline": "dense + lexical + hybrid merge + reranker",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "collection_name": MILVUS_COLLECTION_NAME,
        "top_k": 3,
    }

    confidence_config = {
        "candidate_match_threshold": round(percentile(successful_candidate_ratios, 0.25), 4),
        "short_answer_match_threshold": round(percentile(successful_short_answer_ratios, 0.25), 4),
        "notes": "Thresholds derived from the 25th percentile of successful evaluation examples."
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(CONFIDENCE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(confidence_config, f, ensure_ascii=False, indent=2)

    print("Evaluation completed.")
    print(json.dumps(summary, indent=2))
    print("Confidence config:")
    print(json.dumps(confidence_config, indent=2))


if __name__ == "__main__":
    main()
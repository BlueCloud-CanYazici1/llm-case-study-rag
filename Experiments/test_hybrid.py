from src.config import (
    CHUNKS_PATH,
    MILVUS_DB_PATH,
    MILVUS_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from src.embeddings import load_embedding_model, embed_texts
from src.vector_store import create_milvus_client, search_similar_chunks, format_search_results
from src.retrieval import load_chunks, lexical_search
from src.reranker import load_reranker, rerank_results


def merge_results(dense_results: list[dict], lexical_results: list[dict], top_k: int = 10):
    merged = {}

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item["chunk_id"]
        dense_score = 1 / rank
        if chunk_id not in merged:
            merged[chunk_id] = {**item, "dense_rank": rank, "lexical_rank": None, "hybrid_score": 0.0}
        merged[chunk_id]["hybrid_score"] += dense_score

    for rank, item in enumerate(lexical_results, start=1):
        chunk_id = item["chunk_id"]
        lexical_score = 1 / rank
        if chunk_id not in merged:
            merged[chunk_id] = {**item, "dense_rank": None, "lexical_rank": rank, "hybrid_score": 0.0}
        merged[chunk_id]["lexical_rank"] = rank
        merged[chunk_id]["hybrid_score"] += lexical_score

    results = list(merged.values())
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:top_k]


def main():
    query = "What is the main legislative body of Veridia?"

    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    reranker = load_reranker()

    query_embedding = embed_texts(embedding_model, [query])[0]

    client = create_milvus_client(MILVUS_DB_PATH)
    dense_raw = search_similar_chunks(client, MILVUS_COLLECTION_NAME, query_embedding, top_k=10)
    dense_results = format_search_results(dense_raw)

    chunks = load_chunks(CHUNKS_PATH)
    lexical_results = lexical_search(query, chunks, top_k=10)

    hybrid_results = merge_results(dense_results, lexical_results, top_k=10)
    reranked_results = rerank_results(query, hybrid_results, reranker, top_k=5)

    print(f"Query: {query}\n")

    print("=== DENSE TOP 3 ===")
    for item in dense_results[:3]:
        print(item["chunk_id"], "|", item["header"])

    print("\n=== LEXICAL TOP 3 ===")
    for item in lexical_results[:3]:
        print(item["chunk_id"], "|", item["header"])

    print("\n=== HYBRID TOP 5 ===")
    for item in hybrid_results[:5]:
        print(item["chunk_id"], "|", item["header"], "| hybrid_score:", round(item["hybrid_score"], 4))

    print("\n=== RERANKED TOP 5 ===")
    for item in reranked_results[:5]:
        print(
            item["chunk_id"],
            "|",
            item["header"],
            "| hybrid_score:",
            round(item["hybrid_score"], 4),
            "| reranker_score:",
            round(item["reranker_score"], 4),
        )


if __name__ == "__main__":
    main()
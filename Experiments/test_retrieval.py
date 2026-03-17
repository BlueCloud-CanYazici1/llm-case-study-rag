from src.config import MILVUS_DB_PATH, MILVUS_COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.embeddings import load_embedding_model, embed_texts
from src.vector_store import create_milvus_client, search_similar_chunks


def main():
    query = "What is the main legislative body of Veridia?"

    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    query_embedding = embed_texts(model, [query])[0]

    client = create_milvus_client(MILVUS_DB_PATH)
    results = search_similar_chunks(
        client,
        MILVUS_COLLECTION_NAME,
        query_embedding,
        top_k=3,
    )

    print(f"Query: {query}\n")

    for rank, hit in enumerate(results[0], start=1):
        entity = hit["entity"]
        print(f"Rank: {rank}")
        print(f"Score: {hit['distance']}")
        print(f"Chunk ID: {entity['chunk_id']}")
        print(f"Header: {entity['header']}")
        print(f"Text: {entity['text']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
# ------------------------------------------------------------
# Vector Store Utilities (Milvus)
#
# This module handles all interactions with the Milvus vector
# database used for semantic retrieval.
#
# It provides utilities to:
# - create the Milvus client
# - create or recreate the collection
# - insert chunk embeddings
# - perform similarity search
#
# Retrieved vectors are converted into structured chunk
# results used by the retrieval pipeline.
# ------------------------------------------------------------

from pathlib import Path
from pymilvus import MilvusClient


def create_milvus_client(db_path):
    return MilvusClient(uri=str(Path(db_path)))


def recreate_collection(client, collection_name: str, dimension: int):
    existing = client.list_collections()

    if collection_name in existing:
        client.drop_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        auto_id=False,
    )


def build_milvus_records(chunks, embeddings):
    records = []

    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
        records.append(
            {
                "id": idx,
                "chunk_id": chunk["chunk_id"],
                "vector": embedding.tolist(),
                "text": chunk["text"],
                "entry_id": chunk["entry_id"],
                "header": chunk["header"],
                "chunk_index": chunk["chunk_index"],
                "word_count": chunk["word_count"],
            }
        )

    return records


def insert_records(client, collection_name: str, records):
    client.insert(collection_name=collection_name, data=records)

def search_similar_chunks(client, collection_name: str, query_vector, top_k: int = 3):
    results = client.search(
        collection_name=collection_name,
        data=[query_vector.tolist()],
        limit=top_k,
        output_fields=["chunk_id", "text", "header", "entry_id", "chunk_index", "word_count"],
    )
    return results

def format_search_results(results):
    formatted = []

    for hit in results[0]:
        entity = hit["entity"]
        formatted.append(
            {
                "score": float(hit["distance"]),
                "chunk_id": entity.get("chunk_id", ""),
                "entry_id": entity.get("entry_id"),
                "header": entity.get("header", ""),
                "chunk_index": entity.get("chunk_index"),
                "word_count": entity.get("word_count"),
                "text": entity.get("text", ""),
            }
        )

    return formatted
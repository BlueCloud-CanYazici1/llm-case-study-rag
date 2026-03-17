# ------------------------------------------------------------
# Data Preparation Pipeline
#
# This script converts the Dr. Voss diary PDF into a searchable
# retrieval index.
#
# Steps:
# 1. Extract and clean text from the PDF
# 2. Segment the document into diary entries
# 3. Split entries into overlapping retrieval chunks
# 4. Generate embeddings for each chunk
# 5. Store vectors and metadata in a Milvus database
#
# This step builds the vector index used by the RAG system.
# ------------------------------------------------------------

import json

from src.config import (
    PDF_PATH,
    CLEANED_PAGES_PATH,
    ENTRIES_PATH,
    ENTRY_BODY_STATS_PATH,
    CHUNKS_PATH,
    ARTIFACTS_DIR,
    TARGET_CHUNK_WORDS,
    CHUNK_OVERLAP_WORDS,
    MIN_CHUNK_WORDS,
    MILVUS_DB_PATH,
    MILVUS_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)
from src.pdf_parser import extract_pages_from_pdf, save_pages_to_json
from src.chunking import (
    combine_raw_pages,
    segment_entries_from_raw_text,
    save_entries_to_json,
    build_entry_body_word_stats,
    build_retrieval_chunks,
    save_chunks_to_json,
)
from src.embeddings import load_embedding_model, embed_texts
from src.vector_store import (
    create_milvus_client,
    recreate_collection,
    build_milvus_records,
    insert_records,
)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    pages = extract_pages_from_pdf(PDF_PATH)
    save_pages_to_json(pages, CLEANED_PAGES_PATH)

    full_raw_text = combine_raw_pages(pages)
    entries = segment_entries_from_raw_text(full_raw_text)
    save_entries_to_json(entries, ENTRIES_PATH)

    body_stats = build_entry_body_word_stats(entries)
    with open(ENTRY_BODY_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(body_stats, f, ensure_ascii=False, indent=2)

    chunks = build_retrieval_chunks(
        entries,
        target_words=TARGET_CHUNK_WORDS,
        overlap_words=CHUNK_OVERLAP_WORDS,
        min_chunk_words=MIN_CHUNK_WORDS,
    )
    save_chunks_to_json(chunks, CHUNKS_PATH)

    model = load_embedding_model(EMBEDDING_MODEL_NAME)
    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(model, chunk_texts)

    client = create_milvus_client(MILVUS_DB_PATH)
    recreate_collection(
        client,
        MILVUS_COLLECTION_NAME,
        dimension=len(embeddings[0]),
    )

    records = build_milvus_records(chunks, embeddings)
    insert_records(client, MILVUS_COLLECTION_NAME, records)

    print("PDF parsed, cleaned, segmented, chunked, and indexed successfully.")
    print(f"Total pages extracted: {len(pages)}")
    print(f"Total entries found: {len(entries)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Embedding model used: {EMBEDDING_MODEL_NAME}")
    print(f"Saved cleaned pages to: {CLEANED_PAGES_PATH}")
    print(f"Saved entries to: {ENTRIES_PATH}")
    print(f"Saved entry body stats to: {ENTRY_BODY_STATS_PATH}")
    print(f"Saved chunks to: {CHUNKS_PATH}")
    print(f"Milvus DB path: {MILVUS_DB_PATH}")
    print(f"Milvus collection: {MILVUS_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
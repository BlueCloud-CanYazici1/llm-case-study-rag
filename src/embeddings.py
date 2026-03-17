# ------------------------------------------------------------
# Embedding Utilities
#
# This module loads the sentence embedding model and converts
# text chunks into vector representations used for retrieval.
#
# Embeddings are normalized so cosine similarity can be used
# efficiently during vector search in Milvus.
# ------------------------------------------------------------

from sentence_transformers import SentenceTransformer


def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name)


def embed_texts(model, texts, batch_size: int = 16):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings
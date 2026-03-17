# ------------------------------------------------------------
# Project Configuration
#
# Central path and model configuration for the RAG pipeline.
# This file defines:
# - data and artifact locations
# - chunking parameters
# - embedding / reranker models
# - evaluation output paths
#
# Chunk settings were chosen after inspecting entry length
# statistics from the diary. Most entries are around 225–235
# words on average, so chunks were sized to preserve semantic
# coherence while still allowing longer entries to be split
# into overlapping retrieval units.
# ------------------------------------------------------------

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

PDF_PATH = DATA_DIR / "dr_voss_diary.pdf"

EXTRACTED_PAGES_PATH = ARTIFACTS_DIR / "extracted_pages.json"
CLEANED_PAGES_PATH = ARTIFACTS_DIR / "cleaned_pages.json"
ENTRIES_PATH = ARTIFACTS_DIR / "entries.json"
ENTRY_BODY_STATS_PATH = ARTIFACTS_DIR / "entry_body_word_stats.json"
#-----------------------------------------------------------------------

CHUNKS_PATH = ARTIFACTS_DIR / "chunks.json"

TARGET_CHUNK_WORDS = 260
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 100

                                    # Reason: 	
                                    #   •	total entry: 125
                                    #	•	min: 123
                                    #	•	max: 376
                                    #	•	avg: 233.6
                                    #	•	med: 225
                                    #	•	above 220 word entry: 69

#-----------------------------------------------------------------------


MILVUS_DB_PATH = "./data/artifacts/milvus.db"
MILVUS_COLLECTION_NAME = "dr_voss_chunks"
EMBEDDING_MODEL_NAME = "Snowflake/snowflake-arctic-embed-s"




                       #EVALUATION#
#-----------------------------------------------------------------------
QUESTIONS_PATH = DATA_DIR / "questions.txt"
ANSWERS_PATH = DATA_DIR / "answers.txt"

EVAL_RESULTS_PATH = ARTIFACTS_DIR / "eval_results.json"
EVAL_SUMMARY_PATH = ARTIFACTS_DIR / "eval_summary.json"
#-----------------------------------------------------------------------

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#-----------------------------------------------------------------------
CONFIDENCE_CONFIG_PATH = ARTIFACTS_DIR / "confidence_config.json"
#-----------------------------------------------------------------------
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#-----------------------------------------------------------------------
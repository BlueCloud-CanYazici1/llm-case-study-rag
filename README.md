# Dr. Voss Diary Retrieval-Augmented Generation System

This project implements a retrieval-first Retrieval-Augmented Generation (RAG) system over the Dr. Voss Diary document.

The system processes the diary, stores semantically meaningful chunks in a vector database, retrieves relevant evidence for a user query, and synthesizes a grounded answer using a local LLM.

Unlike naive RAG systems, this implementation prioritizes retrieval quality and answer reliability over generative flexibility.

The LLM is not used for knowledge discovery, but strictly for rewriting answers from retrieved evidence.

⸻

# Key Features

The system is designed to maximize answer accuracy and minimize hallucinations through:
	•	diary-aware semantic chunking
	•	hybrid retrieval (dense + lexical)
	•	cross-encoder reranking
	•	retrieval-first architecture
	•	controlled LLM answer synthesis
	•	sentence-level evidence extraction
	•	confidence-based abstention
	•	reproducible evaluation pipeline

⸻

# Quick Start

Install dependencies and prepare the data:

pip install -r requirements.txt
PYTHONPATH=. python scripts/prepare_data.py

Start the API server:

python -m uvicorn app:app --reload

Open the interactive API:

http://127.0.0.1:8000/docs


⸻

# Design Goals

This system was built with the following engineering goals:
	•	improve retrieval accuracy beyond simple vector search
	•	ensure answers remain grounded in the source document
	•	minimize hallucinated responses
	•	provide transparent supporting evidence
	•	enable measurable and reproducible evaluation

⸻

# Retrieval-First RAG Design

The system follows a retrieval-first architecture.

Instead of relying on the LLM to infer or search for information, the system first constructs a high-quality retrieval pipeline, and only then invokes the LLM.

Why Retrieval-First?
	•	LLMs are unreliable at implicit retrieval
	•	hallucination risk increases without grounded context
	•	retrieval quality directly determines answer correctness

This design ensures:
	•	strong grounding in source text
	•	reduced hallucination risk
	•	transparent reasoning via evidence
	•	predictable system behavior

⸻

# System Architecture

The system is divided into two stages:

⸻

1. Offline Indexing Stage

The document is processed once and indexed.

PDF Document
      ↓
PDF Parsing
      ↓
Chunking (Diary-aligned segments)
      ↓
Embedding Generation
      ↓
Milvus Vector Index
      ↓
chunks.json


⸻

2. Online Query Stage

At query time, the system executes the retrieval pipeline.

User Question
      ↓
Question Embedding
      ↓
Hybrid Retrieval (Dense + Lexical)
      ↓
Cross-Encoder Reranking
      ↓
Top-K Chunk Selection
      ↓
Context Construction
      ↓
LLM Answer Synthesis
      ↓
Evidence Extraction
      ↓
Confidence Check
      ↓
Final Answer or Abstain


⸻

Query Pipeline (Detailed)
	1.	Query Embedding
Generated using Snowflake Arctic.
	2.	Dense Retrieval
Semantic search via Milvus vector similarity.
	3.	Lexical Retrieval
Token-overlap keyword matching.
	4.	Hybrid Merge
Combines dense and lexical results.
	5.	Cross-Encoder Reranking
Model: ms-marco-MiniLM-L-6-v2
Evaluates (query, chunk) pairs jointly.
	6.	Context Construction
Top-ranked chunks are concatenated to form the LLM input context.
	7.	LLM Answer Synthesis
The LLM rewrites the answer strictly from retrieved evidence.
	8.	Evidence Extraction
The most relevant supporting sentence is extracted.
	9.	Confidence Check
Token overlap is used to decide whether to answer or abstain.

⸻

# Document Processing

Source:

data/dr_voss_diary.pdf


⸻

# Chunking Strategy

The diary is structured as daily entries.

Chunks were aligned with these entries.
	•	average chunk size: 180–250 words
	•	preserves semantic coherence
	•	improves retrieval quality

⸻

# Embedding Model

Snowflake/snowflake-arctic-embed-s


⸻

# Vector Database

Milvus

Provides efficient similarity search over embeddings.

⸻

# Hybrid Retrieval 

Why Hybrid?

Dense retrieval captures semantic meaning but may miss exact terms.
Lexical retrieval captures exact matches but lacks semantic understanding.

Combining both:
	•	improves recall (dense)
	•	improves precision (lexical)

⸻

# Cross-Encoder Reranking

Initial candidates are reranked using:

cross-encoder/ms-marco-MiniLM-L-6-v2

Why Reranking?

Bi-encoder embeddings are approximate.
Cross-encoders provide fine-grained relevance scoring, improving precision.

⸻

# LLM Answer Generation

The LLM is used only for controlled answer synthesis.

Prompt Constraints

The model is explicitly instructed to:
	•	answer only using the provided context
	•	avoid external knowledge
	•	avoid speculation
	•	return no answer if evidence is insufficient

⸻

# Evidence Extraction

The system extracts the most relevant sentence from the selected chunk.

This ensures:
	•	transparency
	•	traceability
	•	verifiability

⸻

# Confidence & Abstention

To prevent hallucinations, the system includes a confidence-based abstention mechanism.

It evaluates:
	•	query ↔ chunk overlap
	•	query ↔ answer overlap

If confidence is low:

"The provided documents do not contain enough information to answer this question reliably."


⸻

# API Usage

Example request:

curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "Which national park in Veridia is known for its ancient forests?",
  "top_k": 3
}'

Example response:

{
  "answer": "...",
  "supporting_sentence": "...",
  "answer_candidate": "...",
  "abstained": false
}


⸻

# Evaluation Pipeline

Run evaluation:

PYTHONPATH=. python scripts/eval.py


⸻

# Evaluation Metric

Sentence-level token overlap:
	•	threshold ≥ 0.6
	•	minimum 2 matching tokens

This metric is simple and reproducible, though it does not capture full semantic equivalence.

⸻

# Final Evaluation Results

After improving retrieval and abstention logic:

Total Questions: 55  
Correct Answers: 53  
Incorrect Answers: 1  
Abstained: 1  

Accuracy (excluding abstained):

53 / 54 ≈ 98.1%


⸻

# System Improvement

Initial system:
	•	Answered: 36
	•	Abstained: 19

Final system:
	•	Answered: 53
	•	Abstained: 1

This improvement was achieved by:
	•	refining confidence thresholds
	•	improving retrieval ranking
	•	better evidence selection

⸻

# Error Analysis

One incorrect answer occurred due to hallucination in the LLM rewriting step.
	•	Retrieved evidence: hydroharmonic farming technology
	•	Generated answer: drones for precision farming

This demonstrates that even with correct retrieval, generation must be tightly controlled.

⸻

# Engineering Trade-offs
	•	precision vs recall (hybrid retrieval)
	•	latency vs accuracy (reranking cost)
	•	abstain vs answer risk
	•	chunk size vs retrieval quality

⸻

# Limitations
	•	answers rely only on the provided document
	•	multi-chunk reasoning is limited
	•	LLM rewriting may introduce minor hallucinations
	•	evaluation metric is token-based, not semantic

⸻

# Future Improvements
	•	semantic similarity evaluation (e.g. embedding-based metrics)
	•	multi-chunk reasoning
	•	stricter answer extraction (fully extractive QA)
	•	lightweight conversational interface

⸻

# Final Note

This project demonstrates a retrieval-engineered RAG system focused on:
	•	reliability
	•	grounding
	•	transparency
	•	measurable performance

rather than a generic LLM-based chatbot.

⸻

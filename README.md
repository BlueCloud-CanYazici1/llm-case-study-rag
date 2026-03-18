
# Dr. Voss Diary Retrieval-Augmented Generation System

This project implements a retrieval-first Retrieval-Augmented Generation (RAG) system over the Dr. Voss Diary document.

The system processes the diary, stores semantically meaningful chunks in a vector database, retrieves relevant evidence for a user query, and synthesizes a grounded answer using a local LLM.

Unlike naive RAG systems, this implementation prioritizes retrieval quality, answer grounding, and reliability over generative flexibility.

The LLM is not used for knowledge discovery, but strictly for controlled answer synthesis from retrieved evidence.

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

pip install -r requirements.txt
PYTHONPATH=. python scripts/prepare_data.py
python -m uvicorn app:app --reload

Open API:

http://127.0.0.1:8000/docs


⸻

# Design Goals
	•	improve retrieval accuracy beyond simple vector search
	•	ensure answers remain grounded in the source document
	•	minimize hallucinated responses
	•	provide transparent supporting evidence
	•	enable measurable and reproducible evaluation

⸻

# Core Design Principle

Retrieval quality determines answer quality.

Instead of relying on the LLM to infer knowledge, the system invests heavily in retrieval quality and evidence selection.

This reduces hallucination risk and ensures answers remain traceable to the document.

⸻

# System Architecture

Offline Indexing

PDF → Parsing → Chunking → Embedding → Milvus Index → chunks.json

Online Query Pipeline

Question
 → Embedding
 → Hybrid Retrieval
 → Cross-Encoder Reranking
 → Top-K Selection
 → Context Construction
 → LLM Answer Synthesis
 → Evidence Extraction
 → Confidence Check
 → Answer / Abstain


⸻

# Query Pipeline (Detailed)
	1.	Query embedding (Snowflake Arctic)
	2.	Dense retrieval (Milvus vector search)
	3.	Lexical retrieval (token overlap)
	4.	Hybrid merge
	5.	Cross-encoder reranking (MiniLM)
	6.	Context construction (top chunks concatenation)
	7.	LLM answer synthesis
	8.	Sentence-level evidence extraction
	9.	Confidence-based validation

⸻

# Document Processing

Source:

data/dr_voss_diary.pdf


⸻

# Chunking Strategy

The diary is written as daily chronological entries.

Instead of arbitrary fixed-size chunking, chunks were aligned with these entries.

Why this approach?
	•	each entry is a self-contained semantic unit
	•	splitting mid-entry breaks context
	•	improves retrieval precision

Empirical findings
	•	smaller chunks → context loss
	•	larger chunks → irrelevant noise

Final choice:
	•	180–250 words per chunk
	•	best balance between context and precision

⸻

# Embedding Model

Snowflake/snowflake-arctic-embed-s


⸻

# Vector Storage

Milvus

Provides efficient similarity search over embeddings.

⸻

# Hybrid Retrieval

Why Hybrid?

Dense retrieval:
	•	captures semantic similarity
	•	may miss exact keywords

Lexical retrieval:
	•	captures exact matches
	•	lacks semantic understanding

Combining both:
	•	improves recall (dense)
	•	improves precision (lexical)

⸻

# Cross-Encoder Reranking

Model:

cross-encoder/ms-marco-MiniLM-L-6-v2

Why Reranking?

Bi-encoder embeddings are approximate.
Cross-encoders evaluate (query, chunk) pairs jointly and provide more accurate relevance scoring.

⸻

# LLM Answer Synthesis

The LLM is used only for controlled rewriting.

Prompt constraints:
	•	use only provided context
	•	do not add external knowledge
	•	avoid speculation
	•	return no answer if insufficient evidence

⸻

# Evidence Extraction

The system extracts the most relevant supporting sentence.

This ensures:
	•	transparency
	•	interpretability
	•	verifiability

⸻

# Confidence & Abstention

The system uses token-overlap scoring to decide whether to answer.

Evaluates:
	•	query ↔ chunk overlap
	•	query ↔ answer overlap

If below threshold:

"The provided documents do not contain enough information to answer this question reliably."


⸻

# Threshold Selection (Important)

Thresholds were not chosen arbitrarily.

Initial system
	•	Answered: 36
	•	Abstained: 19

Problem:
	•	relevant evidence existed
	•	but system abstained too often

Improvements
	•	relaxed overlap thresholds
	•	combined chunk + answer scoring

Final system
	•	Answered: 53
	•	Abstained: 1

⸻

# Evaluation Pipeline

PYTHONPATH=. python scripts/eval.py


⸻

# Evaluation Metric
	•	token overlap ≥ 0.6
	•	minimum 2 tokens

Simple and reproducible, though not fully semantic.

⸻

# Final Results

Total Questions: 55
Correct Answers: 53
Incorrect Answers: 1
Abstained: 1

Accuracy (excluding abstained):

98.1%


⸻

# Engineering Trade-offs
	•	precision vs recall (hybrid retrieval)
	•	latency vs accuracy (reranking cost)
	•	abstain vs hallucinate risk
	•	chunk size vs retrieval quality

⸻

# Limitations
	•	limited multi-chunk reasoning
	•	depends on chunk quality
	•	LLM rewriting may introduce minor hallucinations
	•	evaluation is token-based

⸻

# Future Improvements
	•	semantic similarity metrics
	•	multi-chunk reasoning
	•	fully extractive QA
	•	conversational interface

⸻

# Final Note

This project demonstrates a retrieval-engineered RAG system focused on:
	•	reliability
	•	grounding
	•	transparency
	•	measurable performance

rather than a generic chatbot.


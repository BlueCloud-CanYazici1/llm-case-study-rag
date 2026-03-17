## Running the Project

This section describes how to install the required dependencies, prepare the dataset, run the API server, and evaluate the RAG pipeline.

The full pipeline consists of four steps:

1. Install dependencies  
2. Process the document and generate embeddings  
3. Launch the FastAPI application  
4. Run the evaluation script  

---

## 1. Install Dependencies

Create and activate a Python virtual environment, then install all required packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This installs all dependencies required for:

- document processing  
- embedding generation  
- vector database interaction  
- FastAPI server  
- evaluation pipeline  

---

## 2. Prepare the Data

Run the data preparation script:

```bash
PYTHONPATH=. python scripts/prepare_data.py
```

This script performs the following steps:

- extracts text from `data/dr_voss_diary.pdf`
- splits the document into semantic chunks
- generates embeddings for each chunk
- stores embeddings and metadata in the Milvus vector database

After this step, the document becomes searchable through the retrieval pipeline.

---

## 3. Start the API Server

Launch the FastAPI application using Uvicorn:

```bash
python -m uvicorn app:app --reload
```

The server will start at:

```
http://127.0.0.1:8000
```

You can send POST requests to the `/query` endpoint with a question payload.

Example request:

```json
{
  "question": "What technologies are used in Veridia?"
}
```

The system retrieves relevant document chunks and returns the most relevant supporting sentence from the retrieved evidence.

Interactive API documentation is available at:

```
http://127.0.0.1:8000/docs
```

---

## 4. Run the Evaluation Pipeline

To evaluate the system performance, run:

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/eval.py
```

The evaluation script:

- reads questions from `data/questions.txt`
- queries the RAG system
- compares generated answers with `data/answers.txt`
- reports the overall accuracy

This provides a simple benchmark to measure retrieval performance.

---

## Complete Execution Pipeline

For convenience, the entire workflow can be executed using the following sequence.

### Terminal 1

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/prepare_data.py
python -m uvicorn app:app --reload
```

### Terminal 2

```bash
source .venv/bin/activate
PYTHONPATH=. python scripts/eval.py
```

The API documentation can be accessed at:

```
http://127.0.0.1:8000/docs
```
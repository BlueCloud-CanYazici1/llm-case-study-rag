import argparse
import json
import time
from pathlib import Path

import requests


def load_results(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_abstained_questions(results):
    abstained = []
    for row in results:
        if row.get("abstained") is True:
            abstained.append(
                {
                    "question_id": row.get("question_id"),
                    "question": row.get("question"),
                }
            )
    return abstained


def query_api(api_url: str, question: str, top_k: int):
    payload = {
        "question": question,
        "top_k": top_k,
    }
    response = requests.post(api_url, json=payload, timeout=180)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        default="experiments/batch_query_outputs/batch_query_results.json",
        help="Path to the batch results JSON file.",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/query",
        help="Query endpoint URL.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/abstained_chunk_analysis",
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    batch_results = load_results(results_path)
    abstained_questions = get_abstained_questions(batch_results)

    print(f"Found {len(abstained_questions)} abstained questions.")

    raw_responses = []
    failures = []

    for i, item in enumerate(abstained_questions, start=1):
        question_id = item["question_id"]
        question = item["question"]

        print(f"[{i}/{len(abstained_questions)}] Re-querying question_id={question_id}")

        try:
            api_response = query_api(args.api_url, question, args.top_k)

            record = {
                "question_id": question_id,
                "question": question,
                "raw_response": api_response,
            }
            raw_responses.append(record)

            time.sleep(0.2)

        except Exception as e:
            failures.append(
                {
                    "question_id": question_id,
                    "question": question,
                    "error": str(e),
                }
            )
            print(f"  error: {e}")

    # Save full raw responses
    raw_json_path = output_dir / "abstained_questions_with_chunks.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)

    # Save a readable text report
    txt_path = output_dir / "abstained_questions_with_chunks_readable.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in raw_responses:
            rr = item["raw_response"]
            f.write("=" * 120 + "\n")
            f.write(f"QUESTION ID: {item['question_id']}\n")
            f.write(f"QUESTION: {item['question']}\n")
            f.write(f"ANSWER: {rr.get('answer')}\n")
            f.write(f"ABSTAINED: {rr.get('abstained')}\n")
            f.write(f"SUPPORTING SENTENCE: {rr.get('supporting_sentence')}\n")
            f.write(f"ANSWER CANDIDATE: {rr.get('answer_candidate')}\n")
            f.write("\nRETRIEVED CHUNKS:\n\n")

            chunks = rr.get("retrieved_chunks", [])
            if not chunks:
                f.write("  No retrieved chunks.\n\n")

            for idx, chunk in enumerate(chunks, start=1):
                f.write(f"  --- CHUNK {idx} ---\n")
                f.write(f"  score      : {chunk.get('score')}\n")
                f.write(f"  chunk_id   : {chunk.get('chunk_id')}\n")
                f.write(f"  entry_id   : {chunk.get('entry_id')}\n")
                f.write(f"  header     : {chunk.get('header')}\n")
                f.write(f"  chunk_index: {chunk.get('chunk_index')}\n")
                f.write(f"  word_count : {chunk.get('word_count')}\n")
                f.write("  text:\n")
                f.write(f"{chunk.get('text', '')}\n\n")

    failures_path = output_dir / "abstained_questions_with_chunks_failures.json"
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)

    summary = {
        "total_abstained_questions": len(abstained_questions),
        "successful_requeries": len(raw_responses),
        "failed_requeries": len(failures),
        "output_json": str(raw_json_path),
        "output_readable_txt": str(txt_path),
    }

    summary_path = output_dir / "abstained_questions_with_chunks_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Saved JSON     : {raw_json_path}")
    print(f"Saved TXT      : {txt_path}")
    print(f"Saved failures : {failures_path}")
    print(f"Saved summary  : {summary_path}")


if __name__ == "__main__":
    main()
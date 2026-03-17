import requests
import json
import csv
import time
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--questions", required=True)
parser.add_argument("--api-url", default="http://127.0.0.1:8000/query")
parser.add_argument("--top-k", type=int, default=3)

args = parser.parse_args()

questions_path = Path(args.questions)
questions = [q.strip() for q in questions_path.read_text(encoding="utf-8").splitlines() if q.strip()]

results = []
failures = []

for i, question in enumerate(questions, 1):
    payload = {
        "question": question,
        "top_k": args.top_k
    }

    try:
        response = requests.post(args.api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()

        retrieved = data.get("retrieved_chunks", [])
        scores = [c.get("score") for c in retrieved]

        result = {
            "question_id": i,
            "question": question,
            "answer": data.get("answer"),
            "abstained": data.get("abstained"),
            "supporting_sentence": data.get("supporting_sentence"),
            "answer_candidate": data.get("answer_candidate"),
            "top1_score": scores[0] if len(scores) > 0 else None,
            "top2_score": scores[1] if len(scores) > 1 else None,
            "top3_score": scores[2] if len(scores) > 2 else None,
        }
        results.append(result)
        print(f"{i}/{len(questions)} processed")

        time.sleep(0.2)

    except Exception as e:
        failures.append({
            "question_id": i,
            "question": question,
            "error": str(e)
        })
        print(f"{i}/{len(questions)} error: {e}")

output_dir = Path("experiments/batch_query_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

(output_dir / "batch_query_results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

if results:
    with open(output_dir / "batch_query_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

summary = {
    "total_questions": len(questions),
    "successful": len(results),
    "failed": len(failures),
    "abstained_count": sum(1 for r in results if r.get("abstained")),
    "answered_count": sum(1 for r in results if not r.get("abstained")),
    "average_top1_score": (
        sum(r["top1_score"] for r in results if r.get("top1_score") is not None) /
        max(1, sum(1 for r in results if r.get("top1_score") is not None))
    )
}

(output_dir / "batch_query_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

(output_dir / "batch_query_failures.json").write_text(
    json.dumps(failures, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Done.")
print(f"Results saved under: {output_dir}")
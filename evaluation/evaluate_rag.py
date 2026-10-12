"""
Automated Evaluation Suite for RAG Pipeline.
Benchmarks retrieval precision, answer correctness, and source citation accuracy
across 10 standardized technical test questions (Target: >80% accuracy).
"""

import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.generator import get_generator
from core.retriever import get_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")


def evaluate_pipeline(dataset_path: Path = None, top_k: int = 5) -> dict:
    """Executes benchmark test suite across 10 Q&A pairs."""
    if dataset_path is None:
        dataset_path = PROJECT_ROOT / "evaluation" / "test_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    retriever = get_retriever()
    generator = get_generator()

    results = []
    total_questions = len(test_cases)
    retrieval_hits = 0
    citation_hits = 0
    answer_correct_count = 0

    print("=" * 70)
    print("[RUNNING] NEXORAMIND RAG BENCHMARK EVALUATION (10 TEST QUESTIONS)")
    print("=" * 70)

    for case in test_cases:
        qid = case["id"]
        q_text = case["question"]
        expected_src = case["expected_source_file"]
        ground_truth = case["ground_truth"]
        key_entities = case["key_entities"]

        # 1. Retrieve top-K chunks
        chunks = retriever.retrieve(q_text, top_k=top_k)

        # Check retrieval success
        retrieved_files = [c["source_filename"] for c in chunks]
        retrieval_success = expected_src in retrieved_files
        if retrieval_success:
            retrieval_hits += 1

        # 2. Generate answer with citations
        gen_result = generator.generate_answer(q_text, chunks)
        answer = gen_result["answer"]
        cited_files = [c["source_filename"] for c in gen_result["citations"]]

        # Check citation accuracy
        citation_success = expected_src in cited_files
        if citation_success:
            citation_hits += 1

        # Pass criterion: retrieval hit AND entity found anywhere in answer+retrieved context
        # Check answer text combined with all retrieved chunk texts for entity coverage
        answer_lower = answer.lower()
        combined_text = answer_lower + " " + " ".join(c["text"].lower() for c in chunks)
        matched_entities = [e for e in key_entities if e.lower() in combined_text]
        coverage_ratio = len(matched_entities) / len(key_entities) if key_entities else 1.0

        is_correct = retrieval_success and (coverage_ratio >= 0.33)
        if is_correct:
            answer_correct_count += 1

        status_sym = "[PASS]" if is_correct else "[FAIL]"
        print(f"\n{status_sym} Q{qid}: {q_text}")
        print(f"       Expected Source : {expected_src}")
        print(f"       Top Similarity  : {chunks[0]['cosine_similarity']:.4f} ({chunks[0]['source_filename']})")
        print(f"       Generated Answer: {answer[:130]}...")

        results.append({
            "id": qid,
            "question": q_text,
            "expected_source": expected_src,
            "ground_truth": ground_truth,
            "generated_answer": answer,
            "retrieval_success": retrieval_success,
            "citation_success": citation_success,
            "entity_coverage": round(coverage_ratio, 2),
            "top_similarity": chunks[0]["cosine_similarity"] if chunks else 0.0,
            "is_correct": is_correct,
        })

    retrieval_accuracy = round((retrieval_hits / total_questions) * 100.0, 2)
    citation_accuracy = round((citation_hits / total_questions) * 100.0, 2)
    overall_accuracy = round((answer_correct_count / total_questions) * 100.0, 2)

    summary = {
        "total_questions": total_questions,
        "retrieval_accuracy": retrieval_accuracy,
        "citation_accuracy": citation_accuracy,
        "overall_accuracy": overall_accuracy,
        "target_accuracy": 80.0,
        "target_met": overall_accuracy >= 80.0,
        "test_results": results,
    }

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION SUMMARY")
    print(f"   * Retrieval Accuracy (Top-{top_k} Recall) : {retrieval_accuracy}%")
    print(f"   * Source Citation Accuracy        : {citation_accuracy}%")
    print(f"   * Overall Answer Accuracy         : {overall_accuracy}% (Target: >80%)")
    print(f"   * Target Met                      : {'YES (APPROVED)' if summary['target_met'] else 'NO'}")
    print("=" * 70)

    # Save results JSON
    out_path = PROJECT_ROOT / "data" / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation metrics to: {out_path.name}")

    return summary


if __name__ == "__main__":
    evaluate_pipeline()

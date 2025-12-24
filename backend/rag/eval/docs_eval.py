"""
RAG Evaluation Harness for Acme CRM docs.

Evaluates the RAG pipeline using:
- Context relevance: Did we retrieve the right documents?
- Answer relevance: Does the answer address the question?
- Groundedness: Is the answer grounded in the context (no hallucinations)?

Uses LLM-as-judge for relevance and groundedness scoring.

Usage:
    python -m backend.rag.eval.docs_eval
"""

import json
import time
from pathlib import Path
from typing import Optional

from backend.rag.retrieval.base import create_backend
from backend.rag.pipeline.docs import answer_question
from backend.rag.eval.models import EvalResult
from backend.rag.eval.judge import judge_response, compute_doc_recall


# =============================================================================
# Configuration
# =============================================================================

EVAL_QUESTIONS_PATH = Path(__file__).parent.parent / "eval_questions.json"


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_eval_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    """Load evaluation questions from JSON file."""
    with open(path) as f:
        return json.load(f)


def evaluate_question(
    question_data: dict,
    backend,
    verbose: bool = False,
) -> EvalResult:
    """
    Evaluate a single question through the RAG pipeline.
    
    Args:
        question_data: Dict with id, question, target_doc_ids
        backend: Initialized RetrievalBackend
        verbose: Print progress
        
    Returns:
        EvalResult with all metrics
    """
    question_id = question_data["id"]
    question = question_data["question"]
    target_doc_ids = question_data["target_doc_ids"]
    
    if verbose:
        print(f"\nEvaluating: {question_id}")
        print(f"  Question: {question[:60]}...")
    
    # Run RAG pipeline
    start_time = time.time()
    result = answer_question(question, backend, k=8, verbose=False)
    total_latency = (time.time() - start_time) * 1000
    
    # Build context string for judge
    context = "\n\n".join([
        f"[{c.doc_id}] {c.text[:500]}"
        for c in result["used_chunks"]
    ])
    
    # Compute doc recall
    doc_recall = compute_doc_recall(target_doc_ids, result["doc_ids_used"])
    
    if verbose:
        print(f"  Retrieved: {result['doc_ids_used']}")
        print(f"  Doc recall: {doc_recall:.2f}")
    
    # Judge the response
    judge_result = judge_response(
        question=question,
        context=context,
        answer=result["answer"],
        doc_ids=result["doc_ids_used"],
    )
    
    if verbose:
        print(f"  Judge: context={judge_result.context_relevance}, "
              f"answer={judge_result.answer_relevance}, "
              f"grounded={judge_result.groundedness}")
    
    return EvalResult(
        question_id=question_id,
        question=question,
        target_doc_ids=target_doc_ids,
        retrieved_doc_ids=result["doc_ids_used"],
        answer=result["answer"],
        judge_result=judge_result,
        doc_recall=doc_recall,
        latency_ms=total_latency,
        total_tokens=result["metrics"]["total_tokens"],
    )


def run_evaluation(
    questions: Optional[list[dict]] = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """
    Run full evaluation over all questions.
    
    Args:
        questions: List of question dicts (or load from file)
        verbose: Print progress
        
    Returns:
        List of EvalResult objects
    """
    if questions is None:
        questions = load_eval_questions()
    
    print("=" * 60)
    print("RAG Evaluation Harness")
    print("=" * 60)
    print(f"Evaluating {len(questions)} questions\n")
    
    # Initialize backend
    backend = create_backend()
    
    results = []
    for q in questions:
        result = evaluate_question(q, backend, verbose=verbose)
        results.append(result)
    
    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print summary statistics from evaluation results."""
    n = len(results)
    
    if n == 0:
        print("No results to summarize")
        return
    
    # Compute aggregates
    context_relevance = sum(r.judge_result.context_relevance for r in results) / n
    answer_relevance = sum(r.judge_result.answer_relevance for r in results) / n
    groundedness = sum(r.judge_result.groundedness for r in results) / n
    needs_review = sum(r.judge_result.needs_human_review for r in results) / n
    
    # RAG triad success (all three = 1)
    triad_success = sum(
        1 for r in results
        if r.judge_result.context_relevance == 1
        and r.judge_result.answer_relevance == 1
        and r.judge_result.groundedness == 1
    ) / n
    
    avg_doc_recall = sum(r.doc_recall for r in results) / n
    avg_latency = sum(r.latency_ms for r in results) / n
    total_tokens = sum(r.total_tokens for r in results)
    
    # Approximate cost (GPT-4.1-mini pricing)
    estimated_cost = (total_tokens * 0.8 * 0.40 + total_tokens * 0.2 * 1.60) / 1_000_000
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Questions evaluated':<25} {n:>10}")
    print(f"{'Context Relevance':<25} {context_relevance:>10.1%}")
    print(f"{'Answer Relevance':<25} {answer_relevance:>10.1%}")
    print(f"{'Groundedness':<25} {groundedness:>10.1%}")
    print(f"{'RAG Triad Success':<25} {triad_success:>10.1%}")
    print(f"{'Needs Human Review':<25} {needs_review:>10.1%}")
    print(f"{'Avg Doc Recall':<25} {avg_doc_recall:>10.1%}")
    print("-" * 40)
    print(f"{'Avg Latency (ms)':<25} {avg_latency:>10.0f}")
    print(f"{'Total Tokens':<25} {total_tokens:>10,}")
    print(f"{'Est. Cost ($)':<25} {estimated_cost:>10.4f}")
    
    # Per-question details
    print("\n" + "-" * 60)
    print("PER-QUESTION RESULTS")
    print("-" * 60)
    print(f"{'ID':<6} {'Ctx':<4} {'Ans':<4} {'Gnd':<4} {'Recall':<8} {'Latency':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r.question_id:<6} "
              f"{r.judge_result.context_relevance:<4} "
              f"{r.judge_result.answer_relevance:<4} "
              f"{r.judge_result.groundedness:<4} "
              f"{r.doc_recall:<8.1%} "
              f"{r.latency_ms:<10.0f}ms")
    
    # Failed questions
    failed = [r for r in results if r.judge_result.groundedness == 0 or r.judge_result.answer_relevance == 0]
    if failed:
        print("\n" + "-" * 60)
        print("QUESTIONS NEEDING ATTENTION")
        print("-" * 60)
        for r in failed:
            print(f"\n{r.question_id}: {r.question}")
            print(f"  Judge: {r.judge_result.explanation}")


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main():
    """Main entrypoint for evaluation."""
    results = run_evaluation(verbose=True)
    print_summary(results)
    
    # Save results to file
    output_path = Path("data/processed/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = [r.model_dump() for r in results]
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

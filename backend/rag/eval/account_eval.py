"""
Evaluation harness for Account-aware RAG (MVP2).

Evaluates account-scoped RAG using:
- RAG triad metrics (context relevance, answer relevance, groundedness)
- Privacy leakage detection (retrieved chunks from wrong company)
- Latency and cost tracking

Usage:
    python -m backend.rag.eval.account_eval
"""

import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from qdrant_client import QdrantClient

from backend.rag.config import get_config
from backend.rag.ingest.text_builder import find_csv_dir
from backend.rag.ingest.private_text import ingest_private_texts
from backend.rag.pipeline.account import answer_account_question, load_companies_df
from backend.rag.eval.models import AccountEvalResult
from backend.rag.eval.judge import judge_account_response, check_privacy_leakage


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_PATH = Path("data/processed/eval_account_results.json")
NUM_QUESTIONS_PER_COMPANY = 3
NUM_COMPANIES = 4
RANDOM_SEED = 42


# =============================================================================
# Question Generation
# =============================================================================

def generate_eval_questions(seed: int = RANDOM_SEED) -> list[dict]:
    """
    Generate evaluation questions from actual CSV data.
    
    Returns list of dicts with:
        - id: question ID
        - company_id: target company
        - company_name: company name
        - question: the question text
        - question_type: category of question
    """
    random.seed(seed)
    
    # Load companies
    df = load_companies_df()
    
    # Filter to active companies with data
    active = df[df["status"].isin(["Active", "Trial"])]
    
    # Select companies (deterministic)
    selected = active.head(NUM_COMPANIES)
    
    questions = []
    q_id = 1
    
    # Question templates
    templates = [
        {
            "type": "history_summary",
            "template": "Summarize the recent interactions and history with {company_name}. What calls, emails, or meetings have occurred?",
        },
        {
            "type": "opportunity_status",
            "template": "What are the current opportunities for {company_name}? What stages are they in and what are the risks or next steps?",
        },
        {
            "type": "attachments",
            "template": "What documents or attachments are associated with {company_name}'s opportunities? Summarize what they contain.",
        },
    ]
    
    for _, company in selected.iterrows():
        company_id = company["company_id"]
        company_name = company["name"]
        
        for tmpl in templates[:NUM_QUESTIONS_PER_COMPANY]:
            question = tmpl["template"].format(
                company_name=company_name,
                company_id=company_id,
            )
            
            questions.append({
                "id": f"acct_q{q_id}",
                "company_id": company_id,
                "company_name": company_name,
                "question": question,
                "question_type": tmpl["type"],
            })
            q_id += 1
    
    return questions


# =============================================================================
# Evaluation
# =============================================================================

def ensure_private_collection_exists() -> None:
    """Ensure private Qdrant collection exists, create if not."""
    config = get_config()
    qdrant = QdrantClient(path=str(config.qdrant_path))
    
    if not qdrant.collection_exists(config.private_collection_name):
        print(f"Collection '{config.private_collection_name}' not found, creating...")
        ingest_private_texts(recreate=True)
    else:
        info = qdrant.get_collection(config.private_collection_name)
        if info.points_count == 0:
            print(f"Collection '{config.private_collection_name}' is empty, rebuilding...")
            ingest_private_texts(recreate=True)
        else:
            print(f"Using existing collection with {info.points_count} points")


def evaluate_question(
    question_data: dict,
    verbose: bool = False,
) -> AccountEvalResult:
    """Evaluate a single account question."""
    q_id = question_data["id"]
    company_id = question_data["company_id"]
    company_name = question_data["company_name"]
    question = question_data["question"]
    q_type = question_data["question_type"]
    
    if verbose:
        print(f"\n  {q_id}: {question[:50]}...")
    
    # Run RAG
    result = answer_account_question(
        question=question,
        company_id=company_id,
        verbose=False,
    )
    
    # Check privacy leakage
    leakage, leaked_ids = check_privacy_leakage(
        company_id, result["raw_private_hits"]
    )
    
    if verbose and leakage:
        print(f"    WARNING: Privacy leakage! Leaked companies: {leaked_ids}")
    
    # Build context string for judge
    context_parts = []
    for hit in result["raw_private_hits"][:5]:
        context_parts.append(f"[{hit['id']}] {hit['text_preview']}")
    context_str = "\n".join(context_parts)
    
    # Judge
    sources = [s["id"] for s in result["sources"]]
    judge = judge_account_response(
        company_id=company_id,
        company_name=company_name,
        question=question,
        context=context_str,
        answer=result["answer"],
        sources=sources,
    )
    
    if verbose:
        print(f"    ctx={judge.context_relevance} ans={judge.answer_relevance} "
              f"gnd={judge.groundedness} leak={leakage}")
    
    return AccountEvalResult(
        question_id=q_id,
        company_id=company_id,
        company_name=company_name,
        question=question,
        question_type=q_type,
        answer=result["answer"],
        judge_result=judge,
        privacy_leakage=leakage,
        leaked_company_ids=leaked_ids,
        num_private_hits=len(result["raw_private_hits"]),
        latency_ms=result["meta"]["latency_ms"],
        total_tokens=result["meta"]["total_tokens"],
        estimated_cost=result["meta"]["estimated_cost"],
    )


def run_evaluation(verbose: bool = True) -> list[AccountEvalResult]:
    """Run full evaluation."""
    print("=" * 60)
    print("Account RAG Evaluation (MVP2)")
    print("=" * 60)
    
    # Ensure collection exists
    ensure_private_collection_exists()
    
    # Generate questions
    questions = generate_eval_questions()
    print(f"\nEvaluating {len(questions)} questions across "
          f"{len(set(q['company_id'] for q in questions))} companies")
    
    results = []
    for q in questions:
        result = evaluate_question(q, verbose=verbose)
        results.append(result)
    
    return results


def print_summary(results: list[AccountEvalResult]) -> None:
    """Print evaluation summary."""
    n = len(results)
    if n == 0:
        print("No results")
        return
    
    # Triad metrics
    ctx_rel = sum(r.judge_result.context_relevance for r in results) / n
    ans_rel = sum(r.judge_result.answer_relevance for r in results) / n
    grounded = sum(r.judge_result.groundedness for r in results) / n
    
    triad_success = sum(
        1 for r in results
        if r.judge_result.context_relevance == 1
        and r.judge_result.answer_relevance == 1
        and r.judge_result.groundedness == 1
    ) / n
    
    # Privacy leakage
    leakage_count = sum(r.privacy_leakage for r in results)
    leakage_rate = leakage_count / n
    
    # Latency and cost
    avg_latency = sum(r.latency_ms for r in results) / n
    total_tokens = sum(r.total_tokens for r in results)
    total_cost = sum(r.estimated_cost for r in results)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 40)
    print(f"{'Questions evaluated':<25} {n:>10}")
    print(f"{'Context Relevance':<25} {ctx_rel:>10.1%}")
    print(f"{'Answer Relevance':<25} {ans_rel:>10.1%}")
    print(f"{'Groundedness':<25} {grounded:>10.1%}")
    print(f"{'RAG Triad Success':<25} {triad_success:>10.1%}")
    print("-" * 40)
    print(f"{'Privacy Leakage Rate':<25} {leakage_rate:>10.1%}")
    print(f"{'Leaked Questions':<25} {leakage_count:>10}")
    print("-" * 40)
    print(f"{'Avg Latency (ms)':<25} {avg_latency:>10.0f}")
    print(f"{'Total Tokens':<25} {total_tokens:>10,}")
    print(f"{'Total Cost ($)':<25} {total_cost:>10.4f}")
    
    # Per-company breakdown
    print("\n" + "-" * 60)
    print("PER-COMPANY RESULTS")
    print("-" * 60)
    
    by_company = {}
    for r in results:
        cid = r.company_id
        if cid not in by_company:
            by_company[cid] = {"results": [], "name": r.company_name}
        by_company[cid]["results"].append(r)
    
    print(f"{'Company':<25} {'Triad':<8} {'Leak':<8} {'Latency':<10}")
    print("-" * 60)
    
    for cid, data in sorted(by_company.items()):
        company_results = data["results"]
        cn = len(company_results)
        
        company_triad = sum(
            1 for r in company_results
            if r.judge_result.context_relevance == 1
            and r.judge_result.answer_relevance == 1
            and r.judge_result.groundedness == 1
        ) / cn
        
        company_leak = sum(r.privacy_leakage for r in company_results) / cn
        company_latency = sum(r.latency_ms for r in company_results) / cn
        
        print(f"{data['name'][:24]:<25} {company_triad:<8.0%} {company_leak:<8.0%} {company_latency:<10.0f}ms")
    
    # Questions with issues
    issues = [r for r in results if r.judge_result.groundedness == 0 or r.privacy_leakage == 1]
    if issues:
        print("\n" + "-" * 60)
        print("QUESTIONS WITH ISSUES")
        print("-" * 60)
        for r in issues:
            flags = []
            if r.judge_result.groundedness == 0:
                flags.append("NOT_GROUNDED")
            if r.privacy_leakage == 1:
                flags.append(f"LEAKED:{r.leaked_company_ids}")
            print(f"\n{r.question_id} [{', '.join(flags)}]")
            print(f"  Company: {r.company_name}")
            print(f"  Question: {r.question[:60]}...")
            print(f"  Judge: {r.judge_result.explanation[:80]}...")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entrypoint."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Account RAG (MVP2)")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_evaluation(verbose=args.verbose)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = [r.model_dump() for r in results]
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

"""
LLM-as-judge evaluation functions.

Provides functions to evaluate RAG responses using LLM as a judge.
"""

import json
import re
import logging
from typing import Optional

from backend.rag.eval.models import JudgeResult
from backend.rag.prompts import (
    EVAL_JUDGE_SYSTEM,
    EVAL_JUDGE_PROMPT,
    ACCOUNT_EVAL_JUDGE_SYSTEM,
    ACCOUNT_EVAL_JUDGE_PROMPT,
)
from backend.common.llm_client import call_llm


logger = logging.getLogger(__name__)


def judge_response(
    question: str,
    context: str,
    answer: str,
    doc_ids: list[str],
) -> JudgeResult:
    """
    Use LLM to judge the quality of a documentation RAG response.
    
    Args:
        question: The original question
        context: The retrieved context
        answer: The generated answer
        doc_ids: List of doc_ids in the context
        
    Returns:
        JudgeResult with scores and explanation
    """
    prompt = EVAL_JUDGE_PROMPT.format(
        question=question,
        doc_ids=", ".join(doc_ids),
        context=context[:2000],  # Truncate for judge
        answer=answer,
    )
    
    try:
        response = call_llm(
            prompt=prompt,
            system_prompt=EVAL_JUDGE_SYSTEM,
            model="gpt-4.1-mini",
            max_tokens=500,
        )
        
        # Parse JSON response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result_dict = json.loads(json_match.group())
        else:
            result_dict = json.loads(response)
        
        return JudgeResult(
            context_relevance=int(result_dict.get("context_relevance", 0)),
            answer_relevance=int(result_dict.get("answer_relevance", 0)),
            groundedness=int(result_dict.get("groundedness", 0)),
            needs_human_review=int(result_dict.get("needs_human_review", 1)),
            confidence=float(result_dict.get("confidence", 0.5)),
            explanation=str(result_dict.get("explanation", "No explanation")),
        )
    
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return JudgeResult(
            context_relevance=0,
            answer_relevance=0,
            groundedness=0,
            needs_human_review=1,
            confidence=0.0,
            explanation=f"Judge error: {str(e)}",
        )


def judge_account_response(
    company_id: str,
    company_name: str,
    question: str,
    context: str,
    answer: str,
    sources: list[str],
) -> JudgeResult:
    """
    Judge an account RAG response.
    
    Args:
        company_id: The target company ID
        company_name: The target company name
        question: The original question
        context: The retrieved context
        answer: The generated answer
        sources: List of source IDs
        
    Returns:
        JudgeResult with scores and explanation
    """
    prompt = ACCOUNT_EVAL_JUDGE_PROMPT.format(
        company_id=company_id,
        company_name=company_name,
        question=question,
        context=context[:2500],  # Truncate
        answer=answer,
    )
    
    try:
        response = call_llm(
            prompt=prompt,
            system_prompt=ACCOUNT_EVAL_JUDGE_SYSTEM,
            model="gpt-4.1-mini",
            max_tokens=300,
        )
        
        # Parse JSON
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(response)
        
        return JudgeResult(
            context_relevance=int(result.get("context_relevance", 0)),
            answer_relevance=int(result.get("answer_relevance", 0)),
            groundedness=int(result.get("groundedness", 0)),
            needs_human_review=int(result.get("needs_human_review", 1)),
            explanation=str(result.get("explanation", "")),
        )
    
    except Exception as e:
        logger.warning(f"Judge failed: {e}")
        return JudgeResult(
            context_relevance=0,
            answer_relevance=0,
            groundedness=0,
            needs_human_review=1,
            explanation=f"Judge error: {e}",
        )


def check_privacy_leakage(
    target_company_id: str,
    raw_hits: list[dict],
) -> tuple[int, list[str]]:
    """
    Check if any retrieved chunks belong to a different company.
    
    Args:
        target_company_id: Expected company ID
        raw_hits: List of raw hit dictionaries with company_id
        
    Returns:
        Tuple of (leakage_flag, list_of_leaked_company_ids)
    """
    leaked = []
    for hit in raw_hits:
        hit_company = hit.get("company_id", "")
        if hit_company and hit_company != target_company_id:
            leaked.append(hit_company)
    
    leakage = 1 if leaked else 0
    return leakage, list(set(leaked))


def compute_doc_recall(target_doc_ids: list[str], retrieved_doc_ids: list[str]) -> float:
    """
    Compute recall of target documents in retrieved documents.
    
    Args:
        target_doc_ids: Expected document IDs
        retrieved_doc_ids: Actually retrieved document IDs
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if not target_doc_ids:
        return 1.0
    
    target_set = set(target_doc_ids)
    retrieved_set = set(retrieved_doc_ids)
    
    hits = len(target_set & retrieved_set)
    return hits / len(target_set)

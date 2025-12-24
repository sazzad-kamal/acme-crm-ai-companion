"""
Evaluation data models.

Defines the result structures for RAG evaluation.
"""

from pydantic import BaseModel


class JudgeResult(BaseModel):
    """Result from LLM judge evaluation."""
    context_relevance: int  # 0 or 1
    answer_relevance: int   # 0 or 1
    groundedness: int       # 0 or 1
    needs_human_review: int # 0 or 1
    confidence: float = 0.5 # 0.0 to 1.0
    explanation: str = ""


class EvalResult(BaseModel):
    """Complete evaluation result for a single documentation question."""
    question_id: str
    question: str
    target_doc_ids: list[str]
    retrieved_doc_ids: list[str]
    answer: str
    judge_result: JudgeResult
    doc_recall: float  # What fraction of target docs were retrieved
    latency_ms: float
    total_tokens: int


class AccountEvalResult(BaseModel):
    """Evaluation result for a single account question."""
    question_id: str
    company_id: str
    company_name: str
    question: str
    question_type: str
    answer: str
    judge_result: JudgeResult
    privacy_leakage: int  # 1 if any retrieved chunk from wrong company
    leaked_company_ids: list[str]
    num_private_hits: int
    latency_ms: float
    total_tokens: int
    estimated_cost: float

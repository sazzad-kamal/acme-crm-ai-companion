"""
Data models for agent evaluation results.
"""

from typing import Optional
from pydantic import BaseModel


class ToolEvalResult(BaseModel):
    """Result from evaluating a single tool call."""
    tool_name: str
    test_case_id: str
    input_params: dict
    expected_found: bool
    actual_found: bool
    expected_count: Optional[int] = None
    actual_count: Optional[int] = None
    expected_company_id: Optional[str] = None
    actual_company_id: Optional[str] = None
    data_correct: bool
    sources_present: bool
    error: Optional[str] = None
    latency_ms: float = 0.0


class RouterEvalResult(BaseModel):
    """Result from evaluating router mode selection."""
    test_case_id: str
    question: str
    expected_mode: str
    actual_mode: str
    expected_company_id: Optional[str] = None
    actual_company_id: Optional[str] = None
    mode_correct: bool
    company_correct: bool
    intent_expected: Optional[str] = None
    intent_actual: Optional[str] = None


class E2EEvalResult(BaseModel):
    """Result from end-to-end agent evaluation."""
    test_case_id: str
    question: str
    category: str
    expected_mode: str
    actual_mode: str
    expected_tools: list[str]
    actual_tools: list[str]
    answer: str
    answer_relevance: int  # 0 or 1
    answer_grounded: int   # 0 or 1
    tool_selection_correct: bool
    has_sources: bool
    latency_ms: float
    total_tokens: int
    error: Optional[str] = None
    judge_explanation: str = ""


class ToolEvalSummary(BaseModel):
    """Summary statistics for tool evaluation."""
    total_tests: int
    passed: int
    failed: int
    accuracy: float
    by_tool: dict[str, dict]  # tool_name -> {passed, failed, accuracy}


class RouterEvalSummary(BaseModel):
    """Summary statistics for router evaluation."""
    total_tests: int
    mode_accuracy: float
    company_extraction_accuracy: float
    by_mode: dict[str, dict]  # mode -> {expected, correct, accuracy}


class E2EEvalSummary(BaseModel):
    """Summary statistics for end-to-end evaluation."""
    total_tests: int
    answer_relevance_rate: float
    groundedness_rate: float
    tool_selection_accuracy: float
    avg_latency_ms: float
    by_category: dict[str, dict]


class AgentEvalSummary(BaseModel):
    """Complete agent evaluation summary."""
    tool_eval: Optional[ToolEvalSummary] = None
    router_eval: Optional[RouterEvalSummary] = None
    e2e_eval: Optional[E2EEvalSummary] = None
    overall_score: float = 0.0  # Weighted composite score

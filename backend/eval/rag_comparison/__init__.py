"""RAG retrieval strategy comparison evaluation module.

Compares multiple retrieval configurations (vector, BM25, hybrid, reranked)
against the same questions using RAGAS metrics to identify the optimal strategy.
"""

from backend.eval.rag_comparison.configs import DEFAULT_CONFIGS, RetrievalConfig, build_query_engine
from backend.eval.rag_comparison.models import ComparisonResults, ConfigCaseResult, ConfigResults
from backend.eval.rag_comparison.output import print_comparison_report, save_comparison_results
from backend.eval.rag_comparison.runner import run_rag_comparison

__all__ = [
    "DEFAULT_CONFIGS",
    "RetrievalConfig",
    "build_query_engine",
    "ComparisonResults",
    "ConfigCaseResult",
    "ConfigResults",
    "print_comparison_report",
    "save_comparison_results",
    "run_rag_comparison",
]

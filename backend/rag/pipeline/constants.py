"""
Constants for the pipeline module.
"""

# LLM Configuration
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 1024
ANSWER_MAX_TOKENS = 800

# Context Building
MAX_CONTEXT_TOKENS = 3000
MAX_CHUNKS_PER_DOC = 3
MAX_CHUNKS_PER_TYPE = 4
MIN_BM25_SCORE_RATIO = 0.1

"""
Shared LLM client helper for all RAG components.

Uses OpenAI API via the official Python client.
Requires OPENAI_API_KEY environment variable.
"""

import os
import time
from typing import Optional

from openai import OpenAI


# Global client instance (lazy initialization)
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it before using the LLM client."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """
    Call an OpenAI chat model and return the assistant's response.
    
    Args:
        prompt: The user message / prompt
        system_prompt: Optional system message to set context
        model: The model to use (default: gpt-4.1-mini)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens in response
        
    Returns:
        The assistant's message content as a string
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set
        openai.APIError: If the API call fails
    """
    client = _get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    return response.choices[0].message.content or ""


def call_llm_with_metrics(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """
    Call an OpenAI chat model and return response with metrics.
    
    Returns a dict with:
        - response: The assistant's message content
        - latency_ms: Time taken for the call
        - prompt_tokens: Approximate prompt token count
        - completion_tokens: Approximate completion token count
        - model: The model used
    """
    client = _get_client()
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "response": response.choices[0].message.content or "",
        "latency_ms": latency_ms,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
        "model": model,
    }

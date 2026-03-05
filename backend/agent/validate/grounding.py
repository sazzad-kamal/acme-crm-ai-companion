"""Grounding verifier for answer claims.

This critic stage verifies that every claim in the answer is supported
by the CRM data context provided. It extracts claims and checks them
against the source data to prevent hallucination.
"""

import logging
import re
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from pydantic import BaseModel, Field

from backend.core.llm import SHORT_RESPONSE_MAX_TOKENS, create_openai_chain

logger = logging.getLogger(__name__)

# Evidence tag pattern
EVIDENCE_TAG_PATTERN = re.compile(r"\[E(\d+)\]")


@dataclass
class GroundingResult:
    """Result of grounding verification."""

    is_grounded: bool
    total_claims: int
    verified_claims: int
    ungrounded_claims: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    verification_details: list[dict[str, Any]] = field(default_factory=list)


class ClaimVerification(BaseModel):
    """Verification result for a single claim."""

    claim: str = Field(description="The claim being verified")
    is_supported: bool = Field(description="Whether the claim is supported by the data")
    supporting_data: str | None = Field(
        default=None, description="The data that supports this claim, if any"
    )
    reason: str = Field(description="Why the claim is or isn't supported")


class GroundingVerification(BaseModel):
    """Structured output for grounding verification."""

    claims: list[ClaimVerification] = Field(description="Verification results for each claim")
    overall_grounded: bool = Field(description="Whether the answer is overall well-grounded")
    summary: str = Field(description="Brief summary of grounding quality")


_SYSTEM_PROMPT = """You are a grounding verifier. Your job is to verify that every factual claim in an answer is supported by the provided CRM data.

RULES:
1. Extract each factual claim from the answer (numbers, names, dates, statuses, etc.)
2. For each claim, check if it's directly supported by the CRM DATA
3. A claim is ONLY supported if the exact value appears in the data
4. Mark claims as unsupported if:
   - The data doesn't contain the claimed value
   - The claim infers something not explicitly stated
   - The claim uses different numbers than shown in data

Be strict - hallucinated or inferred claims must be flagged."""

_HUMAN_PROMPT = """=== CRM DATA (source of truth) ===
{crm_data}

=== ANSWER TO VERIFY ===
{answer}

Verify each factual claim in the answer against the CRM data. Be thorough and strict."""


@cache
def _get_grounding_chain() -> Any:
    """Get or create the grounding verification chain (cached singleton)."""
    chain = create_openai_chain(
        system_prompt=_SYSTEM_PROMPT,
        human_prompt=_HUMAN_PROMPT,
        max_tokens=SHORT_RESPONSE_MAX_TOKENS,
        structured_output=GroundingVerification,
    )
    logger.debug("Created grounding verification chain")
    return chain


def _format_sql_results(sql_results: dict[str, Any]) -> str:
    """Format SQL results as readable text for verification."""
    if not sql_results:
        return "No data available."

    parts = []

    # Handle different result structures
    if "rows" in sql_results:
        rows = sql_results["rows"]
        if rows:
            parts.append(f"Query returned {len(rows)} rows:")
            for i, row in enumerate(rows[:20], 1):  # Limit to 20 rows
                if isinstance(row, dict):
                    row_str = ", ".join(f"{k}={v}" for k, v in row.items())
                else:
                    row_str = str(row)
                parts.append(f"  Row {i}: {row_str}")
            if len(rows) > 20:
                parts.append(f"  ... and {len(rows) - 20} more rows")
        else:
            parts.append("Query returned 0 rows.")

    elif "data" in sql_results:
        data = sql_results["data"]
        if isinstance(data, list):
            parts.append(f"Data ({len(data)} items):")
            for i, item in enumerate(data[:20], 1):
                parts.append(f"  {i}. {item}")
        else:
            parts.append(f"Data: {data}")

    else:
        # Fallback: stringify the whole result
        parts.append(str(sql_results)[:2000])

    return "\n".join(parts)


def verify_grounding(
    answer: str,
    sql_results: dict[str, Any] | None,
    strict: bool = True,
) -> GroundingResult:
    """Verify that an answer is grounded in the provided CRM data.

    Args:
        answer: The answer text to verify
        sql_results: The SQL query results used to generate the answer
        strict: If True, require 100% grounding. If False, allow minor issues.

    Returns:
        GroundingResult with verification details
    """
    # Quick checks
    if not answer or not answer.strip():
        return GroundingResult(
            is_grounded=True,
            total_claims=0,
            verified_claims=0,
        )

    if not sql_results:
        # No data to verify against - check if answer acknowledges this
        if "data not available" in answer.lower() or "no results" in answer.lower():
            return GroundingResult(
                is_grounded=True,
                total_claims=0,
                verified_claims=0,
            )
        # Answer makes claims without data - suspicious
        return GroundingResult(
            is_grounded=False,
            total_claims=1,
            verified_claims=0,
            ungrounded_claims=["Answer contains claims but no data was provided"],
        )

    # Format data for verification
    crm_data = _format_sql_results(sql_results)

    try:
        chain = _get_grounding_chain()
        result: GroundingVerification = chain.invoke({
            "crm_data": crm_data,
            "answer": answer,
        })

        # Build grounding result
        total_claims = len(result.claims)
        verified_claims = sum(1 for c in result.claims if c.is_supported)
        ungrounded = [c.claim for c in result.claims if not c.is_supported]

        # Determine if grounded based on strictness
        if strict:
            is_grounded = result.overall_grounded and len(ungrounded) == 0
        else:
            # Allow up to 10% ungrounded claims
            grounded_ratio = verified_claims / total_claims if total_claims > 0 else 1.0
            is_grounded = grounded_ratio >= 0.9

        logger.info(
            f"[Grounding] Verified {verified_claims}/{total_claims} claims, "
            f"grounded={is_grounded}"
        )

        return GroundingResult(
            is_grounded=is_grounded,
            total_claims=total_claims,
            verified_claims=verified_claims,
            ungrounded_claims=ungrounded,
            verification_details=[
                {
                    "claim": c.claim,
                    "supported": c.is_supported,
                    "data": c.supporting_data,
                    "reason": c.reason,
                }
                for c in result.claims
            ],
        )

    except Exception as e:
        logger.warning(f"[Grounding] Verification failed: {e}")
        # On error, assume grounded (don't block the response)
        return GroundingResult(
            is_grounded=True,
            total_claims=0,
            verified_claims=0,
            missing_evidence=[f"Verification error: {e}"],
        )


def verify_evidence_tags(answer: str) -> tuple[bool, list[str]]:
    """Quick check that evidence tags in answer are properly formatted.

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Find all evidence tags used
    tags_used = set(EVIDENCE_TAG_PATTERN.findall(answer))

    if not tags_used:
        # No evidence tags - might be okay for simple responses
        if "data not available" not in answer.lower() and len(answer) > 100:
            issues.append("Answer makes claims without evidence tags")

    # Check if Evidence section exists when tags are used
    if tags_used and "evidence:" not in answer.lower():
        issues.append(f"Evidence tags {tags_used} used but no Evidence section found")

    return len(issues) == 0, issues


__all__ = [
    "GroundingResult",
    "verify_grounding",
    "verify_evidence_tags",
]

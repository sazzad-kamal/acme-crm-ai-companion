"""
Agent orchestration for answering CRM questions.

Coordinates:
1. Router - determine mode and extract parameters (LLM or heuristic)
2. Data tools - fetch CRM data
3. Docs RAG - fetch documentation (reuses backend.rag)
4. LLM - generate grounded answer

Enhanced features:
- Structured logging throughout
- Retry logic for LLM calls
- LLM-based routing option
- Dynamic progress tracking
- Audit logging
"""

import logging
from typing import Optional

from backend.agent.config import get_config
from backend.agent.schemas import (
    ChatResponse, Source, Step, RawData, MetaInfo, RouterResult
)
from backend.agent.llm_router import route_question
from backend.agent.audit import AgentAuditLogger
from backend.agent.progress import AgentProgress
from backend.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    COMPANY_NOT_FOUND_PROMPT,
    DATA_ANSWER_PROMPT,
)
from backend.agent.formatters import (
    format_company_section,
    format_activities_section,
    format_history_section,
    format_pipeline_section,
    format_renewals_section,
    format_docs_section,
)
from backend.agent.llm_helpers import (
    call_llm,
    call_docs_rag,
    generate_follow_up_suggestions,
)
from backend.agent.tools import (
    tool_company_lookup,
    tool_recent_activity,
    tool_recent_history,
    tool_pipeline,
    tool_upcoming_renewals,
)


# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Main Agent Function
# =============================================================================

def answer_question(
    question: str,
    mode: str = "auto",
    company_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> dict:
    """
    Answer a CRM question using the agent pipeline.
    
    Args:
        question: The user's question
        mode: Mode override ("auto", "docs", "data", "data+docs")
        company_id: Pre-specified company ID
        session_id: Optional session ID (for future stateful use)
        user_id: Optional user ID (for future personalization)
        
    Returns:
        Dict matching ChatResponse schema:
        {
            "answer": str,
            "sources": list[Source],
            "steps": list[Step],
            "raw_data": RawData,
            "meta": MetaInfo
        }
    """
    config = get_config()
    progress = AgentProgress()
    audit = AgentAuditLogger()
    
    logger.info(f"Processing question: {question[:100]}...")
    
    sources: list[Source] = []
    raw_data = {
        "companies": [],
        "activities": [],
        "opportunities": [],
        "history": [],
        "renewals": [],
        "pipeline_summary": None,
    }
    
    # -------------------------------------------------------------------------
    # Step 1: Router (LLM or heuristic based on config)
    # -------------------------------------------------------------------------
    progress.add_step("router", "Understanding your question")
    
    try:
        router_result = route_question(question, mode=mode, company_id=company_id)
        logger.info(
            f"Routing complete: mode={router_result.mode_used}, "
            f"company={router_result.company_id}, intent={router_result.intent}"
        )
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        progress.add_step("router_error", f"Routing failed: {e}", status="error")
        return _build_error_response(
            f"Failed to understand question: {e}",
            progress, sources, raw_data, "unknown"
        )
    
    mode_used = router_result.mode_used
    resolved_company_id = router_result.company_id
    days = router_result.days
    intent = router_result.intent
    
    # -------------------------------------------------------------------------
    # Step 2: Data gathering (if mode includes data)
    # -------------------------------------------------------------------------
    company_data = None
    activities_data = None
    history_data = None
    pipeline_data = None
    renewals_data = None
    
    if "data" in mode_used:
        progress.add_step("data", "Fetching CRM data")
        
        try:
            # Handle renewals intent (no specific company)
            if intent == "renewals" and not resolved_company_id:
                logger.debug(f"Fetching renewals for next {days} days")
                renewals_result = tool_upcoming_renewals(days=days)
                renewals_data = renewals_result.data
                sources.extend(renewals_result.sources)
                raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
            
            # Handle company-specific queries
            elif resolved_company_id or router_result.company_name_query:
                # Lookup company
                query = resolved_company_id or router_result.company_name_query
                logger.debug(f"Looking up company: {query}")
                company_result = tool_company_lookup(query or "")
                
                if company_result.data.get("found"):
                    company_data = company_result.data
                    sources.extend(company_result.sources)
                    resolved_company_id = company_data["company"]["company_id"]
                    raw_data["companies"] = [company_data["company"]]
                    
                    logger.debug(f"Fetching data for company {resolved_company_id}")
                    
                    # Get activities
                    activities_result = tool_recent_activity(resolved_company_id, days=days)
                    activities_data = activities_result.data
                    sources.extend(activities_result.sources)
                    raw_data["activities"] = activities_data.get("activities", [])[:8]
                    
                    # Get history
                    history_result = tool_recent_history(resolved_company_id, days=days)
                    history_data = history_result.data
                    sources.extend(history_result.sources)
                    raw_data["history"] = history_data.get("history", [])[:8]
                    
                    # Get pipeline
                    pipeline_result = tool_pipeline(resolved_company_id)
                    pipeline_data = pipeline_result.data
                    sources.extend(pipeline_result.sources)
                    raw_data["opportunities"] = pipeline_data.get("opportunities", [])[:8]
                    raw_data["pipeline_summary"] = pipeline_data.get("summary")
                    
                    logger.info(
                        f"Data fetched: activities={len(activities_data.get('activities', []))}, "
                        f"history={len(history_data.get('history', []))}, "
                        f"opps={len(pipeline_data.get('opportunities', []))}"
                    )
                
                else:
                    # Company not found - we'll handle this in the answer step
                    company_data = company_result.data
                    logger.info(f"Company not found: {query}")
            
            else:
                # No company specified - get general renewals
                logger.debug("No company specified, fetching general renewals")
                renewals_result = tool_upcoming_renewals(days=days)
                renewals_data = renewals_result.data
                sources.extend(renewals_result.sources)
                raw_data["renewals"] = renewals_data.get("renewals", [])[:8]
                
        except Exception as e:
            logger.error(f"Data gathering failed: {e}")
            progress.add_step("data_error", f"Error: {str(e)[:50]}", status="error")
    
    # -------------------------------------------------------------------------
    # Step 3: Docs RAG (if mode includes docs)
    # -------------------------------------------------------------------------
    docs_answer = ""
    docs_sources: list[Source] = []
    
    if "docs" in mode_used:
        progress.add_step("docs", "Checking documentation")
        
        try:
            docs_answer, docs_sources = call_docs_rag(question)
            sources.extend(docs_sources)
            logger.info(f"Docs RAG returned {len(docs_sources)} sources")
        except Exception as e:
            logger.error(f"Docs RAG failed: {e}")
            progress.add_step("docs_error", f"Error: {str(e)[:50]}", status="error")
    else:
        progress.add_step("docs", "Skipped (data-only query)", status="skipped")
    
    # -------------------------------------------------------------------------
    # Step 4: Generate answer (LLM)
    # -------------------------------------------------------------------------
    progress.add_step("answer", "Synthesizing answer")
    
    try:
        # Handle company not found case
        if company_data and not company_data.get("found"):
            matches_text = "\n".join([
                f"- {m.get('name')} ({m.get('company_id')})"
                for m in company_data.get("close_matches", [])[:5]
            ]) or "No similar companies found."
            
            prompt = COMPANY_NOT_FOUND_PROMPT.format(
                question=question,
                query=company_data.get("query", "unknown"),
                matches=matches_text,
            )
            
            answer, llm_latency = call_llm(prompt, AGENT_SYSTEM_PROMPT)
        
        else:
            # Build context sections
            company_section = format_company_section(company_data)
            activities_section = format_activities_section(activities_data)
            history_section = format_history_section(history_data)
            pipeline_section = format_pipeline_section(pipeline_data)
            renewals_section = format_renewals_section(renewals_data)
            docs_section = format_docs_section(docs_answer)
            
            prompt = DATA_ANSWER_PROMPT.format(
                question=question,
                company_section=company_section,
                activities_section=activities_section,
                history_section=history_section,
                pipeline_section=pipeline_section,
                renewals_section=renewals_section,
                docs_section=docs_section,
            )
            
            answer, llm_latency = call_llm(prompt, AGENT_SYSTEM_PROMPT)
        
        logger.info(f"Answer synthesized in {llm_latency}ms")
    
    except Exception as e:
        logger.error(f"Answer synthesis failed: {e}")
        progress.add_step("answer_error", f"Error: {str(e)[:50]}", status="error")
        answer = f"I encountered an error generating the answer: {str(e)}"
    
    # -------------------------------------------------------------------------
    # Audit Logging
    # -------------------------------------------------------------------------
    audit.log_query(
        question=question,
        mode_used=mode_used,
        company_id=resolved_company_id,
        latency_ms=progress.get_elapsed_ms(),
        source_count=len(sources),
        user_id=user_id,
        session_id=session_id,
    )
    
    # -------------------------------------------------------------------------
    # Step 5: Generate follow-up suggestions (if enabled)
    # -------------------------------------------------------------------------
    follow_up_suggestions: list[str] = []
    
    if config.enable_follow_up_suggestions:
        progress.add_step("follow_ups", "Generating suggestions")
        try:
            follow_up_suggestions = generate_follow_up_suggestions(
                question=question,
                mode=mode_used,
                company_id=resolved_company_id,
            )
            logger.debug(f"Generated {len(follow_up_suggestions)} follow-up suggestions")
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            # Non-critical - continue without suggestions
    
    # -------------------------------------------------------------------------
    # Build response
    # -------------------------------------------------------------------------
    return {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "steps": progress.to_list(),
        "raw_data": raw_data,
        "follow_up_suggestions": follow_up_suggestions,
        "meta": {
            "mode_used": mode_used,
            "latency_ms": progress.get_elapsed_ms(),
            "company_id": resolved_company_id,
            "days": days,
        }
    }


def _build_error_response(
    error_msg: str,
    progress: AgentProgress,
    sources: list[Source],
    raw_data: dict,
    mode_used: str,
) -> dict:
    """Build an error response."""
    return {
        "answer": f"I'm sorry, I encountered an error: {error_msg}",
        "sources": [s.model_dump() for s in sources],
        "steps": progress.to_list(),
        "raw_data": raw_data,
        "follow_up_suggestions": [],
        "meta": {
            "mode_used": mode_used,
            "latency_ms": progress.get_elapsed_ms(),
        }
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import os
    
    # Enable logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    
    # Enable mock mode for testing
    os.environ["MOCK_LLM"] = "1"
    
    print("Testing Agent (MOCK_LLM=1)")
    print("=" * 60)
    
    questions = [
        "What's going on with Acme Manufacturing in the last 90 days?",
        "Which accounts have upcoming renewals in the next 90 days?",
        "How do I create a new opportunity?",
    ]
    
    for q in questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print("-" * 60)
        
        result = answer_question(q)
        
        print(f"Mode: {result['meta']['mode_used']}")
        print(f"Company: {result['meta'].get('company_id')}")
        print(f"Latency: {result['meta']['latency_ms']}ms")
        print(f"\nSteps:")
        for step in result['steps']:
            print(f"  - {step['id']}: {step['label']} [{step['status']}]")
        print(f"\nSources ({len(result['sources'])}):")
        for src in result['sources'][:3]:
            print(f"  - {src['type']}: {src['label']}")
        print(f"\nAnswer:\n{result['answer'][:300]}...")

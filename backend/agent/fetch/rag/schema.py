"""RAG knowledge base schema description for planner prompt."""


def get_rag_schema() -> str:
    """Return RAG knowledge base description for the SQL planner.

    This tells the planner what contextual information is available
    via RAG search, helping it decide when to set needs_rag=true.
    """
    return """| Entity | Contains | Common Use Cases |
|--------|----------|------------------|
| company | Key contacts, decision dynamics, adoption status, renewal concerns, win-back notes, attached docs | "Why is X at risk?", "What's the background on X?", "How do we approach renewal?" |
| contact | Communication preferences, concerns, objections, influence, technical requirements | "How should I approach Beth?", "What are Joe's concerns?", "Who is the champion?" |
| opportunity | Deal risks, blockers, recommended next steps, dependencies, proposals/contracts | "What's blocking this deal?", "Why is this stuck?", "What should I do next?" |
| activity | Call/meeting notes with context, concerns raised, action items, prep notes | "What was discussed in the call?", "What did we agree on?", "Any prep notes?" |
| history | Past interaction summaries, outcomes, what was communicated | "What happened last time?", "Any previous discussions?", "Email history?" |"""


__all__ = ["get_rag_schema"]

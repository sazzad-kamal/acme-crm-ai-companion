"""
Base evaluation utilities for agent evaluation.

Re-exports shared utilities from backend.common.eval_base
for use by agent eval modules.
"""


def ensure_qdrant_collections() -> None:
    """
    Ensure Qdrant collections exist, ingesting data if needed.
    Shared by e2e_eval and flow_eval.
    """
    from backend.agent.rag.client import close_qdrant_client, get_qdrant_client
    from backend.agent.rag.config import PRIVATE_COLLECTION, QDRANT_PATH
    from backend.agent.rag.ingest import ingest_private_texts

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = get_qdrant_client()

    private_exists = (
        qdrant.collection_exists(PRIVATE_COLLECTION)
        and (qdrant.get_collection(PRIVATE_COLLECTION).points_count or 0) > 0
    )

    if private_exists:
        print("Qdrant collections ready.")
        return

    # Close singleton before ingest (ingest needs exclusive access to local storage)
    close_qdrant_client()

    print("Ingesting private texts into Qdrant...")
    ingest_private_texts()

    # Verify collection was created (this creates a fresh singleton)
    qdrant = get_qdrant_client()
    if not qdrant.collection_exists(PRIVATE_COLLECTION):
        raise RuntimeError(f"Failed to create collection {PRIVATE_COLLECTION}")
    count = qdrant.get_collection(PRIVATE_COLLECTION).points_count or 0
    print(f"  Private collection created ({count} points)")


# Re-export utilities from focused modules
from backend.eval.formatting import (
    console,
    create_summary_table,
    format_check_mark,
    format_percentage,
    print_eval_header,
)
from backend.eval.shared import (
    REGRESSION_THRESHOLD,
    compare_to_baseline,
    print_baseline_comparison,
    save_baseline,
)

__all__ = [
    "ensure_qdrant_collections",
    "console",
    "create_summary_table",
    "format_check_mark",
    "format_percentage",
    "print_eval_header",
    "compare_to_baseline",
    "save_baseline",
    "print_baseline_comparison",
    "REGRESSION_THRESHOLD",
]

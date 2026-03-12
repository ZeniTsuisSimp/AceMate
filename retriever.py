"""
retriever.py — Semantic search over the Endee vector database.

Embeds user queries and searches the examprep index for the most
relevant chunks from the student's notes and syllabus.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from endee import Endee

from embedder import embed_text

load_dotenv()

logger = logging.getLogger(__name__)

INDEX_NAME = "examprep"


# ---------------------------------------------------------------------------
# Endee client helper
# ---------------------------------------------------------------------------


def _get_client() -> Endee:
    """Create and configure an Endee client.

    Returns:
        Endee: A connected client instance.
    """
    auth_token = os.getenv("ENDEE_AUTH_TOKEN", "")
    client = Endee(auth_token) if auth_token else Endee()

    base_url = os.getenv("ENDEE_URL", "http://localhost:8080")
    if not base_url.endswith("/api/v1"):
        base_url = base_url.rstrip("/") + "/api/v1"
    client.set_base_url(base_url)

    return client


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Perform semantic search for a free-text query.

    Args:
        query: Natural-language question or search string.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: ``id``, ``score``, ``text``,
        ``source_file``, ``chunk_index``, ``topic``, ``type``,
        ``page_range``.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided.")
        return []

    try:
        query_vector = embed_text(query)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        return []

    try:
        client = _get_client()
        index = client.get_index(name=INDEX_NAME)
        raw_results = index.query(vector=query_vector, top_k=top_k)
    except Exception as exc:
        logger.error("Endee query failed: %s", exc)
        return []

    return _format_results(raw_results)


def retrieve_by_topic(topic: str, top_k: int = 10) -> list[dict]:
    """Retrieve chunks related to a specific topic.

    Embeds the topic name as a query and returns the most relevant
    chunks from the vector store.

    Args:
        topic: The topic name to search for.
        top_k: Maximum number of results.

    Returns:
        List of formatted result dicts.
    """
    if not topic or not topic.strip():
        logger.warning("Empty topic provided.")
        return []

    # Use the topic itself as the search query for semantic matching
    search_query = f"Notes about {topic}. Key concepts and details of {topic}."
    return retrieve(search_query, top_k=top_k)


def retrieve_by_subject(subject: str, query: str = "", top_k: int = 10) -> list[dict]:
    """Retrieve chunks filtered to a specific subject.

    Uses Endee's filter to restrict results to the given subject,
    preventing cross-subject contamination.

    Args:
        subject: The subject name to filter by.
        query:   Optional search query (defaults to subject name).
        top_k:   Maximum number of results.

    Returns:
        List of formatted result dicts, all belonging to the subject.
    """
    if not subject or not subject.strip():
        logger.warning("Empty subject provided.")
        return []

    search_text = query.strip() if query and query.strip() else (
        f"Notes about {subject}. Key concepts and details of {subject}."
    )

    try:
        query_vector = embed_text(search_text)
    except Exception as exc:
        logger.error("Failed to embed query: %s", exc)
        return []

    try:
        client = _get_client()
        index = client.get_index(name=INDEX_NAME)
        raw_results = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=[{"subject": {"$eq": subject}}],
        )
    except Exception as exc:
        logger.error("Endee filtered query failed: %s", exc)
        # Fallback to unfiltered search if filter fails
        return retrieve_by_topic(subject, top_k=top_k)

    results = _format_results(raw_results)

    # If filtered search returned nothing, fall back to unfiltered
    if not results:
        logger.info("No results with subject filter '%s', falling back to unfiltered.", subject)
        return retrieve_by_topic(subject, top_k=top_k)

    return results

# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


def _format_results(raw_results) -> list[dict]:
    """Normalize Endee query results into a consistent dict format.

    Args:
        raw_results: Raw response from ``index.query()``.

    Returns:
        List of dicts with standardized keys.
    """
    formatted = []

    if not raw_results:
        return formatted

    # Endee query returns a list of result objects
    for result in raw_results:
        try:
            meta = {}
            # Handle different possible result formats from the SDK
            if hasattr(result, "meta"):
                meta = result.meta if isinstance(result.meta, dict) else {}
            elif isinstance(result, dict):
                meta = result.get("meta", {})

            result_id = getattr(result, "id", None) or (
                result.get("id") if isinstance(result, dict) else ""
            )
            score = getattr(result, "similarity", None) or getattr(
                result, "score", None
            ) or (
                result.get("similarity", result.get("score", 0.0))
                if isinstance(result, dict)
                else 0.0
            )

            formatted.append({
                "id": str(result_id),
                "score": float(score) if score else 0.0,
                "text": meta.get("text", ""),
                "source_file": meta.get("source_file", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "topic": meta.get("topic", "general"),
                "type": meta.get("type", "notes"),
                "page_range": meta.get("page_range", ""),
            })
        except Exception as exc:
            logger.warning("Failed to format result: %s", exc)
            continue

    return formatted

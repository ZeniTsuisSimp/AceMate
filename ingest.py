"""
ingest.py — Ingestion pipeline for loading PDF notes and syllabus into Endee.

Parses PDFs into chunks, generates embeddings via sentence-transformers,
and stores them in the Endee vector database with rich metadata.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from endee import Endee, Precision

from embedder import parse_pdf, parse_file, embed_batch, extract_topics, flatten_topics

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_NAME = "examprep"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output dimension


# ---------------------------------------------------------------------------
# Endee client helpers
# ---------------------------------------------------------------------------


def _get_client() -> Endee:
    """Create and configure an Endee client using environment variables.

    Returns:
        Endee: A connected Endee client instance.

    Raises:
        ConnectionError: If the Endee server is unreachable.
    """
    auth_token = os.getenv("ENDEE_AUTH_TOKEN", "")
    client = Endee(auth_token) if auth_token else Endee()

    base_url = os.getenv("ENDEE_URL", "http://localhost:8080")
    if not base_url:
        raise ConnectionError(
            "ENDEE_URL is not set. Add it to your environment variables. "
            "On Railway: set ENDEE_URL=http://<endee-service-private-hostname>:8080"
        )
    # Endee SDK expects the full API path
    if not base_url.endswith("/api/v1"):
        base_url = base_url.rstrip("/") + "/api/v1"
    client.set_base_url(base_url)

    return client


def _ensure_index(client: Endee) -> None:
    """Create the examprep index if it doesn't already exist.

    Args:
        client: An Endee client instance.
    """
    try:
        client.get_index(name=INDEX_NAME)
        logger.info("Index '%s' already exists.", INDEX_NAME)
    except Exception:
        logger.info("Creating index '%s' …", INDEX_NAME)
        client.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            space_type="cosine",
            precision=Precision.INT8,
        )
        logger.info("Index '%s' created successfully.", INDEX_NAME)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_documents(
    files: list[tuple[str, bytes]],
    doc_type: str = "notes",
    topics: Optional[list[str]] = None,
    grouped_topics: Optional[dict[str, list[str]]] = None,
    progress_callback=None,
) -> int:
    """Parse, embed, and store PDF/image documents in Endee.

    Args:
        files:             List of (filename, file_bytes) tuples.
        doc_type:          Either "notes" or "syllabus".
        topics:            Flat topic list (used for keyword matching).
        grouped_topics:    Subject-to-topics dict (used to assign subject).
        progress_callback: Optional callable(current, total) for progress.

    Returns:
        Total number of chunks ingested across all files.

    Raises:
        ConnectionError: If Endee cannot be reached.
    """
    if not files:
        logger.warning("No files provided for ingestion.")
        return 0

    try:
        client = _get_client()
        _ensure_index(client)
        index = client.get_index(name=INDEX_NAME)
    except Exception as exc:
        logger.error("Failed to connect to Endee: %s", exc)
        raise ConnectionError(
            f"Could not connect to Endee vector database: {exc}"
        ) from exc

    total_chunks = 0
    all_files_count = len(files)

    for file_idx, (filename, file_bytes) in enumerate(files):
        # Parse file into chunks (handles both PDFs and images)
        chunks = parse_file(file_bytes, filename=filename)
        if not chunks:
            logger.warning("No chunks extracted from '%s'. Skipping.", filename)
            if progress_callback:
                progress_callback(file_idx + 1, all_files_count)
            continue

        # Assign topic and subject to each chunk by matching against known topics
        for chunk in chunks:
            chunk["type"] = doc_type
            chunk["topic"] = _match_topic(chunk["text"], topics) if topics else "general"
            chunk["subject"] = _match_subject(chunk["text"], grouped_topics) if grouped_topics else "general"

        # Embed all chunk texts in batch
        texts = [c["text"] for c in chunks]
        vectors = embed_batch(texts)

        # Prepare upsert payload
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            record_id = f"{filename}__chunk_{chunk['chunk_index']}"
            records.append({
                "id": record_id,
                "vector": vector,
                "meta": {
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "page_range": chunk.get("page_range", ""),
                    "topic": chunk.get("topic", "general"),
                    "subject": chunk.get("subject", "general"),
                    "type": chunk["type"],
                },
                "filter": {
                    "topic": chunk.get("topic", "general"),
                    "subject": chunk.get("subject", "general"),
                    "type": chunk["type"],
                },
            })

        # Upsert in batches of 100
        batch_size = 100
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start : batch_start + batch_size]
            try:
                index.upsert(batch)
            except Exception as exc:
                logger.error(
                    "Failed to upsert batch %d–%d for '%s': %s",
                    batch_start, batch_start + len(batch), filename, exc,
                )
                raise

        total_chunks += len(chunks)
        logger.info(
            "Ingested '%s': %d chunks (%d/%d files done).",
            filename, len(chunks), file_idx + 1, all_files_count,
        )

        if progress_callback:
            progress_callback(file_idx + 1, all_files_count)

    logger.info("Ingestion complete. Total chunks: %d", total_chunks)
    return total_chunks


# ---------------------------------------------------------------------------
# Topic matching helper
# ---------------------------------------------------------------------------


def _match_topic(text: str, topics: Optional[list[str]]) -> str:
    """Find the best-matching topic for a text chunk (simple keyword match).

    Args:
        text:   The chunk text.
        topics: List of known topic names.

    Returns:
        The matched topic name, or "general" if no match is found.
    """
    if not topics:
        return "general"

    text_lower = text.lower()
    best_topic = "general"
    best_score = 0

    for topic in topics:
        # Count how many words from the topic appear in the chunk
        topic_words = topic.lower().split()
        score = sum(1 for w in topic_words if w in text_lower)
        if score > best_score:
            best_score = score
            best_topic = topic

    return best_topic if best_score > 0 else "general"


def _match_subject(text: str, grouped_topics: Optional[dict[str, list[str]]]) -> str:
    """Find the best-matching subject for a text chunk.

    Matches against both subject names and their topic lists.

    Args:
        text:            The chunk text.
        grouped_topics:  Dict mapping subject names to topic lists.

    Returns:
        The matched subject name, or "general" if no match is found.
    """
    if not grouped_topics:
        return "general"

    text_lower = text.lower()
    best_subject = "general"
    best_score = 0

    for subject, topic_list in grouped_topics.items():
        score = 0
        # Score from subject name words
        for w in subject.lower().split():
            if w in text_lower:
                score += 2  # subject name match weighted higher
        # Score from topic words under this subject
        for topic in topic_list:
            for w in topic.lower().split():
                if w in text_lower:
                    score += 1
        if score > best_score:
            best_score = score
            best_subject = subject

    return best_subject if best_score > 0 else "general"


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


def clear_index() -> None:
    """Delete all vectors from the examprep index in Endee."""
    try:
        client = _get_client()
        
        # Get the index
        index = client.get_index(name=INDEX_NAME)
        logger.info("Connected to index '%s'.", INDEX_NAME)
        
        # Get all vector IDs and delete them
        try:
            # Try to delete the entire index first
            index.delete()
            logger.info("Deleted index '%s'.", INDEX_NAME)
            
            # Recreate empty index
            _ensure_index(client)
            logger.info("Index '%s' recreated empty.", INDEX_NAME)
        except Exception as delete_err:
            logger.warning("Could not delete index: %s. Trying vector deletion...", delete_err)
            
            # Fallback: try to delete all vectors by deleting and recreating
            try:
                client.delete_index(name=INDEX_NAME)
                logger.info("Deleted index via client method")
                _ensure_index(client)
                logger.info("Index recreated")
            except Exception as fallback_err:
                logger.error("Fallback deletion failed: %s", fallback_err)
                raise
                
    except Exception as exc:
        logger.error("Failed to clear index: %s", exc)
        raise ConnectionError(
            f"Could not clear Endee index: {exc}"
        ) from exc


def get_index_stats() -> dict:
    """Retrieve basic stats about the examprep index.

    Returns:
        Dict with index metadata (or error information).
    """
    try:
        client = _get_client()
        index = client.get_index(name=INDEX_NAME)
        return {"name": INDEX_NAME, "status": "connected", "index": str(index)}
    except Exception as exc:
        return {"name": INDEX_NAME, "status": "error", "error": str(exc)}

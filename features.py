"""
features.py — Core AI features powered by Sarvam AI (sarvam-m model).

Provides four main capabilities:
  1. answer_question   — RAG-based Q&A from student notes
  2. predict_exam_questions — Generate likely exam questions for a topic
  3. summarize_topic   — Concise 5-point topic summary
  4. generate_mcq      — Multiple-choice quiz generation
"""

import os
import json
import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sarvam AI helper
# ---------------------------------------------------------------------------


def call_sarvam(
    prompt: str,
    system_prompt: str = "You are a helpful study assistant.",
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> str:
    """Call Sarvam AI's chat completion endpoint (OpenAI-compatible format).

    Args:
        prompt:        User message content.
        system_prompt: System-level instruction.
        max_tokens:    Maximum response tokens.
        temperature:   Sampling temperature (0 = deterministic, 1 = creative).

    Returns:
        The assistant's reply as a string.

    Raises:
        RuntimeError: On missing API key, network errors, or bad responses.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Please add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sarvam-m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        # Strip <think>...</think> reasoning blocks from sarvam-m output
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        return cleaned
    except requests.exceptions.Timeout:
        logger.error("Sarvam API request timed out.")
        raise RuntimeError("Sarvam API request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Sarvam API.")
        raise RuntimeError(
            "Could not connect to Sarvam AI. Check your internet connection."
        )
    except requests.exceptions.HTTPError as exc:
        logger.error("Sarvam API HTTP error: %s", exc)
        raise RuntimeError(f"Sarvam API error: {exc}")
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Sarvam API response: %s", exc)
        raise RuntimeError("Unexpected response format from Sarvam AI.")


# ---------------------------------------------------------------------------
# Helper: format retrieved chunks for context
# ---------------------------------------------------------------------------


def _build_context(chunks: list[dict], max_chars: int = 6000) -> str:
    """Join retrieved chunks into a single context string for the LLM.

    Args:
        chunks:    List of chunk dicts (must have a 'text' key).
        max_chars: Maximum total character length of the context.

    Returns:
        Formatted context string with source references.
    """
    if not chunks:
        return "No notes available."

    parts = []
    total = 0
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source_file", "unknown")
        pages = chunk.get("page_range", "")
        ref = f"[Source: {source}, Pages: {pages}]" if pages else f"[Source: {source}]"

        entry = f"--- Chunk {i + 1} {ref} ---\n{text}\n"
        if total + len(entry) > max_chars:
            break
        parts.append(entry)
        total += len(entry)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Feature 1: Answer a question from notes (RAG)
# ---------------------------------------------------------------------------


def answer_question(question: str, retrieved_chunks: list[dict]) -> dict:
    """Answer a student's question using only the retrieved notes.

    Args:
        question:         The student's question.
        retrieved_chunks: Chunks retrieved from the vector store.

    Returns:
        Dict with keys: ``answer``, ``sources`` (list of source references).
    """
    context = _build_context(retrieved_chunks)

    system_prompt = (
        "You are a precise academic assistant. Answer the student's question "
        "using ONLY the provided notes. If the answer is not found in the notes, "
        "respond with: 'Not found in your notes.' Always cite which source and "
        "page range your answer comes from."
    )

    prompt = (
        f"## Student's Notes:\n{context}\n\n"
        f"## Question:\n{question}\n\n"
        "Provide a clear, well-structured answer based strictly on the notes above. "
        "At the end, list the sources (filename and page range) you used."
    )

    try:
        raw_answer = call_sarvam(prompt, system_prompt=system_prompt)
    except RuntimeError as exc:
        return {"answer": f"Error: {exc}", "sources": []}

    # Extract source references from chunks
    sources = []
    for chunk in retrieved_chunks:
        src = {
            "file": chunk.get("source_file", "unknown"),
            "pages": chunk.get("page_range", ""),
            "topic": chunk.get("topic", "general"),
        }
        if src not in sources:
            sources.append(src)

    return {"answer": raw_answer, "sources": sources}


# ---------------------------------------------------------------------------
# Feature 2: Predict exam questions
# ---------------------------------------------------------------------------


def predict_exam_questions(topic: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Generate likely exam questions for a given topic.

    Args:
        topic:            The topic name.
        retrieved_chunks: Chunks related to this topic.

    Returns:
        List of dicts with keys: ``question``, ``type``
        (Short Answer / Long Answer / MCQ), ``difficulty``
        (Easy / Medium / Hard).
    """
    context = _build_context(retrieved_chunks)

    system_prompt = (
        "You are an experienced exam paper setter. Analyze the study material "
        "and generate realistic exam questions."
    )

    prompt = (
        f"## Topic: {topic}\n\n"
        f"## Study Material:\n{context}\n\n"
        "Based on the study material above, generate exactly 5 likely exam questions.\n"
        "For each question, specify:\n"
        "- The question text\n"
        "- Type: Short Answer, Long Answer, or MCQ\n"
        "- Difficulty: Easy, Medium, or Hard\n\n"
        "Return ONLY a valid JSON array with objects having keys: "
        '"question", "type", "difficulty".\n'
        "No markdown formatting, just the raw JSON array."
    )

    try:
        raw = call_sarvam(prompt, system_prompt=system_prompt, temperature=0.5)
    except RuntimeError as exc:
        logger.error("Exam question prediction failed: %s", exc)
        return []

    return _parse_json_list(raw, expected_keys=["question", "type", "difficulty"])


# ---------------------------------------------------------------------------
# Feature 3: Summarize a topic
# ---------------------------------------------------------------------------


def summarize_topic(topic: str, retrieved_chunks: list[dict]) -> str:
    """Generate a concise 5-point summary of a topic from notes.

    Args:
        topic:            The topic to summarize.
        retrieved_chunks: Chunks related to the topic.

    Returns:
        Formatted bullet-point summary string.
    """
    context = _build_context(retrieved_chunks)

    system_prompt = (
        "You are an academic content summarizer. Your job is to explain the "
        "actual subject matter — the core concepts, definitions, theories, "
        "formulas, and facts — NOT to describe what a course or syllabus covers. "
        "Never say things like 'The course focuses on…' or 'This topic covers…'. "
        "Instead, directly explain the knowledge itself as if teaching a student."
    )

    prompt = (
        f"## Topic: {topic}\n\n"
        f"## Study Material:\n{context}\n\n"
        "Summarize the actual knowledge content of this topic in exactly 5 detailed "
        "bullet points. Each point must EXPLAIN a concept, definition, fact, or "
        "principle — do NOT just list what the syllabus/course covers. "
        "Write as if you are teaching the student the material, not describing a course. "
        "Be specific: include names, definitions, formulas, examples where available. "
        "Format as a numbered list (1. … 2. … etc.)."
    )

    try:
        summary = call_sarvam(prompt, system_prompt=system_prompt, temperature=0.4)
        return summary
    except RuntimeError as exc:
        return f"Error generating summary: {exc}"


# ---------------------------------------------------------------------------
# Feature 4: Generate MCQs
# ---------------------------------------------------------------------------


def generate_mcq(topic: str, retrieved_chunks: list[dict]) -> list[dict]:
    """Generate multiple-choice questions for quiz / weak-topic practice.

    Args:
        topic:            The topic to generate MCQs for.
        retrieved_chunks: Chunks related to the topic.

    Returns:
        List of dicts with keys: ``question``, ``options`` (list of 4 strings),
        ``correct_answer`` (letter A/B/C/D), ``explanation``.
    """
    context = _build_context(retrieved_chunks)

    system_prompt = (
        "You are a quiz designer for academic exams. Create well-crafted "
        "multiple-choice questions from the provided study material."
    )

    prompt = (
        f"## Topic: {topic}\n\n"
        f"## Study Material:\n{context}\n\n"
        "Generate exactly 5 multiple-choice questions (MCQs).\n"
        "For each MCQ:\n"
        "- Write a clear question\n"
        "- Provide exactly 4 options labeled A, B, C, D\n"
        "- Indicate the correct answer (A, B, C, or D)\n"
        "- Provide a brief explanation of why the answer is correct\n\n"
        "Return ONLY a valid JSON array. Each object must have keys:\n"
        '"question" (string), "options" (object with keys A, B, C, D), '
        '"correct_answer" (string, one of A/B/C/D), "explanation" (string).\n'
        "No markdown, no extra text. Just the JSON array."
    )

    try:
        raw = call_sarvam(prompt, system_prompt=system_prompt, temperature=0.5)
    except RuntimeError as exc:
        logger.error("MCQ generation failed: %s", exc)
        return []

    return _parse_json_list(
        raw, expected_keys=["question", "options", "correct_answer", "explanation"]
    )


# ---------------------------------------------------------------------------
# Feature 5: Private Tutor Chat (subject-scoped)
# ---------------------------------------------------------------------------


def tutor_chat(
    message: str,
    subject: str,
    retrieved_chunks: list[dict],
    conversation_history: list[dict],
) -> str:
    """Respond to a student's message as a private tutor for a specific subject.

    The tutor only answers within the context of the selected subject
    and uses the retrieved notes as its knowledge base.

    Args:
        message:              The student's latest message.
        subject:              The selected subject/topic to stay within.
        retrieved_chunks:     Chunks retrieved for this subject.
        conversation_history: List of previous messages [{role, content}, ...].

    Returns:
        The tutor's reply as a string.
    """
    context = _build_context(retrieved_chunks, max_chars=4000)

    system_prompt = (
        f"You are a friendly, expert private tutor specializing in '{subject}'. "
        "Your role is to help the student understand concepts, solve problems, "
        "and prepare for exams — but ONLY for this subject. "
        "If the student asks about something outside this subject, politely "
        "redirect them back to the topic. "
        "Use the provided study notes as your primary knowledge source. "
        "Explain concepts clearly with examples when helpful. "
        "Be encouraging and supportive.\n\n"
        f"## Student's Notes on {subject}:\n{context}"
    )

    # Build the messages array: single system + history + new message
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (limit to last 10 exchanges to stay within token limits)
    for msg in conversation_history[-10:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Add the new user message
    messages.append({"role": "user", "content": message})

    # Call Sarvam with full message history
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        return "Error: SARVAM_API_KEY is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "sarvam-m",
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        return cleaned
    except Exception as exc:
        logger.error("Tutor chat failed: %s", exc)
        return f"Sorry, I encountered an error: {exc}"


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def _parse_json_list(raw: str, expected_keys: list[str]) -> list[dict]:
    """Attempt to parse a JSON array from LLM output, with cleanup.

    Args:
        raw:           Raw LLM response string.
        expected_keys: Keys each object should contain.

    Returns:
        Parsed list of dicts, or empty list on failure.
    """
    import re
    cleaned = raw.strip()

    # Remove <think>...</think> blocks (Sarvam-m uses reasoning tags)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
    if cleaned.endswith("```"):
        cleaned = cleaned[: cleaned.rfind("```")]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON array inside the text
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("Could not parse JSON from LLM response.")
                return []
        else:
            logger.warning("No JSON array found in LLM response.")
            return []

    if not isinstance(parsed, list):
        return []

    # Validate that each item has the expected keys
    valid = []
    for item in parsed:
        if isinstance(item, dict) and all(k in item for k in expected_keys):
            valid.append(item)

    return valid

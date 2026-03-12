"""
embedder.py — PDF/image parsing, text chunking, and embedding logic.

Uses sentence-transformers (all-MiniLM-L6-v2) for 384-dim embeddings,
PyMuPDF (fitz) for PDF text extraction, and RapidOCR (ONNX Runtime)
for OCR on image files. Topic extraction from syllabus text is
handled via Sarvam AI.
"""

import io
import os
import json
import logging
import fitz  # PyMuPDF
import requests
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_model = None  # cached singleton


def load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model (all-MiniLM-L6-v2).

    Returns:
        SentenceTransformer: The embedding model (384-dimensional output).
    """
    global _model
    if _model is None:
        logger.info("Loading sentence-transformer model: all-MiniLM-L6-v2 …")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Embedding functions
# ---------------------------------------------------------------------------


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a 384-dim vector.

    Args:
        text: The input string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = load_model()
    vector = model.encode(text, show_progress_bar=False)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []
    model = load_model()
    vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# PDF parsing & chunking
# ---------------------------------------------------------------------------


def parse_pdf(file_bytes: bytes, filename: str = "document.pdf",
              chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Extract text from a PDF and split into overlapping word-level chunks.

    Args:
        file_bytes: Raw bytes of the PDF file.
        filename:   Original filename (stored in chunk metadata).
        chunk_size: Number of words per chunk.
        overlap:    Number of overlapping words between consecutive chunks.

    Returns:
        List of dicts, each with keys: ``text``, ``source_file``,
        ``chunk_index``, ``page_range``.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", filename, exc)
        raise ValueError(f"Could not open PDF file '{filename}': {exc}") from exc

    # Collect all words with their page numbers
    all_words: list[str] = []
    word_pages: list[int] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        words = page_text.split()
        all_words.extend(words)
        word_pages.extend([page_num + 1] * len(words))

    doc.close()

    if not all_words:
        logger.warning("PDF '%s' contains no extractable text.", filename)
        return []

    # Build sliding-window chunks
    chunks: list[dict] = []
    step = max(chunk_size - overlap, 1)
    idx = 0

    for start in range(0, len(all_words), step):
        end = min(start + chunk_size, len(all_words))
        chunk_words = all_words[start:end]
        chunk_text = " ".join(chunk_words)

        # Determine page range covered by this chunk
        pages_in_chunk = set(word_pages[start:end])
        page_range = f"{min(pages_in_chunk)}-{max(pages_in_chunk)}"

        chunks.append({
            "text": chunk_text,
            "source_file": filename,
            "chunk_index": idx,
            "page_range": page_range,
        })
        idx += 1

        if end >= len(all_words):
            break

    logger.info("Parsed '%s': %d words → %d chunks.", filename, len(all_words), len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Image parsing & chunking (OCR)
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_image(file_bytes: bytes, filename: str = "image.png",
                chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Extract text from an image via OCR and split into overlapping chunks.

    Args:
        file_bytes: Raw bytes of the image file.
        filename:   Original filename (stored in chunk metadata).
        chunk_size: Number of words per chunk.
        overlap:    Number of overlapping words between consecutive chunks.

    Returns:
        List of dicts with keys: ``text``, ``source_file``,
        ``chunk_index``, ``page_range``.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_array = np.array(image)
    except Exception as exc:
        logger.error("Failed to open image '%s': %s", filename, exc)
        raise ValueError(f"Could not open image file '{filename}': {exc}") from exc

    try:
        ocr = RapidOCR()
        result, _ = ocr(img_array)
        full_text = "\n".join(line[1] for line in result) if result else ""
    except Exception as exc:
        logger.error("OCR failed for '%s': %s", filename, exc)
        raise ValueError(
            f"OCR failed for '{filename}': {exc}"
        ) from exc

    all_words = full_text.split()
    if not all_words:
        logger.warning("Image '%s' produced no OCR text.", filename)
        return []

    # Build sliding-window chunks (single page for images)
    chunks: list[dict] = []
    step = max(chunk_size - overlap, 1)
    idx = 0

    for start in range(0, len(all_words), step):
        end = min(start + chunk_size, len(all_words))
        chunk_text = " ".join(all_words[start:end])

        chunks.append({
            "text": chunk_text,
            "source_file": filename,
            "chunk_index": idx,
            "page_range": "1-1",
        })
        idx += 1

        if end >= len(all_words):
            break

    logger.info("Parsed image '%s': %d words → %d chunks.", filename, len(all_words), len(chunks))
    return chunks


def parse_file(file_bytes: bytes, filename: str,
               chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Dispatch to the correct parser based on file extension.

    Supports PDFs and common image formats (jpg, png, bmp, tiff, webp).

    Args:
        file_bytes: Raw bytes of the file.
        filename:   Original filename.
        chunk_size: Number of words per chunk.
        overlap:    Overlapping words between consecutive chunks.

    Returns:
        List of chunk dicts.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        chunks = parse_pdf(file_bytes, filename, chunk_size, overlap)
        # Fallback: if PDF has no selectable text (scanned image PDF), try OCR
        if not chunks:
            logger.info("PDF '%s' has no text — attempting OCR fallback.", filename)
            try:
                import fitz as _fitz
                doc = _fitz.open(stream=file_bytes, filetype="pdf")
                all_chunks = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    page_chunks = parse_image(img_bytes, f"{filename}_page{page_num+1}.png", chunk_size, overlap)
                    # Fix page range for each chunk
                    for c in page_chunks:
                        c["source_file"] = filename
                        c["page_range"] = f"{page_num+1}-{page_num+1}"
                    all_chunks.extend(page_chunks)
                doc.close()
                chunks = all_chunks
            except Exception as exc:
                logger.warning("OCR fallback for PDF '%s' failed: %s", filename, exc)
        return chunks
    elif ext in IMAGE_EXTENSIONS:
        return parse_image(file_bytes, filename, chunk_size, overlap)
    else:
        raise ValueError(f"Unsupported file type '{ext}' for '{filename}'.")


# ---------------------------------------------------------------------------
# Sarvam AI helper (used for topic extraction)
# ---------------------------------------------------------------------------


def _call_sarvam(prompt: str,
                 system_prompt: str = "You are a helpful study assistant.",
                 max_tokens: int = 1000,
                 temperature: float = 0.7) -> str:
    """Call Sarvam AI's chat completion endpoint.

    Args:
        prompt:        User message.
        system_prompt: System-level instruction.
        max_tokens:    Maximum tokens in the response.
        temperature:   Sampling temperature.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        RuntimeError: If the API call fails or the key is missing.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. Add it to your .env file."
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
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        raw_content = data["choices"][0]["message"]["content"]
        # Strip <think>...</think> reasoning blocks from sarvam-m output
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        return cleaned
    except requests.exceptions.RequestException as exc:
        logger.error("Sarvam API request failed: %s", exc)
        raise RuntimeError(f"Sarvam API call failed: {exc}") from exc
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Sarvam API response format: %s", exc)
        raise RuntimeError(
            "Unexpected response from Sarvam API."
        ) from exc


# ---------------------------------------------------------------------------
# Topic extraction from syllabus
# ---------------------------------------------------------------------------


def extract_topics(syllabus_text: str) -> dict[str, list[str]]:
    """Use Sarvam AI to extract topics grouped by subject from syllabus text.

    Args:
        syllabus_text: Full text of the syllabus (or a large portion).

    Returns:
        A dict mapping subject/unit names to lists of topic strings.
        Example: {"Linear Algebra": ["Matrices", "Determinants", "Eigenvalues"]}
    """
    if not syllabus_text or not syllabus_text.strip():
        logger.warning("Empty syllabus text provided.")
        return {}

    # Split long syllabi into smaller overlapping chunks for thorough extraction
    chunk_limit = 4000
    overlap_chars = 500
    text = syllabus_text.strip()
    text_chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_limit, len(text))
        text_chunks.append(text[start:end])
        start += chunk_limit - overlap_chars
        if end >= len(text):
            break

    merged: dict[str, list[str]] = {}

    for chunk_idx, chunk_text in enumerate(text_chunks):
        prompt = (
            "You are given a portion of an academic syllabus. Extract topics "
            "GROUPED BY SUBJECT / UNIT / MODULE exactly as they appear in the syllabus.\n\n"
            "Rules:\n"
            "- Identify each subject, unit, or module as a top-level key\n"
            "- Under each key, list ALL topics, subtopics, concepts, theorems, "
            "laws, methods, techniques mentioned for that subject\n"
            "- Preserve the grouping from the syllabus — do NOT mix subjects\n"
            "- Be exhaustive — include every single topic and subtopic\n"
            "- If no clear subject grouping exists, use 'General' as the key\n\n"
            "Return ONLY a valid JSON object. Example:\n"
            '{"Linear Algebra": ["Matrices", "Determinants", "Eigenvalues"], '
            '"Calculus": ["Differentiation", "Integration", "Partial Derivatives"]}\n\n'
            f"Syllabus text (part {chunk_idx + 1}/{len(text_chunks)}):\n{chunk_text}"
        )

        try:
            logger.info(
                "Calling Sarvam AI for grouped topic extraction chunk %d/%d (text length: %d)…",
                chunk_idx + 1, len(text_chunks), len(chunk_text),
            )
            raw = _call_sarvam(
                prompt,
                system_prompt=(
                    "You are an extremely thorough academic syllabus parser. "
                    "Extract every topic grouped by subject/unit/module. "
                    "Return only a JSON object mapping subject names to arrays of topic strings. "
                    "No markdown, no explanation."
                ),
                max_tokens=4000,
                temperature=0.2,
            )
            logger.info("Sarvam raw response for grouped topics (chunk %d): %s", chunk_idx + 1, raw[:500])

            cleaned = _clean_llm_response(raw)
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict):
                for subject, topic_list in parsed.items():
                    subject = str(subject).strip()
                    if not subject:
                        continue
                    if subject not in merged:
                        merged[subject] = []
                    if isinstance(topic_list, list):
                        for t in topic_list:
                            t_str = str(t).strip()
                            if t_str and t_str.lower() not in {
                                x.lower() for x in merged[subject]
                            }:
                                merged[subject].append(t_str)
                    elif isinstance(topic_list, str):
                        t_str = topic_list.strip()
                        if t_str and t_str.lower() not in {
                            x.lower() for x in merged[subject]
                        }:
                            merged[subject].append(t_str)
            elif isinstance(parsed, list):
                # Fallback: if LLM returns a flat list, group under "General"
                if "General" not in merged:
                    merged["General"] = []
                for t in parsed:
                    t_str = str(t).strip()
                    if t_str and t_str.lower() not in {
                        x.lower() for x in merged["General"]
                    }:
                        merged["General"].append(t_str)

        except json.JSONDecodeError:
            # Fallback: try to find a JSON object or array in the response
            try:
                brace_start = raw.find("{")
                brace_end = raw.rfind("}")
                bracket_start = raw.find("[")
                bracket_end = raw.rfind("]")

                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    parsed = json.loads(raw[brace_start : brace_end + 1])
                    if isinstance(parsed, dict):
                        for subject, topic_list in parsed.items():
                            subject = str(subject).strip()
                            if not subject:
                                continue
                            if subject not in merged:
                                merged[subject] = []
                            if isinstance(topic_list, list):
                                for t in topic_list:
                                    t_str = str(t).strip()
                                    if t_str and t_str.lower() not in {
                                        x.lower() for x in merged[subject]
                                    }:
                                        merged[subject].append(t_str)
                elif bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
                    parsed = json.loads(raw[bracket_start : bracket_end + 1])
                    if isinstance(parsed, list):
                        if "General" not in merged:
                            merged["General"] = []
                        for t in parsed:
                            t_str = str(t).strip()
                            if t_str and t_str.lower() not in {
                                x.lower() for x in merged["General"]
                            }:
                                merged["General"].append(t_str)
            except (json.JSONDecodeError, Exception):
                pass
            logger.error("Grouped topic extraction JSON parse failed for chunk %d: %s", chunk_idx + 1, raw[:300])
        except RuntimeError as exc:
            logger.error("Topic extraction API call failed for chunk %d: %s", chunk_idx + 1, exc)

    total_topics = sum(len(v) for v in merged.values())
    logger.info("Extracted %d subjects with %d total topics: %s",
                len(merged), total_topics, {k: len(v) for k, v in merged.items()})
    return merged


def flatten_topics(grouped_topics: dict[str, list[str]]) -> list[str]:
    """Flatten a grouped topics dict into a deduplicated list of all topics.

    Includes both subject names and their subtopics for maximum matching.

    Args:
        grouped_topics: Dict mapping subjects to topic lists.

    Returns:
        Flat list of unique topic strings.
    """
    flat: list[str] = []
    seen: set[str] = set()
    for subject, topic_list in grouped_topics.items():
        # Include the subject name itself
        if subject.lower() not in seen:
            seen.add(subject.lower())
            flat.append(subject)
        for t in topic_list:
            if t.lower() not in seen:
                seen.add(t.lower())
                flat.append(t)
    return flat


def _clean_llm_response(raw: str) -> str:
    """Strip <think> tags, markdown fences, and whitespace from LLM output.

    Args:
        raw: Raw LLM response string.

    Returns:
        Cleaned string ready for JSON parsing.
    """
    import re
    cleaned = raw.strip()

    # Remove <think>...</think> blocks (Sarvam-m uses reasoning tags)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # Handle unclosed <think> tag — strip everything from <think> to the
    # first JSON structure ({...} or [...])
    if "<think>" in cleaned:
        # Find where JSON starts
        brace = cleaned.find("{")
        bracket = cleaned.find("[")
        candidates = [i for i in [brace, bracket] if i != -1]
        if candidates:
            cleaned = cleaned[min(candidates):]
        else:
            # No JSON found — strip the <think> tag entirely
            cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()

    # Remove markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    return cleaned

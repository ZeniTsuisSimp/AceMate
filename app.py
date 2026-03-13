"""
app.py — Streamlit UI for AceMate.

Provides a 6-page sidebar navigation:
  1. Upload — Upload and ingest PDFs / images into Endee
  2. Ask from Notes — RAG-based Q&A
  3. Exam Questions — Predicted exam questions by topic
  4. Topic Summary — 5-point topic summaries
  5. Weak Topics — Performance tracking + MCQ practice
  6. Private Tutor — Subject-scoped chatbot tutor
"""

import os
import streamlit as st
import logging

from ingest import ingest_documents, clear_index
from embedder import extract_topics, flatten_topics, parse_pdf, parse_file
from retriever import retrieve, retrieve_by_topic, retrieve_by_subject, get_index_count
from features import answer_question, predict_exam_questions, summarize_topic, generate_mcq, tutor_chat
from tracker import ScoreTracker

# ---------------------------------------------------------------------------
# Page config & custom styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AceMate",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }

    /* Cards */
    .card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .card h3 {
        color: #a78bfa;
        margin-top: 0;
    }
    .card p {
        color: #d1d5db;
    }

    /* Source chips */
    .source-chip {
        display: inline-block;
        background: rgba(167, 139, 250, 0.15);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #c4b5fd;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        margin: 2px 4px;
    }

    /* Stats row */
    .stat-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-box h2 {
        color: #818cf8;
        margin: 0;
        font-size: 2rem;
    }
    .stat-box p {
        color: #9ca3af;
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
    }

    /* Question cards */
    .q-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-left: 4px solid #818cf8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .q-card .q-type {
        display: inline-block;
        background: rgba(129, 140, 248, 0.2);
        color: #a5b4fc;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        margin-right: 8px;
    }
    .q-card .q-diff {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
    }
    .diff-easy { background: rgba(52, 211, 153, 0.2); color: #6ee7b7; }
    .diff-medium { background: rgba(251, 191, 36, 0.2); color: #fcd34d; }
    .diff-hard { background: rgba(248, 113, 113, 0.2); color: #fca5a5; }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .hero h1 {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .hero p {
        color: #9ca3af;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "topics" not in st.session_state:
    st.session_state.topics = {}
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []  # list of filenames indexed this session

tracker = ScoreTracker()

# ---------------------------------------------------------------------------
# Endee connectivity check
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _check_endee():
    import requests as _req
    url = os.environ.get("ENDEE_URL", "http://localhost:8080").rstrip("/")
    try:
        _req.get(f"{url}/health", timeout=5)
        return True, None
    except Exception as exc:
        return False, str(exc)

_endee_ok, _endee_err = _check_endee()
if not _endee_ok:
    endee_url = os.environ.get("ENDEE_URL", "http://localhost:8080")
    st.error(
        f"⚠️ **Cannot reach the Endee vector database** at `{endee_url}`.\n\n"
        "**On Railway:** Go to your project → add an Endee service (Docker image: "
        "`endeeio/endee-server:latest`), then set `ENDEE_URL` in your app's Variables "
        "to the Endee service's private hostname, e.g. `http://endee.railway.internal:8080`.\n\n"
        f"Error: `{_endee_err}`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎓 AceMate")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "📤 Upload",
            "❓ Ask from Notes",
            "📝 Exam Questions",
            "📋 Topic Summary",
            "📊 Weak Topics",
            "🧑‍🏫 Private Tutor",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b7280'>Powered by <b>Endee</b> + <b>Sarvam AI</b></small>",
        unsafe_allow_html=True,
    )
    # Show indexed files for this session
    indexed = st.session_state.get("indexed_files", [])
    if indexed:
        st.markdown("---")
        st.markdown("<small style='color:#6b7280'><b>📁 Indexed this session:</b></small>", unsafe_allow_html=True)
        for fname in indexed:
            st.markdown(f"<small style='color:#9ca3af'>• {fname}</small>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Upload
# ═══════════════════════════════════════════════════════════════════════════

if page == "📤 Upload":
    st.markdown(
        '<div class="hero"><h1>Upload Your Study Material</h1>'
        "<p>Upload notes as PDFs or photos and your syllabus to build your personal knowledge base.</p></div>",
        unsafe_allow_html=True,
    )

    # Show current index status
    _idx_count = get_index_count()
    if _idx_count > 0:
        st.info(f"📦 Endee index currently holds **{_idx_count} vectors** from previous uploads. New uploads will add to these.")
    elif _idx_count == 0:
        st.warning("📭 Index is empty — please upload your notes and syllabus below.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📚 Notes (PDFs or Images)")
        notes_files = st.file_uploader(
            "Upload notes as PDFs or photos of handwritten/printed notes",
            type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
            accept_multiple_files=True,
            key="notes_uploader",
        )

    with col2:
        st.markdown("### 📋 Syllabus (PDF or Image)")
        syllabus_file = st.file_uploader(
            "Upload your syllabus as PDF or photo",
            type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
            accept_multiple_files=False,
            key="syllabus_uploader",
        )

    st.markdown("---")

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        process_btn = st.button(
            "🚀 Process & Index",
            width="stretch",
            type="primary",
        )

    if process_btn:
        if not notes_files and not syllabus_file:
            st.error("⚠️ Please upload at least one file (PDF or image).")
        else:
            progress_bar = st.progress(0, text="Starting ingestion…")
            total_chunks = 0

            try:
                # --- Process syllabus first to extract topics ---
                topics = {}
                if syllabus_file:
                    with st.spinner("📋 Extracting topics from syllabus…"):
                        syl_bytes = syllabus_file.read()
                        syl_chunks = parse_file(syl_bytes, syllabus_file.name)
                        syl_full_text = " ".join(c["text"] for c in syl_chunks)
                        logger.info("Syllabus parsed: %d chunks, %d chars of text", len(syl_chunks), len(syl_full_text))

                        if not syl_full_text.strip():
                            st.error("❌ Could not extract any text from the syllabus file. If it's a photo, ensure the text is clearly readable.")
                        else:
                            topics = extract_topics(syl_full_text)
                            st.session_state.topics = topics

                        if topics:
                            total_t = sum(len(v) for v in topics.values())
                            st.success(f"✅ Extracted {len(topics)} subjects with {total_t} topics from syllabus.")
                        else:
                            st.warning("⚠️ Could not extract topics. Tagging as 'general'.")

                    # Flatten for ingestion
                    flat_topics = flatten_topics(topics) if topics else None

                    # Ingest syllabus
                    with st.spinner("📋 Indexing syllabus…"):
                        syl_count = ingest_documents(
                            [(syllabus_file.name, syl_bytes)],
                            doc_type="syllabus",
                            topics=flat_topics,
                            grouped_topics=topics if topics else None,
                        )
                        total_chunks += syl_count
                    progress_bar.progress(30, text="Syllabus indexed.")

                # --- Process notes ---
                if notes_files:
                    files_data = []
                    for f in notes_files:
                        files_data.append((f.name, f.read()))

                    total_notes = len(files_data)

                    def update_progress(current, total):
                        pct = 30 + int((current / total) * 70)
                        progress_bar.progress(
                            pct,
                            text=f"Processing note {current}/{total}…",
                        )

                    with st.spinner("📚 Indexing notes… This may take a moment."):
                        flat_topics = flatten_topics(topics) if topics else None
                        notes_count = ingest_documents(
                            files_data,
                            doc_type="notes",
                            topics=flat_topics,
                            grouped_topics=topics if topics else None,
                            progress_callback=update_progress,
                        )
                        total_chunks += notes_count

                progress_bar.progress(100, text="✅ Ingestion complete!")
                st.session_state.ingested = True
                st.session_state.chunk_count = total_chunks

                # Remember which files were indexed this session
                if syllabus_file:
                    if syllabus_file.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(syllabus_file.name)
                for f_name, _ in (files_data if notes_files else []):
                    if f_name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(f_name)

                # Success metrics
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f'<div class="stat-box"><h2>{total_chunks}</h2>'
                        "<p>Chunks Indexed</p></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    files_count = len(notes_files or []) + (1 if syllabus_file else 0)
                    st.markdown(
                        f'<div class="stat-box"><h2>{files_count}</h2>'
                        "<p>Files Processed</p></div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    total_t = sum(len(v) for v in topics.values()) if topics else 0
                    st.markdown(
                        f'<div class="stat-box"><h2>{total_t}</h2>'
                        "<p>Topics Found</p></div>",
                        unsafe_allow_html=True,
                    )

            except ConnectionError as exc:
                st.error(f"🔌 Endee connection error: {exc}")
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")
                logger.exception("Ingestion error")

    # Privacy & Security Section
    st.markdown("---")
    st.markdown("### 🔒 Privacy & Security")
    st.warning(
        "⚠️ **Session Notes:** Your uploaded notes are stored in the shared Endee database. "
        "They will be visible to the next user unless you clear them. "
        "Quiz scores are NOT saved (session-only)."
    )

    col_clear_1, col_clear_2 = st.columns([1, 3])
    with col_clear_1:
        if st.button("🗑️ Clear All Notes", use_container_width=True, type="secondary"):
            with st.spinner("Clearing index…"):
                try:
                    clear_index()
                    st.session_state.indexed_files = []
                    st.session_state.topics = {}
                    st.success("✅ All uploaded notes cleared from database!")
                except Exception as e:
                    st.error(f"❌ Failed to clear: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Ask from Notes
# ═══════════════════════════════════════════════════════════════════════════

elif page == "❓ Ask from Notes":
    st.markdown(
        '<div class="hero"><h1>Ask from Your Notes</h1>'
        "<p>Ask any question — answers are backed by your uploaded study material.</p></div>",
        unsafe_allow_html=True,
    )

    # Proactive index check
    _idx_count = get_index_count()
    if _idx_count == 0:
        st.error("📭 Your index is empty. Go to **📤 Upload** first and upload your notes.")
        st.stop()
    else:
        st.caption(f"🔍 Searching across {_idx_count} indexed chunks from your notes.")

    question = st.text_input(
        "💬 Type your question",
        placeholder="e.g. What are the properties of electromagnetic waves?",
    )

    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Searching your notes and generating answer…"):
            # Retrieve relevant chunks
            chunks = retrieve(question, top_k=5)

            if not chunks:
                st.warning("📭 No relevant content found in your notes. Please upload notes first.")
            else:
                result = answer_question(question, chunks)

                # Display answer
                st.markdown("### 💡 Answer")
                st.markdown(
                    f'<div class="card"><p>{result["answer"]}</p></div>',
                    unsafe_allow_html=True,
                )

                # Display sources
                if result["sources"]:
                    st.markdown("### 📎 Sources Used")
                    for src in result["sources"]:
                        st.markdown(
                            f'<span class="source-chip">📄 {src["file"]} '
                            f'(Pages {src["pages"]})</span>',
                            unsafe_allow_html=True,
                        )

                # Show retrieved chunks in expander
                with st.expander("🔍 View Retrieved Note Excerpts"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(
                            f"**Chunk {i + 1}** — {chunk['source_file']} "
                            f"(Pages {chunk['page_range']}) | "
                            f"Score: {chunk['score']:.3f}"
                        )
                        st.text(chunk["text"][:500])
                        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Exam Questions
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📝 Exam Questions":
    st.markdown(
        '<div class="hero"><h1>Predict Exam Questions</h1>'
        "<p>AI analyzes your notes to predict likely exam questions for each topic.</p></div>",
        unsafe_allow_html=True,
    )

    topics = st.session_state.get("topics", {})
    if not topics:
        st.info(
            "📋 No topics found. Upload a syllabus first on the Upload page "
            "to extract topics, or type a topic below."
        )
        topic_input = st.text_input("Enter a topic manually")
        selected_topic = topic_input.strip() if topic_input else None
    else:
        subjects = list(topics.keys())
        selected_subject = st.selectbox("Select a subject", subjects, key="eq_subject")
        subject_topics = topics.get(selected_subject, [])
        if subject_topics:
            selected_topic = st.selectbox("Select a topic", subject_topics, key="eq_topic")
        else:
            selected_topic = selected_subject

    if st.button("🎯 Predict Questions", type="primary") and selected_topic:
        with st.spinner(f"Analyzing notes for '{selected_topic}'…"):
            chunks = retrieve_by_subject(selected_subject, query=selected_topic, top_k=10) if topics else retrieve_by_topic(selected_topic, top_k=10)

            if not chunks:
                st.warning(f"📭 No notes found for topic '{selected_topic}'.")
            else:
                questions = predict_exam_questions(selected_topic, chunks)

                if not questions:
                    st.warning("Could not generate questions. Try again.")
                else:
                    st.markdown(f"### 📝 Predicted Questions for: *{selected_topic}*")

                    # Group by type
                    by_type: dict[str, list] = {}
                    for q in questions:
                        qtype = q.get("type", "Other")
                        by_type.setdefault(qtype, []).append(q)

                    for qtype, items in by_type.items():
                        st.markdown(f"#### {qtype}")
                        for q in items:
                            diff = q.get("difficulty", "Medium").lower()
                            diff_class = f"diff-{diff}" if diff in ("easy", "medium", "hard") else "diff-medium"
                            st.markdown(
                                f'<div class="q-card">'
                                f'<span class="q-type">{qtype}</span>'
                                f'<span class="q-diff {diff_class}">{q.get("difficulty", "Medium")}</span>'
                                f'<p style="margin-top:8px; color:#e5e7eb;">{q["question"]}</p>'
                                f"</div>",
                                unsafe_allow_html=True,
                            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: Topic Summary
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📋 Topic Summary":
    st.markdown(
        '<div class="hero"><h1>Topic Summary</h1>'
        "<p>Get concise, exam-focused summaries of any topic from your notes.</p></div>",
        unsafe_allow_html=True,
    )

    topics = st.session_state.get("topics", {})
    if not topics:
        st.info(
            "📋 No topics found. Upload a syllabus first, or type a topic below."
        )
        topic_input = st.text_input("Enter a topic manually")
        selected_topic = topic_input.strip() if topic_input else None
    else:
        subjects = list(topics.keys())
        selected_subject = st.selectbox("Select a subject", subjects, key="ts_subject")
        subject_topics = topics.get(selected_subject, [])
        if subject_topics:
            selected_topic = st.selectbox("Select a topic", subject_topics, key="ts_topic")
        else:
            selected_topic = selected_subject

    if st.button("📋 Summarize", type="primary") and selected_topic:
        with st.spinner(f"Summarizing '{selected_topic}' from your notes…"):
            chunks = retrieve_by_subject(selected_subject, query=selected_topic, top_k=10) if topics else retrieve_by_topic(selected_topic, top_k=10)

            if not chunks:
                st.warning(f"📭 No notes found for topic '{selected_topic}'.")
            else:
                summary = summarize_topic(selected_topic, chunks)
                st.markdown(f"### 📋 Summary: *{selected_topic}*")
                st.markdown(
                    f'<div class="card"><p>{summary}</p></div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: Weak Topics
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📊 Weak Topics":
    st.markdown(
        '<div class="hero"><h1>Weak Topic Tracker</h1>'
        "<p>Track your performance, identify weak areas, and practice with MCQs.</p></div>",
        unsafe_allow_html=True,
    )

    # --- Performance overview ---
    performance = tracker.get_performance_summary()

    if performance:
        st.markdown("### 📊 Performance Overview")

        # Bar chart
        import pandas as pd

        df = pd.DataFrame([
            {"Topic": topic, "Score (%)": stats["score_pct"]}
            for topic, stats in performance.items()
        ])
        df = df.sort_values("Score (%)", ascending=True)
        st.bar_chart(df.set_index("Topic"), y="Score (%)", width="stretch")

        # Weak topics list
        weak = tracker.get_weak_topics()
        if weak:
            st.markdown("### 🔻 Weak Topics (Lowest Score First)")
            for w in weak:
                emoji = "🔴" if w["score_pct"] < 40 else ("🟡" if w["score_pct"] < 70 else "🟢")
                st.markdown(
                    f"{emoji} **{w['topic']}** — {w['score_pct']}% "
                    f"({w['correct']}/{w['total']} correct)"
                )
    else:
        st.info("📊 No quiz data yet. Practice some MCQs below to start tracking!")

    st.markdown("---")

    # --- MCQ Practice ---
    st.markdown("### 🧠 Practice MCQs")

    topics = st.session_state.get("topics", {})
    if not topics:
        topic_input = st.text_input("Enter a topic to practice")
        practice_topic = topic_input.strip() if topic_input else None
    else:
        # Flatten all topics for weak-topic matching
        all_flat_topics = flatten_topics(topics)

        # Suggest weak topics at the top
        weak_topic_names = [w["topic"] for w in tracker.get_weak_topics() if w["topic"] in all_flat_topics]
        other_topic_names = [t for t in all_flat_topics if t not in weak_topic_names]
        ordered_all = weak_topic_names + other_topic_names

        subjects = list(topics.keys())
        selected_subject = st.selectbox("Select a subject", subjects, key="wt_subject")
        subject_topics = topics.get(selected_subject, [])
        if subject_topics:
            # Re-order subject topics: weak first
            weak_in_subject = [t for t in subject_topics if t in weak_topic_names]
            other_in_subject = [t for t in subject_topics if t not in weak_topic_names]
            ordered_subject = weak_in_subject + other_in_subject
            practice_topic = st.selectbox(
                "Select topic (weak topics listed first)",
                ordered_subject,
                key="wt_topic",
            )
        else:
            practice_topic = selected_subject

    if st.button("🎲 Generate MCQs", type="primary") and practice_topic:
        with st.spinner(f"Generating MCQs for '{practice_topic}'…"):
            chunks = retrieve_by_subject(selected_subject, query=practice_topic, top_k=10) if topics else retrieve_by_topic(practice_topic, top_k=10)

            if not chunks:
                st.warning(f"📭 No notes found for '{practice_topic}'.")
            else:
                mcqs = generate_mcq(practice_topic, chunks)

                if not mcqs:
                    st.warning("Could not generate MCQs. Try again.")
                else:
                    st.session_state["current_mcqs"] = mcqs
                    st.session_state["mcq_topic"] = practice_topic
                    st.session_state["mcq_answers"] = {}

    # Display active MCQs
    if "current_mcqs" in st.session_state and st.session_state["current_mcqs"]:
        mcqs = st.session_state["current_mcqs"]
        mcq_topic = st.session_state.get("mcq_topic", "")

        st.markdown(f"### 📝 MCQ Quiz: *{mcq_topic}*")

        for idx, mcq in enumerate(mcqs):
            st.markdown(f"**Q{idx + 1}.** {mcq['question']}")

            options = mcq.get("options", {})
            if isinstance(options, dict):
                option_labels = [f"{k}: {v}" for k, v in options.items()]
                option_keys = list(options.keys())
            elif isinstance(options, list):
                option_labels = [f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options)]
                option_keys = [chr(65 + i) for i in range(len(options))]
            else:
                continue

            selected = st.radio(
                f"Your answer for Q{idx + 1}",
                option_labels,
                key=f"mcq_{idx}",
                label_visibility="collapsed",
            )

            if selected:
                # Extract the letter
                answer_letter = selected.split(":")[0].strip()
                st.session_state["mcq_answers"][idx] = answer_letter

            st.markdown("---")

        # Submit button
        if st.button("✅ Submit Answers", type="primary"):
            answers = st.session_state.get("mcq_answers", {})
            correct_count = 0
            total_answered = len(answers)

            for idx, mcq in enumerate(mcqs):
                if idx in answers:
                    user_ans = answers[idx]
                    correct_ans = mcq.get("correct_answer", "").strip().upper()
                    is_correct = user_ans.upper() == correct_ans

                    if is_correct:
                        correct_count += 1
                        st.success(f"Q{idx + 1}: ✅ Correct!")
                    else:
                        st.error(
                            f"Q{idx + 1}: ❌ Wrong. Correct answer: {correct_ans}"
                        )
                        st.info(f"💡 {mcq.get('explanation', '')}")

                    # Update tracker
                    tracker.update_score(mcq_topic, is_correct)

            # Show result
            if total_answered > 0:
                pct = round((correct_count / total_answered) * 100, 1)
                st.markdown(
                    f'<div class="stat-box" style="margin-top:1rem">'
                    f"<h2>{correct_count}/{total_answered}</h2>"
                    f"<p>Score: {pct}%</p></div>",
                    unsafe_allow_html=True,
                )

            # Clear MCQs after submission
            st.session_state["current_mcqs"] = []
            st.session_state["mcq_answers"] = {}


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: Private Tutor
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🧑‍🏫 Private Tutor":
    st.markdown(
        '<div class="hero"><h1>Private Tutor</h1>'
        "<p>Choose a subject and chat with your AI tutor — answers stay within that subject.</p></div>",
        unsafe_allow_html=True,
    )

    # --- Subject selection ---
    topics = st.session_state.get("topics", {})

    if not topics:
        st.info(
            "📋 No syllabus topics found. Upload a syllabus on the Upload page first, "
            "or type a subject below."
        )
        manual_subject = st.text_input("Enter a subject to study")
        tutor_subject = manual_subject.strip() if manual_subject else None
    else:
        subjects = list(topics.keys())
        tutor_subject = st.selectbox(
            "📚 Select a subject to study",
            subjects,
            key="tutor_subject_select",
        )
        # Show the topics under the selected subject for context
        if tutor_subject and topics.get(tutor_subject):
            with st.expander(f"📋 Topics under {tutor_subject}"):
                for t in topics[tutor_subject]:
                    st.markdown(f"- {t}")

    # --- Initialise chat state ---
    if "tutor_messages" not in st.session_state:
        st.session_state.tutor_messages = []
    if "tutor_active_subject" not in st.session_state:
        st.session_state.tutor_active_subject = None

    # Reset chat when subject changes
    if tutor_subject and tutor_subject != st.session_state.tutor_active_subject:
        st.session_state.tutor_messages = []
        st.session_state.tutor_active_subject = tutor_subject

    if not tutor_subject:
        st.warning("Please select or enter a subject to start chatting.")
    else:
        st.markdown(f"### 💬 Chatting about: *{tutor_subject}*")

        # Display conversation history
        for msg in st.session_state.tutor_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input(f"Ask anything about {tutor_subject}…")

        if user_input:
            # Append user message and rerun to show it immediately
            st.session_state.tutor_messages.append({"role": "user", "content": user_input})
            st.session_state["_tutor_pending"] = True
            st.rerun()

        # Process pending message (after rerun)
        if st.session_state.get("_tutor_pending"):
            st.session_state["_tutor_pending"] = False
            last_user_msg = st.session_state.tutor_messages[-1]["content"]

            with st.spinner("Thinking…"):
                chunks = retrieve_by_subject(tutor_subject, query=last_user_msg, top_k=8)

                reply = tutor_chat(
                    message=last_user_msg,
                    subject=tutor_subject,
                    retrieved_chunks=chunks,
                    conversation_history=st.session_state.tutor_messages[:-1],
                )

            st.session_state.tutor_messages.append({"role": "assistant", "content": reply})
            st.rerun()

        # Clear chat button
        if st.session_state.tutor_messages:
            if st.button("🗑️ Clear Chat", key="clear_tutor_chat"):
                st.session_state.tutor_messages = []
                st.rerun()

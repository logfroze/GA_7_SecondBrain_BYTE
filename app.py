import os
import streamlit as st
from pathlib import Path

from src.rag_pipeline import SecondBrainRAG
from src.config import EMBEDDING_MODEL, LLM_MODEL, GEMINI_API_KEY, CHROMA_PERSIST_DIR

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Second Brain · Document RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Clean, Restrained Developer UI Styling
# Solid colors, subtle borders, sharp typography, zero AI/template decorations.
# ==============================================================================
DEV_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace !important;
}

/* Hide ONLY the Deploy button */
.stAppDeployButton,
div[data-testid="stToolbar"] .stAppDeployButton,
div[data-testid="stToolbar"] > div:has(.stAppDeployButton) {
    display: none !important;
    visibility: hidden !important;
}

/* Header, 3-dots menu, and sidebar toggle arrow */
header[data-testid="stHeader"] {
    background: transparent !important;
    color: #e6edf3 !important;
}

#MainMenu,
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: block !important;
    visibility: visible !important;
}

/* Ensure 3 dots and sidebar arrow icons are clean and visible */
header[data-testid="stHeader"] button,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    color: #c9d1d9 !important;
    background-color: transparent !important;
}

header[data-testid="stHeader"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    color: #ffffff !important;
    background-color: #21262d !important;
}

/* Solid Neutral Developer Dark Theme */
.stApp {
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #090d13 !important;
    border-right: 1px solid #21262d !important;
}

/* App Header */
.dev-header {
    padding: 4px 0 16px 0;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}

.dev-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #f0f6fc;
    margin: 0;
    letter-spacing: -0.01em;
}

.dev-subtitle {
    font-size: 0.825rem;
    color: #8b949e;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Sidebar Section Headers */
.sidebar-heading {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #8b949e;
    margin-top: 14px;
    margin-bottom: 8px;
}

/* File Uploader Dropzone - Seamless dark styling across themes */
[data-testid="stFileUploadDropzone"] {
    background-color: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
    color: #8b949e !important;
}

[data-testid="stFileUploadDropzone"] button {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
}

/* Status Table */
.status-table {
    width: 100%;
    border-collapse: collapse;
    background-color: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-size: 0.8rem;
    margin-top: 6px;
    overflow: hidden;
}

.status-table tr {
    border-bottom: 1px solid #21262d;
}

.status-table tr:last-child {
    border-bottom: none;
}

.status-table td {
    padding: 7px 10px;
}

.status-table td.label {
    color: #8b949e;
    font-weight: 500;
    width: 42%;
}

.status-table td.val {
    color: #c9d1d9;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    text-align: right;
}

.status-indicator {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: #2ea043;
    margin-right: 5px;
    vertical-align: middle;
}

/* Solid Developer Buttons */
div.stButton > button {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.8125rem !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

div.stButton > button:hover {
    background-color: #30363d !important;
    border-color: #8b949e !important;
    color: #ffffff !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Primary Index Button */
div.stButton > button[kind="primary"] {
    background-color: #238636 !important;
    border-color: rgba(240, 246, 252, 0.1) !important;
    color: #ffffff !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #2ea043 !important;
}

/* Danger / Reset Button Specific Styling */
.danger-action button {
    background-color: #21262d !important;
    color: #f85149 !important;
    border-color: #30363d !important;
}

.danger-action button:hover {
    background-color: #b62324 !important;
    color: #ffffff !important;
    border-color: #b62324 !important;
}

/* Citations Tags */
.citation-tag {
    display: inline-block;
    padding: 1px 6px;
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #58a6ff;
    margin-right: 6px;
    margin-top: 4px;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 6px !important;
    padding: 14px !important;
    margin-bottom: 10px !important;
}

/* Bottom Floating Container & Chat Input Bar */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
.stChatFloatingInputContainer {
    background-color: #0d1117 !important;
    background: #0d1117 !important;
}

.stChatInputContainer {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

.stChatInputContainer:focus-within {
    border-color: #58a6ff !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
    color: #8b949e !important;
}
</style>
"""
st.markdown(DEV_CSS, unsafe_allow_html=True)


# ==============================================================================
# Pipeline & Session State Initialization
# ==============================================================================
if "rag" not in st.session_state:
    try:
        st.session_state.rag = SecondBrainRAG()
    except Exception as e:
        st.error(f"Initialization error: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ==============================================================================
# Sidebar: Ingestion & Status
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-heading">Document Ingestion</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / uploaded_file.name

        if st.button("Index Document", use_container_width=True, type="primary"):
            with st.spinner("Processing chunks & embeddings..."):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    summary = st.session_state.rag.index_pdf(str(file_path))
                    st.success(f"Indexed {summary['source']}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pages", summary["total_pages"])
                    with col2:
                        st.metric("Chunks", summary["indexed_count"])
                except Exception as e:
                    st.error(f"Index error: {e}")

    st.markdown('<div class="sidebar-heading">System Parameters</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <table class="status-table">
        <tr>
            <td class="label">Status</td>
            <td class="val"><span class="status-indicator"></span>Ready</td>
        </tr>
        <tr>
            <td class="label">LLM</td>
            <td class="val">{LLM_MODEL}</td>
        </tr>
        <tr>
            <td class="label">Embeddings</td>
            <td class="val">{EMBEDDING_MODEL}</td>
        </tr>
        <tr>
            <td class="label">Vector DB</td>
            <td class="val">ChromaDB</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="danger-action">', unsafe_allow_html=True)
    if st.button("Reset Knowledge Base", use_container_width=True):
        if "rag" in st.session_state:
            st.session_state.rag.reset_knowledge_base()
            st.session_state.messages = []
            st.info("Vector store reset.")
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# Main Interface: Clean Header & Query Interface
# ==============================================================================
st.markdown("""
<div class="dev-header">
    <h1 class="dev-title">Second Brain</h1>
    <p class="dev-subtitle">Local Document Retrieval-Augmented Generation · PyMuPDF · ChromaDB · Page Citations</p>
</div>
""", unsafe_allow_html=True)


# Quick Reference Queries (Clean Developer List)
if not st.session_state.messages:
    st.markdown('<div class="sidebar-heading">Sample Queries</div>', unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)

    with q_col1:
        if st.button("What is artificial intelligence?", use_container_width=True):
            st.session_state.pending_query = "What is artificial intelligence and what are its historical foundations?"

    with q_col2:
        if st.button("When was Dartmouth Conference held?", use_container_width=True):
            st.session_state.pending_query = "When was the Dartmouth Conference held and who coined the term AI?"

    with q_col3:
        if st.button("What AI applications are discussed?", use_container_width=True):
            st.session_state.pending_query = "What are the real-world applications of AI covered in the document?"


# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        citations = msg.get("citations", [])
        if citations:
            tags = "".join([f'<span class="citation-tag">Page {p}</span>' for p in citations])
            st.markdown(f'<div style="margin-top: 8px;"><strong>Citations:</strong> {tags}</div>', unsafe_allow_html=True)

        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"Context Excerpts ({len(sources)})"):
                for idx, s in enumerate(sources, 1):
                    page = s.get("page", "N/A")
                    src = s.get("source", "Document")
                    sim = s.get("similarity")
                    sim_str = f" · similarity: {round(sim, 3)}" if sim is not None else ""

                    st.markdown(f"`[{idx}]` **{src}** — `Page {page}`{sim_str}")
                    st.code(s.get("text_preview", ""), language="text")


# Chat Query Execution
pending = st.session_state.pending_query
st.session_state.pending_query = None

chat_in = st.chat_input("Query indexed document knowledge base...")
active_query = pending if pending else chat_in

if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer..."):
            try:
                result = st.session_state.rag.query(active_query)
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                sources = result.get("sources", [])

                st.markdown(answer)

                if citations:
                    tags = "".join([f'<span class="citation-tag">Page {p}</span>' for p in citations])
                    st.markdown(f'<div style="margin-top: 8px;"><strong>Citations:</strong> {tags}</div>', unsafe_allow_html=True)

                if sources:
                    with st.expander(f"Context Excerpts ({len(sources)})"):
                        for idx, s in enumerate(sources, 1):
                            page = s.get("page", "N/A")
                            src = s.get("source", "Document")
                            sim = s.get("similarity")
                            sim_str = f" · similarity: {round(sim, 3)}" if sim is not None else ""

                            st.markdown(f"`[{idx}]` **{src}** — `Page {page}`{sim_str}")
                            st.code(s.get("text_preview", ""), language="text")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Query error: {e}")

import os
import streamlit as st
import streamlit.components.v1 as components
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
# Dynamic Adaptive Theming: Dark Mode & Full Light Mode System
# High-contrast, clean developer aesthetic across all surfaces and states.
# ==============================================================================
DEV_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Default Dark Theme Design Tokens */
:root, [data-theme="dark"] {
    --app-bg: #0d1117;
    --sidebar-bg: #090d13;
    --card-bg: #161b22;
    --border-color: #21262d;
    --border-subtle: #30363d;
    --border-hover: #8b949e;
    --text-primary: #f0f6fc;
    --text-secondary: #8b949e;
    --text-val: #c9d1d9;
    --btn-bg: #21262d;
    --btn-border: #30363d;
    --btn-text: #e6edf3;
    --btn-hover-bg: #30363d;
    --danger-text: #f85149;
    --citation-bg: #161b22;
    --citation-border: #30363d;
    --citation-text: #58a6ff;
    --header-icon: #c9d1d9;
    --header-hover-bg: #21262d;
    --accent-color: #58a6ff;
    --code-bg: #161b22;
    --code-text: #e6edf3;
}

/* Light Theme Design Tokens (Full view coverage, high contrast) */
:root[data-theme="light"],
body[data-theme="light"],
.stApp[data-theme="light"],
[data-theme="light"] {
    --app-bg: #ffffff;
    --sidebar-bg: #f6f8fa;
    --card-bg: #ffffff;
    --border-color: #d0d7de;
    --border-subtle: #d8dee4;
    --border-hover: #656d76;
    --text-primary: #1f2328;
    --text-secondary: #59636e;
    --text-val: #24292f;
    --btn-bg: #f6f8fa;
    --btn-border: #d0d7de;
    --btn-text: #24292f;
    --btn-hover-bg: #eaeef2;
    --danger-text: #cf222e;
    --citation-bg: #ddf4ff;
    --citation-border: #54aeff80;
    --citation-text: #0969da;
    --header-icon: #59636e;
    --header-hover-bg: #eaeef2;
    --accent-color: #0969da;
    --code-bg: #f6f8fa;
    --code-text: #24292f;
}

/* System preference fallback */
@media (prefers-color-scheme: light) {
    :root:not([data-theme="dark"]),
    .stApp:not([data-theme="dark"]) {
        --app-bg: #ffffff;
        --sidebar-bg: #f6f8fa;
        --card-bg: #ffffff;
        --border-color: #d0d7de;
        --border-subtle: #d8dee4;
        --border-hover: #656d76;
        --text-primary: #1f2328;
        --text-secondary: #59636e;
        --text-val: #24292f;
        --btn-bg: #f6f8fa;
        --btn-border: #d0d7de;
        --btn-text: #24292f;
        --btn-hover-bg: #eaeef2;
        --danger-text: #cf222e;
        --citation-bg: #ddf4ff;
        --citation-border: #54aeff80;
        --citation-text: #0969da;
        --header-icon: #59636e;
        --header-hover-bg: #eaeef2;
        --accent-color: #0969da;
        --code-bg: #f6f8fa;
        --code-text: #24292f;
    }
}

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
    color: var(--text-primary) !important;
}

#MainMenu,
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    display: block !important;
    visibility: visible !important;
}

header[data-testid="stHeader"] button,
[data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    color: var(--header-icon) !important;
    background-color: transparent !important;
}

header[data-testid="stHeader"] button:hover,
[data-testid="collapsedControl"] button:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    color: var(--text-primary) !important;
    background-color: var(--header-hover-bg) !important;
}

/* Main App Canvas */
.stApp {
    background-color: var(--app-bg) !important;
    color: var(--text-primary) !important;
    transition: background-color 0.2s ease, color 0.2s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color) !important;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}

/* App Header */
.dev-header {
    padding: 4px 0 16px 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 20px;
}

.dev-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}

.dev-subtitle {
    font-size: 0.825rem;
    color: var(--text-secondary);
    margin-top: 4px;
    margin-bottom: 0;
}

/* Sidebar Section Headers */
.sidebar-heading {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-top: 14px;
    margin-bottom: 8px;
}

/* File Uploader Dropzone */
[data-testid="stFileUploadDropzone"] {
    background-color: var(--card-bg) !important;
    border: 1px dashed var(--border-subtle) !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploadDropzone"] div,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] small {
    color: var(--text-secondary) !important;
}

[data-testid="stFileUploadDropzone"] button {
    background-color: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    border: 1px solid var(--btn-border) !important;
}

/* Status Table */
.status-table {
    width: 100%;
    border-collapse: collapse;
    background-color: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    font-size: 0.8rem;
    margin-top: 6px;
    overflow: hidden;
}

.status-table tr {
    border-bottom: 1px solid var(--border-color);
}

.status-table tr:last-child {
    border-bottom: none;
}

.status-table td {
    padding: 7px 10px;
}

.status-table td.label {
    color: var(--text-secondary);
    font-weight: 500;
    width: 42%;
}

.status-table td.val {
    color: var(--text-val);
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

/* Action Buttons */
div.stButton > button {
    background-color: var(--btn-bg) !important;
    color: var(--btn-text) !important;
    border: 1px solid var(--btn-border) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.8125rem !important;
    padding: 6px 14px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}

div.stButton > button:hover {
    background-color: var(--btn-hover-bg) !important;
    border-color: var(--border-hover) !important;
    color: var(--text-primary) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Primary Index Button */
div.stButton > button[kind="primary"] {
    background-color: #1f883d !important;
    border-color: rgba(31, 136, 61, 0.4) !important;
    color: #ffffff !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #1a7f37 !important;
}

/* Danger / Reset Button */
.danger-action button {
    background-color: var(--btn-bg) !important;
    color: var(--danger-text) !important;
    border-color: var(--btn-border) !important;
}

.danger-action button:hover {
    background-color: #cf222e !important;
    color: #ffffff !important;
    border-color: #cf222e !important;
}

/* Citations Tags */
.citation-tag {
    display: inline-block;
    padding: 2px 8px;
    background-color: var(--citation-bg);
    border: 1px solid var(--citation-border);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--citation-text);
    font-weight: 500;
    margin-right: 6px;
    margin-top: 4px;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 14px !important;
    margin-bottom: 10px !important;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: var(--text-primary) !important;
}

/* Bottom Floating Container & Chat Input Bar */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
.stChatFloatingInputContainer {
    background-color: var(--app-bg) !important;
    background: var(--app-bg) !important;
}

.stChatInputContainer {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

.stChatInputContainer:focus-within {
    border-color: var(--accent-color) !important;
}

.stChatInputContainer textarea {
    color: var(--text-primary) !important;
    background-color: transparent !important;
}

/* Expander & Code Blocks */
.streamlit-expanderHeader {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
}

[data-testid="stCodeBlock"] {
    background-color: var(--code-bg) !important;
    border: 1px solid var(--border-color) !important;
}

/* Invisible Theme Sync Component */
div[data-testid="stCustomComponentV1"]:has(iframe),
div:has(> iframe[height="0"]),
iframe[height="0"] {
    display: none !important;
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
}
</style>
"""
st.markdown(DEV_CSS, unsafe_allow_html=True)

# Theme Synchronizer Component
# Monitors Streamlit's internal theme changes (Light / Dark) and syncs data-theme attribute
components.html(
    """
    <script>
    (function() {
        function syncTheme() {
            try {
                const parentDoc = window.parent.document;
                const stApp = parentDoc.querySelector('.stApp');
                if (!stApp) return;

                let theme = null;

                // 1. Check Streamlit's active theme in settings radio menu if selected
                const checkedRadio = parentDoc.querySelector('[data-testid="stThemeSwitcher"] [aria-checked="true"]');
                if (checkedRadio) {
                    const txt = (checkedRadio.innerText || checkedRadio.getAttribute('aria-label') || '').toLowerCase();
                    if (txt.includes('light')) theme = 'light';
                    else if (txt.includes('dark')) theme = 'dark';
                }

                // 2. Check computed colorScheme on stApp
                if (!theme) {
                    const comp = window.parent.getComputedStyle(stApp);
                    if (comp.colorScheme === 'light') {
                        theme = 'light';
                    } else if (comp.colorScheme === 'dark') {
                        theme = 'dark';
                    }
                }

                // 3. Check system media preference as fallback
                if (!theme) {
                    if (window.parent.matchMedia && window.parent.matchMedia('(prefers-color-scheme: light)').matches) {
                        theme = 'light';
                    } else {
                        theme = 'dark';
                    }
                }

                // Synchronize data-theme across DOM roots
                if (parentDoc.documentElement.getAttribute('data-theme') !== theme) {
                    parentDoc.documentElement.setAttribute('data-theme', theme);
                }
                if (parentDoc.body && parentDoc.body.getAttribute('data-theme') !== theme) {
                    parentDoc.body.setAttribute('data-theme', theme);
                }
                if (stApp.getAttribute('data-theme') !== theme) {
                    stApp.setAttribute('data-theme', theme);
                }
            } catch (e) {}
        }
        syncTheme();
        setInterval(syncTheme, 250);
    })();
    </script>
    """,
    height=0,
    width=0,
)


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
                if hasattr(st.session_state.rag, "query"):
                    result = st.session_state.rag.query(active_query)
                else:
                    result = st.session_state.rag.ask(active_query)
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

# BYTE AVIP 2026 — Generative AI Task 7
## Second Brain / Document Interrogator — Progress & Engineering Notes

**Track:** Generative AI  
**Task:** 7 — Second Brain / Document Interrogator  
**Status:** Complete  

---

## What This Project Does

A local Retrieval-Augmented Generation (RAG) system. The user uploads a PDF document, which is indexed into a persistent ChromaDB vector database. The user then asks natural language questions about the document and receives answers grounded strictly in the document text, with exact page number citations.

---

## Tech Stack

| Component | Library / Tool |
| :--- | :--- |
| PDF text extraction | PyMuPDF (`pymupdf >= 1.24.0`) |
| Text chunking | Pure Python custom recursive splitter (`src/chunker.py`) |
| Local embeddings | `all-MiniLM-L6-v2` via SentenceTransformers / ONNX Runtime |
| Vector database | ChromaDB (`chromadb >= 0.5.0`, cosine similarity, persistent) |
| Language model | Google Gemini `gemini-2.5-flash` (`google-genai >= 1.0.0`) |
| Web UI | Streamlit (`streamlit >= 1.35.0`) |
| CLI | Rich (`rich >= 13.7.0`) |
| Configuration | `python-dotenv` |
| Language | Python 3.11 / 3.12 |

---

## Architecture

```
PDF Upload
    ↓
PyMuPDF extraction (page-by-page, 1-indexed)
    ↓
Pure Python recursive chunker (chunk_size=800, overlap=150)
    ↓
all-MiniLM-L6-v2 local embedding (CPU, no external API)
    ↓
ChromaDB upsert (cosine, persistent at data/chroma_db/)
    ↓
User query → local embedding → similarity search (top-k=4)
    ↓
Gemini grounded answer generation (strict no-hallucination prompt)
    ↓
Answer + page citations (from chunk metadata, not LLM text)
```

---

## Engineering Challenges

### 1. Windows DLL initialization failure (ONNX Runtime / torch)

**Problem:** `onnxruntime` fails with `DLL initialization failed` on Windows Python 3.11/3.12. Error points to `torch` rather than `onnxruntime`, making root cause non-obvious. Appears at runtime, not pip install time.

**Solution:**
- Documented workaround in README: `pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall`
- Rewrote text chunker in pure Python (`src/chunker.py`) — eliminates the LangChain dependency that pulled in a conflicting torch version
- `requirements.txt` excludes torch entirely

**Result:** Embedding engine loads reliably on Windows CPU.

---

### 2. LLM hallucinating page citations

**Problem:** Early LLM responses generated `[Page X]` citations with page numbers not present in the retrieved context.

**Solution:**
- Context blocks formatted with `--- EXCERPT N [Source: X | Page: Y] ---` markers
- System prompt contains explicit rule: "NEVER invent or hallucinate page numbers. Only cite page numbers that explicitly appear in the excerpt headers."
- `citations` field in the API response is derived from `retrieved_chunks[i]["metadata"]["page"]` — NOT from parsing LLM text output

**Result:** Citation tags always match actual retrieved document pages regardless of what the LLM writes.

---

### 3. Streamlit dark CSS broken by native light theme

**Problem:** Custom dark CSS injected via `st.markdown(DEV_CSS, unsafe_allow_html=True)` was overridden by Streamlit's native light theme. Top header bar, file uploader dropzone, and bottom chat input container rendered white.

**Solution:**
- `.streamlit/config.toml` updated with `base = "dark"` and explicit hex overrides
- Custom CSS targets `[data-testid="stBottom"]`, `[data-testid="stFileUploadDropzone"]`, `header[data-testid="stHeader"]` with `!important`

**Result:** Consistent dark theme in all scenarios.

---

## Verified Deliverables

- [x] PDF upload, indexing, and persistent ChromaDB vector storage
- [x] Local CPU-only embedding (all-MiniLM-L6-v2, no external API)
- [x] Grounded question answering via Google Gemini with exact page citations
- [x] Streamlit web UI with file uploader, chat interface, citation tags
- [x] Rich-powered CLI (`cli.py`)
- [x] Graceful LLM fallback chain (model chain → extractive fallback)
- [x] Sample Q&A: `sample_qa.md`

---

## Source Files

| File | Purpose |
| :--- | :--- |
| `src/loader.py` | PDF ingestion via PyMuPDF |
| `src/chunker.py` | Pure Python recursive text chunker |
| `src/embeddings.py` | Local SentenceTransformer embedding engine |
| `src/vectorstore.py` | ChromaDB persistent vector store |
| `src/generator.py` | Gemini-powered grounded answer generator |
| `src/rag_pipeline.py` | End-to-end pipeline orchestration |
| `src/config.py` | Configuration loading (dotenv + st.secrets) |
| `app.py` | Streamlit web UI |
| `cli.py` | Rich CLI interface |
| `.streamlit/config.toml` | Streamlit theme configuration |

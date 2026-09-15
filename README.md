# 🧠 GA_7_SecondBrain_BYTE

> **BYTE Arithmatrix Generative AI Internship — Task 7: Second Brain**
>
> A local Retrieval-Augmented Generation (RAG) system that accepts a PDF document, indexes it into a persistent vector store, and answers questions using strictly retrieved context with exact page citations. Zero hallucinated citations.

---

## Architecture Overview

```
PDF Document
    ↓
[loader.py]   → Extract text + exact 1-indexed page numbers (PyMuPDF)
    ↓
[chunker.py]  → Context-aware chunking, preserve page metadata (LangChain splitter)
    ↓
[embeddings.py] → Local dense embeddings: all-MiniLM-L6-v2 (SentenceTransformers, CPU)
    ↓
[vectorstore.py] → Upsert + persist in ChromaDB (cosine similarity)
    ↓
User asks question
    ↓
[vectorstore.py] → Semantic similarity search → top-k chunks
    ↓
[generator.py] → Grounded answer via Google Gemini (gemini-2.5-flash)
    ↓
Answer + Exact [Page X] citations
```

### Module Breakdown

| File | Purpose |
|------|---------|
| `src/config.py` | Loads `.env`, centralises all settings |
| `src/loader.py` | PyMuPDF-based PDF text extraction per page |
| `src/chunker.py` | Recursive character splitter; preserves page number in metadata |
| `src/embeddings.py` | Local `all-MiniLM-L6-v2` embedding engine (offline, free) |
| `src/vectorstore.py` | ChromaDB persistent client: index & similarity search |
| `src/generator.py` | Gemini grounded QA with strict no-hallucination prompting |
| `src/rag_pipeline.py` | End-to-end orchestrator: `index_pdf()` and `ask()` |
| `app.py` | Streamlit web UI |
| `cli.py` | Rich-powered terminal interface |

---

## Setup Instructions

### 1. Prerequisites

- Python 3.11 or 3.12
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/<your-username>/GA_7_SecondBrain_BYTE.git
cd GA_7_SecondBrain_BYTE
```

### 3. Create & Activate Virtual Environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If you get a `DLL initialization failed` error for torch, install the stable CPU-only version:
> ```bash
> pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu --force-reinstall
> ```

### 5. Configure API Key

Copy the example environment file and add your Gemini API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
GEMINI_API_KEY=your_actual_api_key_here
```

Get a **free** Gemini API key at: https://aistudio.google.com/app/apikey

---

## Usage

### Streamlit Web App (Recommended)

```bash
streamlit run app.py
```

1. Open the URL shown in the terminal (usually `http://localhost:8501`)
2. Upload a PDF via the left sidebar → click **Index Document**
3. Type your question in the chat input box
4. Get grounded answers with exact page citations

### Terminal / CLI

**Interactive mode** (recommended for CLI):

```bash
python cli.py interactive
```

Commands inside interactive mode:

```
index <path/to/document.pdf>   → Index a PDF
<your question>                → Ask anything
clear                          → Reset the vector store
exit / quit                    → Exit
```

**One-shot commands:**

```bash
# Index a PDF
python cli.py index --pdf sample_data/sample_document.pdf

# Ask a question
python cli.py ask --query "What are the main topics covered in this document?"
```

---

## Sample Queries

After indexing `sample_data/sample_document.pdf`, try:

```
What is artificial intelligence?
How does machine learning differ from deep learning?
What is a neural network?
What applications does natural language processing have?
What are the ethical considerations in AI?
```

See `sample_qa.md` for verified grounded answers with exact page citations.

---

## Model & License Information

| Component | Model / Library | License |
|-----------|----------------|---------|
| PDF Extraction | PyMuPDF (`fitz`) | AGPL-3.0 / Commercial |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) | Apache 2.0 |
| Vector Store | ChromaDB | Apache 2.0 |
| Language Model | Google Gemini (`gemini-2.5-flash`) | Google AI Terms |
| Text Splitting | LangChain Text Splitters | MIT |
| UI | Streamlit | Apache 2.0 |
| CLI | Rich | MIT |

**This project itself** is licensed under the MIT License — see `LICENSE`.

---

## Quality Guarantees

- ✅ **No hallucinated citations** — The LLM is strictly instructed to only cite page numbers that appear in the retrieved context excerpts
- ✅ **Grounded answers only** — If context doesn't contain the answer, the system says so
- ✅ **No API keys committed** — `.env` is git-ignored; only `.env.example` is versioned
- ✅ **Fully local embeddings** — `all-MiniLM-L6-v2` runs offline on CPU, no external embedding API needed
- ✅ **Persistent vector store** — ChromaDB persists to `data/chroma_db/`; re-indexing is not required on restart

---

## Project Structure

```
GA_7_SecondBrain_BYTE/
├── src/
│   ├── __init__.py
│   ├── config.py          # Centralised config + env loading
│   ├── loader.py          # PyMuPDF PDF loader
│   ├── chunker.py         # Text chunker with page metadata
│   ├── embeddings.py      # Local SentenceTransformer engine
│   ├── vectorstore.py     # ChromaDB vector store manager
│   ├── generator.py       # Gemini grounded answer generator
│   └── rag_pipeline.py    # End-to-end RAG orchestrator
├── sample_data/
│   └── sample_document.pdf   # Multi-page sample PDF for demo
├── data/
│   └── chroma_db/            # Persistent vector DB (auto-created, git-ignored)
├── screenshots/              # UI screenshots (git-tracked)
├── app.py                    # Streamlit web application
├── cli.py                    # Terminal CLI
├── sample_qa.md              # Verified sample Q&A with page citations
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Internship: BYTE Arithmatrix AVIP 2026 — Generative AI Track

**Submitted by:** Muhammad Umar  
**Task:** GA_7 — Second Brain (Local RAG System)

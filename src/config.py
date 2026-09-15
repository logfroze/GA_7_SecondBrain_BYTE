import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

def get_setting(key: str, default: str = "") -> str:
    """Retrieve setting from .env / os.environ, falling back to st.secrets, then default."""
    val = os.getenv(key)
    if val is not None and val.strip() != "":
        return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default

# API Keys & Configuration
GEMINI_API_KEY = get_setting("GEMINI_API_KEY", "")

# Model Configurations
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = get_setting("LLM_MODEL", "gemini-2.5-flash")

# Vector Store Configurations
CHROMA_PERSIST_DIR = get_setting("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))

# Document Chunking Settings
CHUNK_SIZE = int(get_setting("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(get_setting("CHUNK_OVERLAP", "150"))

# Retrieval Settings
TOP_K_RESULTS = int(get_setting("TOP_K_RESULTS", "4"))

def validate_config():
    """Validates that necessary configuration and API keys are present."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set it in your .env file or environment variables."
        )

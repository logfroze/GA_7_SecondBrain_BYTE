from pathlib import Path
from typing import Dict, Any, Optional

from src.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    validate_config,
)
from src.loader import PDFLoader
from src.chunker import DocumentChunker
from src.embeddings import LocalEmbeddingEngine
from src.vectorstore import ChromaVectorStore
from src.generator import GroundedAnswerGenerator


class SecondBrainRAG:
    """End-to-End Local RAG pipeline for the Second Brain application."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        self.persist_dir = persist_dir or CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.llm_model_name = llm_model or LLM_MODEL

        # Initialize components
        self.embedding_engine = LocalEmbeddingEngine(model_name=self.embedding_model_name)
        self.vector_store = ChromaVectorStore(
            persist_directory=self.persist_dir,
            embedding_engine=self.embedding_engine,
        )
        self.chunker = DocumentChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self._generator = None

    @property
    def generator(self) -> GroundedAnswerGenerator:
        if self._generator is None:
            if not self.api_key:
                validate_config()
            self._generator = GroundedAnswerGenerator(
                api_key=self.api_key,
                model_name=self.llm_model_name,
            )
        return self._generator

    def index_pdf(self, pdf_path: str, collection_name: str = "second_brain_docs") -> Dict[str, Any]:
        """
        Loads, chunks, and indexes a PDF document into ChromaDB.
        Returns indexing summary.
        """
        path = Path(pdf_path)
        loader = PDFLoader(str(path))
        page_docs = loader.load()

        chunks = self.chunker.chunk_documents(page_docs)
        indexed_count = self.vector_store.index_chunks(chunks, collection_name=collection_name)

        return {
            "source": path.name,
            "total_pages": len(page_docs),
            "total_chunks": len(chunks),
            "indexed_count": indexed_count,
            "collection_name": collection_name,
        }

    def ask(
        self,
        question: str,
        k: int = TOP_K_RESULTS,
        collection_name: str = "second_brain_docs"
    ) -> Dict[str, Any]:
        """
        Queries the indexed knowledge base and generates a grounded response.
        Returns the answer, citation page list, and supporting excerpt sources.
        """
        retrieved_chunks = self.vector_store.similarity_search(
            query=question,
            k=k,
            collection_name=collection_name,
        )

        return self.generator.generate_answer(query=question, retrieved_chunks=retrieved_chunks)

    def reset_knowledge_base(self, collection_name: str = "second_brain_docs"):
        """Clears the stored vector collection."""
        self.vector_store.clear_collection(collection_name)

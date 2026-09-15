from typing import List, Optional


class LocalEmbeddingEngine:
    """
    Generates dense vector representations locally using the all-MiniLM-L6-v2 model.
    Uses Chroma's native high-performance ONNX runtime engine with fallback to SentenceTransformer.
    Runs 100% locally and offline.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            try:
                from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
                self._engine = ONNXMiniLM_L6_V2()
            except Exception:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._engine = SentenceTransformer(self.model_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")
        return self._engine

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of text chunks into 384-dimensional vectors."""
        if not texts:
            return []
        eng = self.engine
        if hasattr(eng, "encode"):
            embeddings = eng.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return embeddings.tolist()
        # Chroma ONNXMiniLM_L6_V2 callable returns list of numpy arrays
        results = eng(texts)
        converted = []
        for vec in results:
            if hasattr(vec, "tolist"):
                converted.append(vec.tolist())
            else:
                converted.append([float(x) for x in vec])
        return converted

    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query string into a 384-dimensional vector."""
        eng = self.engine
        if hasattr(eng, "encode"):
            embedding = eng.encode(query, show_progress_bar=False, normalize_embeddings=True)
            return embedding.tolist()
        # Chroma ONNXMiniLM_L6_V2 callable takes a list of strings
        results = eng([query])
        vec = results[0]
        if hasattr(vec, "tolist"):
            return vec.tolist()
        return [float(x) for x in vec]

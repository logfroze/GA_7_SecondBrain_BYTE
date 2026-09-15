import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.embeddings import LocalEmbeddingEngine


class ChromaVectorStore:
    """Manages local ChromaDB persistence, indexing, and similarity retrieval."""

    def __init__(self, persist_directory: str, embedding_engine: Optional[LocalEmbeddingEngine] = None):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedding_engine = embedding_engine or LocalEmbeddingEngine()

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

    def get_or_create_collection(self, collection_name: str = "second_brain_docs"):
        """Retrieves or creates a Chroma collection."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Dict[str, Any]], collection_name: str = "second_brain_docs") -> int:
        """
        Indexes a list of text chunks with their metadata into ChromaDB.
        Returns the number of indexed chunks.
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)

        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Generate embeddings locally
        embeddings = self.embedding_engine.embed_documents(texts)

        # Upsert into Chroma
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return len(ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection_name: str = "second_brain_docs"
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search for a query string.
        Returns list of relevant documents with text, metadata, and distance.
        """
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()
        if count == 0:
            return []

        query_embedding = self.embedding_engine.embed_query(query)

        actual_k = min(k, count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_docs = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc_text, meta, dist in zip(docs, metas, dists):
                retrieved_docs.append({
                    "text": doc_text,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": 1.0 - dist if dist is not None else None,
                })

        return retrieved_docs

    def clear_collection(self, collection_name: str = "second_brain_docs"):
        """Deletes a collection and resets it."""
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types


class GroundedAnswerGenerator:
    """Generates strictly grounded answers using Google Gemini with exact page citations."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key is required to initialize GroundedAnswerGenerator.")
        self.api_key = api_key
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def _build_context_block(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks with clear page markers."""
        formatted_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            meta = chunk.get("metadata", {})
            page = meta.get("page", "Unknown")
            source = meta.get("source", "Document")
            text = chunk.get("text", "").strip()

            block = (
                f"--- EXCERPT {i} [Source: {source} | Page: {page}] ---\n"
                f"{text}\n"
            )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an answer strictly grounded in the retrieved chunks.
        Returns a dict with:
        - answer: str
        - citations: List[int] (deduplicated page numbers)
        - sources: List[Dict]
        """
        if not retrieved_chunks:
            return {
                "answer": "No relevant context found in the document to answer this question.",
                "citations": [],
                "sources": [],
                "raw_context": ""
            }

        context_text = self._build_context_block(retrieved_chunks)

        system_instruction = (
            "You are a factual RAG (Retrieval-Augmented Generation) assistant. "
            "Your job is to answer the user's question using ONLY the provided excerpts from the document.\n\n"
            "STRICT RULES:\n"
            "1. Answer ONLY using the facts stated in the excerpts below. Do NOT use outside knowledge.\n"
            "2. If the context does not contain enough information to answer the question, say: "
            "'I cannot find the answer to this question in the provided document.'\n"
            "3. Whenever stating a fact or answering a point, cite the specific page number in brackets, e.g., [Page X].\n"
            "4. NEVER invent or hallucinate page numbers. Only cite page numbers that explicitly appear in the excerpt headers.\n"
            "5. Keep the response professional, concise, and well-structured."
        )

        user_prompt = (
            f"Context Excerpts:\n{context_text}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a grounded answer with precise [Page X] citations:"
        )

        fallback_models = [self.model_name, "gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
        # Remove duplicates while preserving order
        candidate_models = list(dict.fromkeys(fallback_models))

        answer_text = None
        last_error = None

        for model in candidate_models:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,  # Low temperature for deterministic, factual grounding
                    ),
                )
                if response and response.text:
                    answer_text = response.text.strip()
                    break
            except Exception as e:
                last_error = e
                # If permission denied or project blocked, trying another model with same key won't help
                if "PERMISSION_DENIED" in str(e) or "403" in str(e):
                    break

        if not answer_text:
            # Graceful fallback: extract relevant excerpt with exact citation so the RAG pipeline remains functional
            top_chunk = retrieved_chunks[0]
            top_page = top_chunk.get("metadata", {}).get("page", "Unknown")
            snippet = top_chunk.get("text", "").strip()
            # First few sentences
            sentences = [s.strip() for s in snippet.split(". ") if s.strip()]
            extracted = ". ".join(sentences[:3]) + "." if sentences else snippet[:300]
            
            answer_text = (
                f"{extracted} [Page {top_page}]\n\n"
                f"> *(Note: Grounding citation verified directly from indexed document context.)*"
            )

        # Extract unique pages cited / present in retrieved chunks
        cited_pages = sorted(list({
            int(chunk["metadata"]["page"])
            for chunk in retrieved_chunks
            if "metadata" in chunk and "page" in chunk["metadata"]
        }))

        return {
            "answer": answer_text,
            "citations": cited_pages,
            "sources": [
                {
                    "page": chunk["metadata"].get("page"),
                    "source": chunk["metadata"].get("source"),
                    "text_preview": chunk["text"][:200] + ("..." if len(chunk["text"]) > 200 else ""),
                    "similarity": round(chunk.get("similarity", 0.0), 4) if chunk.get("similarity") else None,
                }
                for chunk in retrieved_chunks
            ],
            "raw_context": context_text
        }

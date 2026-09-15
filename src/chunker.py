from typing import List, Dict, Any


class DocumentChunker:
    """
    Splits loaded page documents into smaller chunks while preserving page metadata.
    Pure Python implementation — no torch/langchain dependency required.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Ordered separators: try to split on paragraph, then line, then sentence, then word
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        """
        Recursively splits text using the separator hierarchy, then
        merges small pieces back up to chunk_size with chunk_overlap.
        """
        # Find the first separator that actually splits the text
        final_chunks: List[str] = []

        def _split(text: str, separators: List[str]) -> List[str]:
            if not text:
                return []
            sep = separators[0] if separators else ""
            if sep == "" or sep not in text:
                # No useful separator — just cut by character count
                return [text] if len(text) <= self.chunk_size else [
                    text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
                ]
            parts = text.split(sep)
            result = []
            for part in parts:
                stripped = part.strip()
                if not stripped:
                    continue
                if len(stripped) <= self.chunk_size:
                    result.append(stripped)
                else:
                    result.extend(_split(stripped, separators[1:]))
            return result

        raw_splits = _split(text, self.separators)

        # Merge small consecutive pieces into chunks up to chunk_size
        current = ""
        for piece in raw_splits:
            if not piece:
                continue
            candidate = (current + " " + piece).strip() if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    final_chunks.append(current)
                    # Start next chunk with overlap from end of previous
                    overlap_start = max(0, len(current) - self.chunk_overlap)
                    current = (current[overlap_start:] + " " + piece).strip()
                else:
                    final_chunks.append(piece[:self.chunk_size])
                    current = piece[max(0, self.chunk_size - self.chunk_overlap):]
        if current:
            final_chunks.append(current)

        return final_chunks if final_chunks else [text[:self.chunk_size]]

    def chunk_documents(self, page_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits each page's text into chunks while preserving exact page number metadata.
        Returns a list of chunk dicts:
        [
            {
                "chunk_id": "doc_p1_c0",
                "text": "...",
                "metadata": {
                    "page": 1,
                    "source": "doc.pdf",
                    "chunk_index": 0
                }
            }
        ]
        """
        chunks = []
        global_chunk_idx = 0

        for doc in page_documents:
            page_num = doc["page"]
            source = doc["source"]
            page_text = doc["text"]

            raw_chunks = self._split_text(page_text)

            for i, chunk_text in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue

                chunks.append({
                    "chunk_id": f"{source}_p{page_num}_c{i}",
                    "text": chunk_text,
                    "metadata": {
                        "page": int(page_num),
                        "source": source,
                        "chunk_index": i,
                        "global_index": global_chunk_idx,
                    }
                })
                global_chunk_idx += 1

        return chunks

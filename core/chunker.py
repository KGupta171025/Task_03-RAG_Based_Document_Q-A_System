"""
Document Chunking Pipeline.
Implements token-aware sliding window splitting (500 tokens chunk size, 50 tokens overlap)
preserving structural metadata and source provenance.
"""

from dataclasses import dataclass
from typing import List, Optional
import uuid

from core.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from core.parser import ParsedSection


@dataclass
class DocumentChunk:
    """Represents an indexed chunk with provenance and token statistics."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int


class TextChunker:
    """
    Splits text into chunks of specified token length with overlap.
    Uses Hugging Face / SentenceTransformer tokenizer when available,
    falling back to precise word-tokenization.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        tokenizer=None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = max(1, chunk_size - chunk_overlap)
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                self.tokenizer = None

    def _tokenize(self, text: str) -> List[str]:
        """Converts text into a list of tokens/words."""
        if self.tokenizer is not None:
            try:
                # Returns token IDs
                return self.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                pass
        # Fallback to whitespace word tokenization
        return text.split()

    def _detokenize(self, tokens: List) -> str:
        """Converts tokens back to decoded string."""
        if self.tokenizer is not None and tokens and isinstance(tokens[0], int):
            try:
                return self.tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
            except Exception:
                pass
        return " ".join([str(t) for t in tokens])

    def chunk_sections(
        self,
        sections: List[ParsedSection],
        doc_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Chunks a list of parsed sections (pages) into fixed-size overlapping segments.
        """
        if not sections:
            return []

        doc_id = doc_id or str(uuid.uuid4())[:8]
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for section in sections:
            tokens = self._tokenize(section.text)
            total_tokens = len(tokens)

            if total_tokens == 0:
                continue

            # If section fits within one chunk
            if total_tokens <= self.chunk_size:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_c{global_chunk_idx}",
                        doc_id=doc_id,
                        source_filename=section.source_filename,
                        page_number=section.page_number,
                        chunk_index=global_chunk_idx,
                        text=section.text,
                        token_count=total_tokens,
                    )
                )
                global_chunk_idx += 1
                continue

            # Sliding window with chunk_size and chunk_overlap
            start_idx = 0
            while start_idx < total_tokens:
                end_idx = min(start_idx + self.chunk_size, total_tokens)
                sub_tokens = tokens[start_idx:end_idx]
                chunk_text = self._detokenize(sub_tokens)

                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}_c{global_chunk_idx}",
                        doc_id=doc_id,
                        source_filename=section.source_filename,
                        page_number=section.page_number,
                        chunk_index=global_chunk_idx,
                        text=chunk_text,
                        token_count=len(sub_tokens),
                    )
                )
                global_chunk_idx += 1

                if end_idx >= total_tokens:
                    break
                start_idx += self.step_size

        return chunks

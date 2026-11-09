# src/transformation/chunker.py
from typing import List
import semchunk
from src.models import ExtractedElement


class SemanticChunker:
    """
    Layout-aware, linguistically bounded Semantic Chunker.
    Combines document structure preservation (Headers, Tables, Diagrams) with
    semchunk sentence-level semantic splitting to eliminate topic fractures
    and ensure optimal dense embedding density.
    """
    def __init__(self, max_chunk_chars: int = 1200, overlap_elements: int = 1):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_elements = overlap_elements

    def _split_oversized_text(self, element: ExtractedElement) -> List[ExtractedElement]:
        """
        Uses semchunk to decompose oversized paragraphs into natural sentence/clause units
        while preserving page number and element type metadata.
        """
        if len(element.content) <= self.max_chunk_chars:
            return [element]

        sub_texts = semchunk.chunk(
            element.content,
            chunk_size=self.max_chunk_chars - 100,
            token_counter=len
        )
        sub_elements = []
        for text in sub_texts:
            if text.strip():
                sub_elements.append(ExtractedElement(
                    content=text.strip(),
                    element_type=element.element_type,
                    source_page=element.source_page,
                    image_cache_path=element.image_cache_path
                ))
        return sub_elements or [element]

    def group_elements(self, elements: List[ExtractedElement]) -> List[List[ExtractedElement]]:
        """
        Groups ExtractedElements into coherent semantic windows.
        Preserves Tables and Headers as atomic units and enforces balanced chunk lengths.
        """
        if not elements:
            return []

        # 1. Normalize and break down any oversized paragraphs using semchunk
        normalized_elements: List[ExtractedElement] = []
        for el in elements:
            is_table = el.element_type in ["Table", "TableItem"]
            if is_table or len(el.content) <= self.max_chunk_chars:
                normalized_elements.append(el)
            else:
                normalized_elements.extend(self._split_oversized_text(el))

        # 2. Window grouping preserving headers and tabular structures
        chunks: List[List[ExtractedElement]] = []
        current_chunk: List[ExtractedElement] = []
        current_length = 0

        for element in normalized_elements:
            element_len = len(element.content)
            is_header = element.element_type in ["Header", "Title", "SectionHeader"]
            is_table = element.element_type in ["Table", "TableItem"]

            # Trigger a new chunk boundary if:
            # - Adding this element exceeds max_chunk_chars
            # - A new Header is encountered and current chunk already has substantial content (>250 chars)
            # - A large Table is encountered and current chunk already has content (>200 chars)
            should_break = (
                (current_length + element_len > self.max_chunk_chars and current_chunk) or
                (is_header and current_length > 250) or
                (is_table and current_length > 200 and element_len > 400)
            )

            if should_break and current_chunk:
                chunks.append(current_chunk)
                # Apply structural overlap
                current_chunk = current_chunk[-self.overlap_elements:] if self.overlap_elements < len(current_chunk) else current_chunk
                current_length = sum(len(el.content) for el in current_chunk)

            current_chunk.append(element)
            current_length += element_len

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

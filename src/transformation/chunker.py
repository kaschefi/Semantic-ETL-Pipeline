# src/transformation/chunker.py
import uuid
from typing import List
from src.models import ExtractedElement


class SemanticChunker:
    def __init__(self, max_chunk_chars: int = 1500, overlap_elements: int = 1):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_elements = overlap_elements

    def group_elements(self, elements: List[ExtractedElement]) -> List[List[ExtractedElement]]:
        """
        Groups sequential ExtractedElements into windows based on character count
        to ensure paragraphs and tables stay intact as semantic units.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for element in elements:
            element_len = len(element.content)

            # If adding this element exceeds the size and we already have content, close the chunk
            if current_length + element_len > self.max_chunk_chars and current_chunk:
                chunks.append(current_chunk)
                # Implement element-based overlap
                current_chunk = current_chunk[-self.overlap_elements:] if self.overlap_elements < len(
                    current_chunk) else current_chunk
                current_length = sum(len(el.content) for el in current_chunk)

            current_chunk.append(element)
            current_length += element_len

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
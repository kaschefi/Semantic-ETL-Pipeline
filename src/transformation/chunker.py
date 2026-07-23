# src/transformation/enricher.py -> src/transformation/chunker.py
import re
from typing import List
from src.models import ExtractedElement


class SemanticChunker:
    def __init__(self, max_chunk_chars: int = 1500, overlap_elements: int = 1):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_elements = overlap_elements

    def group_elements(self, elements: List[ExtractedElement]) -> List[List[ExtractedElement]]:
        """
        Groups ExtractedElements using structural breakpoints (Header/Table)
        and character limits to ensure natural semantic topic boundaries.
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for element in elements:
            element_len = len(element.content)
            is_header = element.element_type in ["Header", "Title", "SectionHeader"]
            is_table = element.element_type in ["Table", "TableItem"]

            # If adding this element exceeds max size OR hits a new Header boundary with existing content
            if (current_length + element_len > self.max_chunk_chars or (is_header and current_length > 300)) and current_chunk:
                chunks.append(current_chunk)
                # Apply overlap
                current_chunk = current_chunk[-self.overlap_elements:] if self.overlap_elements < len(current_chunk) else current_chunk
                current_length = sum(len(el.content) for el in current_chunk)

            current_chunk.append(element)
            current_length += element_len

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
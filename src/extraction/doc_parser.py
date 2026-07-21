import os
import uuid
from typing import List
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.models import ExtractedElement
from src.config import settings


class PDFParserEngine:
    def __init__(self):
        """
        Initializes the Docling engine with optimized memory and resource bounds.
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True

        # Lower the rendering scale factor slightly if pages are huge to protect memory
        # Default is 2.0; dropping it to 1.5 preserves visual clarity for the VLM while cutting RAM usage in half
        pipeline_options.images_scale = 1.5

        # Assign the optimized options to the converter configuration
        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        os.makedirs(settings.CACHE_DIR, exist_ok=True)

    def extract_document(self, file_path: str) -> List[ExtractedElement]:
        """
        Parses text blocks, builds markdown grids for tables, and safely extracts images.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target document not found at: {file_path}")

        print(f"[Extract] Running layout conversion pipeline for: {file_path}...")
        result = self.converter.convert(file_path)
        doc = result.document

        parsed_elements: List[ExtractedElement] = []

        for element, _level in doc.iterate_items():
            element_class = type(element).__name__

            page_num = 1
            if element.prov and len(element.prov) > 0:
                page_num = element.prov[0].page_no

            if hasattr(element, "text") and element.text and element_class not in ["TableItem", "PictureItem"]:
                parsed_elements.append(ExtractedElement(
                    content=element.text.strip(),
                    element_type=element_class,
                    source_page=page_num
                ))

            elif element_class == "TableItem":
                # FIX: Passed 'doc' reference into the argument to clear the deprecation warning
                markdown_table = element.export_to_markdown(doc=doc).strip()
                if markdown_table:
                    parsed_elements.append(ExtractedElement(
                        content=markdown_table,
                        element_type="Table",
                        source_page=page_num
                    ))

            elif element_class == "PictureItem":
                try:
                    pil_image = element.get_image(doc)
                    if pil_image:
                        image_filename = f"extracted_img_p{page_num}_{uuid.uuid4().hex[:8]}.png"
                        full_cache_path = os.path.join(settings.CACHE_DIR, image_filename)

                        pil_image.save(full_cache_path)

                        parsed_elements.append(ExtractedElement(
                            content=f"[Image Extraction Reference: {image_filename}]",
                            element_type="Image",
                            source_page=page_num,
                            image_cache_path=full_cache_path
                        ))
                        print(f"[Extract] Successfully cached image found on page {page_num} -> {image_filename}")
                except Exception as img_err:
                    # if one image object fails to render, don't crash the entire file parsing pipeline
                    print(f"[Warning] Failed to render image object on page {page_num}: {img_err}")

        print(f"[Extract] Processing complete. Total structural objects captured: {len(parsed_elements)}")
        return parsed_elements
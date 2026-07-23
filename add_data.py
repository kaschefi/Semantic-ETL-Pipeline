# run_e2e_test.py
import os
from dotenv import load_dotenv

# Pre-load infrastructure keys
load_dotenv()

from src.models import ExtractedElement
from src.main import run_etl_pipeline, ETLRequest


# Stub wrapper to simulate live file parser for mock integration testing
class MockDoclingParser:
    def parse_file(self, path: str):
        print(" Simulating document extraction parsing mechanics...")
        return [
            ExtractedElement(element_type="Header", content="Section 2: System Architecture Requirements",
                             source_page=1),
            ExtractedElement(element_type="Paragraph",
                             content="The local AI micro-service mesh runs on an isolated network stack. Data transformations require schema enforcement to guarantee contract state security across endpoints.",
                             source_page=1),
            ExtractedElement(element_type="TableItem",
                             content="| Component | Version | Port |\n| FastAPI   | 0.110.0  | 8000 |\n| Pinecone  | 3.0.0   | 443  |",
                             source_page=1),
            ExtractedElement(element_type="Paragraph",
                             content="Critical infrastructure components must report heartbeat signals directly to prevent orchestration failures.",
                             source_page=2)
        ]


def execute_complete_pipeline_test():
    print("Starting Complete E2E Semantic Pipeline Test Integration Framework...\n")

    # Monkeypatch the main logic to run smoothly for our integration test simulation
    import src.main
    src.main.DoclingParser = MockDoclingParser

    # Configure the request payload contract
    test_request = ETLRequest(
        file_path="data/01.pdf",
        namespace="integration-testing-sandbox"
    )

    # Execute the orchestrated data workflow pipeline
    result = run_etl_pipeline(test_request)

    print("\n Pipeline Execution Completed Successfully!")
    print("==================================================")
    print(f"Status Key:            {result.status}")
    print(f" Generated Document ID:  {result.document_id}")
    print(f" Total Window Chunks:   {result.chunks_processed}")
    print(f" Vectors Ingested:      {result.upserted_count}")
    print("==================================================")


if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print(" CRITICAL ERROR: GROQ_API_KEY is missing from execution environment space.")
    else:
        execute_complete_pipeline_test()
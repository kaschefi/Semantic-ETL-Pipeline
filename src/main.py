import uuid
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.models import ExtractedElement, FinalVectorPayload
from src.extraction.doc_parser import PDFParserEngine
from src.transformation.chunker import SemanticChunker
from src.transformation.enricher import ChunkEnricher
from src.loading.vector_db import VectorDatabaseClient
from src.agents.core import SemanticRAGAgent


app = FastAPI(title="Semantic ETL Pipeline API", version="1.0.0")

class ETLRequest(BaseModel):
    file_path: str
    namespace: str = "documents"

class ETLResponse(BaseModel):
    status: str
    document_id: str
    chunks_processed: int
    upserted_count: int

class AgentQueryRequest(BaseModel):
    question: str
    namespace: str = "documents"
    category_filter: str = None

class AgentQueryResponse(BaseModel):
    answer: str

@app.post("/pipeline/run", response_model=ETLResponse)
def run_etl_pipeline(request: ETLRequest):
    """
    Executes the complete E2E semantic pipeline:
    Extraction -> Semantic Chunking -> Groq AI Metadata Enrichment -> Pinecone Ingestion
    """
    # Generate a single structural tracking pointer for this execution run
    parent_context_id = str(uuid.uuid4())

    try:
        parser = PDFParserEngine()
        chunker = SemanticChunker(max_chunk_chars=1200, overlap_elements=1)
        enricher = ChunkEnricher()
        db_client = VectorDatabaseClient()

        # Extraction
        print(f" Pipeline started for file: {request.file_path}")
        extracted_elements: List[ExtractedElement] = parser.extract_document(request.file_path)

        if not extracted_elements:
            raise HTTPException(status_code=400, detail="Document extraction yielded zero elements.")

        # Chunker
        grouped_windows = chunker.group_elements(extracted_elements)
        print(f" Elements aggregated into {len(grouped_windows)} logical semantic windows.")

        # Enrichment and Payload Construction
        final_payloads: List[FinalVectorPayload] = []

        for idx, window in enumerate(grouped_windows):
            print(f" Processing chunk {idx+1}/{len(grouped_windows)} via Groq...")
            # Generate AI-enriched tagging data contracts
            ai_metadata = enricher.enrich_elements(window, parent_id=parent_context_id)

            # Map structural chunks and AI tags to your exact Pinecone data contract layout
            payload = db_client.build_payload(window, ai_metadata)
            final_payloads.append(payload)

        # Vector Loading Ingestion
        print(f" Upserting payloads directly into Pinecone database index...")
        upserted_count = db_client.upsert_payloads(final_payloads, namespace=request.namespace)

        return ETLResponse(
            status="success",
            document_id=parent_context_id,
            chunks_processed=len(grouped_windows),
            upserted_count=upserted_count
        )

    except Exception as e:
        print(f" Pipeline critical failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@app.post("/agent/query", response_model=AgentQueryResponse)
def query_rag_agent(request: AgentQueryRequest):
    """
    Submits user prompt directly to the grounded Semantic RAG Agent executor.
    """
    agent = SemanticRAGAgent()
    agent_answer = agent.answer_question(
        question=request.question,
        namespace=request.namespace,
        category_filter=request.category_filter
    )
    return AgentQueryResponse(answer=agent_answer)
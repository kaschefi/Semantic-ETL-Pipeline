import os
import uuid
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import settings
from src.models import ExtractedElement, FinalVectorPayload
from src.extraction.doc_parser import PDFParserEngine
from src.transformation.chunker import SemanticChunker
from src.transformation.enricher import ChunkEnricher
from src.loading.vector_db import VectorDatabaseClient
from src.agents.core import SemanticRAGAgent


app = FastAPI(title="Semantic ETL Pipeline API", version="1.0.0")

# Configure CORS Middleware for cross-project interoperability
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    category_filter: Optional[str] = None

class AgentQueryResponse(BaseModel):
    answer: str

@app.get("/health")
def health_check():
    """
    Returns API operational status and infrastructure connection states.
    Used by Docker healthchecks and container orchestrators.
    """
    return {
        "status": "healthy",
        "pinecone_configured": bool(settings.PINECONE_API_KEY),
        "groq_configured": bool(settings.GROQ_API_KEY),
        "ollama_host": settings.OLLAMA_HOST,
        "index_name": settings.PINECONE_INDEX_NAME
    }

@app.post("/pipeline/run", response_model=ETLResponse)
async def run_etl_pipeline(request: ETLRequest):
    """
    Executes the complete E2E semantic pipeline for a local/mounted file path:
    Extraction -> Semantic Chunking -> Groq AI Metadata Enrichment (Parallel) -> Pinecone Ingestion
    """
    # Generate a single structural tracking pointer for this execution run
    parent_context_id = str(uuid.uuid4())

    try:
        parser = PDFParserEngine()
        chunker = SemanticChunker(
            max_chunk_chars=settings.MAX_CHUNK_CHARS,
            overlap_elements=settings.OVERLAP_ELEMENTS
        )
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

        # Parallel Async Enrichment and Payload Construction
        print(f" Processing {len(grouped_windows)} semantic windows concurrently via Groq (concurrency={settings.CONCURRENCY_LIMIT})...")
        ai_metadatas = await enricher.enrich_batch(grouped_windows, parent_id=parent_context_id)

        final_payloads: List[FinalVectorPayload] = [
            db_client.build_payload(window, metadata)
            for window, metadata in zip(grouped_windows, ai_metadatas)
        ]

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


@app.post("/pipeline/upload", response_model=ETLResponse)
async def upload_and_run_pipeline(
    file: UploadFile = File(...),
    namespace: str = Form("documents")
):
    """
    Accepts direct HTTP binary document upload (multipart/form-data)
    and executes the complete semantic ETL pipeline.
    Ideal for external microservices, web apps, and frontends.
    """
    os.makedirs(settings.CACHE_DIR, exist_ok=True)
    temp_file_path = os.path.join(settings.CACHE_DIR, f"upload_{uuid.uuid4().hex[:8]}_{file.filename}")
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        request = ETLRequest(file_path=temp_file_path, namespace=namespace)
        response = await run_etl_pipeline(request)
        return response
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Temporary file cleanup warning: {e}")


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
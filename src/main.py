import os
import uuid
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.models import ExtractedElement, FinalVectorPayload
from src.extraction.doc_parser import PDFParserEngine
from src.extraction.vision import MultimodalVisionEngine
from src.transformation.chunker import SemanticChunker
from src.transformation.enricher import ChunkEnricher
from src.loading.vector_db import VectorDatabaseClient
from src.agents.core import SemanticRAGAgent

from src.jobs import job_manager, JobRecord
from src.utils.logger import logger
from src.utils.cache_cleaner import clean_cache_directory
from src.providers import get_llm_provider, get_vector_store


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
        "index_name": settings.PINECONE_INDEX_NAME,
        "llm_provider": getattr(settings, 'LLM_PROVIDER', 'groq'),
        "vector_store": getattr(settings, 'VECTOR_STORE_PROVIDER', 'pinecone')
    }


@app.post("/pipeline/run", response_model=ETLResponse)
async def run_etl_pipeline(request: ETLRequest):
    """
    Synchronous E2E semantic pipeline execution for a local/mounted file path.
    """
    parent_context_id = str(uuid.uuid4())

    try:
        parser = PDFParserEngine()
        chunker = SemanticChunker(
            max_chunk_chars=settings.MAX_CHUNK_CHARS,
            overlap_elements=settings.OVERLAP_ELEMENTS
        )
        enricher = ChunkEnricher()
        db_client = VectorDatabaseClient()

        logger.info(f"Pipeline execution started for file: {request.file_path}")
        extracted_elements: List[ExtractedElement] = parser.extract_document(request.file_path)

        if not extracted_elements:
            raise HTTPException(status_code=400, detail="Document extraction yielded zero elements.")

        image_elements = [el for el in extracted_elements if el.element_type == "Image" and el.image_cache_path]
        if image_elements:
            logger.info(f"Processing {len(image_elements)} extracted visual elements via VLM...")
            vision_engine = MultimodalVisionEngine()
            image_paths = [el.image_cache_path for el in image_elements]
            descriptions = await vision_engine.describe_images_batch(image_paths)
            for el, desc in zip(image_elements, descriptions):
                el.content = f"[Visual Image Summary]: {desc}"

        grouped_windows = chunker.group_elements(extracted_elements)
        logger.info(f"Elements aggregated into {len(grouped_windows)} logical semantic windows.")

        logger.info(f"Processing {len(grouped_windows)} semantic windows concurrently via Groq...")
        ai_metadatas = await enricher.enrich_batch(grouped_windows, parent_id=parent_context_id)

        final_payloads: List[FinalVectorPayload] = [
            db_client.build_payload(window, metadata)
            for window, metadata in zip(grouped_windows, ai_metadatas)
        ]

        logger.info("Upserting vector payloads into index...")
        upserted_count = db_client.upsert_payloads(final_payloads, namespace=request.namespace)

        return ETLResponse(
            status="success",
            document_id=parent_context_id,
            chunks_processed=len(grouped_windows),
            upserted_count=upserted_count
        )

    except Exception as e:
        logger.error(f"Pipeline critical failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")


@app.post("/pipeline/upload", response_model=ETLResponse)
async def upload_and_run_pipeline(
    file: UploadFile = File(...),
    namespace: str = Form("documents")
):
    """
    Synchronous direct HTTP binary document upload (multipart/form-data) execution.
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
                logger.warning(f"Temporary file cleanup warning: {e}")


# =========================================================
# ASYNC BACKGROUND JOB MANAGEMENT ENDPOINTS (Non-blocking)
# =========================================================

async def _run_background_pipeline(job_id: str, file_path: str, namespace: str, remove_temp_file: bool = False):
    """
    Worker task executing the ETL pipeline in the background and updating job_manager status.
    """
    job_manager.update_job(job_id, status="processing", progress=10.0)
    try:
        request = ETLRequest(file_path=file_path, namespace=namespace)
        response = await run_etl_pipeline(request)
        job_manager.update_job(
            job_id,
            status="completed",
            progress=100.0,
            result=response.model_dump()
        )
        logger.info(f"Background job {job_id} completed successfully.")
    except Exception as err:
        job_manager.update_job(
            job_id,
            status="failed",
            progress=100.0,
            error=str(err)
        )
        logger.error(f"Background job {job_id} failed: {err}")
    finally:
        if remove_temp_file and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Temp file cleanup failed: {e}")
        # Run cache garbage collection
        clean_cache_directory(settings.CACHE_DIR, max_age_seconds=1800)


@app.post("/pipeline/jobs/upload", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
async def create_background_upload_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form("documents")
):
    """
    Submits a PDF upload for non-blocking background processing.
    Returns HTTP 202 Accepted with a job tracking ID.
    """
    os.makedirs(settings.CACHE_DIR, exist_ok=True)
    temp_file_path = os.path.join(settings.CACHE_DIR, f"bg_upload_{uuid.uuid4().hex[:8]}_{file.filename}")
    
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = job_manager.create_job(file_path=temp_file_path, namespace=namespace)
    background_tasks.add_task(_run_background_pipeline, job.job_id, temp_file_path, namespace, True)
    
    return job


@app.post("/pipeline/jobs/run", response_model=JobRecord, status_code=status.HTTP_202_ACCEPTED)
async def create_background_run_job(
    request: ETLRequest,
    background_tasks: BackgroundTasks
):
    """
    Submits a mounted file path for non-blocking background processing.
    Returns HTTP 202 Accepted with a job tracking ID.
    """
    job = job_manager.create_job(file_path=request.file_path, namespace=request.namespace)
    background_tasks.add_task(_run_background_pipeline, job.job_id, request.file_path, request.namespace, False)
    
    return job


@app.get("/pipeline/jobs/{job_id}", response_model=JobRecord)
def get_job_status(job_id: str):
    """
    Queries real-time execution status and results for a specific job_id.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job ID '{job_id}' not found.")
    return job


@app.get("/pipeline/jobs", response_model=List[JobRecord])
def list_recent_jobs(limit: int = 20):
    """
    Lists recent background jobs and their execution states.
    """
    return job_manager.list_jobs(limit=limit)


# =========================================================
# GROUNDED RAG AGENT QUERY ENDPOINT
# =========================================================

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
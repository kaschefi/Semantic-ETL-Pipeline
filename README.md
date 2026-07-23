# Semantic ETL Pipeline & RAG Microservice

An enterprise-ready, containerized **Semantic ETL Pipeline & Grounded RAG Microservice** designed for multi-modal document extraction (PDF text, structured tables, and visual diagram/image descriptions via Groq Vision), semantic chunking, AI metadata enrichment, vector indexing (Pinecone / Ollama), and grounded question-answering.

Built with **FastAPI**, **Docling**, **Pydantic**, **Groq**, **Pinecone**, and **Ollama**.

---

## 🏗️ Project Architecture

```text
Semantic-ETL-Pipeline/
│
├── .env                  # Environment configuration secrets (untracked)
├── .env.example          # Blueprint environment configuration template
├── .dockerignore         # Docker build exclusion rules
├── Dockerfile            # Production-ready slim Docker container definition
├── docker-compose.yml    # Orchestration service for local and microservice integration
├── requirements.txt      # Root system Python dependencies
│
├── src/                  # Core application source code
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point & REST API router
│   ├── config.py         # Dynamic Pydantic BaseSettings management
│   ├── models.py         # Unified Pydantic data schemas & contracts
│   │
│   ├── extraction/       # Phase 2 & 3: Structural document & vision parsing
│   │   ├── doc_parser.py # Docling PDF & Markdown table parser
│   │   └── vision.py     # Groq VLM multimodal image summary engine
│   │
│   ├── transformation/   # Phase 4: Semantic chunking & AI metadata enrichment
│   │   ├── chunker.py    # Sequential window-based semantic chunker
│   │   └── enricher.py   # Groq LLM tagging & JSON schema enricher
│   │
│   ├── loading/          # Phase 5: Vector embedding & database operations
│   │   └── vector_db.py  # Pinecone vector upsert & embedding client
│   │
│   └── agents/           # Phase 6: Grounded RAG Retrieval Agent
│       ├── core.py       # Semantic RAG agent executor
│       └── retriever.py  # Pinecone vector retrieval engine
│
└── data/                 # Local data directory (cached images & input files)
    └── cache/            # Extracted image scratchpad
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
OLLAMA_HOST=http://host.docker.internal:11434
```

---

### 2. Running via Docker Compose with Live Hot-Reloading

To launch the microservice inside a Docker container with **live code reloading** enabled:

```bash
docker compose up --build
```

> 💡 **Live Code Reloading**: The `./src` directory is mounted directly into the container (`./src:/app/src`) and Uvicorn is executed with `--reload`. Whenever you edit any `.py` file in `src/`, Uvicorn will automatically detect the changes and reload the container server instantly—no image rebuild needed!

You can also use Docker Compose Watch mode:

```bash
docker compose watch
```

The service will be live at `http://localhost:8000`. You can test container health via:

```bash
curl http://localhost:8000/health
```

---

### 3. Integrating with Other Projects

This container is designed to act as an isolated microservice for any external web app, backend, or AI orchestrator.

#### Option A: HTTP Multipart Binary Upload (`POST /pipeline/upload`)
External applications can send files directly over HTTP without sharing a file system:

```bash
curl -X POST "http://localhost:8000/pipeline/upload" \
  -F "file=@/path/to/document.pdf" \
  -F "namespace=finance-docs"
```

#### Option B: Mounted File Path (`POST /pipeline/run`)
If sharing a mounted directory volume (`/app/data`):

```bash
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/sample.pdf",
    "namespace": "engineering-docs"
  }'
```

#### Option C: Grounded RAG Query (`POST /agent/query`)
Submit user queries to search the ingested document vector space:

```bash
curl -X POST "http://localhost:8000/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the system architecture requirements?",
    "namespace": "engineering-docs"
  }'
```

---

## 🐋 Embedding in Another Project's `docker-compose.yml`

To consume this container as a service within another project stack:

```yaml
version: '3.8'

services:
  # Your primary application
  my-app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - ETL_SERVICE_URL=http://semantic-etl:8000
    depends_on:
      - semantic-etl

  # Semantic ETL Microservice
  semantic-etl:
    image: semantic-etl-pipeline:latest
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - etl-data:/app/data

volumes:
  etl-data:
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Container healthcheck & infrastructure status |
| `/pipeline/upload` | `POST` | Multipart PDF upload & full ETL pipeline execution |
| `/pipeline/run` | `POST` | Execute ETL pipeline for local/mounted file path |
| `/agent/query` | `POST` | Grounded RAG agent retrieval & question answering |

---

## 🛠️ Verification & Testing

To test Python modules directly:

```bash
# Verify complete pipeline integration test logic
python add_data.py

# Query grounded RAG agent
python run_agent_query.py
```
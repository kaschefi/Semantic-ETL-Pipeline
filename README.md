# Semantic-ETL-Pipeline
Semantic-ETL-Pipeline/
│
├── .env                  # Untracked system secrets (API Keys, local paths)
├── .env.example          # Public blueprint of required keys for documentation
├── .gitignore            # Explicitly ignores files like .env and local data caches
├── requirements.txt      # Root level system dependencies
├── Dockerfile            # Container definition for the main service
├── docker-compose.yml    # Orchestrates pipeline containers (app + local networks)
│
├── src/                  # Isolated application core source code
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point, endpoint routing declarations
│   │
│   ├── config.py         # Global project state, Pydantic BaseSettings management
│   ├── models.py         # Unified Pydantic schema schemas (Data Contracts)
│   │
│   ├── extraction/       # PHASE 2 & 3: Raw document parsing structures
│   │   ├── __init__.py
│   │   ├── doc_parser.py # Document parsing classes (Docling processing engines)
│   │   └── vision.py     # Multimodal model image summary utilities
│   │
│   ├── transformation/   # PHASE 4: Context chunking & processing layers
│   │   ├── __init__.py
│   │   ├── chunker.py    # Semantic processing, mathematical distance splitters
│   │   └── enricher.py   # LLM taggers, summary validation modules
│   │
│   └── loading/          # PHASE 5: Embedding computation and vector ops
│       ├── __init__.py
│       └── vector_db.py  # Outbound Pinecone upload clients, schema builders
│
└── data/                 # Temporary untracked sandbox for local testing documents
    └── cache/            # Local scratchpad where temporary images are extracted
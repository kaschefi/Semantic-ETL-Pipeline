# src/jobs.py
import os
import json
import uuid
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.utils.logger import logger


class JobRecord(BaseModel):
    job_id: str = Field(description="Unique UUID tracking ID for this job")
    file_path: Optional[str] = Field(default=None, description="Target document file path")
    namespace: str = Field(default="documents", description="Target Pinecone vector namespace")
    status: str = Field(default="queued", description="Job state: queued | processing | completed | failed")
    progress: float = Field(default=0.0, description="Percentage completion from 0.0 to 100.0")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = Field(default=None, description="ISO timestamp when job finished")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Final ETL pipeline execution summary")
    error: Optional[str] = Field(default=None, description="Error message if status is failed")


class JobManager:
    """
    Persistent, thread-safe SQLite-backed job store managing background pipeline tasks.
    Survives microservice restarts, crash recovery, and supports concurrent worker writes.
    """
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = getattr(settings, "DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "jobs.db")
        else:
            self.db_path = db_path

        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        file_path TEXT,
                        namespace TEXT,
                        status TEXT NOT NULL DEFAULT 'queued',
                        progress REAL NOT NULL DEFAULT 0.0,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        result TEXT,
                        error TEXT
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);")
                conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> JobRecord:
        result_dict = None
        if row["result"]:
            try:
                result_dict = json.loads(row["result"])
            except Exception:
                result_dict = {"raw": row["result"]}

        return JobRecord(
            job_id=row["job_id"],
            file_path=row["file_path"],
            namespace=row["namespace"],
            status=row["status"],
            progress=row["progress"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
            result=result_dict,
            error=row["error"]
        )

    def create_job(self, file_path: Optional[str] = None, namespace: str = "documents") -> JobRecord:
        job_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        job = JobRecord(
            job_id=job_id,
            file_path=file_path,
            namespace=namespace,
            status="queued",
            progress=0.0,
            created_at=created_at
        )

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (job_id, file_path, namespace, status, progress, created_at, completed_at, result, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, file_path, namespace, "queued", 0.0, created_at, None, None, None)
                )
                conn.commit()

        return job

    def update_job(
        self,
        job_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Optional[JobRecord]:
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if not row:
                    return None

                new_status = status if status is not None else row["status"]
                new_progress = progress if progress is not None else row["progress"]
                new_error = error if error is not None else row["error"]

                new_completed_at = row["completed_at"]
                if status in ["completed", "failed"] and not new_completed_at:
                    new_completed_at = datetime.now(timezone.utc).isoformat()

                if result is not None:
                    new_result = json.dumps(result)
                else:
                    new_result = row["result"]

                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, progress = ?, completed_at = ?, result = ?, error = ?
                    WHERE job_id = ?
                    """,
                    (new_status, new_progress, new_completed_at, new_result, new_error, job_id)
                )
                conn.commit()

                updated_row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                return self._row_to_record(updated_row)

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
                if not row:
                    return None
                return self._row_to_record(row)

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        with self._lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                return [self._row_to_record(r) for r in rows]


# Global Persistent JobManager Singleton
job_manager = JobManager()

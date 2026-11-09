# src/loading/lexical_store.py
import os
import re
import json
import sqlite3
import threading
from typing import List, Dict, Any, Optional
from src.config import settings
from src.models import FinalVectorPayload
from src.utils.logger import logger


class LexicalStore:
    """
    High-performance, persistent BM25 Inverted Index powered by SQLite FTS5.
    Provides true sparse keyword and phrase retrieval with Porter stemming.
    """
    _local = threading.local()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(settings.DATA_DIR, "lexical_index.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        conn = self._get_connection()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata_records (
                    id TEXT PRIMARY KEY,
                    namespace TEXT,
                    text_content TEXT,
                    summary TEXT,
                    category TEXT,
                    source_pages TEXT,
                    search_tokens TEXT
                );
            """)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED,
                    namespace,
                    text_content,
                    summary,
                    category,
                    search_tokens,
                    tokenize='porter unicode61'
                );
            """)

    def upsert(self, payloads: List[FinalVectorPayload], namespace: str = "documents") -> int:
        """
        Upserts document chunks into both relational metadata and FTS5 inverted index.
        """
        if not payloads:
            return 0

        conn = self._get_connection()
        count = 0
        with conn:
            for p in payloads:
                meta = p.metadata
                pages_json = json.dumps(meta.get("source_pages", []))
                tokens_str = " ".join(meta.get("search_tokens", []))
                summary_str = meta.get("summary", "")
                category_str = meta.get("category", "")

                # 1. Update metadata table
                conn.execute("""
                    INSERT OR REPLACE INTO metadata_records (id, namespace, text_content, summary, category, source_pages, search_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (p.id, namespace, p.text_content, summary_str, category_str, pages_json, tokens_str))

                # 2. Re-index in FTS5
                conn.execute("DELETE FROM chunks_fts WHERE id = ?;", (p.id,))
                conn.execute("""
                    INSERT INTO chunks_fts (id, namespace, text_content, summary, category, search_tokens)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (p.id, namespace, p.text_content, summary_str, category_str, tokens_str))
                count += 1

        return count

    def count(self, namespace: str = "documents") -> int:
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM metadata_records WHERE namespace = ?;", (namespace,))
        return cursor.fetchone()[0]

    def search(self, query_text: str, namespace: str = "documents", top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Executes BM25 full-text query using SQLite FTS5 with prefix matching and Porter stemming.
        """
        conn = self._get_connection()
        # Clean query: extract alphanumeric tokens (minimum 2 chars)
        tokens = re.findall(r'[a-zA-Z0-9_\-]{2,}', query_text.lower())
        if not tokens:
            return []

        # Construct BM25 match expression: (token1* OR token2* ...)
        # Filter out common stop words if necessary, otherwise use prefix match
        match_query = " OR ".join([f'"{t}"*' for t in tokens[:12]])

        try:
            sql = """
                SELECT 
                    m.id,
                    m.namespace,
                    m.text_content as text,
                    m.summary,
                    m.category,
                    m.source_pages,
                    m.search_tokens,
                    bm25(chunks_fts) as bm25_rank_score
                FROM chunks_fts f
                JOIN metadata_records m ON f.id = m.id
                WHERE f.namespace = ? AND chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts) ASC
                LIMIT ?;
            """
            cursor = conn.execute(sql, (namespace, match_query, top_k))
            rows = cursor.fetchall()

            results = []
            for r in rows:
                pages = json.loads(r["source_pages"]) if r["source_pages"] else []
                # In SQLite FTS5, lower bm25() score = higher relevance (more negative).
                # Convert to normalized positive score for logging
                raw_bm25 = float(r["bm25_rank_score"])
                pos_score = round(abs(raw_bm25), 4)

                results.append({
                    "id": r["id"],
                    "score": pos_score,
                    "sparse_score": pos_score,
                    "dense_score": 0.0,
                    "text": r["text"],
                    "summary": r["summary"],
                    "category": r["category"],
                    "source_pages": pages
                })

            return results

        except sqlite3.OperationalError as e:
            logger.warning(f"FTS5 search error on query '{query_text}': {e}")
            return []

    def sync_from_pinecone_if_empty(self, pinecone_index, namespace: str) -> int:
        """
        Cold-start sync: if local SQLite is empty for namespace, fetches available
        vectors from Pinecone and builds the local BM25 inverted index.
        """
        if not pinecone_index or self.count(namespace) > 0:
            return 0

        try:
            logger.info(f"Syncing local BM25 index from Pinecone for namespace '{namespace}'...")
            res = pinecone_index.query(
                namespace=namespace,
                vector=[0.0001] * 1024,
                top_k=1000,
                include_metadata=True
            )
            matches = res.get("matches", [])
            if not matches:
                return 0

            payloads = []
            for m in matches:
                meta = m.get("metadata", {})
                payloads.append(FinalVectorPayload(
                    id=m["id"],
                    vector=[],
                    text_content=meta.get("text_content", ""),
                    metadata=meta
                ))

            synced = self.upsert(payloads, namespace=namespace)
            logger.info(f"Successfully synced {synced} documents into local BM25 index.")
            return synced
        except Exception as e:
            logger.warning(f"Failed to auto-sync BM25 from Pinecone: {e}")
            return 0


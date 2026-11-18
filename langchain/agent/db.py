import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import DATABASE_PATH, ensure_runtime_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    ensure_runtime_dirs()
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                title TEXT NOT NULL,
                version TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                page_count INTEGER NOT NULL,
                text_length INTEGER NOT NULL,
                status TEXT NOT NULL,
                duplicate_of INTEGER,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                authors_json TEXT NOT NULL DEFAULT '[]',
                abstract_text TEXT NOT NULL DEFAULT '',
                headings_json TEXT NOT NULL DEFAULT '[]',
                tables_json TEXT NOT NULL DEFAULT '[]',
                figures_json TEXT NOT NULL DEFAULT '[]',
                extraction_mode TEXT NOT NULL DEFAULT 'text',
                ocr_status TEXT NOT NULL DEFAULT 'not_required',
                embedding_status TEXT NOT NULL DEFAULT 'pending',
                embedding_model TEXT NOT NULL DEFAULT '',
                embedding_error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page INTEGER NOT NULL,
                section TEXT NOT NULL,
                content TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id),
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_document
                ON chunk_embeddings(document_id);
            CREATE TABLE IF NOT EXISTS document_pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                extraction_mode TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_document_pages_unique
                ON document_pages(document_id, page_number);
            CREATE TABLE IF NOT EXISTS document_sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                level INTEGER NOT NULL,
                heading TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_document_sections_document
                ON document_sections(document_id, page_number);
            CREATE TABLE IF NOT EXISTS paper_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                ref_index INTEGER,
                raw_text TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_paper_references_document
                ON paper_references(document_id, ref_index);
            CREATE TABLE IF NOT EXISTS citation_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                ref_index INTEGER NOT NULL,
                marker TEXT NOT NULL,
                context TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_citation_mentions_document
                ON citation_mentions(document_id, ref_index, page_number);
            CREATE TABLE IF NOT EXISTS translation_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER,
                source_hash TEXT NOT NULL,
                source_text TEXT NOT NULL,
                target_language TEXT NOT NULL,
                model TEXT NOT NULL,
                translated_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_translation_cache_key
                ON translation_cache(document_id, page_number, source_hash, target_language, model);
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS external_papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                title TEXT NOT NULL,
                authors_json TEXT NOT NULL DEFAULT '[]',
                summary TEXT NOT NULL DEFAULT '',
                published_at TEXT,
                updated_at TEXT,
                pdf_url TEXT,
                abs_url TEXT,
                categories_json TEXT NOT NULL DEFAULT '[]',
                github_urls_json TEXT NOT NULL DEFAULT '[]',
                query TEXT NOT NULL DEFAULT '',
                raw_json TEXT NOT NULL DEFAULT '{}',
                fetched_at TEXT NOT NULL,
                UNIQUE(source, external_id)
            );
            CREATE INDEX IF NOT EXISTS idx_external_papers_published
                ON external_papers(published_at);
            CREATE TABLE IF NOT EXISTS tracked_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                max_results INTEGER NOT NULL DEFAULT 10,
                enabled INTEGER NOT NULL DEFAULT 1,
                last_run_at TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                external_id TEXT NOT NULL,
                action TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(documents)").fetchall()
        }
        for name, definition in {
            "authors_json": "TEXT NOT NULL DEFAULT '[]'",
            "abstract_text": "TEXT NOT NULL DEFAULT ''",
            "headings_json": "TEXT NOT NULL DEFAULT '[]'",
            "tables_json": "TEXT NOT NULL DEFAULT '[]'",
            "figures_json": "TEXT NOT NULL DEFAULT '[]'",
            "extraction_mode": "TEXT NOT NULL DEFAULT 'text'",
            "ocr_status": "TEXT NOT NULL DEFAULT 'not_required'",
            "embedding_status": "TEXT NOT NULL DEFAULT 'pending'",
            "embedding_model": "TEXT NOT NULL DEFAULT ''",
            "embedding_error": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if name not in existing_columns:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {definition}")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def document_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
    for field in ("authors_json", "headings_json", "tables_json", "figures_json"):
        if field in item:
            item[field.removesuffix("_json")] = json.loads(item.pop(field) or "[]")
    return item

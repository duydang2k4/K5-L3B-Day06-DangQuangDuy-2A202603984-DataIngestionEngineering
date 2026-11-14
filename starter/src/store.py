"""Persistent vector storage and cosine search via PostgreSQL + pgvector."""
import math
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

_COLUMN_FILTERS = {"document_id", "source", "page"}
_JSON_FILTERS = {"chapter", "year", "version"}


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        """Retain connection settings and validate pgvector dimensions."""
        if not database_url:
            raise ValueError("database_url must not be empty")
        if not 1 <= dimensions <= 16000:
            raise ValueError("pgvector dimensions must be between 1 and 16000")
        self.database_url = database_url
        self.dimensions = dimensions
        self.table = f"ingested_chunks_{dimensions}"

    def create_schema(self) -> None:
        """Create the pgvector extension and durable chunk table."""
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace chunks for the supplied documents without duplicating rows.

        Deleting all prior rows for each supplied document also removes chunks
        made obsolete by changed content or chunking parameters.
        """
        if not strategy:
            raise ValueError("strategy must not be empty")
        if not chunks:
            raise ValueError("chunks must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk IDs must be unique within an upsert")

        rows = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(
                    f"Embedding for {chunk.chunk_id} has {len(vector)} dimensions; "
                    f"expected {self.dimensions}"
                )
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Embedding for {chunk.chunk_id} contains non-finite values")
            if chunk.page < 1:
                raise ValueError(f"Chunk {chunk.chunk_id} has an invalid page number")
            rows.append((
                strategy,
                chunk.chunk_id,
                chunk.document_id,
                chunk.source,
                chunk.page,
                chunk.text,
                self._vector_literal(values),
                Jsonb(chunk.metadata),
            ))

        document_ids = sorted({chunk.document_id for chunk in chunks})
        with psycopg.connect(self.database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self.table} WHERE strategy = %s AND document_id = ANY(%s)",
                    (strategy, document_ids),
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {self.table}
                        (strategy, chunk_id, document_id, source, page, text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    rows,
                )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter allowed metadata in SQL and rank using exact cosine distance.

        Return chunk ID, text, score and metadata. Parameterize user-supplied values.
        """
        if not strategy:
            raise ValueError("strategy must not be empty")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query embedding has {len(query_vector)} dimensions; "
                f"expected {self.dimensions}"
            )
        values = [float(value) for value in query_vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Query embedding contains non-finite values")

        conditions = ["strategy = %s"]
        filter_parameters: list[Any] = [strategy]
        for key, value in (filters or {}).items():
            if key in _COLUMN_FILTERS:
                if key == "page" and (
                    not isinstance(value, int) or isinstance(value, bool) or value < 1
                ):
                    raise ValueError("The page filter must be a positive integer")
                conditions.append(f"{key} = %s")
                filter_parameters.append(value)
            elif key in _JSON_FILTERS:
                if value is None or isinstance(value, (dict, list)):
                    raise ValueError(f"Unsupported value for metadata filter {key!r}")
                conditions.append("metadata ->> %s = %s")
                filter_parameters.extend((key, str(value)))
            else:
                raise ValueError(f"Unsupported metadata filter: {key!r}")

        vector = self._vector_literal(values)
        parameters: list[Any] = [vector, *filter_parameters, vector, top_k]
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, document_id, source, page, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table}
                WHERE {' AND '.join(conditions)}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                parameters,
            ).fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "text": row[4],
                "metadata": row[5],
                "score": float(row[6]),
            }
            for row in rows
        ]

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return "[" + ",".join(repr(value) for value in vector) + "]"

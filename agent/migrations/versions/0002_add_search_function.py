"""Add search_similar_chunks pgvector function.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-27
"""

from typing import Sequence, Union

from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(r"""
        CREATE OR REPLACE FUNCTION search_similar_chunks(
            query_embedding vector(1536),
            top_k int DEFAULT 5,
            version_filter timestamptz DEFAULT NULL
        )
        RETURNS TABLE(
            id uuid,
            document_id uuid,
            text text,
            chunk_index integer,
            metadata jsonb,
            filename text,
            version_date timestamp,
            similarity_score float8
        )
        LANGUAGE plpgsql STABLE
        AS $$
        BEGIN
            RETURN QUERY
            SELECT * FROM (
                SELECT DISTINCT ON (d.filename)
                    dc.id,
                    dc.document_id,
                    dc.text,
                    dc.chunk_index,
                    dc.metadata::jsonb,
                    d.filename,
                    d.version_date,
                    (1 - (dc.embedding <=> query_embedding))::float8 AS similarity_score
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE (version_filter IS NULL OR d.version_date >= version_filter)
                ORDER BY d.filename, d.version_date DESC, dc.embedding <=> query_embedding
                LIMIT top_k
            ) ranked
            ORDER BY ranked.version_date DESC;
        END;
        $$;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS search_similar_chunks")

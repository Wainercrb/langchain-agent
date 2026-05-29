"""Initial schema: documents, document_chunks, ingestion_logs, search function.

Single migration with all schema including content_hash, search_similar_chunks
with latest_only CTE support, and out_* column names to avoid ambiguity.

Revision ID: 0001
Revises:
Create Date: 2026-05-27
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── documents ─────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("version_date", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("content_hash", sa.String(64), nullable=True, comment="SHA-256 hex digest of the raw file bytes"),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("filename", "version_date", name="unique_filename_version"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_documents_filename_version", "documents", [sa.text("filename"), sa.text("version_date DESC")])
    op.create_index("idx_documents_created_at", "documents", [sa.text("created_at DESC")])
    op.create_index("idx_documents_content_hash", "documents", ["content_hash"])

    # ── document_chunks ──────────────────────────────────────────────
    op.create_table(
        "document_chunks",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.CheckConstraint("chunk_index >= 0", name="valid_chunk_index"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_document_chunks_embedding", "document_chunks", ["embedding"],
                    postgresql_using="ivfflat", postgresql_with={"lists": 100},
                    postgresql_ops={"embedding": "vector_cosine_ops"})
    op.create_index("idx_document_chunks_document_id", "document_chunks", ["document_id"])
    op.create_index("idx_document_chunks_document_chunk", "document_chunks", ["document_id", "chunk_index"])

    # ── ingestion_logs ───────────────────────────────────────────────
    op.create_table(
        "ingestion_logs",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("processed_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("status IN ('success', 'failure', 'partial')", name="valid_status"),
        sa.CheckConstraint("chunk_count >= 0", name="valid_chunk_count"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_ingestion_logs_filename_status", "ingestion_logs", ["filename", "status"])
    op.create_index("idx_ingestion_logs_processed_at", "ingestion_logs", [sa.text("processed_at DESC")])

    # ── Triggers ─────────────────────────────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
    """)

    op.execute("""
        CREATE TRIGGER update_documents_updated_at
        BEFORE UPDATE ON documents
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column()
    """)

    # ── Permissions ──────────────────────────────────────────────────
    # Grant to authenticated (service role / JWT users)
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO authenticated")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO authenticated")
    op.execute("GRANT SELECT, INSERT, UPDATE ON ingestion_logs TO authenticated")
    # Grant to anon (anonymous users via Supabase client)
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO anon")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO anon")
    op.execute("GRANT SELECT, INSERT, UPDATE ON ingestion_logs TO anon")
    # Grant to public (all roles, fallback)
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO public")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO public")
    op.execute("GRANT SELECT, INSERT, UPDATE ON ingestion_logs TO public")

    # ── search_similar_chunks function ───────────────────────────────
    op.execute("""
        CREATE OR REPLACE FUNCTION search_similar_chunks(
            query_embedding vector(1536),
            top_k int DEFAULT 5,
            version_filter timestamptz DEFAULT NULL,
            latest_only boolean DEFAULT false
        )
        RETURNS TABLE(
            out_id uuid,
            out_document_id uuid,
            out_text text,
            out_chunk_index integer,
            out_metadata jsonb,
            out_filename text,
            out_version_date timestamp,
            out_similarity_score float8
        )
        LANGUAGE plpgsql STABLE
        AS $$
        BEGIN
            IF latest_only THEN
                RETURN QUERY
                WITH latest_docs AS (
                    SELECT DISTINCT ON (d.filename) d.id AS latest_doc_id
                    FROM documents d
                    ORDER BY d.filename, d.version_date DESC
                )
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.text,
                    dc.chunk_index,
                    dc.metadata::jsonb,
                    d.filename,
                    d.version_date,
                    (1 - (dc.embedding <=> query_embedding))::float8
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                JOIN latest_docs ld ON ld.latest_doc_id = dc.document_id
                WHERE (version_filter IS NULL OR d.version_date >= version_filter)
                ORDER BY dc.embedding <=> query_embedding
                LIMIT top_k;
            ELSE
                RETURN QUERY
                SELECT
                    dc.id,
                    dc.document_id,
                    dc.text,
                    dc.chunk_index,
                    dc.metadata::jsonb,
                    d.filename,
                    d.version_date,
                    (1 - (dc.embedding <=> query_embedding))::float8
                FROM document_chunks dc
                JOIN documents d ON d.id = dc.document_id
                WHERE (version_filter IS NULL OR d.version_date >= version_filter)
                ORDER BY dc.embedding <=> query_embedding
                LIMIT top_k;
            END IF;
        END;
        $$;
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS search_similar_chunks(vector, int, timestamptz, boolean)")
    op.execute("DROP TRIGGER IF EXISTS update_documents_updated_at ON documents")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    op.drop_table("ingestion_logs")
    op.drop_table("document_chunks")
    op.drop_table("documents")

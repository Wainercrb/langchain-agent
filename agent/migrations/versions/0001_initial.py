"""Initial schema: documents, document_chunks, ingestion_logs, version_cache.

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
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("version_date", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("filename", "version_date", name="unique_filename_version"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_documents_filename_version", "documents", [sa.text("filename"), sa.text("version_date DESC")])
    op.create_index("idx_documents_created_at", "documents", [sa.text("created_at DESC")])

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

    op.create_table(
        "version_cache",
        sa.Column("id", sa.UUID(), nullable=False, server_default=sa.text("gen_random_uuid()")),
        sa.Column("filename", sa.Text(), nullable=False, unique=True),
        sa.Column("document_id", sa.UUID(), nullable=False),
        sa.Column("version_date", sa.DateTime(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("cached_at", sa.DateTime(), nullable=False, server_default=sa.text("NOW()")),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("idx_version_cache_filename", "version_cache", ["filename"])

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

    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO authenticated")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO authenticated")
    op.execute("GRANT SELECT, INSERT, UPDATE ON ingestion_logs TO authenticated")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON version_cache TO authenticated")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS update_documents_updated_at ON documents")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    op.drop_table("version_cache")
    op.drop_table("ingestion_logs")
    op.drop_table("document_chunks")
    op.drop_table("documents")

-- Supabase Schema Setup for RAG Vector Database
-- Run this SQL in your Supabase SQL Editor

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table: stores document metadata with versioning
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    version_date TIMESTAMP NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_filename_version UNIQUE(filename, version_date)
);

-- Create index for efficient version queries (MAX date per filename)
CREATE INDEX IF NOT EXISTS idx_documents_filename_version 
ON documents(filename, version_date DESC);

-- Create index for quick lookups by creation time
CREATE INDEX IF NOT EXISTS idx_documents_created_at 
ON documents(created_at DESC);

-- Document chunks table: stores text chunks with embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_chunk_index CHECK (chunk_index >= 0)
);

-- Create HNSW index for fast vector similarity search
-- This index is optimized for L2 distance (cosine similarity)
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
ON document_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index for quick lookup by document_id
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
ON document_chunks(document_id);

-- Create index for chunk ordering
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_chunk 
ON document_chunks(document_id, chunk_index);

-- Ingestion logs table: audit trail for document ingestion
CREATE TABLE IF NOT EXISTS ingestion_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('success', 'failure', 'partial')),
    error_message TEXT,
    chunk_count INT DEFAULT 0,
    processed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_chunk_count CHECK (chunk_count >= 0)
);

-- Create index for quick lookup by filename and status
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_filename_status 
ON ingestion_logs(filename, status);

-- Create index for time-based queries
CREATE INDEX IF NOT EXISTS idx_ingestion_logs_processed_at 
ON ingestion_logs(processed_at DESC);

-- Version cache table: stores latest version info for quick lookups
CREATE TABLE IF NOT EXISTS version_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL UNIQUE,
    document_id UUID NOT NULL REFERENCES documents(id),
    version_date TIMESTAMP NOT NULL,
    chunk_count INT DEFAULT 0,
    cached_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index for quick lookups
CREATE INDEX IF NOT EXISTS idx_version_cache_filename 
ON version_cache(filename);

-- Function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for documents.updated_at
CREATE TRIGGER update_documents_updated_at 
BEFORE UPDATE ON documents
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Grants (for Supabase public use, adjust as needed)
GRANT SELECT, INSERT, UPDATE, DELETE ON documents TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON document_chunks TO authenticated;
GRANT SELECT, INSERT, UPDATE ON ingestion_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON version_cache TO authenticated;

-- Add comments for documentation
COMMENT ON TABLE documents IS 'Stores document metadata with version tracking for RAG system';
COMMENT ON TABLE document_chunks IS 'Stores text chunks with embeddings for vector similarity search';
COMMENT ON TABLE ingestion_logs IS 'Audit log for document ingestion operations';
COMMENT ON TABLE version_cache IS 'Cache for latest document version lookups (performance optimization)';
COMMENT ON COLUMN document_chunks.embedding IS 'Vector embedding (1536-dimensional) from Google Embeddings API';

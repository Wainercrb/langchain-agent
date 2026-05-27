#!/usr/bin/env python
"""Test document ingestion directly."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from supabase import create_client

from config import settings
from rag.embeddings import GoogleEmbeddingsWrapper
from rag.vector_store import VectorStore
from services.document_ingester import DocumentIngester
from services.document_processor import DocumentProcessor
from services.file_manager import FileManager
from services.state_tracker import StateTracker
from services.version_manager import VersionManager
from utils.logging import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# Initialize components
logger.info("Initializing components for ingestion test...")

embeddings = GoogleEmbeddingsWrapper(settings.google_api_key)
processor = DocumentProcessor()
supabase_client = create_client(settings.supabase_url, settings.supabase_key)
vector_store = VectorStore(supabase_client)
version_manager = VersionManager()
state_tracker = StateTracker()
file_manager = FileManager(state_tracker)
ingester = DocumentIngester(processor, embeddings, vector_store, version_manager, state_tracker)

# Get files that need processing
logger.info("\nScanning for documents...")
raw_files = file_manager.scan_raw_docs()
new_files, file_hashes = file_manager.detect_new_files(raw_files)

if not new_files:
    logger.info("No new files to process")
    sys.exit(0)

logger.info(f"Found {len(new_files)} files to process:")
for file_path in new_files:
    logger.info(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")

# Process each file
logger.info("\nProcessing files...")
for file_path in new_files:
    md5_hash = file_hashes[file_path.name]
    logger.info(f"\nProcessing: {file_path.name}")
    try:
        result = ingester.ingest_file(file_path, md5_hash)
        logger.info(f"  Status: {result['status']}")
        if result["status"] == "success":
            logger.info(f"  Document ID: {result['document_id']}")
            logger.info(f"  Chunks: {result['chunk_count']}")
        else:
            logger.error(f"  Error: {result.get('error_message', 'Unknown error')}")
    except Exception as e:
        logger.error(f"  Exception: {str(e)}", exc_info=True)

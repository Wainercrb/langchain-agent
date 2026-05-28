"""Watch knowledge/ for new documents, process and store them."""

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from services.container import logger, embeddings, vector_store
from services.parsers.parser import ParserFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_file(file_path: Path) -> None:
    start = time.time()
    size = file_path.stat().st_size
    logger.info(f"Processing: {file_path.name} ({size:,} bytes)")

    # ── Parse ─────────────────────────────────────────────────────────
    ext = file_path.suffix.lower()
    try:
        parser = ParserFactory.get_parser(ext)
        text = parser.parse(file_path)
    except Exception as e:
        logger.error(f"Parse failed: {file_path.name} — {e}")
        shutil.move(str(file_path), str(settings.failed_dir / file_path.name))
        vector_store.log_ingestion(file_path.name, "failure", error_message=str(e))
        return

    logger.debug(f"Parsed {file_path.name}: {len(text)} characters")

    # ── Chunk ─────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if c.strip()]

    if not chunks:
        logger.warning(f"No content after splitting: {file_path.name}")
        shutil.move(str(file_path), str(settings.failed_dir / file_path.name))
        return

    logger.debug(f"Chunked {file_path.name}: {len(chunks)} chunks, "
                 f"avg {len(text) // len(chunks)} chars/chunk")

    # ── Embed ─────────────────────────────────────────────────────────
    try:
        vectors = embeddings.embed_documents(chunks)
    except Exception as e:
        logger.error(f"Embedding failed: {file_path.name} — {e}")
        shutil.move(str(file_path), str(settings.failed_dir / file_path.name))
        vector_store.log_ingestion(file_path.name, "failure", error_message=str(e))
        return

    logger.debug(f"Embedded {file_path.name}: {len(vectors)} vectors, "
                 f"dim={len(vectors[0]) if vectors else '?'}")

    # ── Store ─────────────────────────────────────────────────────────
    doc_id = vector_store.insert_document(file_path.name)
    chunk_records = [
        {"text": chunks[i], "embedding": vectors[i]}
        for i in range(len(chunks))
    ]
    count = vector_store.insert_chunks(doc_id, chunk_records)
    vector_store.log_ingestion(file_path.name, "success", chunk_count=count)

    shutil.move(str(file_path), str(settings.processed_dir / file_path.name))

    elapsed = time.time() - start
    logger.info(f"Done: {file_path.name} ({count} chunks, {elapsed:.1f}s)")


def main() -> None:
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.failed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Watching {settings.knowledge_dir} every {settings.cron_interval_minutes}min")

    while True:
        files = sorted([
            f for f in settings.knowledge_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

        if files:
            logger.info(f"Cycle: {len(files)} file(s) found")
            for f in files:
                process_file(f)
        else:
            logger.debug("Cycle: no files to process")

        time.sleep(settings.cron_interval_minutes * 60)


if __name__ == "__main__":
    main()

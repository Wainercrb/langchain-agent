"""Watch knowledge/ for new documents, process and store them."""

import hashlib
import shutil
import time
from pathlib import Path

from config import settings
from services.container import embeddings, vector_store
from services.parsers.parser import ParserFactory
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_file(file_path: Path) -> None:
    start = time.time()
    print(f"Processing: {file_path.name}")

    raw_bytes = file_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    existing = vector_store.find_document_by_hash(content_hash)
    if existing is not None:
        print("  Skipping (unchanged)")
        shutil.move(str(file_path), str(settings.processed_dir / file_path.name))
        return

    ext = file_path.suffix.lower()
    try:
        parser = ParserFactory.get_parser(ext)
        text = parser.parse(file_path)
    except Exception as e:
        print(f"  Failed: {e}")
        shutil.move(str(file_path), str(settings.failed_dir / file_path.name))
        vector_store.log_ingestion(file_path.name, "failure", error_message=str(e))
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    chunks = [c for c in chunks if c.strip()]

    if not chunks:
        print("  Failed: No content after splitting")
        shutil.move(str(file_path), str(settings.failed_dir / file_path.name))
        return

    try:
        vectors = embeddings.embed_documents(chunks)
    except Exception as e:
        print(f"  Failed: {e}")
        shutil.move(str(file_path), str(settings.failed_dir / file_path.name))
        vector_store.log_ingestion(file_path.name, "failure", error_message=str(e))
        return

    doc_id = vector_store.insert_document(file_path.name, content_hash=content_hash)
    chunk_records = [
        {"text": chunks[i], "embedding": vectors[i]} for i in range(len(chunks))
    ]
    count = vector_store.insert_chunks(doc_id, chunk_records)
    vector_store.log_ingestion(file_path.name, "success", chunk_count=count)

    shutil.move(str(file_path), str(settings.processed_dir / file_path.name))

    elapsed = time.time() - start
    print(f"  Done ({count} chunks, {elapsed:.1f}s)")


def main() -> None:
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    settings.failed_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Watching {settings.knowledge_dir} every {settings.cron_interval_minutes}min"
    )

    while True:
        files = sorted(
            [
                f
                for f in settings.knowledge_dir.iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )

        if files:
            for f in files:
                process_file(f)

        time.sleep(settings.cron_interval_minutes * 60)


if __name__ == "__main__":
    main()

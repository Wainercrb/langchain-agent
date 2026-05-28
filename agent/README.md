# Langchain Agent - RAG Document Ingestion System

Complete batch document ingestion system with automatic scheduling, vector embeddings, and Supabase pgvector storage.

## 🎯 Features

- ✅ **Cron-Based Ingestion**: Automatic processing every 5 minutes
- ✅ **Multi-Format Support**: .txt, .md, .pdf, .docx, .csv
- ✅ **Smart Deduplication**: MD5-based tracking prevents re-processing
- ✅ **Version Management**: Multiple versions per document with timestamp-based retrieval
- ✅ **Batch Embeddings**: Google Generative AI with automatic retry & backoff
- ✅ **pgvector Search**: HNSW indexes for fast similarity search
- ✅ **Alert Notifications**: Console + Discord webhook support
- ✅ **Production Ready**: Structured logging, error handling, async support

---

## 📋 Quick Start (5 minutes)

### 1️⃣ Prerequisites

- **Python 3.10+** (verify: `python --version`)
- **Supabase Account** (free tier: https://supabase.com)
- **Google API Key** (get from: https://aistudio.google.com/app/apikeys)
- **Git** (for cloning)

### 2️⃣ Install & Setup

```bash
# Clone repository
cd c:\oper\me\langchain-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.\.venv\Scripts\Activate.ps1

# Or on macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run complete setup (handles everything)
python scripts/setup.py
```

### 3️⃣ Configure Environment

During setup, you'll be prompted to configure `.env`. Edit it with your credentials:

```bash
# .env file (REQUIRED)
GOOGLE_API_KEY=your_google_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_service_role_key_here

# Optional: Discord alerts
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/...

# Defaults (can be customized)
CRON_INTERVAL_MINUTES=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

### 4️⃣ Verify Setup

```bash
# Check configuration
python -c "from config import settings; print(settings)"

# Test Supabase connection
python scripts/setup_db.py

# Check dependencies
pip list | grep -E "langchain|supabase|apscheduler"
```

### 5️⃣ Start Processing

```bash
# Start the ingestion system
python main.py

# System will now:
# - Run every 5 minutes
# - Scan /knowledge/raw_docs/
# - Process new files
# - Move to /knowledge/processed/ or /knowledge/failed/
# - Send alerts on completion
```

### 6️⃣ Test with Sample Files

```bash
# Drop a test file
echo "This is a test document about Python programming." > knowledge/raw_docs/test.txt

# Watch the logs
# System will process it in the next 5-minute cycle
# Or manually trigger: python -c "from main import *"
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│ main.py                                │
│ ├─ APScheduler (5-min interval)        │
│ └─ IngestionJob                        │
│    ├─ FileManager.scan_raw_docs()      │
│    ├─ FileManager.detect_new_files()   │
│    ├─ DocumentIngester.ingest_batch()  │
│    │  ├─ DocumentProcessor.parse_file()│
│    │  ├─ RecursiveCharacterTextSplitter│
│    │  ├─ GoogleEmbeddingsWrapper       │
│    │  └─ VectorStore.insert_chunks()   │
│    ├─ FileManager.move_to_processed()  │
│    └─ AlertService.send_alerts()       │
├─ Supabase pgvector (backend)           │
│                                      │
└─ /knowledge/ (document folders)        │
```

---

## 📂 Project Structure

```
langchain-agent/
├── config.py                    # Configuration & environment
├── main.py                      # Entry point & scheduler
├── requirements.txt       # Python dependencies
├── schema.sql                   # Supabase schema
├── .env.example                 # Environment template
├── .env                         # Your actual config (git-ignored)
 │
├── rag/
│   ├── embeddings.py           # Google Embeddings API wrapper
│   └── vector_store.py         # Supabase pgvector operations
│
├── services/
│   ├── document_processor.py   # Multi-format file parsing
│   ├── document_ingester.py    # Orchestration pipeline
│   ├── file_manager.py         # File scanning & movement
│   ├── state_tracker.py        # MD5 deduplication
│   ├── version_manager.py      # Version date handling
│   ├── alert_service.py        # Notifications (Discord/email)
│   └── cron_scheduler.py       # APScheduler wrapper
│
├── api/                         # FastAPI routes (future)
├── utils/
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── logging.py              # JSON structured logging
│   └── file_utils.py           # File utility functions
│
├── scripts/
│   ├── setup.py                # Complete project setup
│   └── setup_db.py             # Database initialization
│
├── knowledge/
│   ├── raw_docs/               # Input: new documents here
│   ├── processed/              # Successful ingestions
│   ├── failed/                 # Failed documents
 │
└── tests/                       # Unit & integration tests (Phase 5)
```

---

## 🚀 Usage Examples

### Example 1: Drop a Document

```bash
# Copy file to input folder
cp my_document.pdf knowledge/raw_docs/

# Wait for next cron run (max 5 minutes)
# Or manually trigger via Discord command / REST API (future)

# Check results
ls knowledge/processed/          # Success
ls knowledge/failed/             # Errors
```

### Example 2: Monitor Processing

```bash
# Watch logs in real-time
# Check Discord channel for alerts
# Review JSON logs for structured data:
# - timestamp, level, logger, message
# - module, function, line
# - error details if exception occurred
```

### Example 3: Check State

```python
# Python REPL
from services.state_tracker import StateTracker

tracker = StateTracker()
stats = tracker.get_stats()
print(f"Processed files: {stats['total_processed']}")
print(f"Total chunks: {stats['total_chunks']}")

# List all tracked files
for filename, info in tracker.get_stats().items():
    print(f"{filename}: {info['chunk_count']} chunks")
```

---

## 🔧 Database Setup

### Automatic (Recommended)

```bash
# Runs as part of: python scripts/setup.py
python scripts/setup_db.py
```

**Methods used (in order):**
1. **Direct PostgreSQL** (psycopg2) — Most reliable
2. **Supabase API** — Fallback
3. **Manual SQL** — Last resort

### Manual Setup

If automated setup fails:

1. Open [Supabase Studio](https://app.supabase.com/)
2. Navigate to: **SQL Editor** → **New Query**
3. Copy contents of `schema.sql`
4. Run the SQL
5. Verify tables exist in **Schema** → **Tables**

---

## 📊 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | — | Google Generative AI key |
| `SUPABASE_URL` | ✅ Yes | — | Supabase project URL |
| `SUPABASE_KEY` | ✅ Yes | — | Supabase service role key |
| `CRON_INTERVAL_MINUTES` | ❌ No | `5` | Run frequency |
| `CHUNK_SIZE` | ❌ No | `1000` | Text chunk size (chars) |
| `CHUNK_OVERLAP` | ❌ No | `200` | Chunk overlap (chars) |
| `EMBEDDING_RETRIES` | ❌ No | `3` | Retry attempts |
| `DISCORD_WEBHOOK_URL` | ❌ No | — | Discord notifications |
| `ALERT_EMAIL` | ❌ No | — | Email notifications |
| `LOG_LEVEL` | ❌ No | `INFO` | Logging verbosity |
| `LOG_FILE` | ❌ No | `app.log` | Log file path |
| `KNOWLEDGE_DIR` | ❌ No | `./knowledge` | Document directory |

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_file_manager.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Manual Testing

```bash
# Test embeddings
python -c "from rag.embeddings import GoogleEmbeddingsWrapper; \
           from config import settings; \
           emb = GoogleEmbeddingsWrapper(settings.google_api_key); \
           result = emb.embed_query('test'); \
           print(f'Embedding dimension: {len(result)}')"

# Test document processor
python -c "from services.document_processor import DocumentProcessor; \
           proc = DocumentProcessor(); \
           chunks = proc.chunk_text('Your test text here...'); \
           print(f'Created {len(chunks)} chunks')"

# Test file manager
python -c "from services.file_manager import FileManager; \
           from services.state_tracker import StateTracker; \
           tracker = StateTracker(); \
           fm = FileManager(tracker); \
           files = fm.scan_raw_docs(); \
           print(f'Found {len(files)} files')"
```

---

## 🐛 Troubleshooting

### Issue: "GOOGLE_API_KEY not found"

**Solution:**
```bash
# 1. Get key from: https://aistudio.google.com/app/apikeys
# 2. Add to .env:
echo "GOOGLE_API_KEY=your_key_here" >> .env
# 3. Restart the application
```

### Issue: "Supabase connection failed"

**Solution:**
```bash
# 1. Verify credentials in .env
# 2. Check Supabase project is active
# 3. Test connection:
python -c "from supabase import create_client; \
           from config import settings; \
           client = create_client(settings.supabase_url, settings.supabase_key); \
           print('Connected!')"
```

### Issue: "Table doesn't exist"

**Solution:**
```bash
# Run database setup
python scripts/setup_db.py

# Or manually in Supabase Studio SQL Editor:
\copy schema.sql
```

### Issue: "Files not processing"

**Solution:**
```bash
# 1. Check .env configuration
python -c "from config import settings; print(settings)"

# 2. Check /knowledge/raw_docs/ has files
ls knowledge/raw_docs/

# 3. Check logs
tail -f app.log

# 4. Test manually
python scripts/setup.py
```

### Issue: "psycopg2 not found"

**Solution:**
```bash
# Install directly
pip install psycopg2-binary

# Or it's in requirements.txt:
pip install -r requirements.txt
```

---

## 📈 Performance Tuning

### Batch Size (Embeddings)

```python
# In config.py, adjust EMBEDDING_BATCH_SIZE
# Larger = fewer API calls, but higher memory/latency
# Default: 10 texts per batch (recommended for Google API)
```

### Chunk Size

```python
# CHUNK_SIZE=500  # Smaller chunks = more precise search
# CHUNK_SIZE=2000 # Larger chunks = more context
# Default: 1000 (balanced)
```

### Cron Interval

```bash
# CRON_INTERVAL_MINUTES=1  # Real-time processing (higher cost)
# CRON_INTERVAL_MINUTES=5  # Balanced (default)
# CRON_INTERVAL_MINUTES=60 # Batch overnight processing
```

---

## 🔐 Security Best Practices

1. **Never commit .env** ✅ (already in .gitignore)
2. **Use service role key** (not anon key for database ops)
3. **Rotate API keys** regularly
4. **Restrict file uploads** to trusted sources
5. **Monitor ingestion logs** for failures
6. **Enable pgvector indexes** for faster queries
7. **Set CRON_INTERVAL_MINUTES** to appropriate value

---

## 🚀 Deployment

### Local Development

```bash
# Already covered in Quick Start above
python main.py
```

### Production (Docker)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

### Cloud Deployment

- **Railway.app**: Supports Python + environment variables
- **Render**: Native support for Python + cron
- **AWS Lambda**: Use with scheduled CloudWatch events
- **Google Cloud Run**: Jobs for scheduled execution
- **Azure Container Instances**: Scheduled tasks

---

## 📚 API Reference

### DocumentIngester

```python
from services.document_ingester import DocumentIngester

# Single file ingestion
result = ingester.ingest_file(
    file_path="path/to/document.pdf",
    md5_hash="abc123...",
    custom_version_date="2026-05-22"
)

# Batch ingestion
results = ingester.ingest_batch(
    files=["file1.txt", "file2.pdf"],
    file_hashes={"file1.txt": "hash1", "file2.pdf": "hash2"}
)
```

### VectorStore

```python
from rag.vector_store import VectorStore

# Insert document
doc_id = vector_store.insert_document(
    filename="report.pdf",
    version_date="2026-05-22",
    metadata={"category": "reports"}
)

# Search
results = vector_store.search_similar(
    query_embedding=[...1536 dimensions...],
    top_k=5
)
```

### FileManager

```python
from services.file_manager import FileManager

# Scan for new files
files = file_manager.scan_raw_docs()

# Detect changes
new_files, hashes = file_manager.detect_new_files(files)

# Move files
file_manager.move_to_processed(Path("knowledge/raw_docs/file.txt"))
file_manager.move_to_failed(Path("knowledge/raw_docs/file.txt"), "Parse error")
```

---

## 📝 Logging

### View Logs

```bash
# Live logs (default console)
python main.py

# File logs
tail -f app.log

# JSON parsing (for monitoring systems)
cat app.log | jq '.'
```

### Log Format

```json
{
  "timestamp": "2026-05-22T14:30:45.123Z",
  "level": "INFO",
  "logger": "services.document_ingester",
  "message": "Successfully ingested report.pdf",
  "module": "document_ingester",
  "function": "ingest_batch",
  "line": 85,
  "exception": null
}
```

---

## 🌐 Retrieval API

Query stored documents and get AI-generated answers with context.

### Starting the API Server

```bash
# From project root
python main.py

# Server runs on http://localhost:8000
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc
```

### API Documentation

- **Interactive Swagger UI**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Endpoints

#### POST /v1/chat — Query Documents with RAG

Query stored documents and get AI-generated answers with sources.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I enroll in the program?",
    "top_k": 5,
    "include_sources": true,
    "temperature": 0.7
  }'
```

**Parameters:**
- `query` (string, required): Your question (1-2000 characters)
- `top_k` (integer, default=5): Number of documents to retrieve (1-20)
- `include_sources` (boolean, default=true): Include source document references
- `temperature` (float, default=0.7): LLM creativity level (0.0=deterministic, 1.0=creative)

**Response (200):**
```json
{
  "response": "To enroll in the program, you need to: 1. Complete the online form... 2. Submit required documentation...",
  "query": "How do I enroll in the program?",
  "sources": [
    {
      "document_id": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "enrollment_guide.pdf",
      "similarity_score": 0.95,
      "version_date": "2025-01-15T10:30:00",
      "content_preview": "Enrollment Guidelines: To enroll in our program, follow these steps...",
      "chunk_id": "550e8400-e29b-41d4-a716-446655440001"
    }
  ],
  "execution_time_ms": 2340.5,
  "model": "gemini-2.5-flash"
}
```

**Error (400) - Query Too Long:**
```json
{
  "detail": {
    "error": "invalid_request",
    "message": "Query exceeds maximum length of 2000 characters",
    "timestamp": "2025-01-15T12:00:00"
  }
}
```

**Error (422) - Invalid Parameters:**
```json
{
  "detail": [
    {
      "loc": ["body", "top_k"],
      "msg": "ensure this value is less than or equal to 20",
      "type": "value_error.number.not_le"
    }
  ]
}
```

#### GET /v1/health — Health Check

Verify API and database connectivity.

**Request:**
```bash
curl http://localhost:8000/v1/health
```

**Response (200):**
```json
{
  "status": "ok",
  "timestamp": "2025-01-15T12:00:00",
  "version": "1.0.0",
  "db_connected": true
}
```

**Response (200) - Error State:**
```json
{
  "status": "error",
  "timestamp": "2025-01-15T12:00:00",
  "version": "1.0.0",
  "db_connected": false
}
```

### Python Client Example

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Query documents
response = requests.post(f"{BASE_URL}/v1/chat", json={
    "query": "What are the enrollment requirements?",
    "top_k": 5,
    "include_sources": True,
    "temperature": 0.7
})

result = response.json()
print(f"Answer: {result['response']}")
print(f"Sources: {len(result.get('sources', []))} documents")
print(f"Time: {result['execution_time_ms']:.0f}ms")

# Check health
health = requests.get(f"{BASE_URL}/v1/health").json()
print(f"API Status: {health['status']}")
print(f"DB Connected: {health['db_connected']}")
```

### JavaScript/Fetch Example

```javascript
// Query documents
const response = await fetch('http://localhost:8000/v1/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'How do I enroll?',
    top_k: 5,
    include_sources: true,
    temperature: 0.7
  })
});

const data = await response.json();
console.log('Answer:', data.response);
console.log('Sources:', data.sources?.length || 0);
console.log('Time:', data.execution_time_ms, 'ms');
```

### Configuration

Set environment variables in `.env`:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0.7
LOG_LEVEL=INFO
```

### Testing API

```bash
# Run all tests
pytest tests/ -v

# Run only API tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=api --cov-report=html
```

### Common Issues

**"Database connection failed"**
```bash
# Verify database setup
python scripts/setup_db.py

# Check Supabase credentials
python -c "from config import settings; print(settings.supabase_url)"
```

**"Query takes too long"**
```bash
# Ensure pgvector index exists
# Check Supabase SQL Editor for index creation

# Try reducing top_k
curl -X POST http://localhost:8000/v1/chat \
  -d '{"query": "...", "top_k": 3}'
```

**"No results returned"**
```bash
# Check that documents have been ingested
# Run batch processor first:
python main.py &

# Wait for documents to be processed, then query API
curl http://localhost:8000/v1/health
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🆘 Support

- **Issues**: Create GitHub issue with details
- **Docs**: See README.md and QUICKSTART.md
- **Discord**: Join our community server
- **Email**: support@example.com

---

## 🎯 Roadmap

- [ ] REST API for manual uploads
- [ ] Web dashboard for monitoring
- [ ] Support for more LLMs (OpenAI, Anthropic)
- [ ] Advanced RAG retrieval (re-ranking, fusion)
- [ ] Multi-tenant support
- [ ] Distributed processing
- [ ] Kubernetes deployment
- [ ] GraphQL API

---

**Happy document processing! 🚀**

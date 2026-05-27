# ⚡ Quick Start — 5 Minutes

Get the RAG system running in 5 minutes. No manual SQL required.

---

## Step 1: Activate Virtual Environment (30 seconds)

```bash
# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

---

## Step 2: Run Automated Setup (3 minutes)

```bash
python scripts/setup.py
```

This script will:
- ✅ Validate Python version
- ✅ Create `.env` from template (if missing)
- ✅ Install and verify dependencies
- ✅ **Automatically create database tables** (no manual SQL!)
- ✅ Create `/knowledge` folders
- ✅ Validate all configurations

**Output:**
```
✅ Python Version
✅ Environment File  
✅ Dependencies
✅ Load Environment
✅ Create Directories
✅ Database Setup
✅ Validate Config

Passed: 7/7
✅ Setup completed successfully!
```

---

## Step 3: Add Your Credentials (1 minute)

If setup paused asking for credentials:

```bash
# Edit .env file
GOOGLE_API_KEY=sk-...your key from https://aistudio.google.com/...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=...service role key from Supabase project settings...
```

Then re-run:
```bash
python scripts/setup.py
```

---

## Step 4: Start Processing (30 seconds)

```bash
python main.py
```

**Output:**
```
============================================================
🤖 Langchain Agent - Batch Document Ingestion System
============================================================
[Config loaded]
✅ Embeddings initialized
✅ Document processor initialized
✅ Vector store initialized
...
Scheduler configured for 5-minute intervals
Starting scheduler (Ctrl+C to stop)...
```

---

## Step 5: Test with a File (1 minute)

Open **another terminal** while main.py is running:

```bash
# Drop a test file
echo "Python is a popular programming language." > knowledge/raw_docs/test.txt

# Wait 5 minutes OR manually trigger
```

**Check results:**
```bash
# Success files
ls knowledge/processed/

# Failed files
ls knowledge/failed/

# Processed state
type .processed_files.json
```

---

## ✅ Done!

Your RAG system is running! 🚀

### Next Steps

1. **Drop PDFs/Docs** into `knowledge/raw_docs/`
2. **System auto-processes** every 5 minutes
3. **Check alerts** in Discord (if configured)
4. **Search embeddings** in Supabase

---

## 🆘 Troubleshooting

### ❌ Setup failed at "Database Setup"

**Reason:** Connection or schema issue

**Fix:**
```bash
# 1. Verify credentials in .env
cat .env | grep SUPABASE

# 2. Try manual database setup
python scripts/setup_db.py

# 3. Or skip and run anyway (may fail on ingestion)
python main.py
```

### ❌ "GOOGLE_API_KEY not found"

**Fix:**
```bash
# 1. Get key: https://aistudio.google.com/app/apikeys
# 2. Add to .env
echo "GOOGLE_API_KEY=your_key" >> .env

# 3. Re-run
python main.py
```

### ❌ Files not processing

**Check:**
```bash
# 1. Is main.py still running?
# 2. Are files in correct folder?
ls knowledge/raw_docs/

# 3. Check logs
tail -f app.log
```

---

## 📊 What Happens Next?

```
Cron Job (every 5 min)
    ↓
Scan /knowledge/raw_docs/
    ↓
Check for new/modified files (MD5)
    ↓
Parse multi-format (PDF, DOCX, etc)
    ↓
Chunk text (RecursiveCharacterTextSplitter)
    ↓
Create embeddings (Google Generative AI)
    ↓
Store in Supabase pgvector
    ↓
Move file to /processed/ or /failed/
    ↓
Send Discord alert
```

---

## 🎯 Next: Add to Slack/Discord

Get alerts when documents are processed:

```bash
# In .env:
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
```

---

## 🚀 Production Ready!

Your system is now:
- ✅ Monitoring `/knowledge/raw_docs/`
- ✅ Auto-deduplicating via MD5
- ✅ Batch embedding with retry
- ✅ Storing in pgvector
- ✅ Alerting on Discord
- ✅ Tracking all state

**Stop with:** `Ctrl+C`

---

**Ready to ingest? Start dropping documents! 📄**

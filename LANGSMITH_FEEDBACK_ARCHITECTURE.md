# LangSmith Feedback Loop Architecture

## Overview

This document explains how the LangSmith feedback loop works in the langchain-agent system, the bug that was breaking it, and how to verify the fix in production.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE (Astro)                          │
│                                                                              │
│  User types query ──► POST /v1/chat ──► Receives ChatResponse with run_id   │
│                                                                              │
│  User clicks 👍/👎 ──► POST /v1/feedback with run_id                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND (Python)                           │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ToolCallingAgent.invoke()                                          │    │
│  │                                                                     │    │
│  │  @traceable(name="ToolCallingAgent.invoke", run_type="chain")     │    │
│  │  def invoke(self, query: str, ...) -> ChatResponse:               │    │
│  │      # LangSmith creates trace automatically                       │    │
│  │      run_tree = get_current_run_tree()  # ← Get active trace      │    │
│  │      run_id = str(run_tree.id) if run_tree else None              │    │
│  │                                                                     │    │
│  │      # ... execute LLM + tools ...                                │    │
│  │                                                                     │    │
│  │      return ChatResponse(                                          │    │
│  │          response=response_text,                                   │    │
│  │          run_id=run_id,  # ← Return LangSmith trace ID            │    │
│  │          ...                                                       │    │
│  │      )                                                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ LangSmithFeedbackProvider.record_feedback()                        │    │
│  │                                                                     │    │
│  │  def record_feedback(self, run_id: str, feedback_type: str):      │    │
│  │      self._client.create_feedback(                                 │    │
│  │          run_id=run_id,  # ← Attach feedback to LangSmith trace   │    │
│  │          key="user-feedback",                                      │    │
│  │          score=1.0 if feedback_type == "like" else 0.0,           │    │
│  │      )                                                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LANGSMITH PLATFORM                              │
│                                                                              │
│  Trace: ToolCallingAgent.invoke (ID: abc-123-def-456)                       │
│  ├── Input: "Find the security policy"                                      │
│  ├── LLM Call #1: Tool selection → search_documents                         │
│  ├── Tool Execution: Retriever → 5 chunks                                   │
│  ├── LLM Call #2: Answer synthesis                                          │
│  ├── Output: "The security policy states..."                                │
│  └── Feedback: 👍 (score=1.0) ← Attached via run_id                        │
│                                                                              │
│  Dashboard: https://smith.langchain.com/o/default/projects/p/langchain-agent│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How LangSmith Tracing Works

### Automatic Trace Creation

When you decorate a function with `@traceable`, LangSmith automatically:

1. **Creates a trace** when the function is called
2. **Generates a unique trace ID** (UUID)
3. **Records inputs, outputs, latency, and token usage**
4. **Sends the trace to LangSmith** asynchronously

```python
from langsmith import traceable

@traceable(name="ToolCallingAgent.invoke", run_type="chain")
def invoke(self, query: str, ...) -> ChatResponse:
    # LangSmith trace is created automatically
    # Trace ID is NOT returned by the decorator
    # You must call get_current_run_tree() to access it
    ...
```

### Accessing the Trace ID

The trace ID is stored in a **context variable** that's accessible from within the traced function:

```python
from langsmith.run_helpers import get_current_run_tree

@traceable
def my_function():
    run_tree = get_current_run_tree()
    
    if run_tree:
        trace_id = run_tree.id  # UUID of the active trace
        trace_name = run_tree.name  # Name of the trace
        print(f"Trace ID: {trace_id}")
    
    return "result"
```

### Why This Matters

The trace ID is the **correlation key** that links:
- The user's query
- The LLM's decision-making process
- The tools that were called
- The documents that were retrieved
- **User feedback** (👍/👎)

Without the correct trace ID, feedback cannot be correlated with the execution that produced the response.

---

## The Bug: Random UUID Instead of Trace ID

### What Was Happening

```python
# BEFORE (Broken)
@traceable(name="ToolCallingAgent.invoke", run_type="chain")
def invoke(self, query: str, ...) -> ChatResponse:
    # ... execute LLM + tools ...
    
    run_id = str(uuid.uuid4())  # ← Random UUID, NOT the LangSmith trace ID
    
    return ChatResponse(
        response=response_text,
        run_id=run_id,  # ← Wrong ID returned to user
    )
```

**Result**:
- LangSmith creates trace with ID `abc-123-def-456`
- Agent returns random UUID `xyz-789-uvw-012` as `run_id`
- User submits feedback with `run_id="xyz-789-uvw-012"`
- LangSmith tries to attach feedback to non-existent trace
- Feedback is silently lost (or logged as "accepted" due to graceful degradation)

### Why It Was Hard to Detect

1. **Graceful Degradation**: The feedback service catches exceptions and returns `status="accepted"` when LangSmith is unreachable or the run_id is invalid. This masks the bug.

2. **No Validation**: The API doesn't validate that the `run_id` exists in LangSmith before accepting feedback.

3. **Silent Failure**: LangSmith doesn't error when you try to attach feedback to a non-existent run — it just ignores it.

4. **Random UUID Looks Valid**: A random UUID is still a valid UUID format, so it passes all validation checks.

---

## The Fix: Capture the Real Trace ID

### What Should Happen

```python
# AFTER (Fixed)
from langsmith.run_helpers import get_current_run_tree

@traceable(name="ToolCallingAgent.invoke", run_type="chain")
def invoke(self, query: str, ...) -> ChatResponse:
    # ... execute LLM + tools ...
    
    # Capture the ACTUAL LangSmith trace ID
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None
    
    logger.debug(f"LangSmith run_id captured: {run_id}")
    
    return ChatResponse(
        response=response_text,
        run_id=run_id,  # ← Correct trace ID returned to user
    )
```

**Result**:
- LangSmith creates trace with ID `abc-123-def-456`
- Agent captures `run_tree.id` = `abc-123-def-456`
- Agent returns `run_id="abc-123-def-456"` to user
- User submits feedback with `run_id="abc-123-def-456"`
- LangSmith attaches feedback to the correct trace ✅
- Feedback appears in LangSmith dashboard, correlated with full execution

---

## Verifying the Fix in Production

### Step 1: Check the API Response

Send a test query and inspect the `run_id` in the response:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find the security policy",
    "top_k": 5,
    "include_sources": true,
    "temperature": 0.7
  }'
```

**Expected Response**:
```json
{
  "response": "The security policy states...",
  "query": "Find the security policy",
  "sources": [...],
  "execution_time_ms": 2340.5,
  "model": "openai/gpt-4o",
  "run_id": "abc-123-def-456-..."  ← Should be a valid UUID
}
```

**Verification**:
- ✅ `run_id` is present and not `null`
- ✅ `run_id` is a valid UUID format
- ✅ `run_id` is NOT the same across multiple requests (each trace has a unique ID)

### Step 2: Verify in LangSmith Dashboard

1. Go to your LangSmith project dashboard:
   ```
   https://smith.langchain.com/o/default/projects/p/langchain-agent
   ```

2. Find the trace for your test query (look for the most recent trace with your query text)

3. Copy the **Trace ID** from the LangSmith UI (it's displayed at the top of the trace details)

4. Compare it with the `run_id` from the API response:
   - ✅ They should be **identical**
   - ❌ If they're different, the fix didn't work

### Step 3: Submit Feedback and Verify Correlation

1. Use the UI to submit feedback (👍 or 👎) for the test query

2. Or submit feedback via API:
   ```bash
   curl -X POST http://localhost:8000/v1/feedback \
     -H "Content-Type: application/json" \
     -d '{
       "run_id": "abc-123-def-456-...",  ← Use the run_id from Step 1
       "feedback_type": "like",
       "comment": "Great answer!"
     }'
   ```

3. Go back to the LangSmith trace from Step 2

4. Look for the **Feedback** section in the trace details:
   - ✅ You should see: `user-feedback: 1.0 (like)` with your comment
   - ❌ If feedback is missing, the correlation is broken

### Step 4: Check Logs

Look for the debug log that confirms the run_id was captured:

```bash
docker logs langchain-agent 2>&1 | grep "LangSmith run_id captured"
```

**Expected Output**:
```
DEBUG | LangSmith run_id captured: abc-123-def-456-...
```

### Step 5: Monitor Feedback Over Time

After deployment, monitor the LangSmith dashboard for:

1. **Feedback Count**: Number of traces with feedback attached
2. **Feedback Distribution**: Ratio of likes vs dislikes
3. **Common Queries**: Queries that receive negative feedback (potential quality issues)

**Expected Behavior**:
- ✅ New traces should have feedback attached within seconds of user submission
- ✅ Feedback should appear in the "Feedback" tab of the LangSmith dashboard
- ✅ You can filter traces by feedback score to find problematic queries

---

## Troubleshooting

### Problem: `run_id` is `null` in API Response

**Cause**: LangSmith tracing is disabled or `get_current_run_tree()` returned `None`

**Solution**:
1. Check that `ENABLE_LANGSMITH_TRACING=true` in `.env`
2. Verify `LANGSMITH_API_KEY` is set correctly
3. Check logs for LangSmith initialization errors:
   ```bash
   docker logs langchain-agent 2>&1 | grep -i langsmith
   ```

### Problem: `run_id` Doesn't Match LangSmith Trace ID

**Cause**: The fix wasn't applied correctly, or there's a caching issue

**Solution**:
1. Verify the code change in `tool_calling.py`:
   ```bash
   grep -A 3 "get_current_run_tree" agent/infrastructure/agent/tool_calling.py
   ```
2. Restart the container to clear any cached code:
   ```bash
   docker-compose restart agent
   ```
3. Check that you're looking at the correct trace in LangSmith (match by query text and timestamp)

### Problem: Feedback Doesn't Appear in LangSmith

**Cause**: LangSmith API key is invalid, or network connectivity issue

**Solution**:
1. Test LangSmith connectivity:
   ```bash
   curl -H "x-api-key: $LANGSMITH_API_KEY" \
     https://api.smith.langchain.com/info
   ```
2. Check feedback service logs:
   ```bash
   docker logs langchain-agent 2>&1 | grep "Feedback"
   ```
3. Verify the `run_id` you're using matches a trace in LangSmith

### Problem: Feedback Returns `status="accepted"` Instead of `status="recorded"`

**Cause**: LangSmith is unreachable or the run_id is invalid (graceful degradation)

**Solution**:
1. Check network connectivity to LangSmith
2. Verify the `run_id` exists in LangSmith dashboard
3. Check logs for feedback errors:
   ```bash
   docker logs langchain-agent 2>&1 | grep "Failed to record feedback"
   ```

---

## Best Practices

### 1. Always Check for `None`

When using `get_current_run_tree()`, always handle the case where tracing is disabled:

```python
run_tree = get_current_run_tree()
run_id = str(run_tree.id) if run_tree else None
```

### 2. Log the Trace ID for Debugging

Add a debug log to confirm the trace ID was captured:

```python
logger.debug(f"LangSmith run_id captured: {run_id}")
```

### 3. Validate Feedback Before Submission

In the UI, check that `run_id` is not `null` before showing feedback buttons:

```typescript
if (message.runId) {
  // Show 👍/👎 buttons
} else {
  // Hide feedback buttons (tracing disabled)
}
```

### 4. Monitor Feedback Quality

Use LangSmith's feedback analytics to:
- Identify queries with low feedback scores
- Find patterns in negative feedback
- Improve prompts and document coverage based on feedback

### 5. Test the Feedback Loop Regularly

Add an integration test that:
1. Sends a query
2. Captures the `run_id`
3. Submits feedback
4. Verifies feedback appears in LangSmith

Run this test in CI/CD to catch regressions.

---

## Related Documentation

- **Fix Plan**: `FIX_RUN_ID_PLAN.md` — Implementation steps and testing strategy
- **LangSmith Docs**: https://docs.smith.langchain.com/
- **Feedback API**: https://docs.smith.langchain.com/how_to_guides/evaluation/feedback
- **Tracing Guide**: https://docs.smith.langchain.com/how_to_guides/tracing

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Production Ready

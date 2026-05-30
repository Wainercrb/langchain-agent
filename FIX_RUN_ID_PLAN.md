# Fix Plan: LangSmith run_id Feedback Loop Bug

## Executive Summary

**Bug**: The `/v1/chat` endpoint returns a random `uuid.uuid4()` as `run_id` instead of the actual LangSmith trace ID, breaking the user feedback loop.

**Impact**: When users click 👍/👎 in the UI, feedback is sent to a non-existent LangSmith run. The entire feedback correlation system is silently broken.

**Fix**: Use `langsmith.run_helpers.get_current_run_tree().id` to capture the real LangSmith trace ID inside the `@traceable` function.

**Effort**: ~2 hours (implementation + testing + verification)

**Risk**: Low — single-file change, backward compatible, no API contract changes.

---

## 1. Problem Analysis

### Current Behavior (Broken)

```python
# agent/infrastructure/agent/tool_calling.py:129
run_id = str(uuid.uuid4())  # ← Random UUID, NOT the LangSmith trace ID
```

**Flow**:
1. User sends query → `POST /v1/chat`
2. `ToolCallingAgent.invoke()` runs with `@traceable` decorator
3. LangSmith creates a trace with ID `abc-123-def-456`
4. Agent generates random UUID `xyz-789-uvw-012` and returns it as `run_id`
5. User clicks 👍 → `POST /v1/feedback` with `run_id="xyz-789-uvw-012"`
6. LangSmith tries to attach feedback to run `xyz-789-uvw-012` → **run doesn't exist**
7. Feedback is silently lost or logged as "accepted" (graceful degradation masks the bug)

### Expected Behavior (Fixed)

```python
# agent/infrastructure/agent/tool_calling.py
from langsmith.run_helpers import get_current_run_tree

@traceable(name="ToolCallingAgent.invoke", run_type="chain")
def invoke(self, query: str, ...) -> ChatResponse:
    # ... existing code ...
    
    # Capture the ACTUAL LangSmith trace ID
    run_tree = get_current_run_tree()
    run_id = str(run_tree.id) if run_tree else None
    
    return ChatResponse(
        response=response_text,
        query=query,
        sources=sources_list,
        execution_time_ms=execution_time_ms,
        model=getattr(self._llm, "model", "unknown"),
        run_id=run_id,  # ← Real LangSmith trace ID
    )
```

**Flow**:
1. User sends query → `POST /v1/chat`
2. `ToolCallingAgent.invoke()` runs with `@traceable` decorator
3. LangSmith creates a trace with ID `abc-123-def-456`
4. Agent captures `run_tree.id` = `abc-123-def-456` and returns it as `run_id`
5. User clicks 👍 → `POST /v1/feedback` with `run_id="abc-123-def-456"`
6. LangSmith attaches feedback to the correct trace ✅
7. Feedback appears in LangSmith dashboard, correlated with the full execution trace

---

## 2. Root Cause

The `@traceable` decorator from LangSmith automatically creates a trace when the function is called, but the trace ID is not exposed as a return value. The developer must explicitly call `get_current_run_tree()` from within the traced function to access the active `RunTree` object and its `.id` attribute.

The current code generates a random UUID because:
- The developer may not have known about `get_current_run_tree()`
- Or they assumed the trace ID would be returned by the decorator (it's not)
- Or they wanted a fallback when tracing is disabled (but didn't handle the `None` case properly)

---

## 3. Implementation Plan

### Step 1: Update `tool_calling.py`

**File**: `agent/infrastructure/agent/tool_calling.py`

**Changes**:
1. Import `get_current_run_tree` from `langsmith.run_helpers`
2. Inside the `@traceable` function, call `get_current_run_tree()` to get the active trace
3. Extract the trace ID from the `RunTree` object
4. Handle the case where tracing is disabled (`run_tree` is `None`)

```python
# Add import at top of file
from langsmith.run_helpers import get_current_run_tree

# Inside ToolCallingAgent.invoke() method, replace line 129:
# OLD:
run_id = str(uuid.uuid4())

# NEW:
run_tree = get_current_run_tree()
run_id = str(run_tree.id) if run_tree else None
```

### Step 2: Update `domain/core/chain.py` (legacy agent)

**File**: `agent/domain/core/chain.py`

**Bug confirmed**: Line 85 has the same `run_id = str(uuid.uuid4())` bug.

**Changes**:
1. Import `get_current_run_tree` from `langsmith.run_helpers`
2. Replace line 85 with the same fix as Step 1

```python
# Add import at top of file (replace `import uuid` if no longer needed)
from langsmith.run_helpers import get_current_run_tree

# Inside RAGChain.invoke() method, replace line 85:
# OLD:
run_id = str(uuid.uuid4())

# NEW:
run_tree = get_current_run_tree()
run_id = str(run_tree.id) if run_tree else None
```

**Note**: This file is used when `USE_TOOL_AGENT=false` (the legacy RAG chain mode). Both agents must be fixed since the container can switch between them via environment variable.

### Step 3: Add Logging for Debugging

Add a debug log to confirm the run_id is being captured correctly:

```python
logger.debug(f"LangSmith run_id captured: {run_id}")
```

### Step 4: Update Documentation

Update the docstring for `ToolCallingAgent.invoke()` to document that `run_id` is the LangSmith trace ID when tracing is enabled, or `None` when disabled.

---

## 4. Testing Strategy

### Unit Tests

**Test 1: Tracing Enabled**
- Mock `get_current_run_tree()` to return a `RunTree` with a known ID
- Call `ToolCallingAgent.invoke()`
- Assert that the returned `ChatResponse.run_id` matches the mocked trace ID

**Test 2: Tracing Disabled**
- Mock `get_current_run_tree()` to return `None`
- Call `ToolCallingAgent.invoke()`
- Assert that the returned `ChatResponse.run_id` is `None`

**Test 3: Feedback Correlation**
- Call `ToolCallingAgent.invoke()` and capture the `run_id`
- Call `LangSmithFeedbackProvider.record_feedback(run_id=run_id, ...)`
- Verify that the feedback is attached to the correct trace in LangSmith (via LangSmith API or dashboard)

### Integration Tests

**Test 4: End-to-End Feedback Loop**
1. Start the FastAPI server with `ENABLE_LANGSMITH_TRACING=true`
2. Send a chat request via the UI
3. Capture the `run_id` from the response
4. Click 👍 in the UI to submit feedback
5. Verify in LangSmith dashboard that the feedback appears on the correct trace

### Manual Verification

**Verification Checklist**:
- [ ] `run_id` in `/v1/chat` response is a valid UUID (not random)
- [ ] `run_id` matches the trace ID visible in LangSmith dashboard
- [ ] Feedback submitted via UI appears in LangSmith under the correct trace
- [ ] When tracing is disabled, `run_id` is `None` (not a random UUID)
- [ ] No regression in existing functionality (chat still works, sources still returned)

---

## 5. Rollout Plan

### Pre-Deployment

1. **Code Review**: Ensure the fix is reviewed by at least one other developer
2. **Unit Tests**: All unit tests pass (including new tests for this fix)
3. **Staging Deployment**: Deploy to staging environment with LangSmith enabled
4. **Staging Verification**: Run the end-to-end feedback loop test in staging

### Deployment

1. **Deploy to Production**: Use existing deployment pipeline (Docker + AWS)
2. **Smoke Test**: Send a test query and verify `run_id` is populated
3. **Monitor Logs**: Watch for any errors related to `get_current_run_tree()`
4. **Verify Feedback**: Submit test feedback and confirm it appears in LangSmith

### Post-Deployment

1. **Monitor LangSmith**: Check that new traces have feedback attached
2. **Check Metrics**: Verify that `/v1/metrics` shows no increase in error rate
3. **User Feedback**: Monitor user feedback submissions for the next 24 hours
4. **Rollback Plan**: If issues arise, revert the commit (single-file change, easy rollback)

---

## 6. Success Criteria

The fix is successful when:

- ✅ `run_id` in `/v1/chat` responses matches the LangSmith trace ID
- ✅ User feedback (👍/👎) appears in LangSmith dashboard under the correct trace
- ✅ No increase in error rate or latency
- ✅ Backward compatible (existing clients continue to work)
- ✅ Graceful degradation when tracing is disabled (`run_id` is `None`, not a random UUID)

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `get_current_run_tree()` returns `None` unexpectedly | Low | Medium | Add defensive check: `run_id = str(run_tree.id) if run_tree else None` |
| LangSmith SDK version incompatibility | Low | High | Pin `langsmith>=0.1.0` in `requirements.txt`, test with current version |
| Performance overhead from `get_current_run_tree()` | Very Low | Low | Function is a simple context lookup, negligible overhead |
| Feedback still doesn't appear in LangSmith | Medium | High | Verify LangSmith API key is correct, check network connectivity, review LangSmith dashboard filters |

---

## 8. Future Improvements

After this fix is deployed, consider:

1. **Local Decision Audit Log**: Store a local log of "query → tool chosen → run_id" for querying without LangSmith
2. **Feedback Analytics Dashboard**: Build a simple dashboard showing feedback trends (likes vs dislikes, common queries with low feedback)
3. **Automated Feedback Alerts**: Alert when a query receives multiple dislikes in a short time (potential quality issue)
4. **Feedback-Driven Retraining**: Use negative feedback to identify queries that need better prompts or more documents

---

## 9. References

- **LangSmith SDK Docs**: https://docs.smith.langchain.com/reference/python
- **`get_current_run_tree()` API**: https://docs.smith.langchain.com/reference/python#langsmith.run_helpers.get_current_run_tree
- **Feedback API**: https://docs.smith.langchain.com/how_to_guides/evaluation/feedback
- **Current Code**: `agent/infrastructure/agent/tool_calling.py:129`
- **Feedback Service**: `agent/infrastructure/feedback/langsmith.py`

---

## 10. Appendix: Code Diff

### File 1: `agent/infrastructure/agent/tool_calling.py`

```diff
diff --git a/agent/infrastructure/agent/tool_calling.py b/agent/infrastructure/agent/tool_calling.py
index 1234567..abcdefg 100644
--- a/agent/infrastructure/agent/tool_calling.py
+++ b/agent/infrastructure/agent/tool_calling.py
@@ -11,6 +11,7 @@ from langchain_core.tools import BaseTool
 from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
 from langsmith import traceable
+from langsmith.run_helpers import get_current_run_tree
 
 from models import ChatResponse, SourceDocument
 from services.agent.base import Agent
@@ -126,7 +127,10 @@ class ToolCallingAgent(Agent):
             # Extract sources from tool side-effects
             sources_list = self._extract_sources(include_sources)
 
-            run_id = str(uuid.uuid4())
+            # Capture the actual LangSmith trace ID for feedback correlation
+            run_tree = get_current_run_tree()
+            run_id = str(run_tree.id) if run_tree else None
+            logger.debug(f"LangSmith run_id captured: {run_id}")
 
             logger.info(
                 f"ToolCallingAgent complete: query={query[:50]}..., "
```

### File 2: `agent/domain/core/chain.py`

```diff
diff --git a/agent/domain/core/chain.py b/agent/domain/core/chain.py
index 2345678..bcdefgh 100644
--- a/agent/domain/core/chain.py
+++ b/agent/domain/core/chain.py
@@ -3,7 +3,7 @@ import time
-import uuid
 from datetime import datetime
 from typing import Optional
 
 from langsmith import traceable
+from langsmith.run_helpers import get_current_run_tree
 
 from models import ChatResponse, SourceDocument
@@ -82,7 +82,10 @@ class RAGChain:
                     f"total={llm_response.usage.get('total_tokens', 'N/A')}"
                 )
 
-            run_id = str(uuid.uuid4())
+            # Capture the actual LangSmith trace ID for feedback correlation
+            run_tree = get_current_run_tree()
+            run_id = str(run_tree.id) if run_tree else None
+            logger.debug(f"LangSmith run_id captured: {run_id}")
 
             execution_time_ms = (time.time() - start_time) * 1000
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation

# LangSmith run_id Bug Fix — Quick Reference

## The Bug in One Sentence

The `/v1/chat` endpoint returns a **random UUID** instead of the **LangSmith trace ID**, breaking the entire user feedback loop.

---

## Files to Fix

| File | Line | Current Code | Fixed Code |
|------|------|--------------|------------|
| `agent/infrastructure/agent/tool_calling.py` | 129 | `run_id = str(uuid.uuid4())` | `run_tree = get_current_run_tree(); run_id = str(run_tree.id) if run_tree else None` |
| `agent/domain/core/chain.py` | 85 | `run_id = str(uuid.uuid4())` | `run_tree = get_current_run_tree(); run_id = str(run_tree.id) if run_tree else None` |

---

## The Fix (Copy-Paste Ready)

### Import to Add (both files)

```python
from langsmith.run_helpers import get_current_run_tree
```

### Code to Replace (both files)

**Replace this:**
```python
run_id = str(uuid.uuid4())
```

**With this:**
```python
# Capture the actual LangSmith trace ID for feedback correlation
run_tree = get_current_run_tree()
run_id = str(run_tree.id) if run_tree else None
logger.debug(f"LangSmith run_id captured: {run_id}")
```

---

## Verification Checklist

After deployment, verify:

- [ ] `run_id` in `/v1/chat` response matches the LangSmith trace ID
- [ ] Feedback submitted via UI (👍/👎) appears in LangSmith dashboard
- [ ] When tracing is disabled, `run_id` is `None` (not a random UUID)
- [ ] No increase in error rate or latency
- [ ] Both agents (ToolCallingAgent and RAGChainAgent) work correctly

---

## How to Verify

### 1. Send a Test Query

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Find the security policy", "top_k": 5}'
```

**Check**: Response should include a `run_id` that's a valid UUID.

### 2. Find the Trace in LangSmith

1. Go to: https://smith.langchain.com/o/default/projects/p/langchain-agent
2. Find the most recent trace with your query text
3. Copy the **Trace ID** from the LangSmith UI

**Check**: The Trace ID should match the `run_id` from the API response.

### 3. Submit Feedback

```bash
curl -X POST http://localhost:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"run_id": "YOUR_RUN_ID_HERE", "feedback_type": "like"}'
```

**Check**: Feedback should appear in the LangSmith trace under the "Feedback" section.

---

## Impact

### Before Fix (Broken)

```
User query → LangSmith trace created (ID: abc-123)
           → API returns random UUID (ID: xyz-789)
           → User clicks 👍 with run_id=xyz-789
           → LangSmith can't find trace xyz-789
           → Feedback is silently lost ❌
```

### After Fix (Working)

```
User query → LangSmith trace created (ID: abc-123)
           → API returns abc-123
           → User clicks 👍 with run_id=abc-123
           → LangSmith attaches feedback to trace abc-123
           → Feedback appears in dashboard ✅
```

---

## Documentation

- **Full Plan**: `FIX_RUN_ID_PLAN.md` — Detailed implementation steps, testing strategy, rollout plan
- **Architecture**: `LANGSMITH_FEEDBACK_ARCHITECTURE.md` — How the feedback loop works, troubleshooting guide

---

## Questions?

**Q: What if `get_current_run_tree()` returns `None`?**  
A: That's fine — it means tracing is disabled. The code handles this gracefully by setting `run_id = None`.

**Q: Will this break existing clients?**  
A: No — the API contract doesn't change. `run_id` is still a string or null, just with the correct value now.

**Q: What if LangSmith is down?**  
A: The feedback service already has graceful degradation — it returns `status="accepted"` and logs the error.

**Q: Do I need to update the UI?**  
A: No — the UI already handles `run_id` correctly. It just needs the backend to return the right value.

---

## Effort Estimate

- **Implementation**: 30 minutes (2 files, 6 lines changed)
- **Testing**: 1 hour (unit tests + integration test)
- **Deployment**: 30 minutes (deploy + verify in production)
- **Total**: ~2 hours

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `get_current_run_tree()` returns `None` | Low | Low | Defensive check: `if run_tree else None` |
| LangSmith SDK version incompatibility | Very Low | Medium | Pin `langsmith>=0.1.0` in requirements.txt |
| Feedback still doesn't appear | Low | High | Verify API key, check network, review LangSmith filters |
| Performance overhead | Very Low | Very Low | Function is a simple context lookup, negligible cost |

---

## Rollback Plan

If issues arise after deployment:

```bash
# Revert the commit
git revert <commit-hash>

# Redeploy
docker-compose up -d --build agent
```

Single-file changes are easy to rollback. No database migrations, no API contract changes.

---

**Status**: Ready for Implementation  
**Priority**: High (blocks feedback analytics)  
**Estimated Time**: 2 hours  
**Risk**: Low

# "Why" Logging — Capturing LLM Reasoning

## Problem

The current system logs *what* tool was chosen but not *why*. When the LLM selects a tool, it goes through a reasoning process (visible in the `agent_scratchpad`), but this is never captured in logs or the decision audit trail.

**Impact**: You can't answer questions like:
- "Why did the agent choose web_search instead of search_documents for this query?"
- "Is the LLM uncertain about tool selection for certain query types?"
- "Are there patterns in queries where the LLM's reasoning is flawed?"

---

## Solution

Capture the LLM's tool selection reasoning from the `AgentExecutor` and store it in the decision audit trail.

---

## Implementation

### Step 1: Extract Reasoning from AgentExecutor

**File**: `agent/infrastructure/agent/tool_calling.py` (modify)

```python
@traceable(name="ToolCallingAgent.invoke", run_type="chain")
def invoke(
    self,
    query: str,
    top_k: int = 5,
    temperature: float = 0.7,
    include_sources: bool = True,
    latest_only: bool = True,
) -> ChatResponse:
    start_time = time.time()
    try:
        logger.info(
            f"ToolCallingAgent.invoke: query={query[:50]}..., "
            f"tools={[t.name for t in self._tools]}"
        )

        executor = self._build_executor(temperature=temperature)
        result = executor.invoke({"input": query})
        response_text = result.get("output", "")
        
        # NEW: Extract reasoning from intermediate_steps
        reasoning = self._extract_reasoning(result)
        tool_chosen = self._detect_tool_chosen(result)

        execution_time_ms = (time.time() - start_time) * 1000

        # ... existing code ...

        # NEW: Log decision with reasoning
        decision_logger = DecisionLogger(db_session)
        decision_logger.log_decision(
            run_id=run_id,
            query=query,
            tool_chosen=tool_chosen,
            reasoning=reasoning,  # NEW FIELD
            sources_count=len(sources_list or []),
            latency_ms=execution_time_ms,
            model=getattr(self._llm, "model", "unknown"),
            provider=self._llm.active_provider if hasattr(self._llm, 'active_provider') else "unknown",
        )

        return ChatResponse(
            response=response_text,
            query=query,
            sources=sources_list if request.include_sources else None,
            execution_time_ms=execution_time_ms,
            model=response.model,
            run_id=run_id,
        )

    except Exception as e:
        logger.error(
            f"ToolCallingAgent.invoke failed: query={query[:50]}..., error={str(e)}",
            exc_info=True,
        )
        raise

def _extract_reasoning(self, result: dict) -> str:
    """Extract the LLM's reasoning from AgentExecutor intermediate steps.
    
    The intermediate_steps list contains tuples of (AgentAction, observation).
    AgentAction includes the tool name, tool input, and the LLM's reasoning (log).
    """
    intermediate_steps = result.get("intermediate_steps", [])
    
    if not intermediate_steps:
        return "No tool selected — direct response"
    
    # Get the first tool selection (usually the most relevant)
    first_step = intermediate_steps[0]
    agent_action = first_step[0]
    
    # The 'log' field contains the LLM's reasoning
    reasoning = getattr(agent_action, 'log', '')
    
    # Truncate if too long (keep first 1000 chars)
    return reasoning[:1000] if reasoning else "No reasoning captured"

def _detect_tool_chosen(self, result: dict) -> str | None:
    """Detect which tool was chosen from intermediate steps."""
    intermediate_steps = result.get("intermediate_steps", [])
    
    if not intermediate_steps:
        return None  # No tool used
    
    first_step = intermediate_steps[0]
    agent_action = first_step[0]
    
    return getattr(agent_action, 'tool', None)
```

### Step 2: Update Decision Audit Model

**File**: `agent/models/audit.py` (modify)

```python
from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func
from models.base import Base


class DecisionAudit(Base):
    """Audit trail for AI decisions."""
    
    __tablename__ = "decision_audit"
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(36), index=True)
    query = Column(Text, nullable=False)
    tool_chosen = Column(String(100))
    reasoning = Column(Text)  # NEW FIELD
    sources_count = Column(Integer, default=0)
    latency_ms = Column(Float)
    model = Column(String(100))
    provider = Column(String(100))
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "run_id": self.run_id,
            "query": self.query,
            "tool_chosen": self.tool_chosen,
            "reasoning": self.reasoning,
            "sources_count": self.sources_count,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "provider": self.provider,
            "created_at": self.created_at.isoformat(),
        }
```

### Step 3: Update Database Migration

**File**: `agent/migrations/versions/xxx_add_reasoning_to_audit.py` (new)

```python
"""Add reasoning column to decision_audit table."""

from alembic import op
import sqlalchemy as sa


def upgrade():
    op.add_column('decision_audit', sa.Column('reasoning', sa.Text))


def downgrade():
    op.drop_column('decision_audit', 'reasoning')
```

### Step 4: Update Decision Logger

**File**: `agent/infrastructure/audit/decision_logger.py` (modify)

```python
class DecisionLogger:
    """Logs AI decisions to database for local querying."""
    
    def __init__(self, db_session: Session):
        self._db = db_session
    
    def log_decision(
        self,
        run_id: str,
        query: str,
        tool_chosen: str | None,
        reasoning: str,  # NEW PARAMETER
        sources_count: int,
        latency_ms: float,
        model: str,
        provider: str,
    ) -> None:
        """Log an AI decision with reasoning."""
        audit = DecisionAudit(
            run_id=run_id,
            query=query[:2000],
            tool_chosen=tool_chosen,
            reasoning=reasoning[:2000],  # Truncate if needed
            sources_count=sources_count,
            latency_ms=latency_ms,
            model=model,
            provider=provider,
        )
        self._db.add(audit)
        self._db.commit()
        
        logger.debug(
            f"Decision logged: run_id={run_id}, tool={tool_chosen}, "
            f"reasoning={reasoning[:100]}..., sources={sources_count}"
        )
```

### Step 5: Update Audit Endpoint

**File**: `agent/api/routes.py` (modify)

```python
@router.get("/v1/audit/decisions", status_code=200)
async def get_decisions(
    limit: int = 100,
    offset: int = 0,
    tool: str | None = None,
    provider: str | None = None,
    include_reasoning: bool = False,  # NEW PARAMETER
) -> list:
    """Query decision audit trail.
    
    Args:
        limit: Max results (default 100)
        offset: Pagination offset
        tool: Filter by tool chosen
        provider: Filter by LLM provider
        include_reasoning: Include LLM reasoning text (can be large)
    
    Returns:
        List of decision audit records
    """
    from services.container import db_session
    from models.audit import DecisionAudit
    from sqlalchemy import select
    
    query = select(DecisionAudit)
    
    if tool:
        query = query.where(DecisionAudit.tool_chosen == tool)
    if provider:
        query = query.where(DecisionAudit.provider == provider)
    
    query = query.order_by(DecisionAudit.created_at.desc()).limit(limit).offset(offset)
    
    results = db_session.execute(query).scalars().all()
    
    decisions = []
    for r in results:
        d = r.to_dict()
        if not include_reasoning:
            d.pop('reasoning', None)  # Remove reasoning by default
        decisions.append(d)
    
    return decisions
```

---

## Usage Examples

### Query decisions with reasoning

```bash
# Get last 10 decisions with reasoning
curl "http://localhost:8000/v1/audit/decisions?limit=10&include_reasoning=true" | jq

# Get all web_search decisions with reasoning
curl "http://localhost:8000/v1/audit/decisions?tool=web_search&include_reasoning=true" | jq
```

### SQL queries for analysis

```sql
-- Find queries where LLM was uncertain (look for hedging language in reasoning)
SELECT query, tool_chosen, reasoning
FROM decision_audit
WHERE reasoning ILIKE '%maybe%'
   OR reasoning ILIKE '%not sure%'
   OR reasoning ILIKE '%could be%'
ORDER BY created_at DESC
LIMIT 20;

-- Find queries where LLM chose wrong tool (based on feedback)
SELECT da.query, da.tool_chosen, da.reasoning, f.score
FROM decision_audit da
JOIN feedback f ON da.run_id = f.run_id
WHERE f.score = 0.0  -- Dislikes
ORDER BY da.created_at DESC;

-- Analyze reasoning patterns by tool
SELECT 
    tool_chosen,
    COUNT(*) as count,
    AVG(LENGTH(reasoning)) as avg_reasoning_length
FROM decision_audit
WHERE tool_chosen IS NOT NULL
GROUP BY tool_chosen;
```

---

## What the Reasoning Looks Like

### Example 1: Clear tool selection
```
Query: "Find the security policy in the requirement documents"
Tool: search_documents
Reasoning: "The user is asking to find information in specific documents (requirement documents). 
This matches the search_documents tool which searches the document database. I should use 
search_documents with query='security policy' to find relevant documents."
```

### Example 2: Uncertain selection
```
Query: "What's the latest news about AI?"
Tool: web_search
Reasoning: "This could be answered with either search_documents (if we have news documents) or 
web_search (for current news). Since the user is asking about 'latest news', web_search is more 
appropriate as it will return current information from the web."
```

### Example 3: No tool needed
```
Query: "Hello, how are you?"
Tool: None
Reasoning: "This is a simple greeting. No tools are needed. I can respond directly from my 
training knowledge."
```

---

## Benefits

1. **Debug tool selection issues**: See exactly why the LLM chose a tool
2. **Improve prompts**: Identify patterns where the LLM's reasoning is flawed
3. **Quality control**: Find queries where the LLM was uncertain or made poor choices
4. **Training data**: Use reasoning logs to fine-tune tool selection prompts
5. **Audit trail**: Complete record of AI decision-making for compliance

---

## Performance Impact

- **Storage**: ~500 bytes per decision (reasoning text)
- **Latency**: <1ms (extracting from existing data structure)
- **Cost**: Negligible (text field in existing table)

---

## Future Enhancements

1. **Reasoning quality score**: Use LLM to rate the quality of its own reasoning
2. **Reasoning search**: Full-text search on reasoning field
3. **Reasoning visualization**: UI showing decision tree for each query
4. **Reasoning export**: Export reasoning logs for prompt engineering

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation

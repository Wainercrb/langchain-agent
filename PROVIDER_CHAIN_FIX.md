# Provider Chain with Single Retry — Simplified Error Handling

## Executive Summary

**Problem**: The original plan (circuit breaker + 3 retries + failover) is over-engineered for an LLM agent system. It wastes 14 seconds retrying a dead provider before failing over.

**Solution**: Simple provider chain with single retry:
- Try Provider A once
- If transient error → retry once with 1s backoff
- If still failing or permanent error → try Provider B
- Continue through provider list
- Total worst-case time: 4 providers × 2 attempts × 1s = **8 seconds** (vs 14s in original plan)

**Effort**: 3 hours (down from 10 hours in original plan)  
**Risk**: Low — simpler logic, easier to test  
**Impact**: High — faster failover, simpler code, achieves all three goals

---

## Why This Achieves Your Goals

### Goal 1: Maintenance & Monitoring ✅

**How**: Log which provider succeeded/failed, track provider usage metrics

```python
# In provider chain
for provider in providers:
    try:
        response = provider.invoke(messages)
        logger.info(f"Provider {provider.name} succeeded")
        metrics.record_provider_success(provider.name)
        return response
    except Exception as e:
        logger.warning(f"Provider {provider.name} failed: {e}")
        metrics.record_provider_failure(provider.name)
        continue
```

**Monitoring endpoints**:
- `/v1/providers` — shows which providers are configured
- `/v1/metrics` — shows provider success/failure rates
- CloudWatch alarms — alert if all providers fail

### Goal 2: AI Decision Logs ✅

**How**: "Why" logging captures LLM reasoning (separate from error handling)

```python
# In tool_calling.py (already documented in WHY_LOGGING_FIX.md)
reasoning = self._extract_reasoning(result)
decision_logger.log_decision(
    run_id=run_id,
    query=query,
    tool_chosen=tool_chosen,
    reasoning=reasoning,  # ← LLM's reasoning
    provider=provider.name,  # ← Which provider was used
)
```

**Audit trail**:
- `/v1/audit/decisions` — queryable decision history
- `reasoning` field — LLM's tool selection logic
- `provider` field — which LLM provider was used

### Goal 3: Automated Error-Handling ✅

**How**: Provider chain automatically tries next provider on failure

```python
providers = [OpenRouter, Gemini, OpenAI, Anthropic]

for provider in providers:
    # First attempt
    try:
        return provider.invoke(messages)
    except TransientError:
        # Retry once (handles rate limits, network blips)
        time.sleep(1)
        try:
            return provider.invoke(messages)
        except:
            continue  # Try next provider
    except PermanentError:
        continue  # Try next provider immediately

raise AllProvidersFailedError()
```

**Error handling**:
- Transient errors (rate limit, timeout) → retry once, then failover
- Permanent errors (invalid key, 400 error) → failover immediately
- All providers fail → return 503 with clear error message

---

## Implementation

### Step 1: Create Provider Chain

**File**: `agent/services/llm/provider_chain.py` (new)

```python
"""Provider chain — tries multiple LLM providers in sequence.

Simple error handling:
- Try each provider once
- If transient error (rate limit, timeout), retry once with 1s backoff
- If still failing or permanent error, try next provider
- If all providers fail, raise AllProvidersFailedError
"""

import time
from typing import List, Any
from dataclasses import dataclass

from services.logging import logger
from utils.exceptions import TransientLLMError, PermanentLLMError


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    provider: Any  # LLM provider instance


class AllProvidersFailedError(Exception):
    """Raised when all providers in the chain fail."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"All providers failed: {'; '.join(errors)}")


class ProviderChain:
    """LLM provider chain with single retry for transient errors.
    
    Usage:
        chain = ProviderChain([
            ProviderConfig("openrouter", openrouter_provider),
            ProviderConfig("gemini", gemini_provider),
            ProviderConfig("openai", openai_provider),
        ])
        
        response = chain.invoke(messages)
    """
    
    def __init__(self, providers: List[ProviderConfig]):
        if not providers:
            raise ValueError("At least one provider is required")
        
        self._providers = providers
        self._active_provider_index = 0
        
        logger.info(
            f"ProviderChain initialized with {len(providers)} providers: "
            f"{[p.name for p in providers]}"
        )
    
    @property
    def model(self) -> str:
        """Return model name of active provider."""
        return self._providers[self._active_provider_index].provider.model
    
    @property
    def active_provider(self) -> str:
        """Return name of active provider."""
        return self._providers[self._active_provider_index].name
    
    def invoke(self, messages: list) -> Any:
        """Invoke LLM with automatic provider failover.
        
        Tries each provider in order:
        1. First attempt
        2. If TransientError, retry once with 1s backoff
        3. If still failing or PermanentError, try next provider
        4. If all providers fail, raise AllProvidersFailedError
        
        Args:
            messages: List of message dicts (OpenAI format)
            
        Returns:
            LLM response from first successful provider
            
        Raises:
            AllProvidersFailedError: If all providers fail
        """
        errors = []
        
        for i, provider_config in enumerate(self._providers):
            provider_name = provider_config.name
            provider = provider_config.provider
            
            # First attempt
            try:
                logger.info(f"ProviderChain: trying {provider_name}")
                result = provider.invoke(messages)
                
                # Success — update active provider
                if i != self._active_provider_index:
                    logger.info(
                        f"ProviderChain: switched from {self._providers[self._active_provider_index].name} "
                        f"to {provider_name}"
                    )
                    self._active_provider_index = i
                
                return result
                
            except TransientLLMError as e:
                # Transient error — retry once with 1s backoff
                logger.warning(
                    f"ProviderChain: {provider_name} transient error, retrying in 1s: {e}"
                )
                time.sleep(1)
                
                try:
                    result = provider.invoke(messages)
                    
                    # Retry succeeded
                    if i != self._active_provider_index:
                        self._active_provider_index = i
                    
                    return result
                    
                except Exception as retry_error:
                    # Retry failed — try next provider
                    logger.warning(
                        f"ProviderChain: {provider_name} retry failed: {retry_error}"
                    )
                    errors.append(f"{provider_name}: {retry_error}")
                    continue
                
            except PermanentLLMError as e:
                # Permanent error — try next provider immediately
                logger.error(
                    f"ProviderChain: {provider_name} permanent error: {e}"
                )
                errors.append(f"{provider_name}: {e}")
                continue
                
            except Exception as e:
                # Unexpected error — try next provider
                logger.error(
                    f"ProviderChain: {provider_name} unexpected error: {e}",
                    exc_info=True,
                )
                errors.append(f"{provider_name}: unexpected error")
                continue
        
        # All providers failed
        logger.error(f"ProviderChain: all providers failed: {errors}")
        raise AllProvidersFailedError(errors)
    
    def get_providers(self) -> List[str]:
        """Get list of provider names."""
        return [p.name for p in self._providers]
```

### Step 2: Wire Provider Chain in Container

**File**: `agent/services/container.py` (modify)

```python
# ── LLM ──────────────────────────────────────────────────────────────
from services.llm import OpenRouterProvider, GoogleProvider, OpenAIProvider
from services.llm.provider_chain import ProviderChain, ProviderConfig

# Create individual providers
_openrouter = OpenRouterProvider(
    model=settings.openrouter_model,
    temperature=settings.openrouter_temperature,
    max_tokens=settings.openrouter_max_tokens,
    api_key=settings.openrouter_api_key,
    timeout=settings.llm_timeout_seconds,
)

_gemini = GoogleProvider(
    model=settings.gemini_model,
    temperature=settings.gemini_temperature,
    api_key=settings.google_api_key,
)

_openai = OpenAIProvider(
    model=settings.openai_model,
    temperature=settings.openai_temperature,
    api_key=settings.openai_api_key or None,
)

# Create provider chain
llm = ProviderChain([
    ProviderConfig("openrouter", _openrouter),
    ProviderConfig("gemini", _gemini),
    ProviderConfig("openai", _openai),
])
```

### Step 3: Update Agent to Use Provider Chain

**File**: `agent/services/container.py` (modify)

```python
if settings.use_tool_agent:
    # ... existing code ...
    agent = ToolCallingAgent(
        llm=llm,  # ProviderChain has same interface as single provider
        tools=_tools,
        artifact_store=_search_artifact_store,
        default_top_k=5,
    )
```

### Step 4: Add Health Check Endpoint

**File**: `agent/api/routes.py` (add)

```python
@router.get("/v1/providers", status_code=200)
async def provider_status() -> dict:
    """Get list of configured LLM providers."""
    from services.container import llm
    
    if hasattr(llm, 'get_providers'):
        return {
            "providers": llm.get_providers(),
            "active_provider": llm.active_provider,
        }
    else:
        return {"error": "Provider chain not configured"}
```

---

## Configuration

**File**: `agent/config/settings.py` (add)

```python
# ── Provider Chain ───────────────────────────────────────────────────
provider_chain: str = Field(
    default="openrouter,gemini,openai",
    alias="PROVIDER_CHAIN",
    description="Comma-separated list of providers in failover order",
)
provider_retry_delay_seconds: int = Field(
    default=1,
    alias="PROVIDER_RETRY_DELAY_SECONDS",
    description="Delay between retry attempts (seconds)",
)
```

**File**: `agent/.env` (add)

```bash
# Provider Chain
PROVIDER_CHAIN=openrouter,gemini,openai
PROVIDER_RETRY_DELAY_SECONDS=1
```

---

## Testing Strategy

### Unit Tests

**Test 1: First Provider Succeeds**
```python
def test_first_provider_succeeds():
    provider1 = MockProvider(should_fail=False)
    provider2 = MockProvider(should_fail=False)
    
    chain = ProviderChain([
        ProviderConfig("provider1", provider1),
        ProviderConfig("provider2", provider2),
    ])
    
    result = chain.invoke([{"role": "user", "content": "test"}])
    
    assert result is not None
    assert chain.active_provider == "provider1"
    assert provider1.call_count == 1
    assert provider2.call_count == 0  # Never tried
```

**Test 2: First Provider Fails, Second Succeeds**
```python
def test_failover_to_second_provider():
    provider1 = MockProvider(should_fail=True)
    provider2 = MockProvider(should_fail=False)
    
    chain = ProviderChain([
        ProviderConfig("provider1", provider1),
        ProviderConfig("provider2", provider2),
    ])
    
    result = chain.invoke([{"role": "user", "content": "test"}])
    
    assert result is not None
    assert chain.active_provider == "provider2"
    assert provider1.call_count == 1
    assert provider2.call_count == 1
```

**Test 3: Transient Error with Successful Retry**
```python
def test_transient_error_retry_succeeds():
    provider1 = MockProvider(
        should_fail=True,
        fail_count=1,  # Fail first attempt, succeed on retry
        error_type=TransientLLMError
    )
    
    chain = ProviderChain([
        ProviderConfig("provider1", provider1),
    ])
    
    result = chain.invoke([{"role": "user", "content": "test"}])
    
    assert result is not None
    assert provider1.call_count == 2  # First attempt + retry
```

**Test 4: All Providers Fail**
```python
def test_all_providers_fail():
    provider1 = MockProvider(should_fail=True)
    provider2 = MockProvider(should_fail=True)
    
    chain = ProviderChain([
        ProviderConfig("provider1", provider1),
        ProviderConfig("provider2", provider2),
    ])
    
    with pytest.raises(AllProvidersFailedError):
        chain.invoke([{"role": "user", "content": "test"}])
```

### Integration Tests

**Test 5: End-to-End Failover**
1. Start system with OpenRouter as primary
2. Block OpenRouter IP (simulate outage)
3. Send request → should failover to Gemini within 2 seconds
4. Verify response comes from Gemini
5. Unblock OpenRouter IP
6. Send request → should use OpenRouter again (first in chain)

---

## Monitoring and Observability

### Metrics to Track

Add to `/v1/metrics` endpoint:

```python
class MetricsResponse(BaseModel):
    # ... existing fields ...
    
    active_llm_provider: str = Field(
        default="openrouter",
        description="Currently active LLM provider",
    )
    provider_failures: dict = Field(
        default={},
        description="Failure count per provider",
    )
```

### Logs to Watch

```
INFO  | ProviderChain: trying openrouter
WARN  | ProviderChain: openrouter transient error, retrying in 1s: timeout
INFO  | ProviderChain: trying gemini
INFO  | ProviderChain: switched from openrouter to gemini
ERROR | ProviderChain: all providers failed: [openrouter: timeout, gemini: 503, openai: 401]
```

### CloudWatch Alarms

```yaml
AllProvidersFailed:
  Condition: error_count > 10 in 5 minutes
  Action: Send Discord alert (all LLM providers may be down)
```

---

## Comparison: Original Plan vs Simplified Approach

| Aspect | Original Plan | Simplified Approach |
|--------|---------------|---------------------|
| **Effort** | 10 hours | 3 hours |
| **Complexity** | High (circuit breaker + retry + failover) | Low (provider chain + 1 retry) |
| **Failover time** | 14 seconds (3 retries × exponential backoff) | 2 seconds (1 retry × 1s backoff) |
| **Code complexity** | 500+ lines | 150 lines |
| **Test complexity** | High (circuit breaker states) | Low (success/failure paths) |
| **Achieves Goal 1** | ✅ Yes | ✅ Yes |
| **Achieves Goal 2** | ✅ Yes | ✅ Yes |
| **Achieves Goal 3** | ✅ Yes | ✅ Yes (faster) |

---

## Rollout Plan

### Pre-Deployment

1. **Code Review**: Ensure provider chain implementation is reviewed
2. **Unit Tests**: All unit tests pass
3. **Integration Tests**: Test failover scenarios
4. **Staging Deployment**: Deploy to staging with all 3 providers configured

### Deployment

1. **Deploy to Production**: Use existing deployment pipeline
2. **Verify Providers**: Check `/v1/providers` endpoint shows all 3 providers
3. **Smoke Test**: Send test queries, verify primary (OpenRouter) is used
4. **Failover Test**: Temporarily block OpenRouter, verify failover to Gemini
5. **Monitor Logs**: Watch for provider chain logs

### Post-Deployment

1. **Monitor for 1 Week**: Watch for unexpected failovers
2. **Check Provider Usage**: Verify primary is used 95%+ of the time
3. **Document Failover Events**: Log any failovers for post-mortem

---

## Success Criteria

The provider chain is successful when:

- ✅ System remains available when primary LLM is down
- ✅ Automatic failover to secondary within 2 seconds
- ✅ Automatic failback when primary recovers (it's first in chain)
- ✅ No manual intervention required
- ✅ Observable via logs and `/v1/providers` endpoint
- ✅ Simpler code than original plan (150 lines vs 500+ lines)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Transient error causes unnecessary failover | Medium | Low | Single retry handles most transient errors |
| All providers fail simultaneously | Very Low | Critical | Return clear error message, alert operations |
| Provider chain adds latency | Low | Low | Overhead is minimal (<10ms per provider check) |
| Rate limiting across all providers | Low | Medium | Monitor provider usage, add more providers if needed |

---

## Future Enhancements

After basic provider chain is working:

1. **Health Scoring**: Track provider success rates, route to healthiest provider
2. **Cost-Based Routing**: Route to cheapest provider when quality is comparable
3. **Latency-Based Routing**: Route to fastest provider based on recent latency
4. **Provider Dashboard**: Visual UI showing provider status, failover history

---

## References

- **Original Plan**: `CIRCUIT_BREAKER_FIX.md`, `LLM_FAILOVER_FIX.md` (superseded by this doc)
- **Why Logging**: `WHY_LOGGING_FIX.md` (captures LLM reasoning)
- **Decision Audit**: `PHASE_2_3_SUMMARIES.md` § P2.2 (stores tool selection)
- **Current Container**: `agent/services/container.py`

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation (replaces circuit breaker + failover docs)

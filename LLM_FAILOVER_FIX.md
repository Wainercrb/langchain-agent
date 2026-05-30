# LLM Provider Failover — Implementation Guide

## Executive Summary

**Problem**: `container.py` hardcodes `OpenRouterProvider`. If OpenRouter is down, the entire system is down — despite having Gemini and OpenAI already configured in settings.

**Solution**: Implement a failover chain that automatically switches to a backup LLM provider when the primary fails:
- Primary: OpenRouter (current)
- Secondary: Gemini (already configured)
- Tertiary: OpenAI (already configured)
- Automatic failover when primary circuit breaker is open
- Automatic failback when primary recovers

**Effort**: 6 hours  
**Risk**: Medium — need to test failover/failback scenarios  
**Impact**: High — ensures availability even when primary LLM is down

---

## 1. Failover Architecture

### Failover Chain

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│      FailoverLLMProvider                │
│                                         │
│  1. Try OpenRouter (primary)            │
│     └─ Circuit open? → Failover         │
│                                         │
│  2. Try Gemini (secondary)              │
│     └─ Circuit open? → Failover         │
│                                         │
│  3. Try OpenAI (tertiary)               │
│     └─ Circuit open? → All providers down│
│                                         │
└─────────────────────────────────────────┘
    │
    ▼
Response (from whichever provider succeeded)
```

### Failover Logic

1. **Primary provider** (OpenRouter):
   - Attempt request
   - If circuit breaker is open → failover to secondary
   - If transient error → retry (existing logic), then failover
   - If permanent error → failover immediately

2. **Secondary provider** (Gemini):
   - Attempt request
   - If circuit breaker is open → failover to tertiary
   - If error → failover to tertiary

3. **Tertiary provider** (OpenAI):
   - Attempt request
   - If error → return 503 (all providers down)

4. **Failback**:
   - Primary provider's circuit breaker half-opens after 60s
   - Test request succeeds → circuit closes
   - Next request uses primary again (automatic failback)

---

## 2. Implementation Plan

### Step 1: Create Failover Provider

**File**: `agent/infrastructure/llm/failover_provider.py` (new)

```python
"""Failover LLM provider — automatically switches to backup when primary fails.

Combines multiple LLM providers with circuit breakers for high availability.
"""

from typing import List, Any, Optional
from dataclasses import dataclass

from services.logging import logger
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from utils.exceptions import TransientLLMError, PermanentLLMError


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider in the failover chain."""
    name: str
    provider: Any  # LLM provider instance (OpenRouterProvider, GoogleProvider, etc.)
    circuit_breaker: CircuitBreaker


class FailoverLLMProvider:
    """LLM provider with automatic failover.
    
    Usage:
        failover = FailoverLLMProvider(
            providers=[
                ProviderConfig("openrouter", openrouter_provider, CircuitBreaker("openrouter")),
                ProviderConfig("gemini", gemini_provider, CircuitBreaker("gemini")),
                ProviderConfig("openai", openai_provider, CircuitBreaker("openai")),
            ]
        )
        
        response = failover.invoke(messages)
    """
    
    def __init__(self, providers: List[ProviderConfig]):
        """Initialize failover provider.
        
        Args:
            providers: Ordered list of providers (primary, secondary, tertiary)
        """
        if not providers:
            raise ValueError("At least one provider is required")
        
        self._providers = providers
        self._active_provider_index = 0
        
        logger.info(
            f"FailoverLLMProvider initialized with {len(providers)} providers: "
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
        """Invoke LLM with automatic failover.
        
        Tries each provider in order. If a provider's circuit breaker is open
        or it raises an exception, fails over to the next provider.
        
        Args:
            messages: List of message dicts (OpenAI format)
            
        Returns:
            LLM response from first successful provider
            
        Raises:
            PermanentLLMError: If all providers fail
        """
        errors = []
        
        for i, provider_config in enumerate(self._providers):
            provider_name = provider_config.name
            circuit_breaker = provider_config.circuit_breaker
            provider = provider_config.provider
            
            # Check if circuit is open
            if circuit_breaker.state.value == "open":
                logger.warning(
                    f"Failover: skipping {provider_name} (circuit breaker open)"
                )
                errors.append(f"{provider_name}: circuit breaker open")
                continue
            
            # Try this provider
            try:
                logger.info(f"Failover: trying {provider_name}")
                result = circuit_breaker.call(provider.invoke, messages)
                
                # Success — update active provider
                if i != self._active_provider_index:
                    logger.info(
                        f"Failover: switched from {self._providers[self._active_provider_index].name} "
                        f"to {provider_name}"
                    )
                    self._active_provider_index = i
                
                return result
                
            except CircuitBreakerOpenError:
                # Circuit opened during call (shouldn't happen, but handle it)
                logger.warning(f"Failover: {provider_name} circuit opened during call")
                errors.append(f"{provider_name}: circuit breaker opened")
                continue
                
            except TransientLLMError as e:
                # Transient error — circuit breaker already retried, failover
                logger.warning(
                    f"Failover: {provider_name} failed with transient error: {e}"
                )
                errors.append(f"{provider_name}: {e}")
                continue
                
            except PermanentLLMError as e:
                # Permanent error — failover immediately
                logger.error(
                    f"Failover: {provider_name} failed with permanent error: {e}"
                )
                errors.append(f"{provider_name}: {e}")
                continue
                
            except Exception as e:
                # Unexpected error — failover
                logger.error(
                    f"Failover: {provider_name} failed with unexpected error: {e}",
                    exc_info=True,
                )
                errors.append(f"{provider_name}: unexpected error")
                continue
        
        # All providers failed
        error_details = "; ".join(errors)
        logger.error(f"Failover: all providers failed: {error_details}")
        raise PermanentLLMError(
            message=f"All LLM providers failed: {error_details}",
            provider="failover",
        )
    
    def get_provider_status(self) -> dict:
        """Get status of all providers (for health checks).
        
        Returns:
            Dict with provider names as keys, circuit breaker states as values
        """
        return {
            config.name: {
                "state": config.circuit_breaker.state.value,
                "failure_count": config.circuit_breaker._failure_count,
                "active": i == self._active_provider_index,
            }
            for i, config in enumerate(self._providers)
        }
```

### Step 2: Wire Failover in Container

**File**: `agent/infrastructure/container.py` (modify)

```python
# ── LLM ──────────────────────────────────────────────────────────────
from services.llm import OpenRouterProvider, GoogleProvider, OpenAIProvider
from services.llm.failover_provider import FailoverLLMProvider, ProviderConfig
from utils.circuit_breaker import CircuitBreaker

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

# Create circuit breakers
_openrouter_cb = CircuitBreaker(
    name="openrouter",
    failure_threshold=settings.circuit_breaker_failure_threshold,
    recovery_timeout=settings.circuit_breaker_recovery_timeout,
)

_gemini_cb = CircuitBreaker(
    name="gemini",
    failure_threshold=settings.circuit_breaker_failure_threshold,
    recovery_timeout=settings.circuit_breaker_recovery_timeout,
)

_openai_cb = CircuitBreaker(
    name="openai",
    failure_threshold=settings.circuit_breaker_failure_threshold,
    recovery_timeout=settings.circuit_breaker_recovery_timeout,
)

# Create failover provider
llm = FailoverLLMProvider(
    providers=[
        ProviderConfig("openrouter", _openrouter, _openrouter_cb),
        ProviderConfig("gemini", _gemini, _gemini_cb),
        ProviderConfig("openai", _openai, _openai_cb),
    ]
)
```

### Step 3: Update Agent to Use Failover Provider

**File**: `agent/infrastructure/agent/tool_calling.py` (modify)

```python
# The agent already calls self._llm.invoke(messages)
# FailoverLLMProvider has the same interface, so no changes needed
# Just verify that llm.invoke() is called (not llm.chat_model.invoke())

# In container.py, change:
# agent = ToolCallingAgent(llm=llm.chat_model, ...)
# To:
# agent = ToolCallingAgent(llm=llm, ...)

# This requires updating the agent to call llm.invoke() directly
```

**File**: `agent/infrastructure/container.py` (modify)

```python
if settings.use_tool_agent:
    # ... existing code ...
    agent = ToolCallingAgent(
        llm=llm,  # Changed from llm.chat_model
        tools=_tools,
        artifact_store=_search_artifact_store,
        default_top_k=5,
    )
```

### Step 4: Add Health Check for Provider Status

**File**: `agent/api/routes.py` (add endpoint)

```python
@router.get("/v1/providers", status_code=200)
async def provider_status() -> dict:
    """Get status of all LLM providers (circuit breaker states).
    
    Returns:
        Dict with provider names, circuit breaker states, and active provider
    """
    from services.container import llm
    
    if hasattr(llm, 'get_provider_status'):
        return llm.get_provider_status()
    else:
        return {"error": "Provider status not available"}
```

---

## 3. Configuration

**File**: `agent/config/settings.py` (add)

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # ── Failover ─────────────────────────────────────────────────────
    enable_llm_failover: bool = Field(
        default=True,
        alias="ENABLE_LLM_FAILOVER",
        description="Enable automatic LLM provider failover",
    )
    failover_providers: str = Field(
        default="openrouter,gemini,openai",
        alias="FAILOVER_PROVIDERS",
        description="Comma-separated list of providers in failover order",
    )
```

**File**: `agent/.env` (add)

```bash
# LLM Failover
ENABLE_LLM_FAILOVER=true
FAILOVER_PROVIDERS=openrouter,gemini,openai

# Circuit Breaker (shared with Phase 1.1)
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60
```

---

## 4. Testing Strategy

### Unit Tests

**Test 1: Failover to Secondary When Primary Fails**
```python
def test_failover_to_secondary():
    # Mock primary to fail
    primary = MockProvider(should_fail=True)
    secondary = MockProvider(should_fail=False)
    
    failover = FailoverLLMProvider([
        ProviderConfig("primary", primary, CircuitBreaker("primary")),
        ProviderConfig("secondary", secondary, CircuitBreaker("secondary")),
    ])
    
    result = failover.invoke([{"role": "user", "content": "test"}])
    
    assert result is not None
    assert failover.active_provider == "secondary"
```

**Test 2: Failback to Primary When Recovered**
```python
def test_failback_to_primary():
    primary = MockProvider(should_fail=True)
    secondary = MockProvider(should_fail=False)
    
    failover = FailoverLLMProvider([
        ProviderConfig("primary", primary, CircuitBreaker("primary", recovery_timeout=1)),
        ProviderConfig("secondary", secondary, CircuitBreaker("secondary")),
    ])
    
    # Failover to secondary
    failover.invoke([{"role": "user", "content": "test"}])
    assert failover.active_provider == "secondary"
    
    # Fix primary and wait for recovery
    primary.should_fail = False
    time.sleep(1.1)  # Wait for circuit breaker to half-open
    
    # Should failback to primary
    failover.invoke([{"role": "user", "content": "test"}])
    assert failover.active_provider == "primary"
```

**Test 3: All Providers Fail**
```python
def test_all_providers_fail():
    primary = MockProvider(should_fail=True)
    secondary = MockProvider(should_fail=True)
    
    failover = FailoverLLMProvider([
        ProviderConfig("primary", primary, CircuitBreaker("primary")),
        ProviderConfig("secondary", secondary, CircuitBreaker("secondary")),
    ])
    
    with pytest.raises(PermanentLLMError):
        failover.invoke([{"role": "user", "content": "test"}])
```

### Integration Tests

**Test 4: End-to-End Failover**
1. Start system with OpenRouter as primary
2. Block OpenRouter IP (simulate outage)
3. Send 5 requests → circuit opens, failover to Gemini
4. Verify responses come from Gemini
5. Unblock OpenRouter IP
6. Wait 60 seconds → circuit half-opens
7. Send request → failback to OpenRouter
8. Verify responses come from OpenRouter again

---

## 5. Monitoring and Observability

### Metrics to Track

Add to `/v1/metrics` endpoint:

```python
class MetricsResponse(BaseModel):
    # ... existing fields ...
    
    active_llm_provider: str = Field(
        default="openrouter",
        description="Currently active LLM provider",
    )
    provider_failover_count: int = Field(
        default=0,
        description="Number of failovers since startup",
    )
```

### Logs to Watch

```
INFO  | Failover: trying openrouter
WARN  | Failover: openrouter failed with transient error: timeout
INFO  | Failover: trying gemini
INFO  | Failover: switched from openrouter to gemini
INFO  | Failover: trying openrouter (circuit half-open)
INFO  | Failover: switched from gemini to openrouter (recovered)
```

### CloudWatch Alarms (Phase 2)

```yaml
LLMFailoverActive:
  Condition: active_llm_provider != "openrouter"
  Duration: >10 minutes
  Action: Send Discord alert (primary provider may be down)
```

---

## 6. Rollout Plan

### Pre-Deployment

1. **Code Review**: Ensure failover logic is reviewed
2. **Unit Tests**: All unit tests pass
3. **Integration Tests**: Test failover/failback scenarios
4. **Staging Deployment**: Deploy to staging with all 3 providers configured
5. **Chaos Testing**: Block each provider IP, verify failover works

### Deployment

1. **Deploy to Production**: Use existing deployment pipeline
2. **Verify All Providers**: Check `/v1/providers` endpoint shows all 3 providers
3. **Smoke Test**: Send test queries, verify primary (OpenRouter) is used
4. **Failover Test**: Temporarily block OpenRouter, verify failover to Gemini
5. **Failback Test**: Unblock OpenRouter, verify automatic failback

### Post-Deployment

1. **Monitor for 1 Week**: Watch for unexpected failovers
2. **Check Provider Usage**: Verify primary is used 95%+ of the time
3. **Document Failover Events**: Log any failovers for post-mortem
4. **Tune Circuit Breakers**: Adjust thresholds based on production data

---

## 7. Success Criteria

The failover system is successful when:

- ✅ System remains available when primary LLM is down
- ✅ Automatic failover to secondary within seconds
- ✅ Automatic failback when primary recovers
- ✅ No manual intervention required
- ✅ Observable via logs and `/v1/providers` endpoint
- ✅ No degradation in response quality (all providers return comparable responses)

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Secondary provider returns lower quality responses | Medium | Medium | Test response quality for all providers, adjust prompts if needed |
| Failover causes inconsistent responses | Low | High | Log provider switches, monitor for quality issues |
| All providers fail simultaneously | Very Low | Critical | Implement graceful degradation (cached responses) in Phase 4 |
| Failover adds latency | Low | Low | Overhead is minimal (<10ms per failover check) |
| Circuit breaker opens too aggressively | Medium | Medium | Tune thresholds based on production data |

---

## 9. Future Improvements

After basic failover is working:

1. **Weighted Routing**: Send 80% to primary, 20% to secondary (load balancing)
2. **Cost-Based Routing**: Route to cheapest provider when quality is comparable
3. **Latency-Based Routing**: Route to fastest provider based on recent latency
4. **Provider Health Dashboard**: Visual UI showing provider status, failover history, response quality
5. **Automatic Provider Addition**: Dynamically add new providers via configuration (no code change)

---

## 10. Cost Considerations

| Provider | Cost per 1K tokens | Monthly Cost (1M tokens) |
|----------|-------------------|--------------------------|
| OpenRouter (GPT-4o) | $0.005 | $5 |
| Gemini (Flash) | $0.0005 | $0.50 |
| OpenAI (GPT-4o-mini) | $0.00015 | $0.15 |

**Failover Cost Impact**: Minimal — secondary/tertiary providers are only used when primary is down (hopefully <1% of the time).

---

## 11. References

- **Circuit Breaker Pattern**: `CIRCUIT_BREAKER_FIX.md` (Phase 1.1)
- **Current Container**: `agent/infrastructure/container.py`
- **LLM Providers**: `agent/infrastructure/llm/` directory
- **Retry Logic**: `agent/utils/retry.py` (works alongside failover)

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation

# Circuit Breaker Pattern — Implementation Guide

## Executive Summary

**Problem**: When OpenRouter (or any LLM provider) is down, every request burns 3 retries × exponential backoff before failing. This causes cascading failures, increased latency, and poor user experience.

**Solution**: Implement the Circuit Breaker pattern to:
- Stop hammering a failing provider after N consecutive failures
- Return fast failures (or cached responses) when circuit is open
- Automatically test if provider has recovered (half-open state)
- Close circuit when provider is healthy again

**Effort**: 4 hours  
**Risk**: Low — well-established pattern, easy to test  
**Impact**: High — prevents cascading failures, improves availability

---

## 1. Circuit Breaker Pattern Explained

### States

```
                    Success
    ┌──────────────────────────────────┐
    │                                  │
    ▼                                  │
┌─────────┐    N failures    ┌─────────┴──┐
│ CLOSED  │ ───────────────► │   OPEN     │
│(normal) │                  │(fast fail) │
└─────────┘    ◄──────────── └─────┬──────┘
                 Success            │
                                    │ Timeout (60s)
                                    ▼
                              ┌─────────────┐
                              │  HALF-OPEN  │
                              │(test 1 req) │
                              └──────┬──────┘
                                     │
                          ┌──────────┴──────────┐
                          │                     │
                      Success               Failure
                          │                     │
                          ▼                     ▼
                     ┌─────────┐          ┌─────────┐
                     │ CLOSED  │          │  OPEN   │
                     └─────────┘          └─────────┘
```

### State Transitions

1. **CLOSED** (normal operation):
   - Requests flow through to LLM provider
   - Track consecutive failures
   - After 5 consecutive failures → transition to OPEN

2. **OPEN** (circuit broken):
   - Requests fail immediately (no LLM call)
   - Return fallback response or cached response
   - After 60 seconds → transition to HALF-OPEN

3. **HALF-OPEN** (testing recovery):
   - Allow 1 test request through
   - If success → transition to CLOSED
   - If failure → transition back to OPEN

---

## 2. Implementation Plan

### Step 1: Create Circuit Breaker Class

**File**: `agent/utils/circuit_breaker.py` (new)

```python
"""Circuit Breaker pattern for LLM provider resilience.

Prevents cascading failures by stopping requests to a failing provider
after N consecutive failures. Automatically tests recovery after a timeout.
"""

import time
from enum import Enum
from typing import Callable, TypeVar, Any
from functools import wraps

from services.logging import logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


T = TypeVar('T')


class CircuitBreaker:
    """Circuit breaker for external service calls.
    
    Usage:
        breaker = CircuitBreaker(
            name="openrouter",
            failure_threshold=5,
            recovery_timeout=60,
        )
        
        try:
            result = breaker.call(llm_provider.invoke, prompt)
        except CircuitBreakerOpenError:
            # Handle fast failure
            result = fallback_response()
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
    ):
        """Initialize circuit breaker.
        
        Args:
            name: Identifier for logging (e.g., "openrouter", "gemini")
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before testing recovery
            success_threshold: Consecutive successes to close circuit from half-open
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state (thread-safe)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        f"Circuit breaker '{self.name}': OPEN → HALF_OPEN "
                        f"(testing recovery after {self.recovery_timeout}s)"
                    )
            return self._state
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function call
            
        Raises:
            CircuitBreakerOpenError: If circuit is open (fast failure)
            Exception: Any exception from the wrapped function
        """
        state = self.state
        
        if state == CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker '{self.name}': OPEN, failing fast "
                f"(failures={self._failure_count}, "
                f"recovery in {self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s)"
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open"
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    logger.info(
                        f"Circuit breaker '{self.name}': HALF_OPEN → CLOSED "
                        f"(recovered after {self._success_count} successes)"
                    )
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            else:
                # Reset failure count on success
                self._failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failed during recovery test, reopen circuit
                logger.warning(
                    f"Circuit breaker '{self.name}': HALF_OPEN → OPEN "
                    f"(recovery test failed)"
                )
                self._state = CircuitState.OPEN
                self._success_count = 0
            elif self._failure_count >= self.failure_threshold:
                # Threshold reached, open circuit
                logger.error(
                    f"Circuit breaker '{self.name}': CLOSED → OPEN "
                    f"({self._failure_count} consecutive failures)"
                )
                self._state = CircuitState.OPEN
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"Circuit breaker '{self.name}': manually reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""
    pass
```

### Step 2: Wrap LLM Provider Calls

**File**: `agent/infrastructure/llm/openrouter.py` (modify)

```python
# Add import
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

class OpenRouterProvider:
    def __init__(self, model: str, temperature: float, max_tokens: int, api_key: str, timeout: int):
        # ... existing code ...
        
        # Add circuit breaker
        self._circuit_breaker = CircuitBreaker(
            name="openrouter",
            failure_threshold=5,
            recovery_timeout=60,
        )
    
    def invoke(self, messages: list) -> Any:
        """Invoke LLM with circuit breaker protection."""
        try:
            return self._circuit_breaker.call(
                self._invoke_internal,
                messages,
            )
        except CircuitBreakerOpenError:
            # Circuit is open, raise a transient error
            # This will be caught by the failover provider (Phase 1.2)
            raise TransientLLMError(
                message="OpenRouter circuit breaker is open",
                provider="openrouter",
            )
    
    def _invoke_internal(self, messages: list) -> Any:
        """Internal invoke method (wrapped by circuit breaker)."""
        # ... existing invoke logic ...
```

### Step 3: Inject Circuit Breaker in Container

**File**: `agent/infrastructure/container.py` (modify)

```python
# Circuit breaker is now part of the provider, no changes needed here
# But you can access it for monitoring:

if settings.use_tool_agent:
    # ... existing code ...
    agent = ToolCallingAgent(
        llm=llm.chat_model,
        tools=_tools,
        artifact_store=_search_artifact_store,
        default_top_k=5,
    )
    
    # Optional: expose circuit breaker for health checks
    # agent.circuit_breaker = llm._circuit_breaker
```

---

## 3. Configuration

**File**: `agent/config/settings.py` (add)

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # ── Circuit Breaker ──────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        alias="CIRCUIT_BREAKER_FAILURE_THRESHOLD",
        description="Consecutive failures before opening circuit",
    )
    circuit_breaker_recovery_timeout: int = Field(
        default=60,
        alias="CIRCUIT_BREAKER_RECOVERY_TIMEOUT",
        description="Seconds to wait before testing recovery",
    )
```

---

## 4. Testing Strategy

### Unit Tests

**Test 1: Circuit Opens After N Failures**
```python
def test_circuit_opens_after_threshold():
    breaker = CircuitBreaker(name="test", failure_threshold=3)
    
    # Simulate 3 failures
    for _ in range(3):
        try:
            breaker.call(lambda: 1/0)  # Raises ZeroDivisionError
        except ZeroDivisionError:
            pass
    
    assert breaker.state == CircuitState.OPEN
    
    # Next call should fail fast
    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(lambda: "success")
```

**Test 2: Circuit Half-Opens After Timeout**
```python
def test_circuit_half_opens_after_timeout():
    breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=1)
    
    # Open circuit
    try:
        breaker.call(lambda: 1/0)
    except ZeroDivisionError:
        pass
    
    assert breaker.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    time.sleep(1.1)
    
    assert breaker.state == CircuitState.HALF_OPEN
```

**Test 3: Circuit Closes After Successful Recovery**
```python
def test_circuit_closes_after_recovery():
    breaker = CircuitBreaker(name="test", failure_threshold=1, recovery_timeout=1, success_threshold=1)
    
    # Open circuit
    try:
        breaker.call(lambda: 1/0)
    except ZeroDivisionError:
        pass
    
    # Wait for recovery timeout
    time.sleep(1.1)
    
    # Successful call should close circuit
    result = breaker.call(lambda: "success")
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED
```

### Integration Tests

**Test 4: End-to-End with Real LLM Provider**
1. Mock OpenRouter to return 500 errors
2. Send 5 requests → circuit opens
3. Send 6th request → fast failure (no LLM call)
4. Wait 60 seconds → circuit half-opens
5. Mock OpenRouter to return 200 → circuit closes
6. Send request → success

---

## 5. Monitoring and Observability

### Metrics to Track

Add to `/v1/metrics` endpoint:

```python
class MetricsResponse(BaseModel):
    # ... existing fields ...
    
    circuit_breaker_state: str = Field(
        default="closed",
        description="Circuit breaker state: closed, open, half_open",
    )
    circuit_breaker_failure_count: int = Field(
        default=0,
        description="Current consecutive failure count",
    )
```

### Logs to Watch

```
INFO  | Circuit breaker 'openrouter': CLOSED → OPEN (5 consecutive failures)
WARN  | Circuit breaker 'openrouter': OPEN, failing fast (recovery in 45s)
INFO  | Circuit breaker 'openrouter': OPEN → HALF_OPEN (testing recovery after 60s)
INFO  | Circuit breaker 'openrouter': HALF_OPEN → CLOSED (recovered after 2 successes)
```

### CloudWatch Alarms (Phase 2)

```yaml
CircuitBreakerOpen:
  Condition: circuit_breaker_state == "open"
  Duration: >5 minutes
  Action: Send Discord alert
```

---

## 6. Rollout Plan

### Pre-Deployment

1. **Code Review**: Ensure circuit breaker implementation is reviewed
2. **Unit Tests**: All unit tests pass
3. **Integration Tests**: Test with mocked LLM provider
4. **Staging Deployment**: Deploy to staging with `CIRCUIT_BREAKER_FAILURE_THRESHOLD=2` (lower for testing)

### Deployment

1. **Deploy to Production**: Use existing deployment pipeline
2. **Monitor Logs**: Watch for circuit breaker state transitions
3. **Smoke Test**: Send test queries, verify normal operation
4. **Chaos Test**: Temporarily block OpenRouter IP, verify circuit opens
5. **Recovery Test**: Unblock IP, verify circuit closes automatically

### Post-Deployment

1. **Monitor for 1 Week**: Watch for false positives (circuit opens too aggressively)
2. **Tune Thresholds**: Adjust `failure_threshold` and `recovery_timeout` based on production data
3. **Document Incidents**: Log any circuit breaker activations for post-mortem

---

## 7. Success Criteria

The circuit breaker is successful when:

- ✅ Cascading failures are prevented (no more 3-retry storms when provider is down)
- ✅ Fast failures when circuit is open (<100ms vs 30+ seconds with retries)
- ✅ Automatic recovery when provider comes back online
- ✅ No false positives (circuit doesn't open during normal operation)
- ✅ Observable via logs and metrics

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Circuit opens too aggressively | Medium | Medium | Tune `failure_threshold` based on production data |
| Circuit stays open too long | Low | High | Reduce `recovery_timeout` if needed |
| Half-open test request fails due to transient issue | Medium | Low | Increase `success_threshold` to 2-3 |
| Circuit breaker adds latency | Very Low | Low | Overhead is negligible (<1ms per call) |

---

## 9. Future Improvements

After basic circuit breaker is working:

1. **Per-Endpoint Circuit Breakers**: Separate breakers for chat, embeddings, etc.
2. **Adaptive Thresholds**: Automatically adjust thresholds based on error rate
3. **Circuit Breaker Dashboard**: Visual UI showing state, failure count, recovery time
4. **Fallback Responses**: Serve cached responses when circuit is open (Phase 4)

---

## 10. References

- **Martin Fowler on Circuit Breaker**: https://martinfowler.com/bliki/CircuitBreaker.html
- **Python Circuit Breaker Library**: https://github.com/fabfuel/circuitbreaker (alternative to custom implementation)
- **Current Code**: `agent/infrastructure/llm/openrouter.py`
- **Retry Logic**: `agent/utils/retry.py` (works alongside circuit breaker)

---

**Document Version**: 1.0  
**Last Updated**: 2026-05-29  
**Author**: AI Assistant  
**Status**: Ready for Implementation

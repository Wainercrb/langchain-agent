"""Observability — active health monitoring, AI decision tracking, and pluggable tracing.

Subpackages:
    - `health`: Health verification checks and background scheduling.
    - `decisions`: Thread-safe bounded log of AI decision metadata.
    - `provider`: Abstract ObservabilityProvider (Strategy Pattern).
    - `langsmith`: LangSmith backend implementation.
    - `noop`: No-op fallback for local dev.
    - `decorator`: Provider-agnostic @trace decorator.
"""

from .decisions import DecisionTracker
from .health import CheckResult, HealthVerifier, MonitoringScheduler
from .provider import (
    ObservabilityProvider,
    get_observability_provider,
    set_observability_provider,
)

__all__ = [
    "CheckResult",
    "DecisionTracker",
    "HealthVerifier",
    "MonitoringScheduler",
    "ObservabilityProvider",
    "get_observability_provider",
    "set_observability_provider",
]

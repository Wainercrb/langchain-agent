"""Domain port protocols — abstraction boundaries for dependency inversion.

Domain code depends on these protocols instead of concrete infrastructure
implementations. Implementations are injected at the composition root.
"""

from .logger import Logger
from .tracing import TracingOrchestrator
from .parsers import FileParser, ParserRegistry

__all__ = ["Logger", "TracingOrchestrator", "FileParser", "ParserRegistry"]

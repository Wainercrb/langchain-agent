"""Logger protocol — abstraction for structured logging.

Domain code depends on this Protocol instead of concrete logging backends.
Any object with debug, info, warning, and error methods satisfies it.
"""

from typing import Protocol


class Logger(Protocol):
    """Protocol for structured loggers.

    Duck-typed: any object with these four methods satisfies the protocol.
    """

    def debug(self, msg: str, *args, **kwargs) -> None:
        ...

    def info(self, msg: str, *args, **kwargs) -> None:
        ...

    def warning(self, msg: str, *args, **kwargs) -> None:
        ...

    def error(self, msg: str, *args, **kwargs) -> None:
        ...

"""Abstract logger backend — defines the contract for any logging system."""

from abc import ABC, abstractmethod


class Logger(ABC):
    """Abstract base for pluggable logging backends.

    Methods mirror logging.Logger (debug, info, warning, error).
    The backend is configured in the constructor and ready to use.
    """

    @abstractmethod
    def debug(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def info(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def warning(self, msg: str, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def error(self, msg: str, *args, **kwargs) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

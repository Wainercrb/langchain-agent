"""Abstract logger backend — define el contrato para cualquier sistema de logs."""

from abc import ABC, abstractmethod


class Logger(ABC):
    """Abstract base for pluggable logging backends.

    Los métodos reflejan logging.Logger (debug, info, warning, error).
    El backend se configura en el constructor y queda listo para usar.
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

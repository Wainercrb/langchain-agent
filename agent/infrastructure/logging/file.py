"""File logger — log en formato JSON a un archivo."""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .base import Logger
from utils.correlation import get_correlation_id


class FileLogger(Logger):
    """Logs en formato JSON a un archivo, una línea por entrada."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log("DEBUG", msg, args, kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log("INFO", msg, args, kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log("WARNING", msg, args, kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log("ERROR", msg, args, kwargs)

    def _log(self, level: str, msg: str, args: tuple, kwargs: dict) -> None:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "correlation_id": get_correlation_id(),
            "message": msg % args if args else msg,
        }
        exc_info = kwargs.pop("exc_info", False)
        if kwargs:
            data["extra"] = kwargs
        if exc_info:
            data["stack_trace"] = traceback.format_exc()
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

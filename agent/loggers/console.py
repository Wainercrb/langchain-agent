"""Console logger — log en formato JSON a stdout."""

import json
import traceback
from datetime import datetime, timezone

from .base import Logger
from shared.correlation import get_correlation_id


class Console(Logger):
    """Logs en formato JSON a stdout."""

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
        # Extract and handle exc_info
        exc_info = kwargs.pop("exc_info", False)
        if kwargs:
            data["extra"] = kwargs
        if exc_info:
            data["stack_trace"] = traceback.format_exc()
        print(json.dumps(data))

"""Console logger — log en formato JSON a stdout. Sin configuración, sin dependencias."""

import json
from datetime import datetime, timezone

from .base import Logger


class Console(Logger):
    """Logs en formato JSON a stdout. Sin configuración."""

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log("DEBUG", msg, args, kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log("INFO", msg, args, kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log("WARNING", msg, args, kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log("ERROR", msg, args, kwargs)

    # ------------------------------------------------------------------
    def _log(self, level: str, msg: str, args: tuple, kwargs: dict) -> None:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": msg % args if args else msg,
        }
        if kwargs:
            data["extra"] = kwargs
        print(json.dumps(data))

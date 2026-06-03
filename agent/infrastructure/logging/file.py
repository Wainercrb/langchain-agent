"""File logger — log en formato JSON a un archivo con rotación automática.

Rotación por tamaño: cuando el archivo supera LOG_MAX_BYTES, se renombra
a .1, .2, etc. hasta LOG_BACKUP_COUNT. El archivo más viejo se elimina.
"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .base import Logger
from utils.correlation import get_correlation_id

# Defaults
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_BACKUP_COUNT = 5


class FileLogger(Logger):
    """Logs en formato JSON a un archivo con rotación por tamaño.

    Args:
        path: Ruta del archivo de log.
        max_bytes: Tamaño máximo antes de rotar (default: 10 MB).
        backup_count: Cantidad de archivos backup a mantener (default: 5).
    """

    def __init__(
        self,
        path: str,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        backup_count: int = _DEFAULT_BACKUP_COUNT,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes
        self._backup_count = backup_count

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log("DEBUG", msg, args, kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log("INFO", msg, args, kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log("WARNING", msg, args, kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log("ERROR", msg, args, kwargs)

    def _rotate_if_needed(self) -> None:
        """Rotate the log file if it exceeds max_bytes.

        Renames current file to .1, shifts existing backups up,
        and deletes the oldest if it exceeds backup_count.
        """
        if not self._path.exists():
            return

        try:
            current_size = self._path.stat().st_size
        except OSError:
            return

        if current_size < self._max_bytes:
            return

        # Delete oldest backup if at capacity
        oldest = Path(f"{self._path}.{self._backup_count}")
        if oldest.exists():
            oldest.unlink()

        # Shift existing backups: .4 -> .5, .3 -> .4, etc.
        for i in range(self._backup_count - 1, 0, -1):
            src = Path(f"{self._path}.{i}")
            dst = Path(f"{self._path}.{i + 1}")
            if src.exists():
                src.rename(dst)

        # Rotate current -> .1
        self._path.rename(Path(f"{self._path}.1"))

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

        self._rotate_if_needed()

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

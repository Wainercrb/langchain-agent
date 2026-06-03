"""CloudWatch logger — log en formato JSON a AWS CloudWatch Logs.

Uses watchtower (https://github.com/kislyuk/watchtower) to send structured
JSON logs to a CloudWatch log group and stream. Supports boto3 credential
resolution: explicit keys, IAM role, or environment variables.
"""

import json
import logging
import traceback
from typing import Optional

import boto3
from watchtower import CloudWatchLogHandler

from config.settings import settings
from .base import Logger
from utils.correlation import get_correlation_id


class CloudWatchLogger(Logger):
    """Logs en formato JSON a AWS CloudWatch Logs.

    Args:
        log_group: CloudWatch log group name.
        stream_name: CloudWatch log stream name.
        aws_access_key_id: Optional explicit AWS access key.
        aws_secret_access_key: Optional explicit AWS secret key.
        aws_region: AWS region for CloudWatch.
    """

    def __init__(
        self,
        log_group: Optional[str] = None,
        stream_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
    ) -> None:
        self._log_group = log_group or settings.cloudwatch_log_group
        self._stream_name = stream_name or settings.cloudwatch_stream_name
        self._region = aws_region or settings.aws_region

        # Build boto3 session with explicit or default credentials
        session_kwargs = {"region_name": self._region}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        session = boto3.session.Session(**session_kwargs)

        # Create a standard Python logger as the backing store
        self._py_logger = logging.getLogger("cloudwatch")
        self._py_logger.setLevel(logging.DEBUG)

        # Watchtower handler sends logs to CloudWatch
        cw_handler = CloudWatchLogHandler(
            log_group=self._log_group,
            stream_name=self._stream_name,
            boto3_session=session,
        )
        self._py_logger.addHandler(cw_handler)

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
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "level": level,
            "correlation_id": get_correlation_id(),
            "message": msg % args if args else msg,
        }
        exc_info = kwargs.pop("exc_info", False)
        if kwargs:
            data["extra"] = kwargs
        if exc_info:
            data["stack_trace"] = traceback.format_exc()

        # Use extra= to pass structured data to the Python logger
        # Watchtower serializes the extra dict as the log event
        log_kwargs = {"extra": data}
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        self._py_logger.log(level_map.get(level, logging.INFO), msg, **log_kwargs)

    def __repr__(self) -> str:
        return (
            f"CloudWatchLogger(group={self._log_group!r}, "
            f"stream={self._stream_name!r})"
        )

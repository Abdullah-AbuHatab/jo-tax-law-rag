"""Logging setup: plain text lines with the ``extra={...}`` fields appended as key=value."""

from __future__ import annotations

import logging

_STANDARD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD_FIELDS}
        if extras:
            line += " " + " ".join(f"{key}={value}" for key, value in extras.items())
        return line


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(KeyValueFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

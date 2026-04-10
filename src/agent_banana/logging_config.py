"""Centralized logging configuration for Agent Banana.

Provides a single ``setup_logging()`` entry-point that configures the root
`agent_banana` logger with:
  - Timestamped, coloured console output
  - Rotating file output (``logs/agent_banana.log``, max 5 MB × 3 backups)

Every module should obtain its logger via:
    ``logger = logging.getLogger(__name__)``

The ``log_function`` decorator automatically logs function entry (with args),
return, and any unhandled exceptions — exactly what the user asked for.
"""

from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import sys
import time
import traceback
from pathlib import Path


# ── Coloured formatter (Windows-friendly via ANSI) ──────────────────────────

class _ColorFormatter(logging.Formatter):
    """Console formatter that adds ANSI colours per log level."""

    _COLORS = {
        logging.DEBUG:    "\033[36m",    # cyan
        logging.INFO:     "\033[32m",    # green
        logging.WARNING:  "\033[33m",    # yellow
        logging.ERROR:    "\033[31m",    # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelno, "")
        formatted = super().format(record)
        return f"{color}{formatted}{self._RESET}" if color else formatted


# ── Public API ──────────────────────────────────────────────────────────────

def setup_logging(
    *,
    level: str | int = "DEBUG",
    log_dir: Path | str | None = None,
    log_filename: str = "agent_banana.log",
) -> logging.Logger:
    """Configure the ``agent_banana`` logger hierarchy.

    Call once at application startup (e.g. from ``server.py`` or ``cli.py``).

    Args:
        level: Global minimum log level (``"DEBUG"``, ``"INFO"``, …).
        log_dir: Directory for the rotating log file.  Defaults to
                 ``<project-root>/logs``.
        log_filename: Name of the log file.

    Returns:
        The configured root ``agent_banana`` logger.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.DEBUG)

    root_logger = logging.getLogger("agent_banana")

    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        return root_logger

    root_logger.setLevel(level)

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(_ColorFormatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────────────
    if log_dir is None:
        # Default: <project-root>/logs
        log_dir = Path(__file__).resolve().parents[2] / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / log_filename,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)  # always write everything to file
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s:%(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    root_logger.info("Logger initialised  (level=%s, file=%s)", logging.getLevelName(level), log_dir / log_filename)
    return root_logger


# ── Function decorator ──────────────────────────────────────────────────────

def log_function(fn=None, *, level: int = logging.DEBUG):
    """Decorator that logs every call, return value, and exception.

    Usage::

        @log_function
        def my_func(x, y):
            ...

        @log_function(level=logging.INFO)
        def important_func():
            ...
    """
    def decorator(func):
        _logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # ---- Build a compact arg summary (avoid huge image payloads) ----
            def _short(v, limit=120):
                s = repr(v)
                return s if len(s) <= limit else s[:limit] + "…"

            arg_parts = [_short(a) for a in args]
            kwarg_parts = [f"{k}={_short(v)}" for k, v in kwargs.items()]
            call_sig = ", ".join(arg_parts + kwarg_parts)

            _logger.log(level, "→ ENTER  %s(%s)", func.__qualname__, call_sig)
            t0 = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                _logger.log(
                    level,
                    "← RETURN %s  [%.1f ms]  result=%s",
                    func.__qualname__, elapsed, _short(result),
                )
                return result
            except Exception:
                elapsed = (time.perf_counter() - t0) * 1000
                _logger.error(
                    "✖ FAILED %s  [%.1f ms]\n%s",
                    func.__qualname__, elapsed, traceback.format_exc(),
                )
                raise

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator

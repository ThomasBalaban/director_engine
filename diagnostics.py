"""
Structured error log at diagnostics/errors.log (JSONL, one event per line).

Why a file: stdout drops on backpressure (see main._install_nonblocking_stdio)
and the launcher's reader can stall, so errors can be lost from the live log.
A file write is independent of either and survives the stall.

Call log_error() from anywhere — it never raises and is thread-safe.
"""
import json
import os
import threading
import time
import traceback as _tb
from typing import Any, Optional

_DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
_ERR_PATH = os.path.join(_DIAG_DIR, "errors.log")
_lock = threading.Lock()
_TRACE_CAP = 2000  # bytes, last N chars of formatted traceback


def _entry(component: str, message: str, exc: Optional[BaseException], context: dict) -> str:
    e: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "monotonic": round(time.monotonic(), 3),
        "component": component,
        "message": message,
    }
    if context:
        e["context"] = {k: (str(v)[:200] if not isinstance(v, (int, float, bool, type(None))) else v)
                        for k, v in context.items()}
    if exc is not None:
        e["exc_type"] = type(exc).__name__
        e["exc_msg"] = str(exc) or repr(exc)
        try:
            tb_str = "".join(_tb.format_exception(type(exc), exc, exc.__traceback__))
            e["trace"] = tb_str[-_TRACE_CAP:]
        except Exception:
            pass
    return json.dumps(e, default=str) + "\n"


def log_error(component: str, message: str, exc: Optional[BaseException] = None, **context) -> None:
    """Append one error event. Never raises."""
    try:
        line = _entry(component, message, exc, context)
        os.makedirs(_DIAG_DIR, exist_ok=True)
        with _lock:
            with open(_ERR_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def install_asyncio_handler() -> None:
    """Route asyncio unhandled exceptions to errors.log. Call once after the loop exists."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return

    def _handler(_loop, context):
        exc = context.get("exception")
        msg = context.get("message", "unhandled asyncio exception")
        extras = {k: v for k, v in context.items() if k not in ("exception", "message")}
        log_error("asyncio", msg, exc=exc, **extras)

    loop.set_exception_handler(_handler)


def install_excepthook() -> None:
    """Capture uncaught exceptions from main and worker threads."""
    import sys
    prev = sys.excepthook

    def _hook(exc_type, exc, tb):
        try:
            log_error("uncaught", f"uncaught {exc_type.__name__}", exc=exc)
        except Exception:
            pass
        prev(exc_type, exc, tb)

    sys.excepthook = _hook

    try:
        prev_t = getattr(threading, "excepthook", None)

        def _thook(args):
            try:
                log_error(
                    "uncaught_thread",
                    f"thread {args.thread.name} raised {args.exc_type.__name__}",
                    exc=args.exc_value,
                )
            except Exception:
                pass
            if prev_t:
                prev_t(args)

        threading.excepthook = _thook
    except Exception:
        pass

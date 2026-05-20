"""
Shared concurrency gate for every Ollama call in director_engine.

The 32B-class model on the M4 serves one inference at a time. Letting
analyst, compressor, summarizer, and thought generation all fire in parallel
just queues them at the runner and they all blow past their wait_for timeout.
This semaphore caps the total in-flight calls across modules.
"""
import asyncio
from typing import Optional

from config import OLLAMA_MAX_CONCURRENT

_gate: Optional[asyncio.Semaphore] = None


def get_ollama_gate() -> asyncio.Semaphore:
    """Lazy init so the semaphore binds to the running event loop."""
    global _gate
    if _gate is None:
        _gate = asyncio.Semaphore(OLLAMA_MAX_CONCURRENT)
    return _gate

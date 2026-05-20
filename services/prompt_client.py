# Save as: director_engine/services/prompt_client.py
"""
Client for communicating with the Prompt Service.

The brain fires speech requests here. The prompt service
decides what actually reaches Nami.
"""

import httpx
from typing import Dict, Any, Optional
from config import PROMPT_SERVICE_URL

_client: Optional[httpx.AsyncClient] = None

# ── In-flight backpressure ────────────────────────────────────────────────────
# Each request_speech spawns an HTTP POST. Under reflex-trigger floods these
# pile up, each one consuming asyncio scheduler time, the GIL during JSON
# serialization, and a socket from the httpx pool. When prompt_service is
# already cooldown-blocking, the requests still cost us. Cap inflight and
# drop excess so the loop can breathe.
_MAX_INFLIGHT_REQUESTS = 4
_inflight = 0
_dropped_total = 0


async def initialize():
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
        print(f"✅ [PromptClient] Connected to {PROMPT_SERVICE_URL}")


async def close():
    global _client
    if _client:
        await _client.aclose()
        _client = None
        print("✅ [PromptClient] Closed")


async def request_speech(
    trigger: str,
    content: str,
    priority: float = 0.5,
    source: str = "DIRECTOR",
    is_interrupt: bool = False,
    event_id: str = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Send a speech request to the prompt service.
    
    Returns the gate result: {"delivered": bool, "gate_result": str}
    """
    if not _client:
        await initialize()

    global _inflight, _dropped_total

    # Backpressure: drop non-interrupt requests when saturated. Interrupts
    # (direct mentions, urgency) bypass the cap — they're the rare-but-important
    # path. Regular reflex-trigger floods get dropped to keep the loop healthy.
    if not is_interrupt and _inflight >= _MAX_INFLIGHT_REQUESTS:
        _dropped_total += 1
        if _dropped_total <= 5 or _dropped_total % 20 == 0:
            print(
                f"⚠️  [PromptClient] DROPPED non-interrupt request "
                f"({_inflight} in-flight, {_dropped_total} total dropped) | trigger={trigger}"
            )
        return {"delivered": False, "gate_result": "director_backpressure"}

    payload = {
        "trigger": trigger,
        "content": content,
        "priority": priority,
        "source": source,
        "is_interrupt": is_interrupt,
        "event_id": event_id,
        "metadata": metadata or {},
    }

    _inflight += 1
    try:
        response = await _client.post(
            f"{PROMPT_SERVICE_URL}/speak",
            json=payload,
            timeout=2.0,
        )
        result = response.json()

        if result.get("delivered"):
            print(f"✅ [PromptClient] Delivered: {trigger}")
        else:
            reason = result.get("gate_result", "unknown")
            if reason != "already_reacted":  # Don't spam logs for dedup
                print(f"🚫 [PromptClient] Blocked: {reason} | {trigger}")

        return result

    except httpx.ConnectError:
        print(f"❌ [PromptClient] Cannot reach prompt service at {PROMPT_SERVICE_URL}")
        return {"delivered": False, "gate_result": "service_unreachable"}
    except httpx.TimeoutException as e:
        # ReadTimeout / PoolTimeout / ConnectTimeout — str(e) is often empty
        kind = type(e).__name__
        print(f"⏱️  [PromptClient] Timeout ({kind}) talking to prompt service | trigger={trigger}")
        return {"delivered": False, "gate_result": f"timeout: {kind}"}
    except Exception as e:
        # Always include the exception type — many httpx errors have empty str()
        detail = str(e) or repr(e)
        print(f"❌ [PromptClient] Error ({type(e).__name__}): {detail} | trigger={trigger}")
        return {"delivered": False, "gate_result": f"error: {type(e).__name__}"}
    finally:
        _inflight -= 1


async def send_interrupt(
    content: str,
    source: str,
    trigger: str = "direct_mention",
    event_id: str = None,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for interrupt requests."""
    return await request_speech(
        trigger=trigger,
        content=content,
        priority=0.0,
        source=source,
        is_interrupt=True,
        event_id=event_id,
        metadata=metadata,
    )


async def notify_user_responded():
    """Tell prompt service: user spoke again, clear awaiting state."""
    if not _client:
        return
    try:
        await _client.post(f"{PROMPT_SERVICE_URL}/user_responded", timeout=1.0)
    except Exception as e:
        print(f"⚠️ [PromptClient] notify_user_responded failed: {e}")


async def notify_bot_response():
    """Tell prompt service: Nami just responded to user. Start cooldown."""
    if not _client:
        return
    try:
        await _client.post(f"{PROMPT_SERVICE_URL}/register_bot_response", timeout=1.0)
    except Exception as e:
        print(f"⚠️ [PromptClient] notify_bot_response failed: {e}")
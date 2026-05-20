# director_engine/main.py

# ── Non-blocking stdout/stderr ────────────────────────────────────────────────
# MUST run before any other imports that might print (uvicorn, ollama, etc).
#
# Without this, the launcher's stdout-reader thread can fall behind under
# heavy logging (heartbeats + thread watchdog + vision events + memory flags
# all going at once). The OS pipe buffer fills (~64KB on macOS), and the next
# print() in this process blocks on write() — holding the GIL while it waits.
# That freezes the asyncio loop, the thread watchdog, the HTTP server, and
# every ticker simultaneously. Observed: full lockup at ~10-15 min runtime,
# with all logs (asyncio + thread) stopping at the exact same moment.
#
# Trade-off: when the pipe is full we silently drop log lines (counted in
# .dropped on the wrapper). The engine keeps running.
def _install_nonblocking_stdio():
    import os, sys, fcntl, io, time, threading

    def _ts() -> str:
        # HH:MM:SS.mmm stamp captured at write-time so the launcher's read-time
        # stamp can't lie when the pipe drains in a burst.
        now = time.time()
        local = time.localtime(now)
        ms = int((now - int(now)) * 1000)
        return f"[{local.tm_hour:02d}:{local.tm_min:02d}:{local.tm_sec:02d}.{ms:03d}] "

    class _NonBlockingWriter:
        def __init__(self, underlying, name):
            self._u = underlying
            self._name = name
            self._buf = []          # partial-line accumulator (no \n yet)
            self._lock = threading.Lock()
            self.dropped = 0
            try:
                fd = underlying.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except (OSError, io.UnsupportedOperation, AttributeError):
                pass

        def write(self, s):
            if not s:
                return 0
            # Line-buffered stamping: accumulate until \n, stamp+flush each line.
            # This makes timestamps reflect when the engine actually wrote, not
            # when the launcher's reader caught up.
            with self._lock:
                slen = len(s)
                start = 0
                while start < slen:
                    nl = s.find('\n', start)
                    if nl < 0:
                        self._buf.append(s[start:])
                        break
                    self._buf.append(s[start:nl + 1])
                    line = ''.join(self._buf)
                    self._buf = []
                    try:
                        self._u.write(_ts() + line)
                    except (BlockingIOError, OSError):
                        self.dropped += 1
                    start = nl + 1
            return slen

        def flush(self):
            with self._lock:
                if self._buf:
                    line = ''.join(self._buf)
                    self._buf = []
                    try:
                        self._u.write(_ts() + line + '\n')
                    except (BlockingIOError, OSError):
                        self.dropped += 1
                try:
                    self._u.flush()
                except (BlockingIOError, OSError):
                    pass

        def __getattr__(self, name):
            return getattr(self._u, name)

    import sys as _sys
    _sys.stdout = _NonBlockingWriter(_sys.stdout, "stdout")
    _sys.stderr = _NonBlockingWriter(_sys.stderr, "stderr")

_install_nonblocking_stdio()
# ──────────────────────────────────────────────────────────────────────────────

# ── Faulthandler ──────────────────────────────────────────────────────────────
# Enables thread-traceback dumps. Use it two ways:
#   1. On crash (uncaught fatal): traceback auto-printed to stderr.
#   2. On demand: `kill -SIGABRT <director_pid>` from another terminal dumps
#      the current stack of every thread (asyncio loop, watchdog, etc).
#      Useful when the engine is wedged and you want to know exactly where.
import faulthandler as _faulthandler
import sys as _sys
_faulthandler.enable(file=_sys.stderr)
# ──────────────────────────────────────────────────────────────────────────────

import uvicorn
import asyncio
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import config
import shared
import core_logic
import services.llm_analyst as llm_analyst
from services.sensor_bridge import SensorBridge
import diagnostics

# Install error capture once Python's stdlib is set up. Asyncio handler is
# installed later, once the running loop exists.
diagnostics.install_excepthook()


# ── File-based diagnostic heartbeat ───────────────────────────────────────────
# Writes a small status file every 5s from a dedicated OS thread. Immune to:
#   - asyncio loop blocks (it's a thread, not a coroutine)
#   - stdout pipe-fill (writes to a file, not stdout)
#   - launcher slowness (no IPC dependency)
#
# When debugging a "freeze":
#   - File mtime keeps updating → process alive, only stdout/asyncio wedged
#   - File mtime stops → whole Python process is truly frozen (OS-level issue)
#
# Tail with: `tail -f director_engine/diagnostics/heartbeat.txt`
def _start_file_heartbeat():
    import os as _os
    import threading as _threading
    import time as _time
    import json as _json

    diag_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "diagnostics")
    _os.makedirs(diag_dir, exist_ok=True)
    heartbeat_path = _os.path.join(diag_dir, "heartbeat.txt")

    def _loop():
        while True:
            try:
                st = core_logic._reflex_state
                now_m = _time.monotonic()
                in_backoff = st.get("backoff_active_until", 0) > now_m
                backoff_remaining = max(0.0, st.get("backoff_active_until", 0) - now_m)

                # Pull prompt_client counters if accessible
                try:
                    from services import prompt_client as _pc
                    pc_inflight = getattr(_pc, "_inflight", None)
                    pc_dropped = getattr(_pc, "_dropped_total", None)
                except Exception:
                    pc_inflight = None
                    pc_dropped = None

                # Pull analyze counters
                try:
                    from services import llm_analyst as _la
                    an_pending = getattr(_la, "_analyze_pending", None)
                    an_dropped = getattr(_la, "_analyze_dropped_total", None)
                except Exception:
                    an_pending = None
                    an_dropped = None

                pending_memory = 0
                try:
                    pending_memory = len(getattr(shared.store, "pending_memories_to_save", []))
                except Exception:
                    pass

                vision_dropped = getattr(core_logic, "_vision_events_dropped_total", 0)

                payload = {
                    "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
                    "monotonic": round(now_m, 2),
                    "reflex_iter": st["iteration"],
                    "reflex_step": st["step"],
                    "last_iter_duration_s": st.get("last_iter_actual_duration_s", 0),
                    "loop_drift_s": st.get("last_iter_loop_drift_s", 0),
                    "recent_thought_timeouts": st.get("recent_thought_timeouts", 0),
                    "backoff_active": in_backoff,
                    "backoff_remaining_s": round(backoff_remaining, 1),
                    "hub_connected": getattr(shared.sio, "connected", None),
                    "thread_count": _threading.active_count(),
                    "promptclient_inflight": pc_inflight,
                    "promptclient_dropped_total": pc_dropped,
                    "analyze_pending": an_pending,
                    "analyze_dropped_total": an_dropped,
                    "pending_memory_saves": pending_memory,
                    "vision_events_dropped_total": vision_dropped,
                }
                # Write+rename for atomicity (avoid partial reads from tail -f)
                tmp_path = heartbeat_path + ".tmp"
                with open(tmp_path, "w") as f:
                    f.write(_json.dumps(payload, indent=2) + "\n")
                _os.replace(tmp_path, heartbeat_path)
            except Exception:
                pass  # never let diagnostics crash the engine
            _time.sleep(5.0)

    t = _threading.Thread(target=_loop, name="file-heartbeat", daemon=True)
    t.start()
# ──────────────────────────────────────────────────────────────────────────────


# ── Stuck-detector ───────────────────────────────────────────────────────────
# Watches reflex_iter. If it doesn't advance for STUCK_AFTER_S seconds, dump
# every thread's stack to diagnostics/stuck_dump.txt via faulthandler. This is
# an OS thread, so it survives an asyncio wedge.
#
# Read the dump file after a freeze to see exactly where the event loop hung.
# Repeated freezes overwrite the file — the most recent freeze wins.
def _start_stuck_detector():
    import os as _os
    import threading as _threading
    import time as _time
    import faulthandler as _faulthandler

    diag_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "diagnostics")
    _os.makedirs(diag_dir, exist_ok=True)
    dump_path = _os.path.join(diag_dir, "stuck_dump.txt")

    STUCK_AFTER_S = 30.0
    POLL_S = 5.0

    def _loop():
        last_iter = -1
        last_advance_at = _time.monotonic()
        dumped_for_this_freeze = False
        while True:
            _time.sleep(POLL_S)
            try:
                st = core_logic._reflex_state
                cur_iter = st["iteration"]
                now = _time.monotonic()
                if cur_iter != last_iter:
                    last_iter = cur_iter
                    last_advance_at = now
                    dumped_for_this_freeze = False
                    continue
                if dumped_for_this_freeze:
                    continue
                stuck_for = now - last_advance_at
                if stuck_for < STUCK_AFTER_S:
                    continue
                # Frozen. Dump everything we can.
                try:
                    with open(dump_path, "w") as f:
                        f.write("=== STUCK DETECTOR DUMP ===\n")
                        f.write(f"wall_time:      {_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"reflex_iter:    {cur_iter} (frozen for {stuck_for:.1f}s)\n")
                        f.write(f"reflex_step:    {st.get('step')}\n")
                        f.write(f"step_started:   {st.get('step_started_at')}\n")
                        f.write(f"last_iter_dur:  {st.get('last_iter_actual_duration_s')}\n")
                        f.write(f"thread_count:   {_threading.active_count()}\n")
                        f.write("\n=== ALL THREAD STACKS ===\n")
                        f.flush()
                        _faulthandler.dump_traceback(file=f, all_threads=True)
                    dumped_for_this_freeze = True
                except Exception:
                    pass
            except Exception:
                pass

    t = _threading.Thread(target=_loop, name="stuck-detector", daemon=True)
    t.start()
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Nami Director Engine")

# Initialize the SensorBridge so it hooks up to the shared.sio Hub connection
sensor_bridge = SensorBridge()
pending_memory_requests = {}

# --- Request Models ---

class ContextRequest(BaseModel):
    trigger: Optional[str] = None
    event_id: Optional[str] = None
    metadata: Optional[dict] = {}


# --- Hub Connection ---

# ── Hub connection state-change handlers (state-change-only, low volume) ─────
# An event-driven flag lets connect_to_hub wake immediately on disconnect
# instead of waiting up to 2s for its next poll.
_hub_reconnect_event = asyncio.Event()


@shared.sio.event
async def connect():
    print(f"🔌 [Hub] CONNECTED to {config.HUB_URL}")


@shared.sio.event
async def disconnect():
    print(f"🔌 [Hub] DISCONNECTED — triggering immediate reconnect")
    _hub_reconnect_event.set()


@shared.sio.event
async def connect_error(data):
    kind = type(data).__name__
    detail = str(data) or repr(data)
    print(f"🔌 [Hub] CONNECT_ERROR ({kind}): {detail}")
    _hub_reconnect_event.set()
# ──────────────────────────────────────────────────────────────────────────────


async def connect_to_hub():
    """
    Maintain connection to the Central Hub.

    Reconnect strategy:
    - Healthy: poll sio.connected every 1s
    - Disconnect event fires → wake immediately and try to reconnect
    - Reconnect attempt: bounded by 10s timeout to prevent the await from
      hanging forever if the hub itself is wedged
    - Failed attempt: exponential-ish backoff (2s, 4s, 8s, max 8s)
    """
    backoff = 2.0
    while shared.server_ready:
        if shared.sio.connected:
            backoff = 2.0  # reset on healthy state
            # Wait either for a disconnect signal or the next health-poll tick
            try:
                await asyncio.wait_for(_hub_reconnect_event.wait(), timeout=1.0)
                _hub_reconnect_event.clear()
            except asyncio.TimeoutError:
                pass
            continue

        # Disconnected — try to reconnect with bounded duration
        try:
            print(f"🔌 [Director Engine] Connecting to Hub at {config.HUB_URL}...")
            await asyncio.wait_for(
                shared.sio.connect(config.HUB_URL, transports=['websocket', 'polling']),
                timeout=10.0,
            )
            _hub_reconnect_event.clear()
        except asyncio.TimeoutError:
            print(f"⚠️ [Director Engine] Hub connect timed out after 10s. Retrying in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
        except Exception as e:
            kind = type(e).__name__
            detail = str(e) or repr(e)
            print(f"⚠️ [Director Engine] Hub connect failed ({kind}): {detail}. Retrying in {backoff:.0f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)


# --- Endpoints ---

@app.get("/health")
async def health():
    """The UI Launcher queries this to check if the Director is online."""
    return {
        "status": "ok",
        "service": "director_engine",
        "hub_connected": shared.sio.connected
    }


@app.post("/context")
async def get_context(payload: ContextRequest = ContextRequest()):
    """
    Called by the prompt service when it needs to build a full prompt.

    Returns the complete structured context block assembled from:
    - Current visual/audio/chat events in the context store
    - Semantic memory retrieval (hybrid scored)
    - Current directive from the decision engine
    - Narrative log, ancient history, thread state
    - Active user profile

    The prompt service passes optional trigger metadata so the director
    can tailor the context to what triggered the speech request.
    """
    directive = shared.store.current_directive

    # Build smart query for semantic memory retrieval
    summary_data = shared.store.get_summary_data()
    memory_query = core_logic.build_smart_memory_query(shared.store, summary_data)

    # --- NEW: Flush pending memories to the Hub before we query ---
    # Snapshot under lock then await emit outside the lock — otherwise
    # promote_to_memory (which acquires store.lock to append) races this
    # iteration. Iterating the list while another task mutates it can raise
    # "list changed size during iteration" or silently skip entries.
    if hasattr(shared.store, 'pending_memories_to_save'):
        with shared.store.lock:
            pending = list(shared.store.pending_memories_to_save)
            shared.store.pending_memories_to_save.clear()
        for mem in pending:
            await shared.sio.emit('save_memory', mem)

    # --- NEW: Fetch memories from the microservice instead of locally ---
    smart_memories = await fetch_memories_for_prompt(memory_query, limit=5)

    try:
        context_block = await shared.prompt_constructor.construct_context_block(
            store=shared.store,
            directive=directive,
            memories=smart_memories
        )
    except Exception as e:
        print(f"❌ [Director /context] Failed to build context: {e}")
        import traceback
        traceback.print_exc()
        context_block = f"[Context build error: {e}]"

    return {
        "context": context_block,
        "mood": shared.store.current_mood,
        "scene": shared.store.current_scene.name,
        "flow": shared.store.current_flow.name,
        "conversation_state": shared.store.current_conversation_state.name,
        "summary": summary_data.get("summary", ""),
        "directive": directive.to_dict() if directive else None,
        "active_user": shared.store.active_user_profile,
        "manual_context": shared.get_manual_context(),
        "current_streamer": shared.get_current_streamer(),
    }


@app.get("/breadcrumbs")
async def breadcrumbs(count: int = 5):
    """
    Lightweight endpoint for the prompt service to poll recent high-interest events
    without triggering a full context build.

    Useful for quick checks (e.g., 'has anything important happened recently?')
    before deciding whether to call /context.
    """
    return shared.store.get_breadcrumbs(count=count)


@app.get("/thread_stats")
async def thread_stats():
    """Debug endpoint to inspect conversation thread state."""
    return shared.store.thread_manager.get_stats()


@app.get("/memory_stats")
async def memory_stats():
    """Debug endpoint to inspect memory store state."""
    
    # 1. Ask the microservice for the top 5 memories. 
    # (Sending an empty query triggers it to just return the highest importance ones)
    top_memories = await fetch_memories_for_prompt("", limit=5)
    
    with shared.store.lock:
        return {
            "total_memories": "Managed by Memory Service",
            "narrative_log_count": len(shared.store.narrative_log),
            "ancient_history_count": len(shared.store.ancient_history_log),
            "pending_saves": len(getattr(shared.store, 'pending_memories_to_save', [])),
            
            # 2. Format them exactly how the UI expects them!
            "top_memories": [
                {
                    "text": (m.get('memory_text') or m.get('text', ''))[:80],
                    "score": round(m.get('importance', 0), 3),
                    "source": "memory_service",
                }
                for m in top_memories
            ]
        }


@shared.sio.on('memory_results')
def on_memory_results(data):
    """Catches the reply from the memory_service."""
    req_id = data.get('request_id')
    if req_id in pending_memory_requests:
        # Resolve the future with the returned memories
        pending_memory_requests[req_id].set_result(data.get('memories', []))

async def fetch_memories_for_prompt(query_text: str, limit: int = 5) -> list:
    """Helper to ask the memory service for context and await the reply."""
    req_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    pending_memory_requests[req_id] = future
    
    # Send the request to the Hub
    await shared.sio.emit('query_memories', {
        'request_id': req_id, 
        'query': query_text, 
        'limit': limit
    })
    
    try:
        # Wait up to 2.5 seconds for the memory_service to reply
        results = await asyncio.wait_for(future, timeout=2.5)
        return results
    except asyncio.TimeoutError:
        print("⚠️ [Memory] Query timed out. Memory service might be offline.")
        return []
    finally:
        pending_memory_requests.pop(req_id, None)

@app.get("/store_stats")
async def store_stats():
    """Debug endpoint to inspect event store layer sizes."""
    layers = shared.store.get_all_events_for_summary()
    return {
        "immediate": len(layers["immediate"]),
        "recent": len(layers["recent"]),
        "background": len(layers["background"]),
        "narrative_log": len(shared.store.narrative_log),
        "ancient_history": len(shared.store.ancient_history_log),
        "current_mood": shared.store.current_mood,
        "current_scene": shared.store.current_scene.name,
        "current_flow": shared.store.current_flow.name,
        "conversation_state": shared.store.current_conversation_state.name,
    }


# --- Server Boot ---

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared.ui_event_loop = loop

    # Route asyncio unhandled exceptions to diagnostics/errors.log
    diagnostics.install_asyncio_handler()

    # Boot up background tasks
    loop.run_until_complete(llm_analyst.create_http_client())
    loop.create_task(connect_to_hub())
    loop.create_task(core_logic.summary_ticker())
    loop.create_task(core_logic.reflex_ticker())
    loop.create_task(core_logic.heartbeat_ticker())
    core_logic.start_thread_watchdog()
    _start_file_heartbeat()
    _start_stuck_detector()

    shared.server_ready = True
    print("✅ Director Engine is READY")
    print(f"   /context      → Full structured context for prompt service")
    print(f"   /breadcrumbs  → Lightweight recent event poll")
    print(f"   /health       → Status check")
    print(f"   /store_stats  → Event layer debug")
    print(f"   /memory_stats → Memory debug")
    print(f"   /thread_stats → Conversation thread debug")

    # Start the HTTP server required by the Nami Launcher UI
    server_config = uvicorn.Config(
        app,
        host=config.DIRECTOR_HOST,
        port=config.DIRECTOR_PORT,
        log_level="warning"
    )
    server = uvicorn.Server(server_config)
    try:
        loop.run_until_complete(server.serve())
    finally:
        loop.close()


if __name__ == "__main__":
    run_server()
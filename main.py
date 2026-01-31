# Save as: director_engine/main.py
import uvicorn
import asyncio
import threading
import webbrowser
import signal as signal_module
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import socketio

import config
from systems import process_manager
import services.llm_analyst as llm_analyst
from services.sensor_bridge import SensorBridge
import shared
import core_logic

# --- APP SETUP ---
app = FastAPI(title="Nami Director Engine")
socket_app = socketio.ASGIApp(shared.sio)
app.mount('/socket.io', socket_app)

base_path = Path(__file__).parent.resolve()
ui_path = base_path / "ui"
audio_path = base_path / "audio_effects"
# NOTE: Static mounts moved to AFTER all API routes - see below

# --- TASKS ---
summary_ticker_task: Optional[asyncio.Task] = None
reflex_ticker_task: Optional[asyncio.Task] = None
sensor_bridge_task: Optional[asyncio.Task] = None

# --- MODELS & ROUTES ---
class EventPayload(BaseModel):
    source_str: str
    text: str
    metadata: Dict[str, Any] = {}
    username: Optional[str] = None

@app.get("/audio_effects/{filename}")
async def serve_audio_effect(filename: str):
    file_path = audio_path / filename
    if not file_path.exists() or not str(file_path.resolve()).startswith(str(audio_path.resolve())):
        return {"error": "Invalid file"}
    return FileResponse(file_path, media_type="audio/wav")

@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = shared.store.get_summary()
    return summary

@app.get("/summary_data")
async def get_summary_data():
    data = shared.store.get_summary_data()
    if data['directive']: data['directive'] = data['directive'].to_dict()
    return data

@app.get("/breadcrumbs")
async def get_breadcrumbs(count: int = 3):
    """
    Returns formatted context for Nami's prompt.
    This is the bridge between Director (Brain 1) and Nami (Brain 2).
    """
    try:
        print(f"📡 [/breadcrumbs] Request received (count={count})")
        
        summary_data = shared.store.get_summary_data()
        print(f"   Summary data keys: {list(summary_data.keys())}")
        
        current_query = summary_data.get('summary', "") or " ".join(summary_data.get('topics', []))
        print(f"   Query for memories: '{current_query[:50]}...' " if current_query else "   Query: (empty)")
        
        smart_memories = shared.memory_optimizer.retrieve_relevant_memories(shared.store, current_query, limit=5)
        print(f"   Retrieved {len(smart_memories)} memories")
        
        # Get the directive - handle None case
        directive = summary_data.get('directive')
        print(f"   Directive: {type(directive).__name__ if directive else 'None'}")
        
        # Build the formatted context
        formatted_context = await shared.prompt_constructor.construct_context_block(
            shared.store, directive, smart_memories[:3]
        )
        
        print(f"   Formatted context length: {len(formatted_context)} chars")
        if len(formatted_context) < 100:
            print(f"   Full context: {formatted_context}")
        else:
            print(f"   Context preview: {formatted_context[:150]}...")
        
        return {"formatted_context": formatted_context}
        
    except Exception as e:
        print(f"❌ [/breadcrumbs] Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Return a meaningful error context instead of crashing
        return {"formatted_context": f"[Director Error: {str(e)}]"}

# --- NEW: Speech state endpoint for HTTP fallback ---
@app.get("/speech_state")
async def get_speech_state():
    return {"is_speaking": shared.is_nami_speaking()}

@app.post("/speech_started")
async def speech_started():
    shared.set_nami_speaking(True)
    return {"status": "ok"}

@app.post("/speech_finished")
async def speech_finished():
    shared.set_nami_speaking(False)
    return {"status": "ok"}

# --- NEW: Lock state endpoints ---
@app.get("/lock_states")
async def get_lock_states():
    return {
        "streamer_locked": shared.is_streamer_locked(),
        "context_locked": shared.is_context_locked()
    }

# --- IMPORTANT: Mount static files AFTER all API routes ---
# This prevents the catch-all "/" from intercepting API requests like /breadcrumbs
app.mount("/static", StaticFiles(directory=ui_path, html=True), name="static")
# The root mount MUST be last - it catches everything not matched above
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

# --- SOCKET HANDLERS ---
@shared.sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        model = EventPayload(**payload)
        await core_logic.process_engine_event(config.InputSource[model.source_str], model.text, model.metadata, model.username)
    except Exception as e:
        print(f"Invalid event payload: {e}")

@shared.sio.on("bot_reply")
async def receive_bot_reply(sid, payload: dict):
    shared.emit_bot_reply(
        payload.get('reply', ''), 
        payload.get('prompt', ''), 
        payload.get('is_censored', False),
        payload.get('censorship_reason'),
        payload.get('filtered_area')
    )

    shared.speech_dispatcher.register_user_response()

@shared.sio.on("speech_started")
async def handle_speech_started(sid, payload: dict = None):
    """Called when Nami starts speaking (TTS begins)"""
    source = payload.get('source', 'UNKNOWN') if payload else 'UNKNOWN'
    shared.set_nami_speaking(True, source=source)

@shared.sio.on("speech_finished")
async def handle_speech_finished(sid, payload: dict = None):
    """Called when Nami finishes speaking (TTS complete)"""
    shared.set_nami_speaking(False)

@shared.sio.on("set_streamer")
async def handle_set_streamer(sid, payload: dict):
    """Called when operator changes the streamer dropdown"""
    streamer_id = payload.get('streamer_id', 'peepingotter')
    shared.set_current_streamer(streamer_id, from_ai=False)

@shared.sio.on("set_manual_context")
async def handle_set_manual_context(sid, payload: dict):
    """Called when operator sets manual context (e.g., 'playing Phasmophobia')"""
    context = payload.get('context', '')
    shared.set_manual_context(context, from_ai=False)

# --- NEW: Lock state handlers ---
@shared.sio.on("set_streamer_lock")
async def handle_set_streamer_lock(sid, payload: dict):
    """Called when operator toggles the streamer lock"""
    locked = payload.get('locked', False)
    shared.set_streamer_locked(locked)

@shared.sio.on("set_context_lock")
async def handle_set_context_lock(sid, payload: dict):
    """Called when operator toggles the context lock"""
    locked = payload.get('locked', False)
    shared.set_context_locked(locked)

# --- SERVER LIFECYCLE ---
def open_browser():
    try: webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except: pass

def run_server():
    global summary_ticker_task, reflex_ticker_task, sensor_bridge_task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    shared.ui_event_loop = loop
    
    loop.run_until_complete(llm_analyst.create_http_client())
    
    # Start Tickers
    summary_ticker_task = loop.create_task(core_logic.summary_ticker())
    reflex_ticker_task = loop.create_task(core_logic.reflex_ticker())
    
    sensor_bridge = SensorBridge(event_callback=core_logic.process_engine_event)
    sensor_bridge_task = loop.create_task(sensor_bridge.run())
    
    shared.server_ready = True
    print("✅ Director Engine is READY")
    print(f"📡 API endpoints available at http://localhost:{config.DIRECTOR_PORT}/breadcrumbs")
    
    server = uvicorn.Server(uvicorn.Config(app, host=config.DIRECTOR_HOST, port=config.DIRECTOR_PORT, log_level="warning", loop="asyncio"))
    
    def signal_handler():
        print("\n⚠️ Shutdown signal received...")
        server.should_exit = True
    
    for sig in (signal_module.SIGINT, signal_module.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt")
    finally:
        print("\n🛑 DIRECTOR ENGINE SHUTDOWN INITIATED")
        shared.server_ready = False
        
        # Cleanup Tasks
        for task in [sensor_bridge_task, summary_ticker_task, reflex_ticker_task]:
            if task and not task.done():
                task.cancel()
                try: loop.run_until_complete(asyncio.wait_for(task, timeout=2.0))
                except: pass
        
        # Cleanup Services
        try: loop.run_until_complete(llm_analyst.close_http_client())
        except: pass
        try: loop.run_until_complete(shared.speech_dispatcher.close())
        except: pass
        
        process_manager.shutdown_vision_app()
        time.sleep(1.0)
        try: loop.close()
        except: pass
        print("✅ SHUTDOWN COMPLETE")

if __name__ == "__main__":
    print("🧠 DIRECTOR ENGINE (Brain 1) - Starting...")
    process_manager.launch_vision_app()
    threading.Timer(2.0, open_browser).start()
    run_server()
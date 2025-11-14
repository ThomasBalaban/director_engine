# Save as: director_engine/main.py
import uvicorn # type: ignore
import asyncio
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.responses import FileResponse, PlainTextResponse # type: ignore
from pydantic import BaseModel, Field # type: ignore
from typing import Dict, Any, List, Optional
import socketio # type: ignore

import config
import llm_analyst
from context_store import ContextStore, EventItem
from scoring import calculate_event_score

# --- Global variables ---
ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
summary_ticker_task: Optional[asyncio.Task] = None
server_ready: bool = False  # NEW: Flag to prevent emissions before loop is ready

async def summary_ticker(store: ContextStore):
    """A background task that periodically generates a new summary."""
    global server_ready
    
    # Wait for the server to be fully ready before starting
    while not server_ready:
        await asyncio.sleep(0.1)
    
    print("✅ Summary ticker starting (server is ready)")
    await asyncio.sleep(5) # Wait for initial context to build
    
    while True:
        try:
            await llm_analyst.generate_summary(store)
            summary, raw_context = store.get_summary()
            emit_current_summary(summary, raw_context)
        except Exception as e:
            print(f"[Director] Error in summary ticker: {e}")
        
        await asyncio.sleep(config.SUMMARY_INTERVAL_SECONDS)

# --- Application Setup ---
app = FastAPI(
    title="Nami Director Engine (Brain 1)",
    description="Receives, scores, and stores all context events. Provides 'breadcrumbs' to Nami (Brain 2)."
)
store = ContextStore()

# --- Socket.IO Setup ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount('/socket.io', socket_app)

# --- Static Files Setup ---
base_path = Path(__file__).parent.resolve()
ui_path = base_path / "ui"
audio_path = base_path / "audio_effects"
app.mount("/static", StaticFiles(directory=ui_path, html=True), name="static")

@app.get("/audio_effects/{filename}")
async def serve_audio_effect(filename: str):
    file_path = audio_path / filename
    if not file_path.exists() or not file_path.is_file():
        print(f"❌ Audio effect not found: {file_path}")
        return {"error": "Audio effect not found"}
    if not str(file_path.resolve()).startswith(str(audio_path.resolve())):
        print(f"🚨 Security violation: Attempted to access {file_path}")
        return {"error": "Access denied"}
    return FileResponse(file_path, media_type="audio/wav")

# --- Pydantic Models ---
class EventPayload(BaseModel):
    source_str: str = Field(..., json_schema_extra={"example": "VISUAL_CHANGE"})
    text: str = Field(..., json_schema_extra={"example": "User opened a new Chrome tab."})
    metadata: Dict[str, Any] = Field(default_factory=dict)
    username: Optional[str] = Field(None, json_schema_extra={"example": "PeepingOtter"})

class EventResponse(BaseModel):
    status: str
    event_id: str
    sieve_score: float
    analysis_queued: bool

class BreadcrumbItem(BaseModel):
    source: str
    text: str
    score: float
    timestamp: float

# --- Thread-Safe UI Emitters ---
def _emit_threadsafe(event, data):
    """
    Thread-safe function to schedule an emit call on the UI's event loop.
    """
    global ui_event_loop, server_ready
    
    # Don't emit until the server is fully ready
    if not server_ready:
        return
    
    # Check if loop exists and is not closed
    if ui_event_loop is None or ui_event_loop.is_closed():
        return
    
    try:
        # Schedule the coroutine to be executed on the UI server's event loop
        asyncio.run_coroutine_threadsafe(
            sio.emit(event, data),
            ui_event_loop
        )
    except RuntimeError as e:
        # Handle "Event loop is closed" gracefully during shutdown
        if "closed" not in str(e).lower():
            print(f"⚠️ UI Emit Error: Could not send event '{event}'. Reason: {e}")
    except Exception as e:
        # Other unexpected errors
        print(f"❌ UI Emit Error: Could not send event '{event}'. Reason: {e}")

def emit_vision_context(context): 
    _emit_threadsafe('vision_context', {'context': context})

def emit_spoken_word_context(context): 
    _emit_threadsafe('spoken_word_context', {'context': context})

def emit_audio_context(context): 
    _emit_threadsafe('audio_context', {'context': context})

def emit_twitch_message(username, message): 
    _emit_threadsafe('twitch_message', {'username': username, 'message': message})

def emit_bot_reply(reply, prompt="", is_censored=False): 
    _emit_threadsafe('bot_reply', {'reply': reply, 'prompt': prompt, 'is_censored': is_censored})

def emit_current_summary(summary: str, raw_context: str):
    _emit_threadsafe('current_summary', {'summary': summary, 'raw_context': raw_context})

# --- Socket.IO Handlers ---
@sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        payload_model = EventPayload(**payload)
        source_enum = config.InputSource[payload_model.source_str]
    except Exception as e:
        print(f"Invalid event payload received: {e}")
        return

    # Emit to UI
    if source_enum == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(payload_model.text)
    elif source_enum in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        emit_spoken_word_context(payload_model.text)
    elif source_enum == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(payload_model.text)
    elif source_enum in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION, config.InputSource.BOT_TWITCH_REPLY]:
        emit_twitch_message(payload_model.username or "Chat", payload_model.text)
    
    if source_enum == config.InputSource.BOT_TWITCH_REPLY:
        store.add_event(source_enum, payload_model.text, payload_model.metadata, 0.0)
        return
    
    sieve_score = calculate_event_score(source_enum, payload_model.metadata, config.SOURCE_WEIGHTS)
    event = store.add_event(source_enum, payload_model.text, payload_model.metadata, sieve_score)
    
    bg_tasks = BackgroundTasks()
    if sieve_score >= config.OLLAMA_TRIGGER_THRESHOLD:
        bg_tasks.add_task(llm_analyst.analyze_and_update_event, event, store)
    elif sieve_score >= config.INTERJECTION_THRESHOLD:
        print(f"[Director] Tier 2 Interjection! Sieve score {sieve_score:.2f} >= {config.INTERJECTION_THRESHOLD}")
        bg_tasks.add_task(llm_analyst.trigger_nami_interjection, event, sieve_score)

    if bg_tasks.tasks:
        task = asyncio.create_task(bg_tasks())
        await task

@sio.on("bot_reply")
async def receive_bot_reply(sid, payload: dict):
    emit_bot_reply(
        reply=payload.get('reply', ''),
        prompt=payload.get('prompt', ''),
        is_censored=payload.get('is_censored', False)
    )

# --- HTTP Endpoints ---
# *** FIX: MOVED THESE ROUTES *BEFORE* THE app.mount("/") CATCH-ALL ***
@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = store.get_summary()
    return summary

@app.get("/breadcrumbs", response_model=List[BreadcrumbItem])
async def get_breadcrumbs(count: int = 3):
    breadcrumbs = store.get_breadcrumbs(count=count)
    return breadcrumbs

# --- This must come AFTER the API routes ---
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")


# --- Browser opener ---
def open_browser():
    try: 
        webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except Exception as e: 
        print(f"Could not open browser: {e}")

def run_server():
    """Run the Uvicorn server in a blocking manner."""
    global ui_event_loop, summary_ticker_task, server_ready
    
    print("🔧 Initializing event loop...")
    
    # 1. Create and set a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # 2. Assign the loop to the global var *before* anything else
    ui_event_loop = loop
    print(f"✅ UI event loop captured: {id(ui_event_loop)}")
    
    # 3. Manually run the startup tasks that were in lifespan
    print("🔧 Initializing HTTP client and starting background tasks...")
    loop.run_until_complete(llm_analyst.create_http_client())
    
    # Start the summary ticker task
    summary_ticker_task = loop.create_task(summary_ticker(store))
    print("✅ Summary generation ticker task created")
    
    # 4. NOW mark the server as ready - this allows emissions to proceed
    server_ready = True
    print("✅ Server marked as READY - emissions now enabled")
    print("✅✅✅ DIRECTOR IS NOW READY TO ACCEPT EVENTS ✅✅✅")
    
    # 5. Configure and run the server
    config_uvicorn = uvicorn.Config(
        app, 
        host=config.DIRECTOR_HOST, 
        port=config.DIRECTOR_PORT, 
        log_level="warning",
        loop="asyncio"
    )
    server = uvicorn.Server(config_uvicorn)
    
    print(f"🚀 Starting Uvicorn server on {config.DIRECTOR_HOST}:{config.DIRECTOR_PORT}")
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("🛑 Shutting down Uvicorn server...")
        # 6. Manually run shutdown tasks
        server_ready = False  # Disable emissions during shutdown
        if summary_ticker_task:
            summary_ticker_task.cancel()
            try:
                loop.run_until_complete(summary_ticker_task)
            except asyncio.CancelledError:
                pass
        loop.run_until_complete(llm_analyst.close_http_client())
        loop.close()
        print("✅ Director Engine server shut down.")

if __name__ == "__main__":
    print("="*60)
    print("🧠 DIRECTOR ENGINE (Brain 1) - Starting...")
    print("="*60)
    print(f"📡 Port: {config.DIRECTOR_PORT}")
    print(f"🤖 Ollama Model: {config.OLLAMA_MODEL}")
    print(f"⏱️  Summary Interval: {config.SUMMARY_INTERVAL_SECONDS}s")
    print(f"🎯 Ollama Trigger: {config.OLLAMA_TRIGGER_THRESHOLD}")
    print(f"🚨 Interjection Threshold: {config.INTERJECTION_THRESHOLD}")
    print("="*60)

    try:
        # Open browser after a delay, but do it in a separate thread
        threading.Timer(2.0, open_browser).start()
        
        # Run the server (this blocks)
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
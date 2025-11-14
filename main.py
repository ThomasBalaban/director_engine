# director_engine/main.py
import uvicorn
import asyncio
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import socketio

import config
import llm_analyst
from context_store import ContextStore, EventItem
from scoring import calculate_event_score

# --- Application Setup ---
app = FastAPI(
    title="Nami Director Engine (Brain 1)",
    description="Receives, scores, and stores all context events. Provides 'breadcrumbs' to Nami (Brain 2)."
)
store = ContextStore()

# --- Socket.IO Setup (from nami/ui/server.py) ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app.mount('/socket.io', socketio.ASGIApp(sio))
ui_event_loop = None

# --- Static Files Setup (from nami/ui/server.py) ---
base_path = Path(__file__).parent.resolve()
ui_path = base_path / "ui"
audio_path = base_path / "audio_effects"

# Serve the index.html from the /ui folder at the root
app.mount("/static", StaticFiles(directory=ui_path, html=True), name="static")
app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")


@app.get("/audio_effects/{filename}")
async def serve_audio_effect(filename: str):
    """Serve audio effect files for SSML audio tags"""
    file_path = audio_path / filename
    
    if not file_path.exists() or not file_path.is_file():
        print(f"❌ Audio effect not found: {file_path}")
        return {"error": "Audio effect not found"}
    
    if not str(file_path.resolve()).startswith(str(audio_path.resolve())):
        print(f"🚨 Security violation: Attempted to access {file_path}")
        return {"error": "Access denied"}
    
    return FileResponse(file_path, media_type="audio/wav")

# --- Pydantic Models (for API data validation) ---

class EventPayload(BaseModel):
    source_str: str = Field(..., example="VISUAL_CHANGE")
    text: str = Field(..., example="User opened a new Chrome tab.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Added username for Twitch messages
    username: str = Field(None, example="PeepingOtter")

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

# --- Thread-Safe UI Emitters (from nami/ui/server.py) ---

def _emit_threadsafe(event, data):
    """A thread-safe function to schedule an emit call on the UI's event loop."""
    global ui_event_loop
    if ui_event_loop:
        try:
            asyncio.run_coroutine_threadsafe(
                sio.emit(event, data),
                ui_event_loop
            )
        except Exception as e:
            print(f"UI Emit Error: Could not send event '{event}'. Reason: {e}")
    else:
        print(f"UI Emit Warning: Event loop not ready. Dropped event '{event}'.")

def emit_log(level, message):
    _emit_threadsafe('log_message', {'level': level, 'message': message})

def emit_vision_context(context):
    _emit_threadsafe('vision_context', {'context': context})

def emit_spoken_word_context(context):
    _emit_threadsafe('spoken_word_context', {'context': context})

def emit_audio_context(context):
    _emit_threadsafe('audio_context', {'context': context})

def emit_twitch_message(username, message):
    _emit_threadsafe('twitch_message', {'username': username, 'message': message})

def emit_bot_reply(reply, prompt="", is_censored=False):
    # This app (Brain 1) does not generate replies, but we provide the
    # function in case we want to proxy Nami's replies here later.
    _emit_threadsafe('bot_reply', {'reply': reply, 'prompt': prompt, 'is_censored': is_censored})


# --- API Endpoints ---

@app.post("/event", response_model=EventResponse)
async def ingest_event(payload: EventPayload, background_tasks: BackgroundTasks):
    """
    Tier 1: Ingests a single ambient event from any source.
    """
    try:
        source_enum = config.InputSource[payload.source_str]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid source_str: {payload.source_str}")

    # --- Emit to UI in real-time ---
    # This is the new UI integration.
    if source_enum == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(payload.text)
    elif source_enum == config.InputSource.MICROPHONE:
        emit_spoken_word_context(payload.text)
    elif source_enum == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(payload.text)
    elif source_enum in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        emit_twitch_message(payload.username or "Chat", payload.text)
    # --- End UI Integration ---

    # 1. Score it with the "Sieve" (fast rules)
    sieve_score = calculate_event_score(source_enum, payload.metadata, config.SOURCE_WEIGHTS)
    
    # 2. Store it immediately
    event = store.add_event(source_enum, payload.text, payload.metadata, sieve_score)
    
    analysis_queued = False
    
    # 3. Check for "Analyst" (LLM) task
    if sieve_score >= config.OLLAMA_TRIGGER_THRESHOLD:
        print(f"[Director] Sieve score {sieve_score:.2f} >= {config.OLLAMA_TRIGGER_THRESHOLD}. Queuing LLM analysis.")
        background_tasks.add_task(llm_analyst.analyze_and_update_event, event, store)
        analysis_queued = True
        
    elif sieve_score >= config.INTERJECTION_THRESHOLD:
        # Tier 2 Interjection (Sieve-based)
        print(f"[Director] Tier 2 Interjection! Sieve score {sieve_score:.2f} >= {config.INTERJECTION_THRESHOLD}")
        background_tasks.add_task(llm_analyst.trigger_nami_interjection, event, sieve_score)

    # 4. Return the "Sieve" score *immediately*
    return EventResponse(
        status="received",
        event_id=event.id,
        sieve_score=sieve_score,
        analysis_queued=analysis_queued
    )


@app.get("/breadcrumbs", response_model=List[BreadcrumbItem])
async def get_breadcrumbs(count: int = 3):
    """
    Tier 3: Provides the Top N "most interesting" events (breadcrumbs)
    to Nami (Brain 2).
    """
    breadcrumbs = store.get_breadcrumbs(count=count)
    return breadcrumbs


# --- Server Lifecycle ---

@app.on_event("startup")
async def app_startup():
    """Get the event loop on startup for thread-safe emits."""
    global ui_event_loop
    ui_event_loop = asyncio.get_running_loop()
    print("UI event loop captured.")

@app.on_event("shutdown")
async def app_shutdown():
    """Clean up the HTTP client on shutdown."""
    await llm_analyst.close_http_client()


# --- Main entrypoint to run the server ---
def open_browser():
    """Open the web browser to the UI."""
    try:
        webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except Exception as e:
        print(f"Could not open browser: {e}")

if __name__ == "__main__":
    print(f"Starting Director Engine (Brain 1) on port {config.DIRECTOR_PORT}...")
    print(f" -> Ollama Analyst Model: {config.OLLAMA_MODEL}")
    print(f" -> Ollama Trigger Threshold: {config.OLLAMA_TRIGGER_THRESHOLD}")
    print(f" -> Interjection Threshold: {config.INTERJECTION_THRESHOLD}")

    # Open the browser 1 second after starting the server
    threading.Timer(1.0, open_browser).start()
    
    # Run the Uvicorn server
    uvicorn.run(
        app,
        host=config.DIRECTOR_HOST,
        port=config.DIRECTOR_PORT,
        log_level="info" # Use "info" or "warning"
    )
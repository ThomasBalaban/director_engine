# Save as: director_engine/main.py
import uvicorn
import asyncio
import threading
import webbrowser
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import socketio

import config
import llm_analyst
from context_store import ContextStore, EventItem
from scoring import calculate_event_score

# --- (Lifespan function is unchanged) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    global ui_event_loop
    await asyncio.sleep(0.1) 
    ui_event_loop = asyncio.get_running_loop()
    print("UI event loop captured.")
    await llm_analyst.create_http_client()
    yield
    print("Shutting down Director Engine...")
    await llm_analyst.close_http_client()

# --- Application Setup ---
app = FastAPI(
    title="Nami Director Engine (Brain 1)",
    description="Receives, scores, and stores all context events. Provides 'breadcrumbs' to Nami (Brain 2).",
    lifespan=lifespan
)
store = ContextStore()

# --- Socket.IO Setup ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount('/socket.io', socket_app)
ui_event_loop = None

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

app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")


# --- (Pydantic Models are unchanged) ---
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

# --- (UI Emitters are unchanged) ---
def _emit_threadsafe(event, data):
    global ui_event_loop
    if ui_event_loop:
        try: asyncio.run_coroutine_threadsafe(sio.emit(event, data), ui_event_loop)
        except Exception as e:
            if "event loop is closed" not in str(e): print(f"UI Emit Error: Could not send event '{event}'. Reason: {e}")
    else: pass
def emit_vision_context(context): _emit_threadsafe('vision_context', {'context': context})
def emit_spoken_word_context(context): _emit_threadsafe('spoken_word_context', {'context': context})
def emit_audio_context(context): _emit_threadsafe('audio_context', {'context': context})
def emit_twitch_message(username, message): _emit_threadsafe('twitch_message', {'username': username, 'message': message})
def emit_bot_reply(reply, prompt="", is_censored=False): _emit_threadsafe('bot_reply', {'reply': reply, 'prompt': prompt, 'is_censored': is_censored})


# --- Socket.IO Event Handlers ---

@sio.on("event")
async def ingest_event(sid, payload: dict):
    """
    Tier 1: Ingests a single ambient event via Socket.IO.
    """
    try:
        payload_model = EventPayload(**payload)
        source_enum = config.InputSource[payload_model.source_str]
    except Exception as e:
        print(f"Invalid event payload received: {e}")
        return

    # --- THIS IS THE FIX ---
    # We now check for both ambient and direct sources to update the UI.
    if source_enum == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(payload_model.text)
        
    elif source_enum in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        # Route both MICROPHONE and DIRECT_MICROPHONE to the "Spoken Word" panel
        emit_spoken_word_context(payload_model.text)
        
    elif source_enum == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(payload_model.text)
        
    elif source_enum in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        # This was already correct, but shown for clarity
        emit_twitch_message(payload_model.username or "Chat", payload_model.text)
    # --- END OF FIX ---
    
    # --- (Scoring logic is unchanged) ---
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
    """Receives a reply from Nami and forwards it to the UI."""
    emit_bot_reply(
        reply=payload.get('reply', ''),
        prompt=payload.get('prompt', ''),
        is_censored=payload.get('is_censored', False)
    )

@app.get("/breadcrumbs", response_model=List[BreadcrumbItem])
async def get_breadcrumbs(count: int = 3):
    """
    Tier 3: Provides the Top N "most interesting" events (breadcrumbs)
    to Nami (Brain 2).
    """
    breadcrumbs = store.get_breadcrumbs(count=count)
    return breadcrumbs

# --- (Main entrypoint is unchanged) ---
def open_browser():
    try: webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except Exception as e: print(f"Could not open browser: {e}")

if __name__ == "__main__":
    print(f"Starting Director Engine (Brain 1) on port {config.DIRECTOR_PORT}...")
    print(f" -> Ollama Analyst Model: {config.OLLAMA_MODEL}")
    print(f" -> Ollama Trigger Threshold: {config.OLLAMA_TRIGGER_THRESHOLD}")
    print(f" -> Interjection Threshold: {config.INTERJECTION_THRESHOLD}")
    threading.Timer(1.0, open_browser).start()
    uvicorn.run(
        "main:app",
        host=config.DIRECTOR_HOST,
        port=config.DIRECTOR_PORT,
        log_level="info"
    )
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
server_ready: bool = False
store = ContextStore()

async def summary_ticker(store: ContextStore):
    """A background task that periodically generates a new summary."""
    global server_ready
    while not server_ready:
        await asyncio.sleep(0.1)
    
    print("✅ Summary ticker starting (server is ready)")
    await asyncio.sleep(5) 
    
    while True:
        try:
            # 1. Generate new summary
            await llm_analyst.generate_summary(store)
            summary_data = store.get_summary_data()
            emit_current_summary(summary_data['summary'], summary_data['raw_context'])

            # 2. Re-analyze one stale event to make graph dynamic
            stale_event = store.get_stale_event_for_analysis()
            if stale_event:
                print(f"[Director] Ticker analyzing stale event: {stale_event.id}")
                await llm_analyst.analyze_and_update_event(stale_event, store, emit_event_scored)

        except Exception as e:
            print(f"[Director] Error in summary ticker: {e}")
        
        await asyncio.sleep(config.SUMMARY_INTERVAL_SECONDS)

# --- Application Setup ---
app = FastAPI(
    title="Nami Director Engine (Brain 1)",
    description="Receives, scores, and stores all context events. Provides 'breadcrumbs' to Nami (Brain 2)."
)
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
        return {"error": "Audio effect not found"}
    if not str(file_path.resolve()).startswith(str(audio_path.resolve())):
        return {"error": "Access denied"}
    return FileResponse(file_path, media_type="audio/wav")

# --- Pydantic Models ---
class EventPayload(BaseModel):
    source_str: str = Field(..., json_schema_extra={"example": "VISUAL_CHANGE"})
    text: str = Field(..., json_schema_extra={"example": "User opened a new Chrome tab."})
    metadata: Dict[str, Any] = Field(default_factory=dict)
    username: Optional[str] = Field(None, json_schema_extra={"example": "PeepingOtter"})

class BreadcrumbItem(BaseModel):
    source: str
    text: str
    score: float
    timestamp: float

# --- Thread-Safe UI Emitters ---
def _emit_threadsafe(event, data):
    global ui_event_loop, server_ready
    if not server_ready or ui_event_loop is None or ui_event_loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(sio.emit(event, data), ui_event_loop)
    except RuntimeError as e:
        if "closed" not in str(e).lower():
            print(f"⚠️ UI Emit Error: Could not send event '{event}'. Reason: {e}")
    except Exception as e:
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

def emit_event_scored(event: EventItem):
    """Emits a scored event to the UI for the graph."""
    _emit_threadsafe('event_scored', {
        'score': event.score,
        'timestamp': event.timestamp,
        'text': event.text,
        'source': event.source.name,
        'id': event.id # *** MODIFIED: Pass the ID for reliable UI updates ***
    })

# --- Socket.IO Handlers ---
@sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        payload_model = EventPayload(**payload)
        source_enum = config.InputSource[payload_model.source_str]
    except Exception as e:
        print(f"Invalid event payload received: {e}")
        return

    # Emit to UI context panels
    if source_enum == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(payload_model.text)
    elif source_enum in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        emit_spoken_word_context(payload_model.text)
    elif source_enum == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(payload_model.text)
    elif source_enum in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        emit_twitch_message(payload_model.username or "Chat", payload_model.text)
    
    if source_enum == config.InputSource.BOT_TWITCH_REPLY:
        store.add_event(source_enum, payload_model.text, payload_model.metadata, 0.0)
        emit_twitch_message(payload_model.username or "Nami", payload_model.text)
        return
    
    sieve_score = calculate_event_score(source_enum, payload_model.metadata, config.SOURCE_WEIGHTS)
    event = store.add_event(source_enum, payload_model.text, payload_model.metadata, sieve_score)
    
    emit_event_scored(event)
    
    bg_tasks = BackgroundTasks()
    bundle_event_created = False
    
    if source_enum in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and sieve_score >= 0.6:
        store.set_pending_speech(event)
    
    elif source_enum not in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and sieve_score >= 0.7:
        pending_speech = store.get_and_clear_pending_speech(max_age_seconds=3.0)
        
        if pending_speech:
            print(f"✅ EVENT BUNDLE: Correlated speech '{pending_speech.text}' with '{event.text}'")
            bundle_text = f"User reacted with '{pending_speech.text}' to the event: '{event.text}'"
            bundle_metadata = {**event.metadata, "is_bundle": True, "speech_text": pending_speech.text, "event_text": event.text}
            bundle_event = store.add_event(event.source, bundle_text, bundle_metadata, 1.0)
            emit_event_scored(bundle_event)
            bg_tasks.add_task(llm_analyst.analyze_and_update_event, bundle_event, store, emit_event_scored)
            bundle_event_created = True

    if not bundle_event_created:
        if sieve_score >= config.OLLAMA_TRIGGER_THRESHOLD:
            bg_tasks.add_task(llm_analyst.analyze_and_update_event, event, store, emit_event_scored)
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
@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = store.get_summary()
    return summary

@app.get("/breadcrumbs", response_model=List[BreadcrumbItem])
async def get_breadcrumbs(count: int = 3):
    breadcrumbs = store.get_breadcrumbs(count=count)
    return breadcrumbs

@app.get("/summary_data")
async def get_summary_data():
    return store.get_summary_data()

app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

# --- Server Lifecycle ---
def open_browser():
    try: 
        webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except Exception as e: 
        print(f"Could not open browser: {e}")

def run_server():
    global ui_event_loop, summary_ticker_task, server_ready
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ui_event_loop = loop
    print(f"✅ UI event loop captured: {id(ui_event_loop)}")
    
    loop.run_until_complete(llm_analyst.create_http_client())
    summary_ticker_task = loop.create_task(summary_ticker(store))
    print("✅ Summary generation ticker task created")
    
    server_ready = True
    print("✅ Server marked as READY - emissions now enabled")
    print("✅✅✅ DIRECTOR IS NOW READY TO ACCEPT EVENTS ✅✅✅")
    
    config_uvicorn = uvicorn.Config(
        app, host=config.DIRECTOR_HOST, port=config.DIRECTOR_PORT, log_level="warning", loop="asyncio"
    )
    server = uvicorn.Server(config_uvicorn)
    
    print(f"🚀 Starting Uvicorn server on {config.DIRECTOR_HOST}:{config.DIRECTOR_PORT}")
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt received")
    finally:
        print("🛑 Shutting down Uvicorn server...")
        server_ready = False
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

    threading.Timer(2.0, open_browser).start()
    run_server()
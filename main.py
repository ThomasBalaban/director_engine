# Save as: director_engine/main.py
import uvicorn # type: ignore
import asyncio
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from fastapi.responses import FileResponse, PlainTextResponse # type: ignore
from pydantic import BaseModel, Field # type: ignore
from typing import Dict, Any, List, Optional
import socketio # type: ignore

import config
import llm_analyst
from context_store import ContextStore, EventItem
from scoring import calculate_event_score
from user_profile_manager import UserProfileManager

# --- Global variables ---
ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
summary_ticker_task: Optional[asyncio.Task] = None
server_ready: bool = False

# Initialize Core Systems
store = ContextStore()
profile_manager = UserProfileManager()

async def summary_ticker(store: ContextStore):
    global server_ready
    while not server_ready:
        await asyncio.sleep(0.1)
    
    print("✅ Summary ticker starting (server is ready)")
    await asyncio.sleep(5) 
    
    while True:
        try:
            # 1. Generate new summary/prediction
            await llm_analyst.generate_summary(store)
            
            # 2. Gather all data for the "Director State" emission
            summary_data = store.get_summary_data()
            breadcrumbs_context = store.get_breadcrumbs(count=5)
            
            # Emit everything to UI
            emit_director_state(
                summary=summary_data['summary'],
                raw_context=summary_data['raw_context'],
                prediction=summary_data['prediction'],
                mood=summary_data['mood'],
                active_user=breadcrumbs_context.get('active_user'),
                memories=breadcrumbs_context.get('memories', [])
            )

            # 3. Re-analyze one stale event
            stale_event = store.get_stale_event_for_analysis()
            if stale_event:
                # Fire and forget stale analysis
                asyncio.create_task(llm_analyst.analyze_and_update_event(
                    stale_event, 
                    store, 
                    profile_manager, 
                    handle_analysis_complete
                ))

        except Exception as e:
            print(f"[Director] Error in summary ticker: {e}")
        
        await asyncio.sleep(config.SUMMARY_INTERVAL_SECONDS)

# --- Application Setup ---
app = FastAPI(title="Nami Director Engine")
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)
app.mount('/socket.io', socket_app)

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
    source_str: str
    text: str
    metadata: Dict[str, Any] = {}
    username: Optional[str] = None

# --- Emitters ---
def _emit_threadsafe(event, data):
    global ui_event_loop, server_ready
    if not server_ready or ui_event_loop is None or ui_event_loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(sio.emit(event, data), ui_event_loop)
    except RuntimeError as e:
        if "closed" not in str(e).lower():
            print(f"⚠️ UI Emit Error: {e}")

def emit_vision_context(context): _emit_threadsafe('vision_context', {'context': context})
def emit_spoken_word_context(context): _emit_threadsafe('spoken_word_context', {'context': context})
def emit_audio_context(context): _emit_threadsafe('audio_context', {'context': context})
def emit_twitch_message(username, message): _emit_threadsafe('twitch_message', {'username': username, 'message': message})
def emit_bot_reply(reply, prompt="", is_censored=False): _emit_threadsafe('bot_reply', {'reply': reply, 'prompt': prompt, 'is_censored': is_censored})

def emit_director_state(summary, raw_context, prediction, mood, active_user, memories):
    _emit_threadsafe('director_state', {
        'summary': summary, 
        'raw_context': raw_context,
        'prediction': prediction,
        'mood': mood,
        'active_user': active_user,
        'memories': memories
    })

def emit_event_scored(event: EventItem):
    _emit_threadsafe('event_scored', {
        'score': event.score,
        'timestamp': event.timestamp,
        'text': event.text,
        'source': event.source.name,
        'id': event.id 
    })

def handle_analysis_complete(event: EventItem):
    """Callback: Updates UI when analysis finishes."""
    emit_event_scored(event)
    # Refresh state to show new memories/profile updates
    summary_data = store.get_summary_data()
    breadcrumbs = store.get_breadcrumbs(count=5)
    emit_director_state(
        summary_data['summary'], 
        summary_data['raw_context'], 
        summary_data['prediction'],
        summary_data['mood'], 
        breadcrumbs.get('active_user'), 
        breadcrumbs.get('memories', [])
    )

# --- Socket.IO Handlers ---
@sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        payload_model = EventPayload(**payload)
        source_enum = config.InputSource[payload_model.source_str]
    except Exception as e:
        print(f"Invalid event payload: {e}")
        return

    # 1. Instant UI Updates (Unblocked!)
    if source_enum == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(payload_model.text)
    elif source_enum in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        emit_spoken_word_context(payload_model.text)
    elif source_enum == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(payload_model.text)
    elif source_enum in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        emit_twitch_message(payload_model.username or "Chat", payload_model.text)
    
    # 2. User Profile
    if payload_model.username:
        profile = profile_manager.get_profile(payload_model.username)
        store.set_active_user(profile)

    if source_enum == config.InputSource.BOT_TWITCH_REPLY:
        store.add_event(source_enum, payload_model.text, payload_model.metadata, 0.0)
        emit_twitch_message(payload_model.username or "Nami", payload_model.text)
        return
    
    # 3. Scoring
    sieve_score = calculate_event_score(source_enum, payload_model.metadata, config.SOURCE_WEIGHTS)
    event = store.add_event(source_enum, payload_model.text, payload_model.metadata, sieve_score)
    
    emit_event_scored(event)
    
    # 4. Analysis (Fire and Forget - No Await!)
    bundle_event_created = False
    
    # Bundling logic
    if source_enum in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and sieve_score >= 0.6:
        store.set_pending_speech(event)
    
    elif source_enum not in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and sieve_score >= 0.7:
        pending_speech = store.get_and_clear_pending_speech(max_age_seconds=3.0)
        if pending_speech:
            print(f"✅ EVENT BUNDLE: {pending_speech.text} + {event.text}")
            bundle_text = f"User reacted with '{pending_speech.text}' to: '{event.text}'"
            bundle_metadata = {**event.metadata, "is_bundle": True, "speech_text": pending_speech.text, "event_text": event.text}
            bundle_event = store.add_event(event.source, bundle_text, bundle_metadata, 1.0)
            emit_event_scored(bundle_event)
            
            # Async Task
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                bundle_event, store, profile_manager, handle_analysis_complete
            ))
            bundle_event_created = True

    if not bundle_event_created:
        if sieve_score >= config.OLLAMA_TRIGGER_THRESHOLD:
            # Async Task
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                event, store, profile_manager, handle_analysis_complete
            ))
        elif sieve_score >= config.INTERJECTION_THRESHOLD:
             asyncio.create_task(llm_analyst.trigger_nami_interjection(event, sieve_score))

@sio.on("bot_reply")
async def receive_bot_reply(sid, payload: dict):
    emit_bot_reply(
        payload.get('reply', ''),
        payload.get('prompt', ''),
        payload.get('is_censored', False)
    )

# --- HTTP Endpoints ---
@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = store.get_summary()
    return summary

@app.get("/breadcrumbs")
async def get_breadcrumbs(count: int = 3):
    return store.get_breadcrumbs(count=count)

@app.get("/summary_data")
async def get_summary_data():
    return store.get_summary_data()

app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

# --- Server Lifecycle ---
def open_browser():
    try: webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except: pass

def run_server():
    global ui_event_loop, summary_ticker_task, server_ready
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ui_event_loop = loop
    print(f"✅ UI event loop captured: {id(ui_event_loop)}")
    
    # Initialize Async Clients
    loop.run_until_complete(llm_analyst.create_http_client())
    
    summary_ticker_task = loop.create_task(summary_ticker(store))
    server_ready = True
    print("✅ Director Engine is READY")
    
    config_uvicorn = uvicorn.Config(app, host=config.DIRECTOR_HOST, port=config.DIRECTOR_PORT, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config_uvicorn)
    
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt")
    finally:
        print("🛑 Shutting down...")
        server_ready = False
        if summary_ticker_task:
            summary_ticker_task.cancel()
            try: loop.run_until_complete(summary_ticker_task)
            except: pass
        loop.run_until_complete(llm_analyst.close_http_client())
        loop.close()

if __name__ == "__main__":
    print("="*60)
    print("🧠 DIRECTOR ENGINE (Brain 1) - Starting...")
    print("="*60)
    threading.Timer(2.0, open_browser).start()
    run_server()
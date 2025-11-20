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
from scoring import calculate_event_score, EventScore
from user_profile_manager import UserProfileManager
from adaptive_controller import AdaptiveController
from correlation_engine import CorrelationEngine
from energy_system import EnergySystem # [NEW]

# --- Global variables ---
ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
summary_ticker_task: Optional[asyncio.Task] = None
server_ready: bool = False

# Initialize Core Systems
store = ContextStore()
profile_manager = UserProfileManager()
adaptive_ctrl = AdaptiveController()
correlation_engine = CorrelationEngine()
energy_system = EnergySystem() # [NEW]

async def summary_ticker(store: ContextStore):
    global server_ready
    while not server_ready:
        await asyncio.sleep(0.1)
    
    print("✅ Summary ticker starting (server is ready)")
    await asyncio.sleep(5) 
    
    while True:
        try:
            # 1. Generate new summary/prediction/state
            await llm_analyst.generate_summary(store)
            
            # 2. Update Adaptive Thresholds
            chat_vel, energy_level = store.get_activity_metrics()
            new_threshold = adaptive_ctrl.update(chat_vel, energy_level)
            
            # 3. Run Cross-Event Correlation
            patterns = correlation_engine.correlate(store)
            for pat in patterns:
                print(f"🧩 [Correlation] {pat['text']}")
                sys_event = store.add_event(
                    config.InputSource.SYSTEM_PATTERN, 
                    pat['text'], 
                    pat['metadata'], 
                    pat['score']
                )
                emit_event_scored(sys_event)
                
                # Check if pattern warrants Nami speaking? 
                # Usually patterns are just context, but "Engagement Void" might need action.
                # For now, we leave that to the "Interjection" logic below.

            # 4. Gather all data for the "Director State" emission
            summary_data = store.get_summary_data()
            breadcrumbs_context = store.get_breadcrumbs(count=5)
            
            emit_director_state(
                summary=summary_data['summary'],
                raw_context=summary_data['raw_context'],
                prediction=summary_data['prediction'],
                mood=summary_data['mood'],
                conversation_state=summary_data['conversation_state'],
                flow_state=summary_data['flow'],      # [NEW]
                user_intent=summary_data['intent'],   # [NEW]
                active_user=breadcrumbs_context.get('active_user'),
                memories=breadcrumbs_context.get('memories', []),
                adaptive_state={
                    "threshold": round(new_threshold, 2),
                    "state": adaptive_ctrl.state_label,
                    "chat_velocity": round(chat_vel, 1),
                    "energy": round(energy_level, 2),
                    "social_battery": energy_system.get_status() # [NEW]
                }
            )

            # 5. Re-analyze one stale event
            stale_event = store.get_stale_event_for_analysis()
            if stale_event:
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

def emit_director_state(summary, raw_context, prediction, mood, conversation_state, flow_state, user_intent, active_user, memories, adaptive_state=None):
    _emit_threadsafe('director_state', {
        'summary': summary, 
        'raw_context': raw_context,
        'prediction': prediction,
        'mood': mood,
        'conversation_state': conversation_state,
        'flow': flow_state,       # [NEW]
        'intent': user_intent,    # [NEW]
        'active_user': active_user,
        'memories': memories,
        'adaptive': adaptive_state or {}
    })

def emit_event_scored(event: EventItem):
    _emit_threadsafe('event_scored', {
        'score': event.score.interestingness, 
        'scores': event.score.to_dict(),
        'timestamp': event.timestamp,
        'text': event.text,
        'source': event.source.name,
        'id': event.id 
    })

def handle_analysis_complete(event: EventItem):
    emit_event_scored(event)
    summary_data = store.get_summary_data()
    breadcrumbs = store.get_breadcrumbs(count=5)
    emit_director_state(
        summary_data['summary'], 
        summary_data['raw_context'], 
        summary_data['prediction'],
        summary_data['mood'], 
        summary_data['conversation_state'],
        summary_data['flow'],
        summary_data['intent'],
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

    if source_enum == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(payload_model.text)
    elif source_enum in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        emit_spoken_word_context(payload_model.text)
    elif source_enum == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(payload_model.text)
    elif source_enum in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        emit_twitch_message(payload_model.username or "Chat", payload_model.text)
    
    if payload_model.username:
        profile = profile_manager.get_profile(payload_model.username)
        store.set_active_user(profile)

    if source_enum == config.InputSource.BOT_TWITCH_REPLY:
        zero_score = EventScore()
        store.add_event(source_enum, payload_model.text, payload_model.metadata, zero_score)
        emit_twitch_message(payload_model.username or "Nami", payload_model.text)
        
        # [ENERGY COST] Reply is cheap but not free
        energy_system.spend(config.ENERGY_COST_REPLY)
        return
    
    heuristic_score: EventScore = calculate_event_score(source_enum, payload_model.metadata, config.SOURCE_WEIGHTS)
    event = store.add_event(source_enum, payload_model.text, payload_model.metadata, heuristic_score)
    
    emit_event_scored(event)
    
    bundle_event_created = False
    primary_score = heuristic_score.interestingness

    if source_enum in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and primary_score >= 0.6:
        store.set_pending_speech(event)
    elif source_enum not in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and primary_score >= 0.7:
        pending_speech = store.get_and_clear_pending_speech(max_age_seconds=3.0)
        if pending_speech:
            print(f"✅ EVENT BUNDLE: {pending_speech.text} + {event.text}")
            bundle_text = f"User reacted with '{pending_speech.text}' to: '{event.text}'"
            bundle_metadata = {**event.metadata, "is_bundle": True, "speech_text": pending_speech.text, "event_text": event.text}
            bundle_score = EventScore(interestingness=1.0, urgency=0.9, conversational_value=1.0, topic_relevance=1.0)
            bundle_event = store.add_event(event.source, bundle_text, bundle_metadata, bundle_score)
            emit_event_scored(bundle_event)
            
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                bundle_event, store, profile_manager, handle_analysis_complete
            ))
            bundle_event_created = True

    if not bundle_event_created:
        if primary_score >= config.OLLAMA_TRIGGER_THRESHOLD:
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                event, store, profile_manager, handle_analysis_complete
            ))
        elif heuristic_score.urgency >= adaptive_ctrl.current_threshold:
             # [ENERGY CHECK] Can we afford this interjection?
             if energy_system.can_afford(config.ENERGY_COST_INTERJECTION):
                 # We analyze first. If analysis confirms high score, THEN we interject.
                 # The actual interjection HTTP call happens inside `llm_analyst.analyze_and_update_event` -> `trigger_nami_interjection`
                 # We should pass the energy system to that flow or check here?
                 # For simplicity: We check budget here. If we have budget, we ALLOW the analysis that MIGHT lead to interjection.
                 # If it actually triggers, we spend the energy then.
                 
                 # Wait, `trigger_nami_interjection` is called inside `analyze_and_update_event`.
                 # Let's modify `trigger_nami_interjection` to check/spend energy.
                 # Ideally we pass energy_system dependency, but for now let's just allow the analysis.
                 asyncio.create_task(llm_analyst.analyze_and_update_event(
                    event, store, profile_manager, handle_analysis_complete
                ))

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
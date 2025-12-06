# Save as: director_engine/main.py
import uvicorn
import asyncio
import threading
import webbrowser
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import socketio

import config
import llm_analyst
from context_store import ContextStore, EventItem
from scoring import calculate_event_score, EventScore
from user_profile_manager import UserProfileManager
from adaptive_controller import AdaptiveController
from correlation_engine import CorrelationEngine
from energy_system import EnergySystem
from behavior_engine import BehaviorEngine
from memory_ops import MemoryOptimizer
from context_compression import ContextCompressor
from scene_manager import SceneManager
from decision_engine import DecisionEngine
from prompt_constructor import PromptConstructor
from sensor_bridge import SensorBridge  # <--- NEW IMPORT

ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
summary_ticker_task: Optional[asyncio.Task] = None
sensor_bridge_task: Optional[asyncio.Task] = None # <--- NEW TASK
server_ready: bool = False

# --- INIT ENGINES ---
store = ContextStore()
profile_manager = UserProfileManager()
adaptive_ctrl = AdaptiveController()
correlation_engine = CorrelationEngine()
energy_system = EnergySystem()
behavior_engine = BehaviorEngine()
memory_optimizer = MemoryOptimizer()
context_compressor = ContextCompressor()
scene_manager = SceneManager()
decision_engine = DecisionEngine()
prompt_constructor = PromptConstructor()

# --- SHARED EVENT PROCESSOR ---
# This is the core logic that handles ALL inputs (SocketIO or Sensor Bridge)
async def process_engine_event(source: config.InputSource, text: str, metadata: Dict[str, Any] = {}, username: Optional[str] = None):
    # 1. UI Emit (Raw Data)
    if source == config.InputSource.VISUAL_CHANGE:
        emit_vision_context(text)
    elif source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
        emit_spoken_word_context(text)
    elif source == config.InputSource.AMBIENT_AUDIO:
        emit_audio_context(text)
    elif source in [config.InputSource.TWITCH_CHAT, config.InputSource.TWITCH_MENTION]:
        emit_twitch_message(username or "Chat", text)
    
    # 2. User Profile Update
    if username:
        profile = profile_manager.get_profile(username)
        store.set_active_user(profile)

    # 3. Handle Bot Self-Reply (Echo)
    if source == config.InputSource.BOT_TWITCH_REPLY:
        zero_score = EventScore()
        store.add_event(source, text, metadata, zero_score)
        emit_twitch_message(username or "Nami", text)
        behavior_engine.register_bot_action(store, text)
        energy_system.spend(config.ENERGY_COST_REPLY)
        return
    
    # 4. Scoring & Storage
    heuristic_score: EventScore = calculate_event_score(source, metadata, config.SOURCE_WEIGHTS)
    event = store.add_event(source, text, metadata, heuristic_score)
    emit_event_scored(event)
    
    # 5. Debt Check (Did user answer a question?)
    if source in [config.InputSource.MICROPHONE, config.InputSource.DIRECT_MICROPHONE]:
         behavior_engine.check_debt_resolution(store, text)

    # 6. Event Bundling (Linking Speech to Visuals)
    bundle_event_created = False
    primary_score = heuristic_score.interestingness

    if source in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and primary_score >= 0.6:
        store.set_pending_speech(event)
    elif source not in [config.InputSource.DIRECT_MICROPHONE, config.InputSource.MICROPHONE] and primary_score >= 0.7:
        # If we see something interesting right after hearing something, link them
        pending_speech = store.get_and_clear_pending_speech(max_age_seconds=3.0)
        if pending_speech:
            print(f"✅ EVENT BUNDLE: {pending_speech.text} + {event.text}")
            bundle_text = f"User reacted with '{pending_speech.text}' to: '{event.text}'"
            bundle_metadata = {**metadata, "is_bundle": True, "speech_text": pending_speech.text, "event_text": event.text}
            bundle_score = EventScore(interestingness=1.0, urgency=0.9, conversational_value=1.0, topic_relevance=1.0)
            bundle_event = store.add_event(event.source, bundle_text, bundle_metadata, bundle_score)
            emit_event_scored(bundle_event)
            # Immediate Analysis on Bundles
            asyncio.create_task(llm_analyst.analyze_and_update_event(
                bundle_event, store, profile_manager, handle_analysis_complete
            ))
            bundle_event_created = True

    # 7. Attention & Analysis Trigger
    if not bundle_event_created:
        attended_event = behavior_engine.direct_attention(store, [event])
        if attended_event:
            # Analyze if high score OR if urgent enough to break silence
            if primary_score >= config.OLLAMA_TRIGGER_THRESHOLD:
                asyncio.create_task(llm_analyst.analyze_and_update_event(
                    event, store, profile_manager, handle_analysis_complete
                ))
            elif heuristic_score.urgency >= adaptive_ctrl.current_threshold:
                 if energy_system.can_afford(config.ENERGY_COST_INTERJECTION):
                     asyncio.create_task(llm_analyst.analyze_and_update_event(
                        event, store, profile_manager, handle_analysis_complete
                    ))

async def summary_ticker(store: ContextStore):
    global server_ready
    while not server_ready:
        await asyncio.sleep(0.1)
    
    print("✅ Summary ticker starting (server is ready)")
    await asyncio.sleep(5) 
    
    while True:
        try:
            await llm_analyst.generate_summary(store)
            
            chat_vel, energy_level = store.get_activity_metrics()
            new_threshold = adaptive_ctrl.update(chat_vel, energy_level)
            adaptive_ctrl.process_feedback(store) 
            
            scene_manager.update_scene(store)
            
            patterns = correlation_engine.correlate(store)
            for pat in patterns:
                sys_event = store.add_event(
                    config.InputSource.SYSTEM_PATTERN, 
                    pat['text'], 
                    pat['metadata'], 
                    pat['score']
                )
                emit_event_scored(sys_event)

            behavior_engine.update_goal(store)
            
            directive = decision_engine.generate_directive(
                store, behavior_engine, adaptive_ctrl, energy_system
            )
            store.set_directive(directive) 
            
            # Curiosity & Callbacks (Bot Thoughts)
            thought_text = behavior_engine.check_curiosity(store, profile_manager)
            if thought_text:
                print(f"💡 [Curiosity] {thought_text}")
                thought_event = store.add_event(
                    config.InputSource.INTERNAL_THOUGHT,
                    thought_text,
                    {"type": "curiosity", "goal": behavior_engine.current_goal.name},
                    EventScore(interestingness=0.6, conversational_value=0.9)
                )
                emit_event_scored(thought_event)
                if energy_system.can_afford(config.ENERGY_COST_INTERJECTION):
                     asyncio.create_task(llm_analyst.analyze_and_update_event(
                        thought_event, store, profile_manager, handle_analysis_complete
                    ))

            callback_text = behavior_engine.check_callbacks(store)
            if callback_text:
                print(f"🕰️ [Callback] {callback_text}")
                cb_event = store.add_event(
                    config.InputSource.INTERNAL_THOUGHT,
                    callback_text,
                    {"type": "callback", "goal": "context_continuity"},
                    EventScore(interestingness=0.7, conversational_value=0.8)
                )
                emit_event_scored(cb_event)
                if energy_system.can_afford(config.ENERGY_COST_INTERJECTION):
                     asyncio.create_task(llm_analyst.analyze_and_update_event(
                        cb_event, store, profile_manager, handle_analysis_complete
                    ))
            
            memory_optimizer.decay_memories(store)
            await context_compressor.run_compression_cycle(store)

            summary_data = store.get_summary_data()
            
            current_context_query = summary_data.get('summary', "")
            if not current_context_query or len(current_context_query) < 5:
                active_topics = summary_data.get('topics', [])
                current_context_query = " ".join(active_topics)

            smart_memories = memory_optimizer.retrieve_relevant_memories(
                store, 
                current_context_query,
                limit=5
            )
            
            memories_list = [{
                "source": m.source.name, 
                "text": m.memory_text or m.text, 
                "score": round(m.score.interestingness, 2), 
                "type": "memory"
            } for m in smart_memories]

            if store.narrative_log:
                recent_history = list(reversed(store.narrative_log[-3:]))
                for i, story in enumerate(recent_history):
                    memories_list.insert(i, {
                        "source": "NARRATIVE_HISTORY",
                        "text": f"Previously: {story}",
                        "score": 1.0, 
                        "type": "narrative"
                    })

            emit_director_state(
                summary=summary_data['summary'],
                raw_context=summary_data['raw_context'],
                prediction=summary_data['prediction'],
                mood=summary_data['mood'],
                conversation_state=summary_data['conversation_state'],
                flow_state=summary_data['flow'],
                user_intent=summary_data['intent'],
                active_user=store.active_user_profile,
                memories=memories_list,
                directive=summary_data['directive'].to_dict() if summary_data['directive'] else None,
                adaptive_state={
                    "threshold": round(new_threshold, 2),
                    "state": adaptive_ctrl.state_label,
                    "chat_velocity": round(chat_vel, 1),
                    "energy": round(energy_level, 2),
                    "social_battery": energy_system.get_status(),
                    "current_goal": behavior_engine.current_goal.name,
                    "current_scene": summary_data['scene']
                }
            )

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

class EventPayload(BaseModel):
    source_str: str
    text: str
    metadata: Dict[str, Any] = {}
    username: Optional[str] = None

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

def emit_director_state(summary, raw_context, prediction, mood, conversation_state, flow_state, user_intent, active_user, memories, directive, adaptive_state=None):
    _emit_threadsafe('director_state', {
        'summary': summary, 
        'raw_context': raw_context,
        'prediction': prediction,
        'mood': mood,
        'conversation_state': conversation_state,
        'flow': flow_state,
        'intent': user_intent,
        'active_user': active_user,
        'memories': memories,
        'directive': directive, 
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

# --- SOCKET.IO HANDLER ---
@sio.on("event")
async def ingest_event(sid, payload: dict):
    try:
        payload_model = EventPayload(**payload)
        source_enum = config.InputSource[payload_model.source_str]
    except Exception as e:
        print(f"Invalid event payload: {e}")
        return
    
    # Use shared processor
    await process_engine_event(
        source_enum, 
        payload_model.text, 
        payload_model.metadata, 
        payload_model.username
    )

@sio.on("bot_reply")
async def receive_bot_reply(sid, payload: dict):
    emit_bot_reply(
        payload.get('reply', ''),
        payload.get('prompt', ''),
        payload.get('is_censored', False)
    )

@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = store.get_summary()
    return summary

@app.get("/breadcrumbs")
async def get_breadcrumbs(count: int = 3):
    summary_data = store.get_summary_data()
    
    current_context_query = summary_data.get('summary', "")
    if not current_context_query or len(current_context_query) < 5:
        active_topics = summary_data.get('topics', [])
        current_context_query = " ".join(active_topics)
    
    smart_memories = memory_optimizer.retrieve_relevant_memories(
        store, 
        current_context_query, 
        limit=5
    )
    
    directive_obj = summary_data.get('directive')
    
    formatted_context = await prompt_constructor.construct_context_block(
        store, 
        directive_obj,
        smart_memories[:3] 
    )
    
    return {"formatted_context": formatted_context}

@app.get("/summary_data")
async def get_summary_data():
    data = store.get_summary_data()
    if data['directive']:
        data['directive'] = data['directive'].to_dict()
    return data

app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

def open_browser():
    try: webbrowser.open(f"http://localhost:{config.DIRECTOR_PORT}")
    except: pass

def run_server():
    global ui_event_loop, summary_ticker_task, server_ready, sensor_bridge_task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ui_event_loop = loop
    print(f"✅ UI event loop captured: {id(ui_event_loop)}")
    
    loop.run_until_complete(llm_analyst.create_http_client())
    
    # 1. Start Summary Ticker
    summary_ticker_task = loop.create_task(summary_ticker(store))
    
    # 2. Start Sensor Bridge (Connect to Vision App)
    sensor_bridge = SensorBridge(event_callback=process_engine_event)
    sensor_bridge_task = loop.create_task(sensor_bridge.run())
    
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
        if sensor_bridge_task:
            sensor_bridge_task.cancel()
        loop.run_until_complete(llm_analyst.close_http_client())
        loop.close()

if __name__ == "__main__":
    print("="*60)
    print("🧠 DIRECTOR ENGINE (Brain 1) - Starting...")
    print("="*60)
    threading.Timer(2.0, open_browser).start()
    run_server()
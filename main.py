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
from energy_system import EnergySystem
from behavior_engine import BehaviorEngine
from memory_ops import MemoryOptimizer
from context_compression import ContextCompressor
from scene_manager import SceneManager

ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
summary_ticker_task: Optional[asyncio.Task] = None
server_ready: bool = False

store = ContextStore()
profile_manager = UserProfileManager()
adaptive_ctrl = AdaptiveController()
correlation_engine = CorrelationEngine()
energy_system = EnergySystem()
behavior_engine = BehaviorEngine()
memory_optimizer = MemoryOptimizer()
context_compressor = ContextCompressor()
scene_manager = SceneManager()

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
            
            # 3. Update Scene
            scene_manager.update_scene(store)
            
            # 4. Run Cross-Event Correlation
            patterns = correlation_engine.correlate(store)
            for pat in patterns:
                sys_event = store.add_event(
                    config.InputSource.SYSTEM_PATTERN, 
                    pat['text'], 
                    pat['metadata'], 
                    pat['score']
                )
                emit_event_scored(sys_event)

            # 5. Run Behavioral Systems
            behavior_engine.update_goal(store)
            
            thought_text = behavior_engine.check_curiosity(store)
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
            
            # 6. Memory & Compression
            memory_optimizer.decay_memories(store)
            await context_compressor.run_compression_cycle(store)

            # 7. Gather Data & Emit
            summary_data = store.get_summary_data()
            active_topics = summary_data.get('topics', [])
            smart_memories = memory_optimizer.retrieve_relevant_memories(store, active_topics)
            
            memories_list = [{
                "source": m.source.name, 
                "text": m.memory_text or m.text, 
                "score": round(m.score.interestingness, 2), 
                "type": "memory"
            } for m in smart_memories]

            # [FIX] Inject Narrative Log OR Fallback Summary
            if store.narrative_log:
                recent_history = list(reversed(store.narrative_log[-3:]))
                
                for i, story in enumerate(recent_history):
                    memories_list.insert(i, {
                        "source": "NARRATIVE_HISTORY",
                        "text": f"Previously: {story}",
                        "score": 1.0, # Always show at top
                        "type": "narrative"
                    })
            elif not memories_list and summary_data['summary']:
                # If absolutely nothing else is in memory, show the current summary
                # so the panel doesn't look broken.
                memories_list.append({
                    "source": "WORKING_MEMORY",
                    "text": f"Current Context: {summary_data['summary']}",
                    "score": 0.5,
                    "type": "working_memory"
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

            # 8. Stale Analysis
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

def emit_director_state(summary, raw_context, prediction, mood, conversation_state, flow_state, user_intent, active_user, memories, adaptive_state=None):
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
             if energy_system.can_afford(config.ENERGY_COST_INTERJECTION):
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

@app.get("/summary", response_class=PlainTextResponse)
async def get_summary():
    summary, _ = store.get_summary()
    return summary

@app.get("/breadcrumbs")
async def get_breadcrumbs(count: int = 3):
    data = store.get_breadcrumbs(count)
    summary_data = store.get_summary_data()
    active_topics = summary_data.get('topics', [])
    smart_memories = memory_optimizer.retrieve_relevant_memories(store, active_topics)
    
    data['memories'] = [{
        "source": m.source.name, 
        "text": m.memory_text or m.text, 
        "score": round(m.score.interestingness, 2), 
        "type": "memory"
    } for m in smart_memories]
    
    if store.narrative_log:
        last_story = store.narrative_log[-1]
        data['memories'].insert(0, {
            "source": "NARRATIVE_HISTORY",
            "text": f"Previously: {last_story}",
            "score": 1.0,
            "type": "narrative"
        })
    elif summary_data['summary']:
         data['memories'].insert(0, {
            "source": "WORKING_MEMORY",
            "text": f"Current Context: {summary_data['summary']}",
            "score": 0.5,
            "type": "working_memory"
        })
        
    data['bot_goal'] = behavior_engine.current_goal.name
    
    return data

@app.get("/summary_data")
async def get_summary_data():
    return store.get_summary_data()

app.mount("/", StaticFiles(directory=ui_path, html=True), name="ui_static")

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
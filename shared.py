# Save as: director_engine/shared.py
import asyncio
import socketio
from typing import Dict, Any, List, Optional
import config
from context_store import ContextStore, EventItem
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
from speech_dispatcher import SpeechDispatcher

# --- GLOBAL SINGLETONS ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
server_ready: bool = False

# --- ENGINE INITIALIZATION ---
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
speech_dispatcher = SpeechDispatcher()

# --- EMITTERS ---
def _emit_threadsafe(event, data):
    global ui_event_loop, server_ready
    if not server_ready or ui_event_loop is None or ui_event_loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(sio.emit(event, data), ui_event_loop)
    except RuntimeError as e:
        if "closed" not in str(e).lower():
            print(f"⚠️ UI Emit Error: {e}")

def emit_vision_context(context): 
    _emit_threadsafe('vision_context', {'context': context})

def emit_spoken_word_context(context): 
    _emit_threadsafe('spoken_word_context', {'context': context})

def emit_audio_context(context, is_partial=False): 
    _emit_threadsafe('audio_context', {'context': context, 'is_partial': is_partial})

def emit_twitch_message(username, message): 
    _emit_threadsafe('twitch_message', {'username': username, 'message': message})

def emit_bot_reply(reply, prompt="", is_censored=False): 
    _emit_threadsafe('bot_reply', {'reply': reply, 'prompt': prompt, 'is_censored': is_censored})

def emit_event_scored(event: EventItem):
    _emit_threadsafe('event_scored', {
        'score': event.score.interestingness, 
        'scores': event.score.to_dict(),
        'timestamp': event.timestamp,
        'text': event.text,
        'source': event.source.name,
        'id': event.id 
    })

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
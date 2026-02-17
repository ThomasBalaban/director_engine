# Save as: director_engine/shared.py
"""
Shared state and singletons for the Director Engine.

NOTE: All speaking state (is_nami_speaking, cooldowns, interrupts)
has been moved to the Prompt Service (port 8001).
The brain no longer tracks whether Nami is speaking.
"""

import asyncio
import socketio
import time
import traceback
from typing import Dict, Any, List, Optional
import config
from context.context_store import ContextStore, EventItem
from context.user_profile_manager import UserProfileManager
from systems.adaptive_controller import AdaptiveController
from systems.correlation_engine import CorrelationEngine
from systems.energy_system import EnergySystem
from systems.behavior_engine import BehaviorEngine
from context.memory_ops import MemoryOptimizer
from context.context_compression import ContextCompressor
from systems.scene_manager import SceneManager
from systems.decision_engine import DecisionEngine
from services.prompt_constructor import PromptConstructor
from services.speech_dispatcher import SpeechDispatcher

# --- GLOBAL SINGLETONS ---
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
server_ready: bool = False

# --- DIRECTOR MANUAL CONTEXT ---
manual_context: str = ""
current_streamer: str = "peepingotter"

# --- Lock states for AI auto-fill ---
streamer_locked: bool = False
context_locked: bool = False

def set_manual_context(context: str, from_ai: bool = False):
    global manual_context, context_locked
    if context_locked and from_ai:
        print(f"🔒 [Director] Context locked - AI update ignored")
        return False
    manual_context = context
    source = "AI" if from_ai else "Manual"
    print(f"📝 [Director] Context set ({source}): {context[:50]}..." if context else f"📝 [Director] Context cleared ({source})")
    return True

def get_manual_context() -> str:
    global manual_context
    return manual_context

def set_current_streamer(streamer_id: str, from_ai: bool = False):
    global current_streamer, streamer_locked
    if streamer_locked and from_ai:
        print(f"🔒 [Director] Streamer locked - AI update ignored")
        return False
    current_streamer = streamer_id
    source = "AI" if from_ai else "Manual"
    print(f"📺 [Director] Now watching ({source}): {streamer_id}")
    return True

def get_current_streamer() -> str:
    global current_streamer
    return current_streamer

def set_streamer_locked(locked: bool):
    global streamer_locked
    streamer_locked = locked
    print(f"{'🔒' if locked else '🔓'} [Director] Streamer lock: {locked}")

def set_context_locked(locked: bool):
    global context_locked
    context_locked = locked
    print(f"{'🔒' if locked else '🔓'} [Director] Context lock: {locked}")

def is_streamer_locked() -> bool:
    global streamer_locked
    return streamer_locked

def is_context_locked() -> bool:
    global context_locked
    return context_locked


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
def _emit_threadsafe(event: str, data: dict):
    """Thread-safe emit with better error handling."""
    global ui_event_loop, server_ready, sio
    
    if not server_ready:
        return
        
    if ui_event_loop is None:
        return
        
    try:
        if ui_event_loop.is_closed():
            return
    except Exception:
        return
    
    try:
        future = asyncio.run_coroutine_threadsafe(sio.emit(event, data), ui_event_loop)
    except RuntimeError as e:
        if "closed" not in str(e).lower() and server_ready:
            print(f"⚠️ UI Emit Error ({event}): {e}")
    except Exception as e:
        if server_ready:
            print(f"⚠️ UI Emit Error ({event}): {type(e).__name__}: {e}")

def emit_vision_context(context): 
    _emit_threadsafe('vision_context', {'context': context})

def emit_spoken_word_context(context): 
    _emit_threadsafe('spoken_word_context', {'context': context})

def emit_audio_context(context, is_partial=False): 
    _emit_threadsafe('audio_context', {'context': context, 'is_partial': is_partial})

def emit_twitch_message(username, message): 
    _emit_threadsafe('twitch_message', {'username': username, 'message': message})

def emit_bot_reply(reply, prompt="", is_censored=False, censorship_reason=None, filtered_area=None): 
    _emit_threadsafe('bot_reply', {
        'reply': reply, 
        'prompt': prompt, 
        'is_censored': is_censored, 
        'censorship_reason': censorship_reason,
        'filtered_area': filtered_area 
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

def emit_ai_context_suggestion(streamer: str = None, context: str = None):
    _emit_threadsafe('ai_context_suggestion', {
        'streamer': streamer,
        'context': context,
        'streamer_locked': is_streamer_locked(),
        'context_locked': is_context_locked()
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
        'adaptive': adaptive_state or {},
        'manual_context': get_manual_context(),
        'current_streamer': get_current_streamer(),
        'streamer_locked': is_streamer_locked(),
        'context_locked': is_context_locked()
    })
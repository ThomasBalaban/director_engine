# Save as: director_engine/shared.py
import asyncio
import socketio
import time
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

# --- SPEECH STATE TRACKING ---
# Prevents Director from sending interjections while Nami is speaking
nami_is_speaking: bool = False
speech_started_time: float = 0.0
SPEECH_TIMEOUT: float = 60.0  # Max time to wait for speech_finished (failsafe)
last_speech_source: Optional[str] = None  # 'USER_DIRECT', 'IDLE_THOUGHT', etc.
awaiting_user_response: bool = False  # True if we just responded to user and are waiting

def set_nami_speaking(is_speaking: bool, source: str = None):
    """Thread-safe setter for speech state."""
    global nami_is_speaking, speech_started_time, last_speech_source, awaiting_user_response
    nami_is_speaking = is_speaking
    if is_speaking:
        speech_started_time = time.time()
        if source:
            last_speech_source = source
        print(f"🔇 [Speech Lock] Nami started speaking (source: {source}) - Director paused")
    else:
        duration = time.time() - speech_started_time if speech_started_time else 0
        print(f"🔊 [Speech Lock] Nami finished speaking ({duration:.1f}s, was: {last_speech_source}) - Director resumed")
        
        # If we just finished responding to user, set awaiting flag
        if last_speech_source == 'USER_DIRECT':
            awaiting_user_response = True
            print(f"⏳ [Speech Lock] Awaiting user response - idle suppressed")

def clear_user_awaiting():
    """Call this when user speaks again, clearing the awaiting state."""
    global awaiting_user_response
    if awaiting_user_response:
        print(f"✅ [Speech Lock] User responded - idle can resume")
    awaiting_user_response = False

def should_suppress_idle() -> bool:
    """Returns True if idle thoughts should be suppressed."""
    global awaiting_user_response, nami_is_speaking
    return nami_is_speaking or awaiting_user_response

def is_nami_speaking() -> bool:
    """Check if Nami is currently speaking (with timeout failsafe)."""
    global nami_is_speaking, speech_started_time
    
    if not nami_is_speaking:
        return False
    
    # Failsafe: If speech has been "ongoing" for too long, assume it finished
    if speech_started_time and (time.time() - speech_started_time) > SPEECH_TIMEOUT:
        print(f"⚠️ [Speech Lock] Timeout reached ({SPEECH_TIMEOUT}s) - forcing unlock")
        nami_is_speaking = False
        return False
    
    # DEBUG: Log when we're blocking
    print(f"🔇 [Speech Lock] Blocking - Nami is speaking (elapsed: {time.time() - speech_started_time:.1f}s)")
    return True

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
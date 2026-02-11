# Save as: director_engine/shared.py
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
# Note: sio will be replaced by main.py with a properly configured server
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

# --- SPEECH STATE TRACKING ---
nami_is_speaking: bool = False
speech_started_time: float = 0.0
SPEECH_TIMEOUT: float = 60.0
last_speech_source: Optional[str] = None
awaiting_user_response: bool = False

# --- INTERRUPT TRACKING ---
last_interrupt_time: float = 0.0
interrupt_count: int = 0

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
    
    if speech_started_time and (time.time() - speech_started_time) > SPEECH_TIMEOUT:
        print(f"⚠️ [Speech Lock] Timeout reached ({SPEECH_TIMEOUT}s) - forcing unlock")
        nami_is_speaking = False
        return False
    
    return True

# --- NEW: INTERRUPT SYSTEM ---
def interrupt_nami(reason: str = "direct_mention") -> bool:
    """
    Force-interrupt Nami's current speech for high-priority direct interactions.
    This clears the speaking lock AND emits an interrupt signal to Nami's TTS.
    
    Returns True if Nami was actually speaking (and got interrupted).
    """
    global nami_is_speaking, awaiting_user_response, last_interrupt_time, interrupt_count
    was_speaking = nami_is_speaking
    nami_is_speaking = False
    awaiting_user_response = False
    last_interrupt_time = time.time()
    interrupt_count += 1
    
    if was_speaking:
        print(f"🛑 [INTERRUPT] Nami interrupted! Reason: {reason}")
        # Signal to Nami's TTS to stop immediately
        _emit_threadsafe('interrupt_speech', {
            'reason': reason,
            'timestamp': time.time()
        })
        # Also signal to UI for debugging
        _emit_threadsafe('nami_interrupted', {
            'reason': reason,
            'timestamp': time.time(),
            'interrupt_count': interrupt_count
        })
    else:
        print(f"🛑 [INTERRUPT] Interrupt requested but Nami wasn't speaking (reason: {reason})")
    
    # Reset the speech dispatcher's cooldown so the reply goes through immediately
    speech_dispatcher.last_speech_time = 0
    speech_dispatcher.last_user_response_time = 0
    
    return was_speaking

def get_interrupt_stats() -> Dict[str, Any]:
    """Get interrupt statistics for debugging."""
    return {
        "total_interrupts": interrupt_count,
        "last_interrupt_time": last_interrupt_time,
        "seconds_since_last": round(time.time() - last_interrupt_time, 1) if last_interrupt_time else None,
        "nami_currently_speaking": is_nami_speaking(),
        "awaiting_user_response": awaiting_user_response,
        "speech_source": last_speech_source
    }

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
        # Don't wait for result - fire and forget
    except RuntimeError as e:
        # Event loop closed or similar - ignore silently during shutdown
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
    """Emit AI-generated context suggestions to the UI."""
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
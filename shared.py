# Save as: director_engine/shared.py
"""
Shared state and singletons for the Director Engine.

NOTE: All speaking state (is_nami_speaking, cooldowns, interrupts)
has been moved to the Prompt Service (port 8001).
The brain no longer tracks whether Nami is speaking.
"""

import asyncio
import json
import socketio
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import config
from context.context_store import ContextStore, EventItem
from context.user_profile_manager import UserProfileManager
from systems.adaptive_controller import AdaptiveController
from systems.correlation_engine import CorrelationEngine
from systems.energy_system import EnergySystem
from systems.behavior_engine import BehaviorEngine
from context.context_compression import ContextCompressor
from systems.scene_manager import SceneManager
from systems.decision_engine import DecisionEngine
from services.prompt_constructor import PromptConstructor
from services.speech_dispatcher import SpeechDispatcher

# --- GLOBAL SINGLETONS ---
sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=0, reconnection_delay=2)
ui_event_loop: Optional[asyncio.AbstractEventLoop] = None
server_ready: bool = False

# --- DIRECTOR MANUAL CONTEXT (defaults; overridden by _load_director_state) ---
manual_context: str = ""
current_streamer: str = "peepingotter"

# --- Lock states for AI auto-fill ---
streamer_locked: bool = False
context_locked: bool = False

# --- Persisted state file ---
# Lets the operator set "watching X / context Y / lock both" before booting
# the services. Without this, UI events sent before director-hub connect
# are dropped, then the director's defaults overwrite the UI on first emit.
_STATE_FILE = Path(__file__).resolve().parent / "director_state.json"

def _save_director_state() -> None:
    try:
        _STATE_FILE.write_text(json.dumps({
            "manual_context": manual_context,
            "current_streamer": current_streamer,
            "streamer_locked": streamer_locked,
            "context_locked": context_locked,
        }, indent=2))
    except Exception as e:
        print(f"⚠️ [Director] Failed to persist state: {e}")

def _load_director_state() -> None:
    global manual_context, current_streamer, streamer_locked, context_locked
    if not _STATE_FILE.exists():
        return
    try:
        data = json.loads(_STATE_FILE.read_text())
        manual_context = data.get("manual_context", manual_context) or ""
        current_streamer = data.get("current_streamer", current_streamer) or current_streamer
        streamer_locked = bool(data.get("streamer_locked", streamer_locked))
        context_locked = bool(data.get("context_locked", context_locked))
        print(f"💾 [Director] Restored state: streamer={current_streamer!r} "
              f"(locked={streamer_locked}), context={'set' if manual_context else 'empty'} "
              f"(locked={context_locked})")
    except Exception as e:
        print(f"⚠️ [Director] Failed to load persisted state: {e}")

# Restore before any setter / getter is called
_load_director_state()

def set_manual_context(context: str, from_ai: bool = False):
    global manual_context, context_locked
    if context_locked and from_ai:
        print(f"🔒 [Director] Context locked - AI update ignored")
        return False
    manual_context = context
    source = "AI" if from_ai else "Manual"
    print(f"📝 [Director] Context set ({source}): {context[:50]}..." if context else f"📝 [Director] Context cleared ({source})")
    _save_director_state()
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
    _save_director_state()
    return True

def get_current_streamer() -> str:
    global current_streamer
    return current_streamer

def set_streamer_locked(locked: bool):
    global streamer_locked
    streamer_locked = locked
    print(f"{'🔒' if locked else '🔓'} [Director] Streamer lock: {locked}")
    _save_director_state()

def set_context_locked(locked: bool):
    global context_locked
    context_locked = locked
    print(f"{'🔒' if locked else '🔓'} [Director] Context lock: {locked}")
    _save_director_state()

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
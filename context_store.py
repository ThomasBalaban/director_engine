# Save as: director_engine/context_store.py
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from config import (
    InputSource, DEFAULT_MOOD, MOOD_WINDOW_SIZE,
    WINDOW_IMMEDIATE, WINDOW_RECENT, WINDOW_BACKGROUND,
    ConversationState, FlowState, UserIntent, SceneType
)
from scoring import EventScore
import config

@dataclass
class EventItem:
    timestamp: float
    source: InputSource
    text: str
    metadata: Dict[str, Any]
    score: EventScore 
    memory_text: Optional[str] = None 
    thread_id: Optional[str] = None 
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class FocusState:
    target_event_id: Optional[str] = None
    topic: Optional[str] = None
    locked_until: float = 0.0
    strength: float = 0.0

@dataclass
class DebtItem:
    text: str
    timestamp: float
    topic: str
    status: str = "unresolved" 

@dataclass
class BotAction:
    timestamp: float
    action_type: str # 'joke', 'question', 'roast', 'support'
    text: str
    outcome_score: float = 0.0 

class ContextStore:
    def __init__(self):
        self.immediate: List[EventItem] = []   
        self.recent: List[EventItem] = []      
        self.background: List[EventItem] = []
        
        # [REQ 10] Tiered History
        self.narrative_log: List[str] = [] # Mid-term (last hour)
        self.ancient_history_log: List[str] = [] # Long-term (compressed)
        
        self.all_memories: List[EventItem] = [] 
        self.lock = threading.Lock()
        
        self.pending_speech_event: Optional[EventItem] = None
        self.pending_speech_lock = threading.Lock()
        
        # --- CORE STATE ---
        self.current_conversation_state = ConversationState.IDLE
        self.current_flow = FlowState.NATURAL
        self.current_intent = UserIntent.CASUAL
        self.current_scene = SceneType.CHILL_CHATTING
        
        # --- ATTENTION SYSTEM ---
        self.focus_state = FocusState()
        
        # --- CONVERSATIONAL DEBT ---
        self.conversation_debt: List[DebtItem] = []
        
        # --- EMOTIONAL MOMENTUM ---
        self.sentiment_history: List[str] = []
        self.emotional_momentum: str = "Stable"
        self.current_mood: str = DEFAULT_MOOD
        
        # --- [REQ 6] RL TRACKING ---
        self.bot_action_log: List[BotAction] = []
        self.action_weights: Dict[str, float] = {
            "joke": 1.0, 
            "question": 1.0, 
            "roast": 1.0, 
            "support": 1.0
        }
        
        self.current_summary: str = "Just starting up."
        self.summary_raw_context: str = "Waiting for events..."
        self.current_topics: List[str] = []
        self.current_entities: List[str] = []
        self.current_prediction: str = "Observing flow..."
        
        self.active_user_profile: Optional[Dict[str, Any]] = None 
        
        self.summary_lock = threading.Lock()

    def add_event(self, source: InputSource, text: str, metadata: Dict[str, Any], score: EventScore) -> EventItem:
        now = time.time()
        item = EventItem(timestamp=now, source=source, text=text, metadata=metadata, score=score)
        with self.lock:
            self.immediate.append(item)
            self._manage_hierarchy_nolock() 
        return item

    # --- [REQ 6] Bot Action Tracking ---
    def log_bot_action(self, action_type: str, text: str):
        with self.lock:
            self.bot_action_log.append(BotAction(
                timestamp=time.time(),
                action_type=action_type,
                text=text
            ))
            # Keep log small
            if len(self.bot_action_log) > 20:
                self.bot_action_log.pop(0)

    def get_recent_bot_action(self, window: float = 30.0) -> Optional[BotAction]:
        with self.lock:
            if not self.bot_action_log: 
                return None
            last = self.bot_action_log[-1]
            if time.time() - last.timestamp <= window:
                return last
            return None

    def update_action_weight(self, action_type: str, delta: float):
        """Adjusts the weight of an action type based on feedback."""
        with self.lock:
            current = self.action_weights.get(action_type, 1.0)
            # Clamp between 0.5 and 2.0 to prevent runaway bias
            self.action_weights[action_type] = max(0.5, min(2.0, current + delta))

    def add_debt(self, text: str, topic: str = "general"):
        with self.lock:
            self.conversation_debt.append(DebtItem(text=text, timestamp=time.time(), topic=topic))
            print(f"🧾 [Debt] Added: '{text}'")

    def resolve_debt(self, topic: str = None) -> Optional[DebtItem]:
        with self.lock:
            if self.conversation_debt:
                item = self.conversation_debt.pop(0)
                print(f"🧾 [Debt] Resolved: '{item.text}'")
                return item
        return None
    
    def add_narrative_segment(self, text: str):
        with self.lock:
            self.narrative_log.append(text)
            # Allow it to grow to triggers compression externally
            print(f"📜 [Context] Added narrative segment: {text[:50]}...")

    # --- [REQ 10] Ancient History ---
    def archive_ancient_history(self, summary_text: str):
        with self.lock:
            self.ancient_history_log.append(summary_text)
            print(f"🏛️ [History] Archived ancient block: {summary_text[:50]}...")

    def promote_to_memory(self, event: EventItem, summary_text: str = None):
        with self.lock:
            if any(m.id == event.id for m in self.all_memories): return
            event.memory_text = summary_text if summary_text else event.text
            print(f"💾 [Memory] Archiving: {event.memory_text[:50]}...")
            self.all_memories.append(event)

    def update_mood(self, sentiment: str):
        if not sentiment: return
        with self.summary_lock:
            self.sentiment_history.append(sentiment)
            if len(self.sentiment_history) > MOOD_WINDOW_SIZE:
                self.sentiment_history.pop(0)
            
            if "scared" in self.sentiment_history or "tense" in self.sentiment_history:
                self.current_mood = "Scared"
                return
            
            pos_count = sum(1 for s in self.sentiment_history if s in ['positive', 'excited', 'happy', 'ecstatic'])
            neg_count = sum(1 for s in self.sentiment_history if s in ['negative', 'frustrated', 'annoyed', 'angry'])
            
            if pos_count >= 3: self.current_mood = "Happy"
            elif neg_count >= 3: self.current_mood = "Annoyed"
            else: self.current_mood = "Neutral"

    def set_active_user(self, profile: Dict[str, Any]):
        with self.summary_lock: self.active_user_profile = profile

    def set_conversation_state(self, state: ConversationState):
        with self.summary_lock:
            if self.current_conversation_state != state:
                self.current_conversation_state = state

    def set_flow_state(self, flow: FlowState):
        with self.summary_lock:
            if self.current_flow != flow:
                self.current_flow = flow

    def set_user_intent(self, intent: UserIntent):
        with self.summary_lock:
            if self.current_intent != intent:
                self.current_intent = intent
    
    def set_scene(self, scene: SceneType):
        with self.summary_lock:
            if self.current_scene != scene:
                print(f"🎬 Scene Change: {self.current_scene.name} -> {scene.name}")
                self.current_scene = scene

    def _manage_hierarchy_nolock(self):
        now = time.time()
        
        # Move Immediate -> Recent
        to_move_recent = []
        keep_immediate = []
        for e in self.immediate:
            if (now - e.timestamp) > WINDOW_IMMEDIATE:
                to_move_recent.append(e)
            else:
                keep_immediate.append(e)
        self.immediate = keep_immediate
        self.recent.extend(to_move_recent)

        # Move Recent -> Background
        to_move_background = []
        keep_recent = []
        for e in self.recent:
            if (now - e.timestamp) > WINDOW_RECENT:
                to_move_background.append(e)
            else:
                keep_recent.append(e)
        self.recent = keep_recent
        self.background.extend(to_move_background)

        # Prune Background
        self.background = [e for e in self.background if (now - e.timestamp) <= WINDOW_BACKGROUND]

    def get_breadcrumbs(self, count: int = 3) -> Dict[str, Any]:
        with self.lock:
            self._manage_hierarchy_nolock()
            active_events = self.immediate + self.recent
            sorted_events = sorted(active_events, key=lambda e: e.score.interestingness, reverse=True)
            
            short_term = [{
                "source": e.source.name, 
                "text": e.text, 
                "score": round(e.score.interestingness, 2), 
                "type": "recent"
            } for e in sorted_events[:count]]

        with self.summary_lock:
            return {
                "recent_events": short_term,
                "memories": [], 
                "prediction": self.current_prediction,
                "current_mood": self.current_mood,
                "emotional_momentum": self.emotional_momentum,
                "focus_topic": self.focus_state.topic,
                "conversation_state": self.current_conversation_state.name,
                "active_user": self.active_user_profile,
                "flow_state": self.current_flow.name,
                "user_intent": self.current_intent.name,
                "scene": self.current_scene.name
            }

    def update_event_score(self, event_id: str, new_score: EventScore) -> bool:
        with self.lock:
            for layer in [self.immediate, self.recent, self.background]:
                for event in layer:
                    if event.id == event_id:
                        event.score = new_score
                        return True
        return False

    def update_event_metadata(self, event_id: str, metadata_update: Dict[str, Any]) -> bool:
        with self.lock:
            for layer in [self.immediate, self.recent, self.background]:
                for event in layer:
                    if event.id == event_id:
                        event.metadata.update(metadata_update)
                        return True
        return False

    def set_pending_speech(self, event: EventItem):
        with self.pending_speech_lock: self.pending_speech_event = event

    def get_and_clear_pending_speech(self, max_age_seconds: float = 3.0) -> Optional[EventItem]:
        with self.pending_speech_lock:
            if not self.pending_speech_event: return None
            if (time.time() - self.pending_speech_event.timestamp) <= max_age_seconds:
                evt = self.pending_speech_event
                self.pending_speech_event = None
                return evt
            self.pending_speech_event = None
            return None

    def get_summary(self) -> Tuple[str, str]:
        with self.summary_lock: return self.current_summary, self.summary_raw_context
            
    def set_summary(self, summary: str, raw_context: str, topics: List[str], entities: List[str], prediction: str):
        with self.summary_lock:
            self.current_summary = summary
            self.summary_raw_context = raw_context
            self.current_topics = topics
            self.current_entities = entities
            self.current_prediction = prediction
            
    def get_all_events_for_summary(self) -> Dict[str, List[EventItem]]:
        with self.lock:
            self._manage_hierarchy_nolock()
            return {
                "immediate": list(self.immediate),
                "recent": list(self.recent),
                "background": list(self.background)
            }

    def get_summary_data(self) -> Dict[str, Any]:
        with self.summary_lock:
            return {
                "summary": self.current_summary,
                "raw_context": self.summary_raw_context,
                "topics": self.current_topics,
                "entities": self.current_entities,
                "prediction": self.current_prediction,
                "mood": self.current_mood,
                "emotional_momentum": self.emotional_momentum,
                "conversation_state": self.current_conversation_state.name,
                "flow": self.current_flow.name,
                "intent": self.current_intent.name,
                "scene": self.current_scene.name
            }

    def get_stale_event_for_analysis(self) -> Optional[EventItem]:
        with self.lock:
            self._manage_hierarchy_nolock()
            candidates = [
                e for e in self.immediate + self.recent 
                if 0.3 < e.score.interestingness < config.OLLAMA_TRIGGER_THRESHOLD 
                and "sentiment" not in e.metadata 
                and "is_bundle" not in e.metadata
            ]
            if not candidates: return None
            return sorted(candidates, key=lambda e: e.timestamp, reverse=True)[0]

    def get_activity_metrics(self) -> Tuple[float, float]:
        with self.lock:
            self._manage_hierarchy_nolock()
            events = self.recent
            if not events: return 0.0, 0.0

            chat_items = [e for e in events if e.source in [InputSource.TWITCH_CHAT, InputSource.TWITCH_MENTION]]
            chat_velocity = len(chat_items) * 2.0

            high_energy_items = [e for e in events if e.score.interestingness > 0.7]
            stream_energy = min(len(high_energy_items) / 5.0, 1.0) 

            return chat_velocity, stream_energy
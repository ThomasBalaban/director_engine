# Save as: director_engine/context_store.py
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from config import (
    InputSource, PRIMARY_MEMORY_COUNT, DEFAULT_MOOD, MOOD_WINDOW_SIZE,
    WINDOW_IMMEDIATE, WINDOW_RECENT, WINDOW_BACKGROUND,
    ConversationState, FlowState, UserIntent, SceneType # [NEW IMPORT]
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
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class ContextStore:
    def __init__(self):
        self.immediate: List[EventItem] = []   
        self.recent: List[EventItem] = []      
        self.background: List[EventItem] = []
        
        # [NEW] Semantic Compression Log
        # Stores strings like "10:00-10:05: User fought the boss and died repeatedly."
        self.narrative_log: List[str] = []
        
        self.all_memories: List[EventItem] = [] 
        self.lock = threading.Lock()
        
        self.pending_speech_event: Optional[EventItem] = None
        self.pending_speech_lock = threading.Lock()
        
        # --- ENHANCED STATE TRACKING ---
        self.current_conversation_state = ConversationState.IDLE
        self.current_flow = FlowState.NATURAL
        self.current_intent = UserIntent.CASUAL
        self.current_scene = SceneType.CHILL_CHATTING # [NEW]
        
        self.current_summary: str = "Just starting up."
        self.summary_raw_context: str = "Waiting for events..."
        self.current_topics: List[str] = []
        self.current_entities: List[str] = []
        self.current_prediction: str = "Observing flow..."
        
        self.current_mood: str = DEFAULT_MOOD
        self.sentiment_history: List[str] = [] 
        self.active_user_profile: Optional[Dict[str, Any]] = None 
        
        self.summary_lock = threading.Lock()

    def add_event(self, source: InputSource, text: str, metadata: Dict[str, Any], score: EventScore) -> EventItem:
        now = time.time()
        item = EventItem(timestamp=now, source=source, text=text, metadata=metadata, score=score)
        with self.lock:
            self.immediate.append(item)
            self._manage_hierarchy_nolock() 
        return item
    
    # [NEW] Narrative Compression
    def add_narrative_segment(self, text: str):
        """Adds a compressed summary of past events to the permanent log."""
        with self.lock:
            self.narrative_log.append(text)
            # Keep last ~50 narrative chunks (approx 2-3 hours of history)
            if len(self.narrative_log) > 50:
                self.narrative_log.pop(0)
            print(f"📜 [Context] Added narrative segment: {text[:50]}...")

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
        with self.summary_lock:
            self.active_user_profile = profile

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
    
    # [NEW] Scene Setter
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

        # Prune Background (BUT events here are read by Compressor before deletion)
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
                # Memories will be populated by memory_ops
                "memories": [], 
                "prediction": self.current_prediction,
                "current_mood": self.current_mood,
                "conversation_state": self.current_conversation_state.name,
                "active_user": self.active_user_profile,
                "flow_state": self.current_flow.name,
                "user_intent": self.current_intent.name,
                "scene": self.current_scene.name # [NEW]
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
            now = time.time()
            if (now - self.pending_speech_event.timestamp) <= max_age_seconds:
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
                "conversation_state": self.current_conversation_state.name,
                "flow": self.current_flow.name,
                "intent": self.current_intent.name,
                "scene": self.current_scene.name # [NEW]
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
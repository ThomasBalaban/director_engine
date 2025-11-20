# Save as: director_engine/context_store.py
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from config import (
    InputSource, PRIMARY_MEMORY_COUNT, DEFAULT_MOOD, MOOD_WINDOW_SIZE,
    WINDOW_IMMEDIATE, WINDOW_RECENT, WINDOW_BACKGROUND
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
        # --- HIERARCHICAL STORAGE ---
        self.immediate: List[EventItem] = []   # < 10s
        self.recent: List[EventItem] = []      # 10s - 30s
        self.background: List[EventItem] = []  # 30s - 5m
        
        self.all_memories: List[EventItem] = [] 
        self.lock = threading.Lock()
        
        self.pending_speech_event: Optional[EventItem] = None
        self.pending_speech_lock = threading.Lock()
        
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
            # All new events enter the Immediate layer
            self.immediate.append(item)
            self._manage_hierarchy_nolock() 
        return item
    
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

    def _manage_hierarchy_nolock(self):
        """
        Moves events between layers based on age.
        Immediate -> Recent -> Background -> Deleted
        """
        now = time.time()
        
        # 1. Move Immediate -> Recent
        to_move_recent = []
        keep_immediate = []
        for e in self.immediate:
            age = now - e.timestamp
            if age > WINDOW_IMMEDIATE:
                to_move_recent.append(e)
            else:
                keep_immediate.append(e)
        self.immediate = keep_immediate
        self.recent.extend(to_move_recent)

        # 2. Move Recent -> Background
        to_move_background = []
        keep_recent = []
        for e in self.recent:
            age = now - e.timestamp
            if age > WINDOW_RECENT:
                to_move_background.append(e)
            else:
                keep_recent.append(e)
        self.recent = keep_recent
        self.background.extend(to_move_background)

        # 3. Prune Background
        self.background = [e for e in self.background if (now - e.timestamp) <= WINDOW_BACKGROUND]

    def get_breadcrumbs(self, count: int = 3) -> Dict[str, Any]:
        with self.lock:
            self._manage_hierarchy_nolock()
            # Flatten immediate + recent for the "short term" breadcrumbs
            active_events = self.immediate + self.recent
            sorted_events = sorted(active_events, key=lambda e: e.score.interestingness, reverse=True)
            
            short_term = [{
                "source": e.source.name, 
                "text": e.text, 
                "score": round(e.score.interestingness, 2), 
                "type": "recent"
            } for e in sorted_events[:count]]

            sorted_memories = sorted(self.all_memories, key=lambda e: e.score.interestingness, reverse=True)
            primary_memories_list = sorted_memories[:PRIMARY_MEMORY_COUNT]
            long_term = [{
                "source": m.source.name, 
                "text": m.memory_text or m.text, 
                "score": round(m.score.interestingness, 2), 
                "type": "memory"
            } for m in primary_memories_list]

        with self.summary_lock:
            return {
                "recent_events": short_term,
                "memories": long_term,
                "prediction": self.current_prediction,
                "current_mood": self.current_mood,
                "active_user": self.active_user_profile
            }

    def update_event_score(self, event_id: str, new_score: EventScore) -> bool:
        with self.lock:
            # Search all layers
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
        """Returns a dictionary of layers for the summarizer"""
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
                "mood": self.current_mood
            }

    def get_stale_event_for_analysis(self) -> Optional[EventItem]:
        with self.lock:
            self._manage_hierarchy_nolock()
            # We probably only want to analyze immediate or recent items that missed the initial analysis
            candidates = [
                e for e in self.immediate + self.recent 
                if 0.3 < e.score.interestingness < config.OLLAMA_TRIGGER_THRESHOLD 
                and "sentiment" not in e.metadata 
                and "is_bundle" not in e.metadata
            ]
            if not candidates: return None
            return sorted(candidates, key=lambda e: e.timestamp, reverse=True)[0]
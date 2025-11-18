# Save as: director_engine/context_store.py
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from config import CONTEXT_TIME_WINDOW_SECONDS, InputSource
import config # Import config to get OLLAMA_TRIGGER_THRESHOLD

@dataclass
class EventItem:
    """Dataclass to hold a scored event in memory."""
    timestamp: float
    source: InputSource
    text: str
    metadata: Dict[str, Any]
    score: float
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class ContextStore:
    """
    Thread-safe in-memory store for all scored context events.
    This is the "God Brain's" memory.
    """
    def __init__(self):
        self.events: List[EventItem] = []
        self.lock = threading.Lock()
        
        self.pending_speech_event: Optional[EventItem] = None
        self.pending_speech_lock = threading.Lock()
        
        self.current_summary: str = "Just starting up, not sure what's happening yet."
        self.summary_raw_context: str = "Waiting for events..."
        self.current_topics: List[str] = []
        self.current_entities: List[str] = []
        self.summary_lock = threading.Lock()

    def add_event(self, source: InputSource, text: str, metadata: Dict[str, Any], score: float) -> EventItem:
        """Adds a new, scored event to the memory."""
        now = time.time()
        item = EventItem(
            timestamp=now,
            source=source,
            text=text,
            metadata=metadata,
            score=score
        )
        
        with self.lock:
            self.events.append(item)
            self._prune_events_nolock() 
            
        return item
        
    def _prune_events_nolock(self):
        """Internal helper to prune events *without* acquiring the lock."""
        now = time.time()
        cutoff_time = now - CONTEXT_TIME_WINDOW_SECONDS
        self.events = [e for e in self.events if e.timestamp >= cutoff_time]

    def get_breadcrumbs(self, count: int = 3) -> List[Dict[str, Any]]:
        """Gets the Top N "most interesting" events from memory."""
        with self.lock:
            self._prune_events_nolock()
            sorted_events = sorted(self.events, key=lambda e: e.score, reverse=True)
            top_events = sorted_events[:count]
            
            return [
                {
                    "source": event.source.name,
                    "text": event.text,
                    "score": round(event.score, 2),
                    "timestamp": event.timestamp
                }
                for event in top_events
            ]

    def update_event_score(self, event_id: str, new_score: float) -> bool:
        """Finds an event by its ID and updates its score."""
        with self.lock:
            for event in self.events:
                if event.id == event_id:
                    event.score = new_score
                    return True
        return False

    def update_event_metadata(self, event_id: str, metadata_update: Dict[str, Any]) -> bool:
        """Finds an event by its ID and updates its metadata dictionary."""
        with self.lock:
            for event in self.events:
                if event.id == event_id:
                    event.metadata.update(metadata_update)
                    return True
        return False

    def set_pending_speech(self, event: EventItem):
        """Sets a high-priority speech event as pending for correlation."""
        with self.pending_speech_lock:
            self.pending_speech_event = event

    def get_and_clear_pending_speech(self, max_age_seconds: float = 3.0) -> Optional[EventItem]:
        """Gets the pending speech event if it's recent, and then clears it."""
        with self.pending_speech_lock:
            if not self.pending_speech_event:
                return None
            
            now = time.time()
            if (now - self.pending_speech_event.timestamp) <= max_age_seconds:
                event_to_return = self.pending_speech_event
                self.pending_speech_event = None
                return event_to_return
            
            self.pending_speech_event = None
            return None

    def get_summary(self) -> Tuple[str, str]:
        """Safely gets the current summary and its raw context."""
        with self.summary_lock:
            return self.current_summary, self.summary_raw_context
            
    def set_summary(self, summary: str, raw_context: str, topics: List[str], entities: List[str]):
        """Safely sets the new summary, context, topics, and entities."""
        with self.summary_lock:
            self.current_summary = summary
            self.summary_raw_context = raw_context
            self.current_topics = topics
            self.current_entities = entities
            
    def get_all_events_for_summary(self) -> List[EventItem]:
        """Gets all current events, sorted by time, for summary generation."""
        with self.lock:
            self._prune_events_nolock()
            return sorted(self.events, key=lambda e: e.timestamp)

    def get_summary_data(self) -> Dict[str, Any]:
        """Gets the full summary object for Nami."""
        with self.summary_lock:
            return {
                "summary": self.current_summary,
                "raw_context": self.summary_raw_context,
                "topics": self.current_topics,
                "entities": self.current_entities
            }

    # --- NEW: Method to find events for re-analysis ---
    def get_stale_event_for_analysis(self) -> Optional[EventItem]:
        """Finds an event that is 'stale' (not yet LLM-analyzed) and has potential."""
        with self.lock:
            self._prune_events_nolock()
            
            # Find potential candidates: score > 0.3, not a bundle, and not yet LLM-analyzed (no sentiment)
            candidates = [
                e for e in self.events 
                if 0.3 < e.score < config.OLLAMA_TRIGGER_THRESHOLD 
                and "sentiment" not in e.metadata 
                and "is_bundle" not in e.metadata
            ]
            
            if not candidates:
                return None
            
            # Return the most recent "interesting" stale event
            return sorted(candidates, key=lambda e: e.timestamp, reverse=True)[0]
    # --- END NEW ---
# Save as: director_engine/context_store.py
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from config import CONTEXT_TIME_WINDOW_SECONDS, InputSource

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
        
        # --- MODIFIED: Summary Storage ---
        self.current_summary: str = "Just starting up, not sure what's happening yet."
        self.summary_raw_context: str = "Waiting for events..."
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
            cutoff_time = now - CONTEXT_TIME_WINDOW_SECONDS
            self.events = [e for e in self.events if e.timestamp >= cutoff_time]
            
        # print(f"[Director] Stored Event (Score: {score:.2f}): {source.name} - {text[:40]}...")
        return item
        
    def _prune_events(self) -> List[EventItem]:
        """Internal helper to prune and return current events."""
        now = time.time()
        cutoff_time = now - CONTEXT_TIME_WINDOW_SECONDS
        self.events = [e for e in self.events if e.timestamp >= cutoff_time]
        return self.events

    def get_breadcrumbs(self, count: int = 3) -> List[Dict[str, Any]]:
        """Gets the Top N "most interesting" events from memory."""
        with self.lock:
            events = self._prune_events()
            sorted_events = sorted(events, key=lambda e: e.score, reverse=True)
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

    # --- MODIFIED: Summary Functions ---
    def get_summary(self) -> Tuple[str, str]:
        """Safely gets the current summary and its raw context."""
        with self.summary_lock:
            return self.current_summary, self.summary_raw_context
            
    def set_summary(self, summary: str, raw_context: str):
        """Safely sets the new summary and its raw context."""
        with self.summary_lock:
            self.current_summary = summary
            self.summary_raw_context = raw_context
            
    def get_all_events_for_summary(self) -> List[EventItem]:
        """Gets all current events, sorted by time, for summary generation."""
        with self.lock:
            events = self._prune_events()
            # Sort by time (oldest to newest) to give context
            return sorted(events, key=lambda e: e.timestamp)
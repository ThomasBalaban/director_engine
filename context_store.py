# director_engine/context_store.py
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

    def add_event(self, source: InputSource, text: str, metadata: Dict[str, Any], score: float) -> EventItem:
        """
        Adds a new, scored event to the memory.
        Returns the created EventItem.
        """
        now = time.time()
        item = EventItem(
            timestamp=now,
            source=source,
            text=text,
            metadata=metadata,
            score=score
        )
        
        with self.lock:
            # Add the new item
            self.events.append(item)
            
            # Prune old events
            cutoff_time = now - CONTEXT_TIME_WINDOW_SECONDS
            self.events = [e for e in self.events if e.timestamp >= cutoff_time]
            
        print(f"[Director] Stored Event (Score: {score:.2f}): {source.name} - {text[:40]}...")
        return item

    def get_breadcrumbs(self, count: int = 3) -> List[Dict[str, Any]]:
        """
        Gets the Top N "most interesting" events from memory.
        This is what Nami (Brain 2) will call.
        """
        with self.lock:
            # Prune one last time just in case
            now = time.time()
            cutoff_time = now - CONTEXT_TIME_WINDOW_SECONDS
            self.events = [e for e in self.events if e.timestamp >= cutoff_time]
            
            # Sort all events by score, descending
            sorted_events = sorted(self.events, key=lambda e: e.score, reverse=True)
            
            # Get the top N
            top_events = sorted_events[:count]
            
            # Format them as simple dicts for the API response
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
        """
        Finds an event by its ID and updates its score.
        This is called by the LLM Analyst (Brain 1B).
        """
        with self.lock:
            for event in self.events:
                if event.id == event_id:
                    event.score = new_score
                    return True
        return False # Event might have expired and been pruned
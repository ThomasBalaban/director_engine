# Save as: director_engine/context_store.py
import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from config import CONTEXT_TIME_WINDOW_SECONDS, InputSource, PRIMARY_MEMORY_COUNT
import config 

@dataclass
class EventItem:
    """Dataclass to hold a scored event in memory."""
    timestamp: float
    source: InputSource
    text: str
    metadata: Dict[str, Any]
    score: float
    # New field for the summarized version
    memory_text: Optional[str] = None 
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

class ContextStore:
    """
    Thread-safe in-memory store for all scored context events.
    """
    def __init__(self):
        self.events: List[EventItem] = [] # Short term flow
        self.all_memories: List[EventItem] = [] # Infinite Long term archive
        self.lock = threading.Lock()
        
        self.pending_speech_event: Optional[EventItem] = None
        self.pending_speech_lock = threading.Lock()
        
        self.current_summary: str = "Just starting up."
        self.summary_raw_context: str = "Waiting for events..."
        self.current_topics: List[str] = []
        self.current_entities: List[str] = []
        self.current_prediction: str = "Observing flow..."
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
    
    def promote_to_memory(self, event: EventItem, summary_text: str = None):
        """Promotes a high-scoring event to the infinite long-term memory archive."""
        with self.lock:
            # Check if already in memories to avoid duplicates
            if any(m.id == event.id for m in self.all_memories):
                return

            # Save the summarized text if provided, otherwise raw text
            event.memory_text = summary_text if summary_text else event.text
            
            print(f"💾 [Memory] Archiving: {event.memory_text[:50]}...")
            self.all_memories.append(event)

    def _prune_events_nolock(self):
        """Internal helper to prune short-term events *without* acquiring the lock."""
        now = time.time()
        cutoff_time = now - CONTEXT_TIME_WINDOW_SECONDS
        self.events = [e for e in self.events if e.timestamp >= cutoff_time]

    def get_breadcrumbs(self, count: int = 3) -> Dict[str, Any]:
        """
        Returns a rich context object.
        """
        with self.lock:
            self._prune_events_nolock()
            
            # 1. Get top short-term events (Use RAW text for immediate context)
            sorted_events = sorted(self.events, key=lambda e: e.score, reverse=True)
            short_term = [
                {"source": e.source.name, "text": e.text, "score": round(e.score, 2), "type": "recent"}
                for e in sorted_events[:count]
            ]
            
            # 2. Get "Primary" memories (Top N from the infinite archive)
            # Sort by score descending
            sorted_memories = sorted(self.all_memories, key=lambda e: e.score, reverse=True)
            primary_memories_list = sorted_memories[:PRIMARY_MEMORY_COUNT]
            
            long_term = []
            for m in primary_memories_list:
                # Use the SUMMARY text for memories if available
                display_text = m.memory_text if m.memory_text else m.text
                long_term.append({
                    "source": m.source.name, 
                    "text": display_text, 
                    "score": round(m.score, 2), 
                    "type": "memory"
                })

        # 3. Get prediction
        with self.summary_lock:
            prediction = self.current_prediction
        
        return {
            "recent_events": short_term,
            "memories": long_term,
            "prediction": prediction
        }

    def update_event_score(self, event_id: str, new_score: float) -> bool:
        with self.lock:
            for event in self.events:
                if event.id == event_id:
                    event.score = new_score
                    return True
        return False

    def update_event_metadata(self, event_id: str, metadata_update: Dict[str, Any]) -> bool:
        with self.lock:
            for event in self.events:
                if event.id == event_id:
                    event.metadata.update(metadata_update)
                    return True
        return False

    def set_pending_speech(self, event: EventItem):
        with self.pending_speech_lock:
            self.pending_speech_event = event

    def get_and_clear_pending_speech(self, max_age_seconds: float = 3.0) -> Optional[EventItem]:
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
        with self.summary_lock:
            return self.current_summary, self.summary_raw_context
            
    def set_summary(self, summary: str, raw_context: str, topics: List[str], entities: List[str], prediction: str):
        with self.summary_lock:
            self.current_summary = summary
            self.summary_raw_context = raw_context
            self.current_topics = topics
            self.current_entities = entities
            self.current_prediction = prediction
            
    def get_all_events_for_summary(self) -> List[EventItem]:
        with self.lock:
            self._prune_events_nolock()
            return sorted(self.events, key=lambda e: e.timestamp)

    def get_summary_data(self) -> Dict[str, Any]:
        with self.summary_lock:
            return {
                "summary": self.current_summary,
                "raw_context": self.summary_raw_context,
                "topics": self.current_topics,
                "entities": self.current_entities,
                "prediction": self.current_prediction
            }

    def get_stale_event_for_analysis(self) -> Optional[EventItem]:
        with self.lock:
            self._prune_events_nolock()
            candidates = [
                e for e in self.events 
                if 0.3 < e.score < config.OLLAMA_TRIGGER_THRESHOLD 
                and "sentiment" not in e.metadata 
                and "is_bundle" not in e.metadata
            ]
            if not candidates:
                return None
            return sorted(candidates, key=lambda e: e.timestamp, reverse=True)[0]
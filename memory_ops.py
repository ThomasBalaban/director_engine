# Save as: director_engine/memory_ops.py
import time
import math
from typing import List, Dict, Any
from context_store import EventItem, ContextStore
from config import MEMORY_DECAY_RATE

class MemoryOptimizer:
    def __init__(self):
        self.last_decay_time = time.time()

    def decay_memories(self, store: ContextStore):
        """
        Reduces the score of memories over time.
        Simulates forgetting irrelevant details.
        """
        now = time.time()
        # Calculate minutes passed since last decay
        minutes_passed = (now - self.last_decay_time) / 60.0
        
        if minutes_passed < 1.0: return # Only run every minute or so
        
        decay_amount = minutes_passed * MEMORY_DECAY_RATE
        self.last_decay_time = now
        
        with store.lock:
            for mem in store.all_memories:
                # Apply decay but keep a floor of 0.1 so we don't delete memories entirely yet
                # High impact memories (score > 0.9) decay slower
                factor = 0.5 if mem.score.interestingness > 0.9 else 1.0
                mem.score.interestingness = max(0.1, mem.score.interestingness - (decay_amount * factor))
                
        # print(f"📉 [Memory] Applied decay. Factor: {decay_amount:.3f}")

    def retrieve_relevant_memories(self, store: ContextStore, query_topics: List[str], limit: int = 5) -> List[EventItem]:
        """
        Scores memories based on Relevance to current context + Importance + Recency.
        This fixes the "Memory Retrieval Scoring" gap.
        """
        if not store.all_memories:
            return []

        scored_memories = []
        
        # If we have no topics, fallback to simple importance
        use_topics = len(query_topics) > 0
        
        for mem in store.all_memories:
            # 1. Base Score (Importance)
            score = mem.score.interestingness * 1.0
            
            # 2. Topic Relevance (Clustering/Matching)
            if use_topics and mem.metadata.get('topics'):
                mem_topics = mem.metadata.get('topics', [])
                # Simple overlap check
                overlap = len(set(mem_topics) & set(query_topics))
                if overlap > 0:
                    score *= (1.0 + (overlap * 0.5)) # Boost relevant memories significantly
            
            # 3. Recency Bonus (Short-term retrieval vs Long-term recall)
            # We actually want OLD memories to be retrievable if relevant, so we don't penalize age too much
            # But we might boost very recent memories slightly
            age_hours = (time.time() - mem.timestamp) / 3600.0
            if age_hours < 1.0:
                score *= 1.1
                
            scored_memories.append((score, mem))
            
        # Sort by calculated retrieval score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [m[1] for m in scored_memories[:limit]]
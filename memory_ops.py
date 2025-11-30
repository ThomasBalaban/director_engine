# Save as: director_engine/memory_ops.py
import time
from typing import List, Dict, Any
from context_store import EventItem, ContextStore
from config import MEMORY_DECAY_RATE
from semantic_memory import SemanticMemoryRetriever

class MemoryOptimizer:
    def __init__(self):
        self.last_decay_time = time.time()
        # Initialize the semantic brain
        self.semantic_retriever = SemanticMemoryRetriever()

    def decay_memories(self, store: ContextStore):
        """
        Reduces the score of memories over time.
        Simulates forgetting irrelevant details.
        """
        now = time.time()
        minutes_passed = (now - self.last_decay_time) / 60.0
        
        if minutes_passed < 1.0: return 
        
        decay_amount = minutes_passed * MEMORY_DECAY_RATE
        self.last_decay_time = now
        
        with store.lock:
            for mem in store.all_memories:
                # High impact memories decay slower
                factor = 0.5 if mem.score.interestingness > 0.9 else 1.0
                mem.score.interestingness = max(0.1, mem.score.interestingness - (decay_amount * factor))

    def retrieve_relevant_memories(self, 
                                 store: ContextStore, 
                                 query_context: str, 
                                 limit: int = 5) -> List[EventItem]:
        """
        Retrieves memories using a hybrid score:
        Hybrid Score = (Semantic Similarity * 0.6) + (Importance * 0.3) + (Recency * 0.1)
        """
        if not store.all_memories:
            return []

        # 1. Get Semantic Scores (0.0 to 1.0)
        # If query_context is empty (e.g., silence), it returns empty list
        semantic_results = self.semantic_retriever.rank_memories(query_context, store.all_memories)
        
        # Create a map for easy lookup: {memory_id: similarity_score}
        semantic_map = {mem.id: score for score, mem in semantic_results}
        
        scored_memories = []
        now = time.time()
        
        for mem in store.all_memories:
            # A. Semantic Score (Default to 0.0 if no match/no query)
            sem_score = semantic_map.get(mem.id, 0.0)
            
            # B. Importance Score (Base interestingness from Director)
            imp_score = mem.score.interestingness
            
            # C. Recency Score (Normalized: 1.0 = now, 0.0 = 24 hours ago)
            age_hours = (now - mem.timestamp) / 3600.0
            recency_score = max(0.0, 1.0 - (age_hours / 24.0))
            
            # --- THE HYBRID FORMULA ---
            # We prioritize Meaning (Semantic) > Importance > Time
            final_score = (sem_score * 0.6) + (imp_score * 0.3) + (recency_score * 0.1)
            
            # Boost very high similarity (exact context matches)
            if sem_score > 0.8:
                final_score += 0.2
                
            scored_memories.append((final_score, mem))
            
        # Sort by final score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Debug print to see what's being picked (Optional)
        # if scored_memories and query_context:
        #    print(f"🔍 [Memory] Context: '{query_context[:30]}...' -> Top: {scored_memories[0][1].text[:30]}... (Score: {scored_memories[0][0]:.2f})")
        
        return [m[1] for m in scored_memories[:limit]]
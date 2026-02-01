# Save as: director_engine/context/memory_ops.py
import time
from typing import List, Dict, Any
from context.context_store import EventItem, ContextStore
from config import MEMORY_DECAY_RATE
from context.semantic_memory import SemanticMemoryRetriever

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
        
        if minutes_passed < 1.0: 
            return 
        
        decay_amount = minutes_passed * MEMORY_DECAY_RATE
        self.last_decay_time = now
        
        decayed_count = 0
        with store.lock:
            for mem in store.all_memories:
                old_score = mem.score.interestingness
                # High impact memories decay slower
                factor = 0.5 if mem.score.interestingness > 0.9 else 1.0
                mem.score.interestingness = max(0.1, mem.score.interestingness - (decay_amount * factor))
                if mem.score.interestingness < old_score:
                    decayed_count += 1
        
        if decayed_count > 0:
            print(f"🧠 [Memory] Decayed {decayed_count} memories by {decay_amount:.3f}")

    def retrieve_relevant_memories(self, 
                                 store: ContextStore, 
                                 query_context: str, 
                                 limit: int = 5) -> List[EventItem]:
        """
        Retrieves memories using a hybrid score:
        Hybrid Score = (Semantic Similarity * 0.6) + (Importance * 0.3) + (Recency * 0.1)
        
        Now with better handling of empty queries and debug output.
        """
        if not store.all_memories:
            return []
        
        # Handle empty or very short queries
        if not query_context or len(query_context.strip()) < 5:
            print(f"🧠 [Memory] Query too short or empty, returning by importance only")
            # Fall back to importance-based retrieval
            sorted_by_importance = sorted(
                store.all_memories, 
                key=lambda m: m.score.interestingness, 
                reverse=True
            )
            return sorted_by_importance[:limit]

        # 1. Get Semantic Scores (0.0 to 1.0)
        semantic_results = self.semantic_retriever.rank_memories(query_context, store.all_memories)
        
        # Create a map for easy lookup: {memory_id: similarity_score}
        semantic_map = {mem.id: score for score, mem in semantic_results}
        
        # Debug: Show semantic matches
        if semantic_results:
            top_semantic = sorted(semantic_results, key=lambda x: x[0], reverse=True)[:3]
            print(f"🧠 [Memory] Top semantic matches:")
            for score, mem in top_semantic:
                content = (mem.memory_text or mem.text)[:40]
                print(f"   • {score:.3f}: {content}...")
        
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
            
            # Boost if the memory contains key terms from the query
            query_lower = query_context.lower()
            mem_text_lower = (mem.memory_text or mem.text).lower()
            
            # Simple keyword boost - if significant words overlap
            query_words = set(w for w in query_lower.split() if len(w) > 4)
            mem_words = set(w for w in mem_text_lower.split() if len(w) > 4)
            overlap = query_words & mem_words
            if overlap:
                keyword_boost = min(0.15, len(overlap) * 0.05)
                final_score += keyword_boost
                
            scored_memories.append((final_score, mem))
            
        # Sort by final score
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Debug: Show final rankings
        if scored_memories:
            print(f"🧠 [Memory] Final rankings (hybrid score):")
            for score, mem in scored_memories[:3]:
                content = (mem.memory_text or mem.text)[:40]
                print(f"   • {score:.3f}: {content}...")
        
        return [m[1] for m in scored_memories[:limit]]
    
    def force_add_memory(self, store: ContextStore, event: EventItem, summary: str = None):
        """
        Force an event into memory regardless of threshold.
        Useful for manually important moments.
        """
        store.promote_to_memory(event, summary_text=summary)
        print(f"🧠 [Memory] Force-added: {(summary or event.text)[:50]}...")
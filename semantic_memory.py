# Save as: director_engine/semantic_memory.py
from sentence_transformers import SentenceTransformer, util
import torch
import time
from typing import List, Tuple, Any

class SemanticMemoryRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"🧠 [Memory] Loading embedding model: {model_name}...")
        # This will download the model on first run (approx 80MB)
        self.model = SentenceTransformer(model_name)
        self.embedding_cache = {} # {event_id: embedding_tensor}
        print(f"✅ [Memory] Semantic model loaded.")

    def _get_embedding(self, text: str):
        """Generates embedding for text."""
        return self.model.encode(text, convert_to_tensor=True)

    def rank_memories(self, query_text: str, memories: List[Any]) -> List[Tuple[float, Any]]:
        """
        Ranks a list of memory events based on semantic similarity to the query.
        Returns list of (similarity_score, memory_event).
        """
        if not memories or not query_text:
            return []
            
        # 1. Maintenance: Clean cache of deleted memories
        current_ids = {m.id for m in memories}
        self.embedding_cache = {k: v for k, v in self.embedding_cache.items() if k in current_ids}
        
        # 2. Generate/Retrieve Embeddings for Memories
        memory_embeddings = []
        valid_memories = []
        
        for mem in memories:
            # Use the summarized memory text if available, otherwise raw text
            content = mem.memory_text or mem.text
            if not content: continue
                
            if mem.id not in self.embedding_cache:
                self.embedding_cache[mem.id] = self._get_embedding(content)
            
            memory_embeddings.append(self.embedding_cache[mem.id])
            valid_memories.append(mem)
            
        if not valid_memories:
            return []

        # 3. Generate Query Embedding
        query_embedding = self._get_embedding(query_text)
        
        # 4. Calculate Cosine Similarities
        # Stack embeddings into a matrix for fast batch calculation
        corpus_embeddings = torch.stack(memory_embeddings)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        # 5. Pair scores with memories
        results = []
        for idx, score in enumerate(cos_scores):
            results.append((float(score), valid_memories[idx]))
            
        return results
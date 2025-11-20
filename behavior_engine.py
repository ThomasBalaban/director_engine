# Save as: director_engine/behavior_engine.py
import time
import random
from typing import List, Optional
from config import BotGoal, InputSource, ConversationState, CURIOSITY_INTERVAL, CALLBACK_INTERVAL
from context_store import ContextStore, EventItem
from scoring import EventScore

class BehaviorEngine:
    def __init__(self):
        self.current_goal = BotGoal.OBSERVE
        self.last_curiosity_check = time.time()
        self.last_callback_check = time.time() # [NEW]
        
    # --- 1. Intention System ---
    def update_goal(self, store: ContextStore):
        chat_vel, stream_energy = store.get_activity_metrics()
        state = store.current_conversation_state
        
        if state == ConversationState.FRUSTRATED:
            self.current_goal = BotGoal.SUPPORT
        elif state == ConversationState.CELEBRATORY:
            self.current_goal = BotGoal.ENTERTAIN
        elif stream_energy > 0.8:
            self.current_goal = BotGoal.ENTERTAIN 
        elif state == ConversationState.IDLE and chat_vel < 2:
            self.current_goal = random.choice([BotGoal.INVESTIGATE, BotGoal.TROLL])
        else:
            self.current_goal = BotGoal.OBSERVE

    # --- 2. Attention Director ---
    def direct_attention(self, events: List[EventItem]) -> Optional[EventItem]:
        if not events: return None
        
        if self.current_goal == BotGoal.SUPPORT:
            candidates = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]
        elif self.current_goal == BotGoal.ENTERTAIN:
            candidates = [e for e in events if e.source in [InputSource.VISUAL_CHANGE, InputSource.SYSTEM_PATTERN]]
        elif self.current_goal == BotGoal.INVESTIGATE:
            candidates = [e for e in events if e.source == InputSource.DIRECT_MICROPHONE]
        else:
            candidates = events
            
        if not candidates: candidates = events
        return max(candidates, key=lambda x: x.score.interestingness)

    # --- 3. Curiosity Engine ---
    def check_curiosity(self, store: ContextStore) -> Optional[str]:
        now = time.time()
        if now - self.last_curiosity_check < CURIOSITY_INTERVAL: return None
        self.last_curiosity_check = now
        
        if store.current_conversation_state not in [ConversationState.IDLE, ConversationState.ENGAGED]:
            return None
            
        user = store.active_user_profile
        if not user: return None
        
        questions = []
        if len(user['facts']) < 3:
            questions.append(f"I don't know much about {user['username']}. I should ask what they do for fun.")
        if not any("game" in f['content'] for f in user['facts']):
             questions.append("I wonder what games they actually like?")
        if store.current_topics:
             questions.append(f"I wonder what they think about {store.current_topics[0]}?")
        
        if questions: return random.choice(questions)
        return None

    # --- 4. [NEW] Callback System ---
    def check_callbacks(self, store: ContextStore) -> Optional[str]:
        """
        Checks narrative history for events relevant to the current context to reference.
        """
        now = time.time()
        if now - self.last_callback_check < CALLBACK_INTERVAL: return None
        self.last_callback_check = now
        
        if not store.narrative_log: return None
        
        # Simple heuristic: If we are "Just Chatting" or "Idle", try to recall the last narrative segment
        # In a real implementation, we'd use embedding similarity here.
        # For now, just grabbing a random recent history segment to "reflect" on.
        
        if store.current_conversation_state in [ConversationState.IDLE, ConversationState.ENGAGED]:
            # Look at history from ~5-10 mins ago (index -3 or -4)
            if len(store.narrative_log) >= 3:
                past_event = store.narrative_log[-3]
                return f"Recall: Remember when this happened? '{past_event}'"
                
        return None
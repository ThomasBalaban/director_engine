# Save as: director_engine/behavior_engine.py
import time
import random
from typing import List, Optional
from config import BotGoal, InputSource, ConversationState, CURIOSITY_INTERVAL, CALLBACK_INTERVAL
from context_store import ContextStore, EventItem
from scoring import EventScore
from user_profile_manager import UserProfileManager

class BehaviorEngine:
    def __init__(self):
        self.current_goal = BotGoal.OBSERVE
        self.last_curiosity_check = time.time()
        self.last_callback_check = time.time()
        self.attention_lock_duration = 5.0 
        
    def update_goal(self, store: ContextStore):
        chat_vel, stream_energy = store.get_activity_metrics()
        state = store.current_conversation_state
        if state == ConversationState.FRUSTRATED: self.current_goal = BotGoal.SUPPORT
        elif state == ConversationState.CELEBRATORY: self.current_goal = BotGoal.ENTERTAIN
        elif stream_energy > 0.8: self.current_goal = BotGoal.ENTERTAIN 
        elif state == ConversationState.IDLE and chat_vel < 2:
            self.current_goal = random.choice([BotGoal.INVESTIGATE, BotGoal.TROLL])
        else: self.current_goal = BotGoal.OBSERVE

    def direct_attention(self, store: ContextStore, events: List[EventItem]) -> Optional[EventItem]:
        if not events: return None
        now = time.time()
        focus = store.focus_state
        
        if self.current_goal == BotGoal.SUPPORT:
            candidates = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]
        elif self.current_goal == BotGoal.ENTERTAIN:
            candidates = [e for e in events if e.source in [InputSource.VISUAL_CHANGE, InputSource.SYSTEM_PATTERN]]
        elif self.current_goal == BotGoal.INVESTIGATE:
            candidates = [e for e in events if e.source == InputSource.DIRECT_MICROPHONE]
        else:
            candidates = events
            
        if not candidates: candidates = events
        candidate = max(candidates, key=lambda x: x.score.interestingness)
        candidate_score = candidate.score.interestingness
        
        # Check Focus Lock
        if focus.target_event_id and now < focus.locked_until:
            switch_cost = 0.3 
            if candidate.id != focus.target_event_id:
                if candidate_score < (focus.strength + switch_cost):
                    return None 
                
        store.focus_state.target_event_id = candidate.id
        store.focus_state.strength = candidate_score
        store.focus_state.locked_until = now + self.attention_lock_duration
        
        print(f"🔒 [Attention] Locked on {candidate.source.name} (Score: {candidate_score:.2f})")
        return candidate

    # --- RL & Debt Hook ---
    def register_bot_action(self, store: ContextStore, text: str):
        """
        Categorizes and logs the action for RL, and registers debt if it's a question.
        """
        action_type = "support"
        if "?" in text: 
            action_type = "question"
            store.add_debt(text)
        elif any(x in text.lower() for x in ["haha", "lol", "lmao", "roast"]):
            action_type = "joke"
        elif "!" in text:
            action_type = "roast"
            
        store.log_bot_action(action_type, text)

    def check_debt_resolution(self, store: ContextStore, user_input: str) -> Optional[str]:
        if store.conversation_debt:
            store.resolve_debt() 
            return None
        now = time.time()
        for debt in store.conversation_debt:
            if now - debt.timestamp > 60.0: 
                store.conversation_debt.remove(debt)
                return f"Wait, you never told me: {debt.text}"
        return None

    # --- [REQ 7] Proactive Topic Suggestion ---
    def check_curiosity(self, store: ContextStore, profile_manager: UserProfileManager) -> Optional[str]:
        now = time.time()
        if now - self.last_curiosity_check < CURIOSITY_INTERVAL: return None
        self.last_curiosity_check = now
        
        if store.current_conversation_state not in [ConversationState.IDLE, ConversationState.ENGAGED]:
            return None
            
        user = store.active_user_profile
        if not user: return None
        
        # 1. Try to revive an old topic
        fact = profile_manager.get_under_discussed_fact(user['username'])
        if fact:
            print(f"💡 [Curiosity] Reviving topic: {fact}")
            profile_manager.mark_fact_used(user['username'], fact)
            return f"Hey, remember when you mentioned '{fact}'? How's that going?"

        # 2. Fallback to generic discovery
        questions = []
        if len(user['facts']) < 3:
            questions.append(f"I don't know much about {user['username']}. I should ask what they do for fun.")
        else:
             questions.append("I wonder what games they actually like?")
        
        if questions: return random.choice(questions)
        return None

    def check_callbacks(self, store: ContextStore) -> Optional[str]:
        now = time.time()
        if now - self.last_callback_check < CALLBACK_INTERVAL: return None
        debt_prompt = self.check_debt_resolution(store, "")
        if debt_prompt: 
            self.last_callback_check = now
            return debt_prompt
        self.last_callback_check = now
        if store.narrative_log and store.current_conversation_state in [ConversationState.IDLE, ConversationState.ENGAGED]:
            if len(store.narrative_log) >= 3:
                past_event = store.narrative_log[-3]
                return f"Recall: Remember when this happened? '{past_event}'"
        return None
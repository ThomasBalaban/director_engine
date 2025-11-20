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
        self.last_callback_check = time.time()
        self.attention_lock_duration = 5.0 # How long focus is "sticky"
        
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

    # --- 2. Attention Director (Competition System) ---
    def direct_attention(self, store: ContextStore, events: List[EventItem]) -> Optional[EventItem]:
        """
        Selects the most important event, respecting the current focus lock.
        Returns None if the event is blocked by the current focus.
        """
        if not events: return None
        
        now = time.time()
        focus = store.focus_state
        
        # 1. Identify the strongest candidate based on Goal
        if self.current_goal == BotGoal.SUPPORT:
            candidates = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]
        elif self.current_goal == BotGoal.ENTERTAIN:
            candidates = [e for e in events if e.source in [InputSource.VISUAL_CHANGE, InputSource.SYSTEM_PATTERN]]
        elif self.current_goal == BotGoal.INVESTIGATE:
            candidates = [e for e in events if e.source == InputSource.DIRECT_MICROPHONE]
        else:
            candidates = events
            
        if not candidates: candidates = events
        
        # Find best candidate
        candidate = max(candidates, key=lambda x: x.score.interestingness)
        candidate_score = candidate.score.interestingness
        
        # 2. Check Focus Lock
        # If we are locked on a target, the new event must be SIGNIFICANTLY better to break focus
        if focus.target_event_id and now < focus.locked_until:
            # Cost to switch attention (Stickiness)
            switch_cost = 0.3 
            
            # Don't lock out our own previous target if we are re-evaluating it
            if candidate.id != focus.target_event_id:
                if candidate_score < (focus.strength + switch_cost):
                    return None # Ignored due to focus lock
                
        # 3. Update Focus
        store.focus_state.target_event_id = candidate.id
        store.focus_state.strength = candidate_score
        store.focus_state.locked_until = now + self.attention_lock_duration
        
        # [ADDED] Log the attention lock
        print(f"🔒 [Attention] Locked on {candidate.source.name} (Score: {candidate_score:.2f})")
        
        return candidate

    # --- 3. Conversational Debt System ---
    def register_bot_action(self, store: ContextStore, text: str):
        """Called when the bot speaks. If it's a question, add to debt."""
        if "?" in text:
            store.add_debt(text)
            
    def check_debt_resolution(self, store: ContextStore, user_input: str) -> Optional[str]:
        """
        Checks if user input resolves a debt. 
        If debt is old, prompts a callback.
        """
        # 1. Check if user answered a pending question (Simple heuristic)
        if store.conversation_debt:
            # In reality, use LLM to verify answer relevance. 
            # For now, assume any direct user response resolves the oldest debt.
            store.resolve_debt() 
            return None

        # 2. Check for Expired Debt (Circle Back)
        now = time.time()
        for debt in store.conversation_debt:
            if now - debt.timestamp > 60.0: # 1 minute old
                store.conversation_debt.remove(debt)
                return f"Wait, you never told me: {debt.text}"
                
        return None

    # --- 4. Curiosity Engine ---
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

    # --- 5. Callback System ---
    def check_callbacks(self, store: ContextStore) -> Optional[str]:
        """
        Checks debt first, then narrative history.
        """
        now = time.time()
        if now - self.last_callback_check < CALLBACK_INTERVAL: return None
        
        # Check debt first
        debt_prompt = self.check_debt_resolution(store, "")
        if debt_prompt: 
            self.last_callback_check = now
            return debt_prompt
        
        self.last_callback_check = now
        
        if not store.narrative_log: return None
        
        if store.current_conversation_state in [ConversationState.IDLE, ConversationState.ENGAGED]:
            # Look at history from ~5-10 mins ago (index -3 or -4)
            if len(store.narrative_log) >= 3:
                past_event = store.narrative_log[-3]
                return f"Recall: Remember when this happened? '{past_event}'"
                
        return None
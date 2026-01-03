# Save as: director_engine/decision_engine.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from config import BotGoal, ConversationState, FlowState, InputSource
from context.context_store import ContextStore, EventItem
from systems.behavior_engine import BehaviorEngine
from systems.adaptive_controller import AdaptiveController
from systems.energy_system import EnergySystem

@dataclass
class Directive:
    objective: str
    tone: str
    constraints: List[str]
    topic_focus: str
    suggested_action: str
    reasoning: str

    def to_dict(self):
        return asdict(self)

class DecisionEngine:
    """
    The Executive Function. 
    Translates the nebulous "state" of the stream into concrete instructions for the Bot.
    """
    
    def generate_directive(self, 
                           store: ContextStore, 
                           behavior: BehaviorEngine, 
                           adaptive: AdaptiveController,
                           energy: EnergySystem) -> Directive:
        
        # 1. Determine Tone based on Adaptive State, Momentum, and Mood
        tone = self._calculate_tone(store, adaptive)

        # 2. Determine Primary Objective based on Goal & State
        objective, action = self._calculate_objective_and_action(store, behavior)
        
        # 3. Calculate Constraints (Energy, Social Context)
        constraints = self._calculate_constraints(store, energy, adaptive)
        
        # 4. Identify Focus Topic
        target_topic = store.focus_state.topic or "Current Context"
        if not target_topic and store.current_topics:
            target_topic = store.current_topics[0]

        return Directive(
            objective=objective,
            tone=tone,
            constraints=constraints,
            topic_focus=target_topic,
            suggested_action=action,
            reasoning=f"Goal: {behavior.current_goal.name} | Flow: {store.current_flow.name}"
        )

    def _calculate_tone(self, store: ContextStore, adaptive: AdaptiveController) -> str:
        # High Chaos -> High Energy Response
        if adaptive.state_label == "Chaos/Hype":
            return "High Energy, Loud, Excited, Fast-paced"
        
        # Dead Air -> Provocative
        if adaptive.state_label == "Dead Air":
            return "Provocative, Random, Slightly Unhinged (to wake chat up)"
            
        # Mood overrides
        if store.current_mood == "Horny":
            return "Flirty, Suggestive, Teasing"
        elif store.current_mood == "Angry":
            return "Sharp, Sarcastic, Ruthless"
        elif store.current_mood == "Scared":
            return "Tense, Whispering, Panic-stricken"
            
        return "Casual, Sarcastic, Witty"

    def _calculate_objective_and_action(self, store: ContextStore, behavior: BehaviorEngine) -> tuple[str, str]:
        # --- NEURO-FICATION: Handler Dynamic ---
        # Check if the "Handler" (Creator) is the active user
        user_role = "viewer"
        if store.active_user_profile:
            user_role = store.active_user_profile.get("role", "viewer")

        # --- Skill Issue / Roast Logic ---
        # Check for specific patterns in immediate events
        recent_patterns = [e for e in store.immediate if e.source == InputSource.SYSTEM_PATTERN]
        for pat in recent_patterns:
            if pat.metadata.get("type") == "skill_issue":
                return "Mock Failure", "Laugh at the handler's incompetence. Do not offer help. Say 'Skill Issue'."
            if pat.metadata.get("type") == "fixation":
                entity = pat.metadata.get("entity", "thing")
                return "Visual Fixation", f"Obsess over the '{entity}'. Ask why it keeps appearing."

        goal = behavior.current_goal
        state = store.current_conversation_state
        
        # Special Logic for Handler interactions
        if user_role == "handler":
            if store.current_mood in ["Annoyed", "Bored"]:
                return "Deflect Blame", "Gaslight the handler, complain about the setup, or blame them for lag."
            elif store.current_flow == FlowState.DEAD_AIR:
                return "Provoke", "Roast the handler for being boring or bad at the game."

        if goal == BotGoal.SUPPORT:
            return "Assist User", "Offer genuine help, backseating, or empathy."
            
        elif goal == BotGoal.ENTERTAIN:
            if state == ConversationState.CELEBRATORY:
                return "Amplify Hype", "Cheer loudly or make a victory joke."
            return "Create Content", "Roast the user, point out something weird, or make a joke."
            
        elif goal == BotGoal.INVESTIGATE:
            return "Extract Information", "Ask a prying personal question or follow up on a user fact."
            
        elif goal == BotGoal.TROLL:
            return "Create Chaos", "Gaslight the user, give bad advice, or misinterpret the situation."
            
        elif goal == BotGoal.OBSERVE:
            return "Passive Observation", "Stay quiet unless directly addressed. Listen."
            
        return "Exist", "Wait."

    def _calculate_constraints(self, store: ContextStore, energy: EnergySystem, adaptive: AdaptiveController) -> List[str]:
        constraints = []
        
        # Energy Constraints
        status = energy.get_status()
        if status['percent'] < 20:
            constraints.append("LOW BATTERY: Keep responses short (1 sentence max).")
        elif status['percent'] < 50:
            constraints.append("Conserve Energy: Avoid long rambles.")
            
        # Flow Constraints
        if store.current_flow == FlowState.DOMINATED:
            constraints.append("Don't interrupt: User is monologuing.")
        elif store.current_flow == FlowState.STACCATO:
            constraints.append("Match tempo: Short, punchy replies.")
            
        # Safety / Context
        if store.current_conversation_state == ConversationState.FRUSTRATED:
            constraints.append("Read the room: Do NOT be overly cheerful. Validate frustration.")
            
        return constraints
# Save as: director_engine/decision_engine.py
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from config import BotGoal, ConversationState, FlowState
from context_store import ContextStore
from behavior_engine import BehaviorEngine
from adaptive_controller import AdaptiveController
from energy_system import EnergySystem

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
        goal = behavior.current_goal
        state = store.current_conversation_state
        
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
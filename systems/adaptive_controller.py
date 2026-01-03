# Save as: director_engine/adaptive_controller.py
from config import INTERJECTION_THRESHOLD
from context.context_store import ContextStore, BotAction

class AdaptiveController:
    def __init__(self):
        self.base_threshold = INTERJECTION_THRESHOLD
        self.current_threshold = self.base_threshold
        self.state_label = "Normal"
        self.last_feedback_time = 0

    def update(self, chat_velocity: float, stream_energy: float) -> float:
        """
        Adjusts the interruption threshold based on stream state.
        """
        if chat_velocity > 30 or stream_energy > 0.8:
            self.state_label = "Chaos/Hype"
            target = 0.95 
        elif chat_velocity < 2 and stream_energy < 0.2:
            self.state_label = "Dead Air"
            target = 0.65 
        else:
            self.state_label = "Normal"
            target = self.base_threshold

        self.current_threshold = (self.current_threshold * 0.8) + (target * 0.2)
        return self.current_threshold

    # --- [REQ 6] Reinforcement Learning Loop ---
    def process_feedback(self, store: ContextStore):
        """
        Evaluates recent bot actions against chat reaction.
        If chat velocity spiked or positive sentiment appeared after an action -> Reinforce.
        """
        last_action = store.get_recent_bot_action(window=10.0)
        if not last_action: return
        
        # We can't re-evaluate the same action indefinitely
        if last_action.timestamp <= self.last_feedback_time: return
        
        chat_vel, energy = store.get_activity_metrics()
        
        # Simple heuristic: Did chat move?
        # (In a real system, we'd check specific sentiment of replies)
        success = False
        
        if chat_vel > 5.0: # Chat is active
            success = True
            print(f"🎓 [RL] Action '{last_action.action_type}' successful (High Velocity)")
            store.update_action_weight(last_action.action_type, +0.1)
        
        # If we were trying to be funny/entertaining and chat is dead -> Failure
        elif last_action.action_type in ['joke', 'roast'] and chat_vel < 1.0:
            print(f"🎓 [RL] Action '{last_action.action_type}' flopped (Dead Air)")
            store.update_action_weight(last_action.action_type, -0.05)
            
        self.last_feedback_time = last_action.timestamp
# Save as: director_engine/adaptive_controller.py
from config import INTERJECTION_THRESHOLD

class AdaptiveController:
    def __init__(self):
        self.base_threshold = INTERJECTION_THRESHOLD
        self.current_threshold = self.base_threshold
        self.state_label = "Normal"

    def update(self, chat_velocity: float, stream_energy: float) -> float:
        """
        Adjusts the interruption threshold based on stream state.
        Returns the new threshold.
        """
        
        # Logic 1: High Energy / Chaos -> Raise Threshold
        # If chat is spamming (velocity > 20) or stream is crazy (energy > 0.8)
        # We want Nami to shut up and observe unless it's CRITICAL.
        if chat_velocity > 30 or stream_energy > 0.8:
            self.state_label = "Chaos/Hype"
            target = 0.95 # Almost impossible to interject
            
        # Logic 2: Slow / Dead Air -> Lower Threshold
        # If nothing is happening, Nami should be more sensitive to keep engagement.
        elif chat_velocity < 2 and stream_energy < 0.2:
            self.state_label = "Dead Air"
            target = 0.65 # Very easy to interject
            
        # Logic 3: Normal Flow
        else:
            self.state_label = "Normal"
            target = self.base_threshold

        # Smooth transition (Linear Interpolation) to prevent jerky behavior
        self.current_threshold = (self.current_threshold * 0.8) + (target * 0.2)
        
        return self.current_threshold
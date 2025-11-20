# Save as: director_engine/scoring.py
import time
from dataclasses import dataclass, field
from typing import Dict, Any
from config import InputSource

@dataclass
class EventScore:
    interestingness: float = 0.0
    urgency: float = 0.0
    conversational_value: float = 0.0
    emotional_intensity: float = 0.0
    topic_relevance: float = 0.0

    def to_dict(self):
        return {
            "interestingness": self.interestingness,
            "urgency": self.urgency,
            "conversational_value": self.conversational_value,
            "emotional_intensity": self.emotional_intensity,
            "topic_relevance": self.topic_relevance
        }
        
    def get_weighted_average(self) -> float:
        # A simple weighted average to use as a single "priority" metric when needed
        return (
            (self.interestingness * 0.4) + 
            (self.urgency * 0.3) + 
            (self.conversational_value * 0.2) + 
            (self.topic_relevance * 0.1)
        )

def calculate_event_score(
    source: InputSource,
    metadata: Dict[str, Any],
    source_weights: Dict[InputSource, float]
) -> EventScore:
    """
    Calculates the initial heuristic 'sieve' score for an event.
    Returns an EventScore object.
    """
    
    # 1. Interestingness (Base from source)
    base_score = source_weights.get(source, 0.1)
    
    # Boosters
    if 'relevance' in metadata: base_score += metadata['relevance'] * 0.3
    if metadata.get('is_summary', False): base_score += 0.1
    if 'confidence' in metadata: base_score += metadata.get('confidence', 0.0) * 0.1
    
    # Clamp
    interestingness = min(max(base_score, 0.0), 1.0)

    # 2. Urgency (Heuristic based on source)
    urgency = 0.1
    if source in [InputSource.DIRECT_MICROPHONE, InputSource.TWITCH_MENTION]:
        urgency = 0.9
    elif source == InputSource.MICROPHONE:
        urgency = 0.6
    if 'urgency' in metadata:
        urgency = max(urgency, metadata['urgency'])

    # 3. Conversational Value (Heuristic)
    conv_value = 0.2
    if source in [InputSource.DIRECT_MICROPHONE, InputSource.MICROPHONE, InputSource.TWITCH_CHAT]:
        conv_value = 0.7
    if len(str(metadata.get('text', ''))) > 10: # Slightly more value if it's not empty
        conv_value += 0.1

    # 4. Emotional Intensity & Topic Relevance
    # Hard to guess heuristically without LLM, start low/neutral
    emotional_intensity = 0.1
    topic_relevance = 0.5 # Assume neutral relevance until analyzed

    return EventScore(
        interestingness=interestingness,
        urgency=urgency,
        conversational_value=conv_value,
        emotional_intensity=emotional_intensity,
        topic_relevance=topic_relevance
    )
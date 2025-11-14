# director_engine/scoring.py
import time
from typing import Dict, Any
from config import InputSource, SOURCE_WEIGHTS

def calculate_event_score(
    source: InputSource,
    metadata: Dict[str, Any],
    source_weights: Dict[InputSource, float]
) -> float:
    """
    Calculates the "interestingness" score for a single ambient event.
    This is a simplified version of Nami's priority_scoring.py,
    focused only on the event itself, not on conversation state.
    """
    
    # Start with base score from source weight
    score = source_weights.get(source, 0.1)
    
    # Apply content relevance factors if available
    # (e.g., from vision_client metadata)
    if 'relevance' in metadata:
        score += metadata['relevance'] * 0.3
    
    # Apply urgency factor if available
    if 'urgency' in metadata:
        score += metadata['urgency'] * 0.2
        
    # Apply confidence (from audio/vision)
    if 'confidence' in metadata:
        score += metadata.get('confidence', 0.0) * 0.2

    # Apply bonus for summaries from vision
    if metadata.get('is_summary', False):
        score += 0.1
        
    # Normalize score to be between 0.0 and 1.0 (or slightly higher)
    # This is a basic clamp, can be refined.
    return min(max(score, 0.0), 1.0)
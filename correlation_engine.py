# Save as: director_engine/correlation_engine.py
import time
from typing import List, Dict, Any
from context_store import ContextStore
from config import InputSource
from scoring import EventScore

class CorrelationEngine:
    def __init__(self):
        self.tilt_level = 0.0
        self.last_pattern_time = 0
        self.cooldown = 10.0 
        
    def _calculate_momentum(self, history: List[str]) -> str:
        if len(history) < 3: return "Stable"
        
        val_map = {
            "excited": 2, "happy": 1, "positive": 1, "neutral": 0, 
            "annoyed": -1, "frustrated": -2, "angry": -3, "scared": -2
        }
        
        values = [val_map.get(h, 0) for h in history[-5:]]
        if not values: return "Stable"
        
        delta = values[-1] - values[0]
        if delta >= 2: return "Escalating_Positive"
        if delta <= -2: return "Escalating_Negative"
        return "Stable"
        
    def correlate(self, store: ContextStore) -> List[Dict[str, Any]]:
        now = time.time()
        store.emotional_momentum = self._calculate_momentum(store.sentiment_history)
        
        if now - self.last_pattern_time < self.cooldown: return []
        patterns = []
        
        layers = store.get_all_events_for_summary()
        events = layers['immediate'] + layers['recent']
        
        if not events: return []

        visuals = [e for e in events if e.source == InputSource.VISUAL_CHANGE]
        chat = [e for e in events if e.source in [InputSource.TWITCH_CHAT, InputSource.TWITCH_MENTION]]
        speech = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]

        # Momentum
        if store.emotional_momentum == "Escalating_Negative":
             patterns.append({
                 "text": "Warning: Mood deteriorating.", 
                 "score": EventScore(0.9, 0.8, 0.9, 0.0, 0.0), 
                 "metadata": {"type": "pattern_momentum_neg"}
             })

        # Meme Moment
        if visuals and len(chat) >= 4:
            best_vis = max(visuals, key=lambda x: x.score.interestingness)
            if best_vis.score.interestingness > 0.6:
                patterns.append({
                    "text": f"Meme Moment: Chat reacting to '{best_vis.text}'", 
                    "score": EventScore(0.95, 0.6, 1.0, 0.0, 0.0), 
                    "metadata": {"type": "pattern_meme", "visual_ref": best_vis.text}
                })

        # Tilt
        tilt_keywords = ["damn", "shit", "fuck", "no", "why", "stupid", "impossible", "dead", "died", "trash"]
        recent_frustration = sum(1 for s in speech if any(k in s.text.lower() for k in tilt_keywords) or s.metadata.get("sentiment") in ["frustrated", "angry"])
        
        if recent_frustration >= 1: self.tilt_level = min(1.0, self.tilt_level + 0.25)
        else: self.tilt_level = max(0.0, self.tilt_level - 0.05)
        
        if self.tilt_level > 0.6:
            patterns.append({
                "text": f"Tilt Warning: Level {int(self.tilt_level * 100)}%", 
                "score": EventScore(0.8, 0.5, 0.9, 0.0, 0.0), 
                "metadata": {"type": "pattern_tilt", "level": self.tilt_level}
            })

        # Victory
        victory_keywords = ["yes", "boom", "let's go", "lets go", "finally", "did it", "won", "beat"]
        clutch_speech = [s for s in speech if any(k in s.text.lower() for k in victory_keywords)]
        if any(s.score.emotional_intensity > 0.7 and s.metadata.get("sentiment") in ["excited", "positive"] for s in clutch_speech):
            patterns.append({
                "text": "Victory Moment!", 
                "score": EventScore(0.9, 0.8, 1.0, 0.0, 0.0), 
                "metadata": {"type": "pattern_victory"}
            })

        # --- [REQ 8] Enhanced Silence Classification ---
        if not layers['immediate']: # Silence in last 10s
            # 1. Technical Silence (Loading/Menus)
            last_vis = layers['recent'][-1] if layers['recent'] and layers['recent'][-1].source == InputSource.VISUAL_CHANGE else None
            if last_vis and any(k in last_vis.text.lower() for k in ["loading", "menu", "pause", "saving"]):
                pass # Ignore technical silence
            
            # 2. Suspense (Horror/Scared context)
            elif store.current_mood == "Scared" or store.emotional_momentum == "Escalating_Negative":
                 pass # Ignore suspenseful silence (don't break tension)

            # 3. Contemplative (User thinking/Menuing without frustration)
            elif store.current_mood == "Neutral" and self.tilt_level < 0.2:
                 pass # User is just chilling/thinking

            # 4. Awkward/Dead Air (Nothing happening, no mood reason for it)
            else:
                avg_score = sum(e.score.interestingness for e in layers['recent']) / len(layers['recent']) if layers['recent'] else 0
                if avg_score < 0.25 and len(layers['recent']) < 3:
                     patterns.append({
                        "text": "Engagement Void: Awkward silence detected.",
                        "score": EventScore(interestingness=0.5, urgency=0.7, conversational_value=0.9), 
                        "metadata": {"type": "pattern_void"}
                    })

        if patterns: self.last_pattern_time = now
        return patterns
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
        
    def correlate(self, store: ContextStore) -> List[Dict[str, Any]]:
        now = time.time()
        if now - self.last_pattern_time < self.cooldown: return []
            
        patterns = []
        layers = store.get_all_events_for_summary()
        events = layers['immediate'] + layers['recent']
        
        if not events: return []

        visuals = [e for e in events if e.source == InputSource.VISUAL_CHANGE]
        chat = [e for e in events if e.source in [InputSource.TWITCH_CHAT, InputSource.TWITCH_MENTION]]
        speech = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]

        # 1. Meme Moment
        if visuals and len(chat) >= 4:
            best_vis = max(visuals, key=lambda x: x.score.interestingness)
            if best_vis.score.interestingness > 0.6:
                patterns.append({
                    "text": f"Meme Moment: Chat is reacting heavily to visual '{best_vis.text}'",
                    "score": EventScore(interestingness=0.95, conversational_value=1.0, urgency=0.6),
                    "metadata": {"type": "pattern_meme", "visual_ref": best_vis.text}
                })

        # 2. Tilt Detection
        tilt_keywords = ["damn", "shit", "fuck", "no", "why", "stupid", "impossible", "dead", "died", "trash"]
        recent_frustration = sum(1 for s in speech if any(k in s.text.lower() for k in tilt_keywords) or s.metadata.get("sentiment") in ["frustrated", "angry"])
        
        if recent_frustration >= 1: self.tilt_level = min(1.0, self.tilt_level + 0.25)
        else: self.tilt_level = max(0.0, self.tilt_level - 0.05)
            
        if self.tilt_level > 0.6:
            patterns.append({
                "text": f"Tilt Warning: User seems frustrated (Tilt Level {int(self.tilt_level * 100)}%).",
                "score": EventScore(interestingness=0.8, emotional_intensity=0.9, urgency=0.5),
                "metadata": {"type": "pattern_tilt", "level": self.tilt_level}
            })

        # 3. Backseating
        backseat_keywords = ["try", "use", "equip", "go left", "go right", "missed", "you need", "press"]
        backseat_chat = [c for c in chat if any(k in c.text.lower() for k in backseat_keywords)]
        user_confused = any(s.metadata.get("sentiment") == "confused" or "?" in s.text for s in speech)
        
        if len(backseat_chat) >= 2 and user_confused:
            patterns.append({
                "text": "Backseating Detected: Chat is giving advice while user is confused.",
                "score": EventScore(interestingness=0.7, conversational_value=0.9, urgency=0.4),
                "metadata": {"type": "pattern_backseating"}
            })

        # 4. Victory
        victory_keywords = ["yes", "boom", "let's go", "lets go", "finally", "did it", "won", "beat"]
        clutch_speech = [s for s in speech if any(k in s.text.lower() for k in victory_keywords)]
        if any(s.score.emotional_intensity > 0.7 and s.metadata.get("sentiment") in ["excited", "positive"] for s in clutch_speech):
            patterns.append({
                "text": "Victory Moment: High energy positive reaction detected.",
                "score": EventScore(interestingness=0.9, emotional_intensity=1.0, urgency=0.8),
                "metadata": {"type": "pattern_victory"}
            })

        # 5. Jumpscare
        scare_speech = [s for s in speech if s.metadata.get("sentiment") == "scared" or s.score.urgency > 0.85]
        if scare_speech:
            patterns.append({
                "text": "Jumpscare/Panic: Sudden fear response detected.",
                "score": EventScore(interestingness=0.85, emotional_intensity=1.0, urgency=0.9),
                "metadata": {"type": "pattern_scare"}
            })

        # --- 6. Silence Profiling (UPDATED) ---
        if not layers['immediate']: # Silence in last 10s
            
            # Check for Technical Silence (Loading screens)
            last_vis = layers['recent'][-1] if layers['recent'] and layers['recent'][-1].source == InputSource.VISUAL_CHANGE else None
            if last_vis and any(k in last_vis.text.lower() for k in ["loading", "menu", "pause", "saving"]):
                # Technical Silence - Ignore
                pass
                
            # Check for Suspense (Game audio high, but no speech)
            # This requires detailed audio metadata we might not fully have yet, but we can infer
            
            # Check for Awkward/Dead Air
            else:
                avg_score = sum(e.score.interestingness for e in layers['recent']) / len(layers['recent']) if layers['recent'] else 0
                if avg_score < 0.25 and len(layers['recent']) < 3:
                     patterns.append({
                        "text": "Engagement Void: Awkward silence detected. Good time to initiate topic.",
                        "score": EventScore(interestingness=0.5, urgency=0.7, conversational_value=0.9), 
                        "metadata": {"type": "pattern_void"}
                    })

        if patterns:
            self.last_pattern_time = now
            
        return patterns
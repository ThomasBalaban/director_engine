# Save as: director_engine/correlation_engine.py
import time
import re
from typing import List, Dict, Any
from context.context_store import ContextStore
from config import InputSource
from scoring import EventScore

# Article-anchored noun extraction — vision descriptions almost always say
# "a person", "the monitor", "an animal", so this catches real subjects
# instead of whatever word happens to appear first.
_ARTICLE_NOUN_RE = re.compile(r"\b(?:a|an|the)\s+([a-z][a-z\-]{2,})\b", re.IGNORECASE)

# Words that look noun-ish after an article but are streaming-room boilerplate
# or framing-language. Anything in here is excluded from fixation candidates.
_FIXATION_STOPWORDS = {
    # Framing / vision-narrator artifacts
    "frame", "frames", "image", "scene", "shot", "view", "clip", "video",
    "background", "foreground", "left", "right", "center", "top", "bottom",
    "side", "edge", "corner", "moment", "second", "while", "bit", "couple",
    "few", "lot", "way", "thing", "things", "kind", "type", "sort", "part",
    "set", "piece", "area", "place",
    # Streaming-room boilerplate (almost always present, not actually fixating)
    "person", "people", "man", "woman", "guy", "character", "streamer", "user",
    "screen", "monitor", "monitors", "display", "computer", "keyboard",
    "mouse", "desk", "chair", "room", "light", "lighting", "ring",
    "camera", "webcam", "microphone",
}

# How many *distinct* recent frames a noun must appear in before it counts.
_FIXATION_MIN_FRAMES = 4
# Fraction of the recent window the noun must cover (caps lock-in on
# permanent fixtures like the streamer).
_FIXATION_MIN_COVERAGE = 0.6
# Don't re-fire the same fixation within this many seconds.
_FIXATION_REFIRE_COOLDOWN = 60.0


class CorrelationEngine:
    def __init__(self):
        self.tilt_level = 0.0
        self.last_pattern_time = 0
        self.cooldown = 15.0  # Increased cooldown to prevent spam
        self.last_fixation_fired: Dict[str, float] = {}  # entity -> last fire time
        self.last_fixation_check = 0

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

    def _extract_frame_nouns(self, text: str) -> set:
        """Pull noun candidates that follow 'a/an/the' — one per frame, dedup'd."""
        nouns = set()
        for m in _ARTICLE_NOUN_RE.finditer(text):
            noun = m.group(1).lower()
            if len(noun) >= 4 and noun not in _FIXATION_STOPWORDS:
                nouns.add(noun)
        return nouns

    def check_visual_fixations(self, store: ContextStore) -> List[Dict[str, Any]]:
        """
        Detect recurring visual entities (Gymbag effect).

        Counts how many *distinct* recent frames mention each candidate noun.
        A fixation requires both an absolute floor (>= _FIXATION_MIN_FRAMES) and
        a coverage ratio (>= _FIXATION_MIN_COVERAGE of recent frames) so that
        always-on fixtures don't dominate and one-off framing words don't fire.
        """
        now = time.time()
        if now - self.last_fixation_check < 10.0:
            return []
        self.last_fixation_check = now

        recent_visuals = [e.text for e in store.recent if e.source == InputSource.VISUAL_CHANGE]
        if len(recent_visuals) < _FIXATION_MIN_FRAMES:
            return []

        frames_per_noun: Dict[str, int] = {}
        for desc in recent_visuals:
            for noun in self._extract_frame_nouns(desc):
                frames_per_noun[noun] = frames_per_noun.get(noun, 0) + 1

        if not frames_per_noun:
            return []

        coverage_threshold = max(
            _FIXATION_MIN_FRAMES,
            int(len(recent_visuals) * _FIXATION_MIN_COVERAGE),
        )
        candidates = [(n, c) for n, c in frames_per_noun.items() if c >= coverage_threshold]
        if not candidates:
            return []

        # Pick the strongest recurrence; respect per-entity refire cooldown
        candidates.sort(key=lambda x: x[1], reverse=True)
        for entity, count in candidates:
            if now - self.last_fixation_fired.get(entity, 0.0) < _FIXATION_REFIRE_COOLDOWN:
                continue
            self.last_fixation_fired[entity] = now
            print(f"👁️ [Fixation] Obsessing over: {entity} ({count}/{len(recent_visuals)} frames)")
            return [{
                "text": f"Visual Fixation: I keep seeing a {entity}.",
                "score": EventScore(interestingness=0.9, urgency=0.7, conversational_value=1.0),
                "metadata": {"type": "fixation", "entity": entity},
            }]

        return []

    def correlate(self, store: ContextStore) -> List[Dict[str, Any]]:
        now = time.time()
        store.emotional_momentum = self._calculate_momentum(store.sentiment_history)
        
        patterns = []
        
        # --- [NEURO-FICATION] Fixations ---
        fixations = self.check_visual_fixations(store)
        patterns.extend(fixations)
        
        layers = store.get_all_events_for_summary()
        events = layers['immediate'] + layers['recent']
        
        if not events: return patterns

        visuals = [e for e in events if e.source == InputSource.VISUAL_CHANGE]
        chat = [e for e in events if e.source in [InputSource.TWITCH_CHAT, InputSource.TWITCH_MENTION]]
        speech = [e for e in events if e.source in [InputSource.MICROPHONE, InputSource.DIRECT_MICROPHONE]]

        # --- [NEURO-FICATION] Skill Issue / Game Over Detection ---
        # 1. Text Triggers (Literal reading)
        fail_text_keywords = ["you died", "game over", "wasted", "mission failed", "retry", "defeat"]
        
        # 2. Visual Triggers (Vibes/Colors for when text is vague)
        # Based on your sample data: "red surrounding hue", "Explosions", "red backdrop"
        fail_visual_keywords = ["red surrounding", "red backdrop", "blood", "damage", "health low", "grey screen"]
        
        recent_visual_text = " ".join([v.text.lower() for v in visuals[-3:]]) # Check last 3 frames
        
        is_fail = False
        fail_type = "unknown"
        
        if any(k in recent_visual_text for k in fail_text_keywords):
            is_fail = True
            fail_type = "text"
        elif any(k in recent_visual_text for k in fail_visual_keywords):
            # Only trigger on visuals if we haven't seen a victory recently
            is_fail = True
            fail_type = "visual_vibe"

        if is_fail:
            # High urgency pattern
            if now - self.last_pattern_time > 20.0: # Longer cooldown for deaths
                print(f"💀 [Pattern] Skill Issue Detected ({fail_type})")
                patterns.append({
                    "text": "EVENT: The Handler just died/failed (Visual/Text Trigger).",
                    "score": EventScore(interestingness=1.0, urgency=1.0, conversational_value=1.0), 
                    "metadata": {"type": "skill_issue"}
                })
                self.last_pattern_time = now 

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
            if best_vis.score.interestingness > 0.6 and (now - self.last_pattern_time > self.cooldown):
                patterns.append({
                    "text": f"Meme Moment: Chat reacting to '{best_vis.text}'", 
                    "score": EventScore(0.95, 0.6, 1.0, 0.0, 0.0), 
                    "metadata": {"type": "pattern_meme", "visual_ref": best_vis.text}
                })
                self.last_pattern_time = now

        # Tilt
        tilt_keywords = ["damn", "shit", "fuck", "no", "why", "stupid", "impossible", "dead", "died", "trash"]
        recent_frustration = sum(1 for s in speech if any(k in s.text.lower() for k in tilt_keywords) or s.metadata.get("sentiment") in ["frustrated", "angry"])
        
        if recent_frustration >= 1: self.tilt_level = min(1.0, self.tilt_level + 0.25)
        else: self.tilt_level = max(0.0, self.tilt_level - 0.05)
        
        if self.tilt_level > 0.6 and (now - self.last_pattern_time > self.cooldown):
            patterns.append({
                "text": f"Tilt Warning: Level {int(self.tilt_level * 100)}%", 
                "score": EventScore(0.8, 0.5, 0.9, 0.0, 0.0), 
                "metadata": {"type": "pattern_tilt", "level": self.tilt_level}
            })
            self.last_pattern_time = now

        # Victory
        victory_keywords = ["yes", "boom", "let's go", "lets go", "finally", "did it", "won", "beat"]
        clutch_speech = [s for s in speech if any(k in s.text.lower() for k in victory_keywords)]
        if any(s.score.emotional_intensity > 0.7 and s.metadata.get("sentiment") in ["excited", "positive"] for s in clutch_speech):
            if now - self.last_pattern_time > self.cooldown:
                patterns.append({
                    "text": "Victory Moment!", 
                    "score": EventScore(0.9, 0.8, 1.0, 0.0, 0.0), 
                    "metadata": {"type": "pattern_victory"}
                })
                self.last_pattern_time = now

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
                if avg_score < 0.25 and len(layers['recent']) < 3 and (now - self.last_pattern_time > self.cooldown):
                     patterns.append({
                        "text": "Engagement Void: Awkward silence detected.",
                        "score": EventScore(interestingness=0.5, urgency=0.7, conversational_value=0.9), 
                        "metadata": {"type": "pattern_void"}
                    })
                     self.last_pattern_time = now

        return patterns
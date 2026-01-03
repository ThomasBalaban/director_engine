# Save as: director_engine/scene_manager.py
import time
from context.context_store import ContextStore
from config import SceneType, InputSource, FlowState, ConversationState

class SceneManager:
    def __init__(self):
        self.last_transition_time = 0
        self.min_scene_duration = 15.0 # Minimum seconds to stay in a scene
        self.cooldowns = {scene: 0 for scene in SceneType}

    def update_scene(self, store: ContextStore):
        now = time.time()
        
        # 1. Lock-in Check: Don't switch if we just switched
        if now - self.last_transition_time < self.min_scene_duration:
            return

        current_scene = store.current_scene
        layers = store.get_all_events_for_summary()
        recent = layers['recent']
        chat_vel, energy = store.get_activity_metrics()
        conv_state = store.current_conversation_state
        
        # 2. Calculate Confidence for Potential Scenes
        scores = {scene: 0.0 for scene in SceneType}
        
        # Baseline
        scores[SceneType.CHILL_CHATTING] = 0.5
        
        # HORROR: Check mood and momentum
        if store.current_mood == "Scared" or store.emotional_momentum == "Escalating_Negative":
            scores[SceneType.HORROR_TENSION] += 0.8
            
        # COMEDY: Check for meme pattern
        if any(e.source == InputSource.SYSTEM_PATTERN and "Meme" in e.text for e in recent):
            scores[SceneType.COMEDY_MOMENT] += 0.9
            
        # COMBAT: High energy + state
        if energy > 0.7 and conv_state in [ConversationState.FRUSTRATED, ConversationState.CELEBRATORY]:
            scores[SceneType.COMBAT_HIGH] += 0.8
            
        # MENUING: Low energy + visuals
        # Heuristic: low energy but visual activity
        visual_count = len([e for e in recent if e.source == InputSource.VISUAL_CHANGE])
        if energy < 0.3 and visual_count > 5:
             scores[SceneType.MENUING] += 0.7
            
        # TECHNICAL: Dead air + low chat
        if store.current_flow == FlowState.DEAD_AIR and chat_vel < 2:
            scores[SceneType.TECHNICAL_DOWNTIME] += 0.6

        # 3. Select Winner
        best_scene = max(scores, key=scores.get)
        best_score = scores[best_scene]
        
        # 4. Transition Thresholds (Hysteresis)
        # Needs higher score to switch away from current scene
        threshold = 0.6 if best_scene != current_scene else 0.0
        
        if best_scene != current_scene and best_score > threshold:
            # Check specific cooldown
            if now > self.cooldowns[best_scene]:
                store.set_scene(best_scene)
                self.last_transition_time = now
                self.cooldowns[current_scene] = now + 30.0 # Cooldown for the scene we just left
                print(f"🎬 [SceneManager] Switched to {best_scene.name} (Score: {best_score:.2f})")
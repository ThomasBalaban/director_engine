# Save as: director_engine/scene_manager.py
from context_store import ContextStore
from config import SceneType, InputSource, FlowState, ConversationState

class SceneManager:
    def update_scene(self, store: ContextStore):
        """
        Heuristically determines the current scene based on metrics and patterns.
        """
        layers = store.get_all_events_for_summary()
        recent = layers['recent']
        
        chat_vel, energy = store.get_activity_metrics()
        conv_state = store.current_conversation_state
        
        # 1. Default to Chill
        new_scene = SceneType.CHILL_CHATTING
        
        # 2. Detection Logic
        
        # HORROR: "Jumpscare" pattern or "Scared" mood
        if store.current_mood == "Scared":
            new_scene = SceneType.HORROR_TENSION
            
        # COMEDY: "Meme Moment" pattern (could check patterns in recent events)
        # If recent events contain SYSTEM_PATTERN with type 'pattern_meme'
        elif any(e.source == InputSource.SYSTEM_PATTERN and "Meme Moment" in e.text for e in recent):
             new_scene = SceneType.COMEDY_MOMENT
             
        # COMBAT: High energy + Visuals + Frustrated/Celebratory state
        elif energy > 0.7 and conv_state in [ConversationState.FRUSTRATED, ConversationState.CELEBRATORY]:
             new_scene = SceneType.COMBAT_HIGH
             
        # MENUING: Visual text contains "Menu", "Inventory", "Settings"
        # We'd need access to visual content details. 
        # For now, we can infer if energy is low but visuals are active? 
        # Or rely on the LLM summary. (Simple heuristic for now: low energy + high visual count = exploration/menuing)
        elif energy < 0.3 and len([e for e in recent if e.source == InputSource.VISUAL_CHANGE]) > 5:
             new_scene = SceneType.MENUING
             
        # TECHNICAL: "Dead Air" flow + low scores
        elif store.current_flow == FlowState.DEAD_AIR:
             new_scene = SceneType.TECHNICAL_DOWNTIME
        
        store.set_scene(new_scene)
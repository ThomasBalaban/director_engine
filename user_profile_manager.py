# Save as: director_engine/user_profile_manager.py
import json
import os
import time
import random
from typing import Dict, Any, List, Optional
from config import PROFILES_DIR, DEFAULT_RELATIONSHIP_TIER, DEFAULT_AFFINITY

class UserProfileManager:
    def __init__(self):
        self.profiles_dir = PROFILES_DIR
        if not os.path.exists(self.profiles_dir):
            os.makedirs(self.profiles_dir)
            print(f"[Profiles] Created directory: {self.profiles_dir}")

    def _get_filepath(self, username: str) -> str:
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).lower()
        return os.path.join(self.profiles_dir, f"{safe_username}.json")

    def get_profile(self, username: str) -> Dict[str, Any]:
        filepath = self._get_filepath(username)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    profile['last_seen'] = time.time()
                    
                    # Backwards compatibility: Ensure role exists if loading old profile
                    if 'role' not in profile:
                        profile['role'] = "handler" if username.lower() == "peepingotter" else "viewer"
                        
                    self._save_json(filepath, profile)
                    return profile
            except Exception as e:
                print(f"[Profiles] Error loading profile for {username}: {e}")
                return self._create_default_profile(username)
        else:
            return self._create_default_profile(username)

    def _create_default_profile(self, username: str) -> Dict[str, Any]:
        print(f"[Profiles] Creating new profile for: {username}")
        now = time.time()
        
        # --- NEURO-FICATION: Identify the Handler ---
        # The 'Handler' is the owner/creator. They get bullied more.
        role = "handler" if username.lower() == "peepingotter" else "viewer"
        
        profile = {
            "username": username,
            "nickname": username,
            "role": role,  # New Field
            "is_adult": True, 
            "created_at": now,
            "last_seen": now,
            "relationship": {
                "tier": DEFAULT_RELATIONSHIP_TIER,
                "affinity": DEFAULT_AFFINITY,
                "vibe": "Neutral"
            },
            "facts": [],
            "nami_opinions": []
        }
        self._save_json(self._get_filepath(username), profile)
        return profile
    
    def _validate_fact(self, fact_text: str, username: str) -> bool:
        text = fact_text.lower().strip()
        username_lower = username.lower()
        if text == username_lower: return False
        trivial_phrases = [f"{username_lower} is a new user", f"{username_lower} is a user", "is a viewer"]
        if any(phrase in text for phrase in trivial_phrases): return False
        garbage_phrases = ["revealed a new fact", "user revealed", "extracted fact"]
        if any(phrase in text for phrase in garbage_phrases): return False
        if len(text) < 5: return False
        return True

    def update_profile(self, username: str, updates: Dict[str, Any]):
        profile = self.get_profile(username)
        modified = False

        if 'new_facts' in updates and updates['new_facts']:
            for fact in updates['new_facts']:
                fact = fact.strip()
                if not self._validate_fact(fact, username):
                    continue

                if not any(f['content'] == fact for f in profile['facts']):
                    profile['facts'].append({
                        "content": fact,
                        "timestamp": time.time(),
                        "category": "learned",
                        "usage_count": 0,      # Track usage for variety
                        "last_used": 0.0
                    })
                    modified = True
                    print(f"[Profiles] ✅ Added valid fact for {username}: {fact}")

        if 'new_opinion' in updates and updates['new_opinion']:
            if updates['new_opinion'] not in profile['nami_opinions']:
                profile['nami_opinions'].append(updates['new_opinion'])
                modified = True

        if 'affinity_change' in updates:
            old_aff = profile['relationship']['affinity']
            profile['relationship']['affinity'] = max(0, min(100, old_aff + updates['affinity_change']))
            modified = True

        if modified:
            self._save_json(self._get_filepath(username), profile)
            return profile
        return profile

    # --- Proactive Topic Suggestion ---
    def get_under_discussed_fact(self, username: str) -> Optional[str]:
        """Returns a fact that hasn't been discussed much, prioritizing variety."""
        profile = self.get_profile(username)
        facts = profile.get('facts', [])
        if not facts: return None
        
        # Filter for facts not used recently (last 1 hour)
        now = time.time()
        available = [f for f in facts if (now - f.get('last_used', 0)) > 3600]
        
        if not available: return None
        
        # Sort by usage count (lowest first)
        available.sort(key=lambda x: x.get('usage_count', 0))
        
        # Pick from the top 3 least used
        candidates = available[:3]
        selected = random.choice(candidates)
        
        return selected['content']

    def mark_fact_used(self, username: str, fact_content: str):
        """Increments usage count for a fact."""
        profile = self.get_profile(username)
        for fact in profile['facts']:
            if fact['content'] == fact_content:
                fact['usage_count'] = fact.get('usage_count', 0) + 1
                fact['last_used'] = time.time()
                self._save_json(self._get_filepath(username), profile)
                return

    def _save_json(self, filepath: str, data: Dict[str, Any]):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Profiles] Error saving profile: {e}")
# Save as: director_engine/user_profile_manager.py
import json
import os
import time
from typing import Dict, Any, List
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
        profile = {
            "username": username,
            "nickname": username,
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
        """
        Filters out garbage facts that are just meta-commentary or trivial.
        """
        text = fact_text.lower().strip()
        username_lower = username.lower()
        
        # 1. Reject exact username match
        if text == username_lower:
            return False
            
        # 2. Reject phrases that just describe the user status
        # "peepingotter is a new user", "peepingotter is a user"
        trivial_phrases = [
            f"{username_lower} is a new user",
            f"{username_lower} is a user",
            f"{username_lower} is a viewer",
            "is a user who",
            "is a new user",
            "is a viewer"
        ]
        if any(phrase in text for phrase in trivial_phrases):
            return False

        # 3. Reject meta-commentary about the act of revealing
        garbage_phrases = [
            "revealed a new fact",
            "user revealed",
            "fact about themselves",
            "extracted fact",
            "user stated",
            "mentioned that"
        ]
        if any(phrase in text for phrase in garbage_phrases):
            return False
            
        # 4. Reject too short
        if len(text) < 5:
            return False
            
        return True

    def update_profile(self, username: str, updates: Dict[str, Any]):
        profile = self.get_profile(username)
        modified = False

        if 'new_facts' in updates and updates['new_facts']:
            for fact in updates['new_facts']:
                fact = fact.strip()
                
                # [VALIDATION STEP]
                if not self._validate_fact(fact, username):
                    print(f"[Profiles] 🗑️ Rejected garbage fact: '{fact}'")
                    continue

                if not any(f['content'] == fact for f in profile['facts']):
                    profile['facts'].append({
                        "content": fact,
                        "timestamp": time.time(),
                        "category": "learned"
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

    def _save_json(self, filepath: str, data: Dict[str, Any]):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Profiles] Error saving profile: {e}")
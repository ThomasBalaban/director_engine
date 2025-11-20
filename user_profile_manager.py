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
        # Sanitize username for filesystem
        safe_username = "".join(c for c in username if c.isalnum() or c in ('-', '_')).lower()
        return os.path.join(self.profiles_dir, f"{safe_username}.json")

    def get_profile(self, username: str) -> Dict[str, Any]:
        """Loads a profile or creates a default one if it doesn't exist."""
        filepath = self._get_filepath(username)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    # Update last seen immediately on load
                    profile['last_seen'] = time.time()
                    self._save_json(filepath, profile)
                    return profile
            except Exception as e:
                print(f"[Profiles] Error loading profile for {username}: {e}")
                return self._create_default_profile(username)
        else:
            return self._create_default_profile(username)

    def _create_default_profile(self, username: str) -> Dict[str, Any]:
        """Creates and saves a new default profile."""
        print(f"[Profiles] Creating new profile for: {username}")
        now = time.time()
        profile = {
            "username": username,
            "nickname": username, # Default nickname is username
            "is_adult": True, # Default assume adult, can be flagged manually
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

    def update_profile(self, username: str, updates: Dict[str, Any]):
        """
        Updates specific fields in a user profile.
        updates can contain: 'facts' (list to append), 'opinion' (str to add), 'affinity_change' (int)
        """
        profile = self.get_profile(username)
        modified = False

        # Add new facts
        if 'new_facts' in updates and updates['new_facts']:
            for fact in updates['new_facts']:
                # Check for dupes (simple text check)
                if not any(f['content'] == fact for f in profile['facts']):
                    profile['facts'].append({
                        "content": fact,
                        "timestamp": time.time(),
                        "category": "learned"
                    })
                    modified = True
                    print(f"[Profiles] Added fact for {username}: {fact}")

        # Add new opinion
        if 'new_opinion' in updates and updates['new_opinion']:
            if updates['new_opinion'] not in profile['nami_opinions']:
                profile['nami_opinions'].append(updates['new_opinion'])
                modified = True

        # Update affinity
        if 'affinity_change' in updates:
            old_aff = profile['relationship']['affinity']
            profile['relationship']['affinity'] = max(0, min(100, old_aff + updates['affinity_change']))
            modified = True
            print(f"[Profiles] {username} affinity: {old_aff} -> {profile['relationship']['affinity']}")

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
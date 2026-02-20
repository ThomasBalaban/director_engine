import httpx
from typing import Dict, Any

PROFILE_SERVICE_URL = "http://localhost:8008"

class UserProfileManager:
    """
    Synchronous Client Proxy: This perfectly mimics your old class so 
    core_logic and llm_analyst don't need to be updated with 'await' keywords.
    """
    def __init__(self):
        # Use the synchronous client so it blocks and returns the dictionary immediately
        self.client = httpx.Client(base_url=PROFILE_SERVICE_URL)
        
    def get_profile(self, username: str) -> Dict[str, Any]:
        try:
            response = self.client.get(f"/profiles/{username}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[Profile Proxy] ⚠️ Error fetching profile for {username}: {e}")
            return {}

    def update_profile(self, username: str, updates: Dict[str, Any]):
        try:
            self.client.post("/profiles/update", json={
                "user_id": username, 
                "data": updates
            })
        except Exception as e:
            print(f"[Profile Proxy] ⚠️ Error updating profile for {username}: {e}")
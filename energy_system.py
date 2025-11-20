# Save as: director_engine/energy_system.py
import time
from config import ENERGY_MAX, ENERGY_REGEN_PER_SEC

class EnergySystem:
    def __init__(self):
        self.current_energy = ENERGY_MAX
        self.last_update = time.time()
        
    def update(self):
        """Regenerate energy based on time passed."""
        now = time.time()
        delta = now - self.last_update
        self.last_update = now
        
        # Regenerate
        self.current_energy = min(ENERGY_MAX, self.current_energy + (ENERGY_REGEN_PER_SEC * delta))
        
    def can_afford(self, cost: float) -> bool:
        self.update() # Ensure fresh state
        return self.current_energy >= cost
        
    def spend(self, cost: float) -> bool:
        if self.can_afford(cost):
            self.current_energy -= cost
            print(f"⚡ [Energy] Spent {cost}. Remaining: {self.current_energy:.1f}")
            return True
        print(f"⚡ [Energy] FAILED to spend {cost}. Current: {self.current_energy:.1f}")
        return False
        
    def get_status(self):
        self.update()
        return {
            "current": round(self.current_energy, 1),
            "max": ENERGY_MAX,
            "percent": round((self.current_energy / ENERGY_MAX) * 100)
        }
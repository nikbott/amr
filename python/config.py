from dataclasses import dataclass
from typing import Tuple

@dataclass
class AMRConfig:
    """
    Simulation Configuration.
    Maps normalized physical coordinates [0.0, 1.0] to integer Morton space.
    """
    # Grid Resolution
    max_level: int = 21  # Max depth (Level 21 allows 2^21 grid in 3D)
    
    # Physics Parameters (Normalized 0.0 to 1.0)
    center: Tuple[float, ...] = (0.5, 0.5, 0.5)
    radius: float = 0.25
    bandwidth: float = 0.05
    
    # Refinement Limits
    coarse_level: int = 4   # Minimum depth everywhere
    fine_level: int = 10    # Maximum depth at features
    
    @property
    def domain_width(self) -> int:
        return 1 << self.max_level
    
    def get_int_center(self, ndim=2):
        w = self.domain_width
        if ndim == 2:
            return (self.center[0] * w, self.center[1] * w)
        return (self.center[0] * w, self.center[1] * w, self.center[2] * w)
        
    def get_int_radius(self):
        return self.radius * self.domain_width
        
    def get_int_bandwidth(self):
        return self.bandwidth * self.domain_width

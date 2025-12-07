import numpy as np
from morton import Morton2D, Morton3D
from config import AMRConfig

class CircleOracle2D:
    def __init__(self, config: AMRConfig):
        self.config = config
        self.cx, self.cy = config.get_int_center(ndim=2)
        self.radius = config.get_int_radius()
        self.bandwidth = config.get_int_bandwidth()
        self.max_depth = config.fine_level
        self.min_depth = config.coarse_level

    def __call__(self, node, max_level_grid: int) -> bool:
        # 1. Decode Position
        x, y = Morton2D.decode(node.code)
        size = 1 << (max_level_grid - node.level)
        
        # 2. Node Center
        node_cx = x + size * 0.5
        node_cy = y + size * 0.5
        
        # 3. Squared Distance (Avoids sqrt)
        dist_sq = (node_cx - self.cx)**2 + (node_cy - self.cy)**2
        
        # 4. Determine Bounds
        # extent = size * sqrt(2) / 2
        extent = size * 0.70710678
        threshold = self.bandwidth + extent
        
        # We want: |dist - radius| < threshold
        # Equivalent to: (radius - threshold)^2 < dist^2 < (radius + threshold)^2
        # Note: Must handle negative lower bound case (dist is always >= 0)
        
        upper_bound = self.radius + threshold
        upper_sq = upper_bound * upper_bound
        
        lower_bound = self.radius - threshold
        # If lower_bound is negative, the condition dist > lower_bound is always true
        lower_sq = 0.0 if lower_bound < 0 else lower_bound * lower_bound

        # 5. Check
        is_refining = (dist_sq < upper_sq) and (dist_sq > lower_sq)
        
        return (node.level < self.min_depth) or \
               (is_refining and node.level < self.max_depth)

class SphereOracle3D:
    def __init__(self, config: AMRConfig):
        self.config = config
        self.cx, self.cy, self.cz = config.get_int_center(ndim=3)
        self.radius = config.get_int_radius()
        self.bandwidth = config.get_int_bandwidth()
        self.max_depth = config.fine_level
        self.min_depth = config.coarse_level

    def __call__(self, node, max_level_grid: int) -> bool:
        # 1. Decode Position
        x, y, z = Morton3D.decode(node.code)
        size = 1 << (max_level_grid - node.level)
        
        # 2. Node Center
        node_cx = x + size * 0.5
        node_cy = y + size * 0.5
        node_cz = z + size * 0.5
        
        # 3. Squared Distance
        dist_sq = (node_cx - self.cx)**2 + (node_cy - self.cy)**2 + (node_cz - self.cz)**2
        
        # 4. Determine Bounds
        # 3D Extent = size * sqrt(3) / 2
        extent = size * 0.8660254
        threshold = self.bandwidth + extent
        
        upper_bound = self.radius + threshold
        upper_sq = upper_bound * upper_bound
        
        lower_bound = self.radius - threshold
        lower_sq = 0.0 if lower_bound < 0 else lower_bound * lower_bound

        # 5. Check
        is_refining = (dist_sq < upper_sq) and (dist_sq > lower_sq)
        
        return (node.level < self.min_depth) or \
               (is_refining and node.level < self.max_depth)
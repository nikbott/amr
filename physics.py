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
        x, y = Morton2D.decode(node.code)
        size = 1 << (max_level_grid - node.level)
        
        # Node Center
        node_cx = x + size * 0.5
        node_cy = y + size * 0.5
        
        # Distance from Grid Center to Node Center
        dist = np.sqrt((node_cx - self.cx)**2 + (node_cy - self.cy)**2)
        
        # [FIX] Box Intersection Logic
        # Calculate the node's semi-diagonal (distance from center to corner)
        # to check if ANY part of the node overlaps the refinement band.
        # extent = size * sqrt(2) / 2 ≈ size * 0.7071
        extent = size * 0.70710678
        
        return (node.level < self.min_depth) or \
               (abs(dist - self.radius) < (self.bandwidth + extent) and node.level < self.max_depth)

class SphereOracle3D:
    def __init__(self, config: AMRConfig):
        self.config = config
        self.cx, self.cy, self.cz = config.get_int_center(ndim=3)
        self.radius = config.get_int_radius()
        self.bandwidth = config.get_int_bandwidth()
        self.max_depth = config.fine_level
        self.min_depth = config.coarse_level

    def __call__(self, node, max_level_grid: int) -> bool:
        x, y, z = Morton3D.decode(node.code)
        size = 1 << (max_level_grid - node.level)
        
        node_cx = x + size * 0.5
        node_cy = y + size * 0.5
        node_cz = z + size * 0.5
        
        dist = np.sqrt((node_cx - self.cx)**2 + (node_cy - self.cy)**2 + (node_cz - self.cz)**2)
        
        # [FIX] 3D Extent = size * sqrt(3) / 2 ≈ size * 0.8660
        extent = size * 0.8660254
        
        return (node.level < self.min_depth) or \
               (abs(dist - self.radius) < (self.bandwidth + extent) and node.level < self.max_depth)

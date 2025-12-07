from abc import ABC, abstractmethod
from typing import List, Tuple, Callable
from dataclasses import dataclass
from morton import Morton2D, Morton3D

@dataclass(slots=True, order=True)
class Node:
    """Lightweight struct representing a leaf in the tree."""
    code: int   # Morton Index (64-bit)
    level: int  # Refinement Depth

class AbstractLinearTree(ABC):
    """
    Abstract Base Class for Linear Trees.
    Implements generic Refine and Balance loops for 2D/3D.
    """
    def __init__(self, max_level: int = 21):
        self.max_level = max_level
        self.leaves: List[Node] = [Node(0, 0)]

    @property
    @abstractmethod
    def ndim(self) -> int: pass

    @property
    def domain_width(self) -> int: return 1 << self.max_level

    @abstractmethod
    def _decode_impl(self, code: int) -> Tuple[int, ...]: pass

    @abstractmethod
    def _encode_impl(self, coords: Tuple[int, ...]) -> int: pass

    @abstractmethod
    def _get_neighbor_code(self, code: int, level: int, offset: Tuple[int, ...]) -> int: pass

    def get_geometry(self, node: Node) -> Tuple[Tuple[int, ...], int]:
        """Returns (coordinates_tuple, size) in integer space."""
        coords = self._decode_impl(node.code)
        size = 1 << (self.max_level - node.level)
        return coords, size

    def refine(self, oracle: Callable[['Node', int], bool]) -> bool:
        """
        Sequential Refinement Algorithm.
        Iterates leaves, applies oracle, splits if necessary, and re-sorts.
        """
        new_leaves = []
        has_changed = False
        
        # Pre-calculate offsets
        if self.ndim == 2:
            base_offsets = [(0,0), (1,0), (0,1), (1,1)]
        else:
            base_offsets = [(0,0,0), (1,0,0), (0,1,0), (1,1,0),
                            (0,0,1), (1,0,1), (0,1,1), (1,1,1)]

        for node in self.leaves:
            if oracle(node, self.max_level):
                has_changed = True
                current_coords = self._decode_impl(node.code)
                new_lvl = node.level + 1
                step = 1 << (self.max_level - new_lvl)

                for offset in base_offsets:
                    # child = parent + offset * step_size
                    child_coords = tuple(c + o * step for c, o in zip(current_coords, offset))
                    new_code = self._encode_impl(child_coords)
                    new_leaves.append(Node(new_code, new_lvl))
            else:
                new_leaves.append(node)

        if has_changed:
            new_leaves.sort(key=lambda n: n.code)
            self.leaves = new_leaves
            
        return has_changed

    def balance(self):
        """
        Enforce 2:1 Balance Constraint (Ripple Effect).
        Ensures no two adjacent cells differ by more than 1 refinement level.
        Batches refinements per ripple pass to minimize sorting overhead.
        """
        max_iter = 20
        # Directions excluding (0,0,...)
        if self.ndim == 2:
            directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if not (dx==0 and dy==0)]
        else:
            directions = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx==0 and dy==0 and dz==0)]

        for _ in range(max_iter):
            # 1. Build Map O(N)
            node_map = {n.code: n for n in self.leaves}
            to_refine_codes = set()

            # 2. Scan all leaves for violations
            for node in self.leaves:
                # We need to check if any neighbor is too coarse (level <= node.level - 2)
                min_valid_level = node.level - 1
                
                # If we are already at level 0 or 1, we can't have a neighbor at -1 or -2
                if min_valid_level < 1:
                    continue

                for direction in directions:
                    # Get neighbor code at MY level first
                    base_n_code = self._get_neighbor_code(node.code, node.level, direction)
                    if base_n_code == -1: continue 

                    # Search "up" the tree (coarser) starting from the violation threshold
                    # We check levels: [node.level - 2, node.level - 3, ... 0]
                    curr_search_level = node.level - 2
                    
                    while curr_search_level >= 0:
                        # Calculate mask for this coarser level
                        shift = (self.max_level - curr_search_level) * self.ndim
                        mask = ~((1 << shift) - 1)
                        
                        # Align the base neighbor code to this coarser grid
                        coarse_n_code = base_n_code & mask
                        
                        # If this neighbor is already marked, stop checking this direction
                        if coarse_n_code in to_refine_codes:
                            break

                        # Check if this coarse neighbor exists in the tree
                        if coarse_n_code in node_map:
                            neighbor = node_map[coarse_n_code]
                            # Confirm it is actually a leaf at this level (not just a prefix of a finer node)
                            if neighbor.level == curr_search_level:
                                to_refine_codes.add(neighbor.code)
                                break # Found the leaf, stop searching coarser levels
                        
                        curr_search_level -= 1

            # 3. Batch Refine
            # If no violations found in this pass, we are balanced.
            if not to_refine_codes:
                break

            # Apply refinement to all marked nodes at once
            self.refine(lambda n, ml: n.code in to_refine_codes)


class Quadtree(AbstractLinearTree):
    @property
    def ndim(self): return 2
    def _decode_impl(self, code): return Morton2D.decode(code)
    def _encode_impl(self, coords): return Morton2D.encode(coords)
    def _get_neighbor_code(self, code, level, offset):
        return Morton2D.get_neighbor(code, level, offset[0], offset[1], self.max_level)


class Octree(AbstractLinearTree):
    @property
    def ndim(self): return 3
    def _decode_impl(self, code): return Morton3D.decode(code)
    def _encode_impl(self, coords): return Morton3D.encode(coords)
    def _get_neighbor_code(self, code, level, offset):
        return Morton3D.get_neighbor(code, level, offset[0], offset[1], offset[2], self.max_level)

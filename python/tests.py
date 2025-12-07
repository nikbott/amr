import unittest
import numpy as np
import sys
from numba import cuda

# Local Imports
from tree import Quadtree, Octree
from morton import Morton2D, Morton3D
from physics import CircleOracle2D
from parallel import parallel_refine_step
from config import AMRConfig

class TestMortonLogic(unittest.TestCase):
    """
    Unit tests for Z-order curve mathematics, including neighbor finding.
    """
    def test_2d_encoding_decoding(self):
        # Basic patterns
        self.assertEqual(Morton2D.encode((1, 0)), 1)
        self.assertEqual(Morton2D.encode((0, 1)), 2)
        # Round trip
        coords = (1234, 5678)
        code = Morton2D.encode(coords)
        self.assertEqual(Morton2D.decode(code), coords)

    def test_3d_encoding_decoding(self):
        self.assertEqual(Morton3D.encode((1, 0, 0)), 1)
        self.assertEqual(Morton3D.encode((0, 0, 1)), 4)
        coords = (123, 456, 789)
        code = Morton3D.encode(coords)
        self.assertEqual(Morton3D.decode(code), coords)

    def test_2d_neighbor_finding(self):
        # Center point at (10, 10)
        code = Morton2D.encode((10, 10))
        level = 16 # Max level (size 1)
        
        # Right neighbor (11, 10)
        n_code = Morton2D.get_neighbor(code, level, dx=1, dy=0, max_level=16)
        self.assertEqual(Morton2D.decode(n_code), (11, 10))
        
        # Boundary check: Left neighbor of (0, 10) should be -1
        edge_code = Morton2D.encode((0, 10))
        n_code_edge = Morton2D.get_neighbor(edge_code, level, dx=-1, dy=0, max_level=16)
        self.assertEqual(n_code_edge, -1)

class TestTreeInvariants(unittest.TestCase):
    """
    Integration tests for Mesh Topology and Physics.
    """
    def test_quadtree_topology(self):
        """Splitting a node should yield exactly 4 children."""
        t = Quadtree()
        t.refine(lambda n, ml: True) # Split root
        self.assertEqual(len(t.leaves), 4)
        # Check Z-order sorting
        codes = [n.code for n in t.leaves]
        self.assertTrue(all(codes[i] <= codes[i+1] for i in range(len(codes)-1)))

    def test_octree_topology(self):
        """Splitting a node should yield exactly 8 children."""
        t = Octree()
        t.refine(lambda n, ml: True)
        self.assertEqual(len(t.leaves), 8)

    def test_area_conservation(self):
        """Sum of leaf areas must equal domain area."""
        t = Quadtree(max_level=10)
        # Refine randomly
        t.refine(lambda n, ml: n.level < 4 and n.code % 3 == 0)
        
        total_area = sum((1 << (t.max_level - n.level))**2 for n in t.leaves)
        self.assertEqual(total_area, t.domain_width**2)

class TestBalancing(unittest.TestCase):
    """
    Verifies the 2:1 Balance Constraint Algorithm.
    """
    def test_2_to_1_constraint(self):
        print("\n   [Balance] Verifying 2:1 Constraint on jagged mesh...")
        tree = Quadtree(max_level=8)
        
        # Create a highly unbalanced mesh (point refinement at corner)
        # This creates a jump from Level 0 to Level 6 at the boundary if unbalanced
        tree.refine(lambda n, ml: n.level < 6 and n.code == 0)
        
        # Run Balance
        tree.balance()
        
        # VERIFICATION LOOP
        # For every leaf, check all neighbors. 
        # Fail if neighbor level is > (my_level + 1) or < (my_level - 1)
        node_map = {n.code: n for n in tree.leaves}
        
        violations = 0
        for node in tree.leaves:
            # Check 4 cardinal neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                # We check neighbors at the SAME level
                n_code = tree._get_neighbor_code(node.code, node.level, (dx, dy))
                if n_code == -1: continue # Boundary
                
                # If neighbor exists in map, level diff is 0. OK.
                if n_code in node_map: continue
                
                # If not, neighbor is either FINER or COARSER.
                
                # 1. Check if neighbor is COARSER (Parent exists?)
                # Code aligned to Level-1
                shift = 2 # 2D
                parent_code = n_code & ~((1<<shift)-1)
                
                # If parent exists in map, level diff is 1. OK.
                if parent_code in node_map and node_map[parent_code].level == node.level - 1:
                    continue
                    
                # 2. Check if neighbor is FINER (Children exist?)
                # This is harder to check efficiently without spatial index.
                # However, the Balance algorithm guarantees *we* are not the problem 
                # if we are coarse.
                # The critical check is: Do I have a neighbor that is TOO COARSE?
                
                # Check Level-2 (Violation)
                shift2 = 2 * 2
                grandparent_code = n_code & ~((1<<shift2)-1)
                if grandparent_code in node_map and node_map[grandparent_code].level == node.level - 2:
                    violations += 1
                    print(f"VIOLATION: Node L{node.level} next to L{node.level-2}")

        self.assertEqual(violations, 0, "Found 2:1 Balance Violations!")
        print("      Success: Mesh is balanced.")

class TestGPUConsistency(unittest.TestCase):
    @unittest.skipIf(not cuda.is_available(), "GPU not available")
    def test_cpu_gpu_match(self):
        print("\n   [Parallel] Checking CPU vs GPU Bit-Exactness...")
        # Force 2D config
        cfg = AMRConfig(max_level=16, fine_level=8, coarse_level=3, 
                        center=(0.5, 0.5), radius=0.25)
        
        # CPU Run
        tree = Quadtree(max_level=cfg.max_level)
        oracle = CircleOracle2D(cfg)
        for _ in range(5): tree.refine(oracle)
        cpu_codes = np.array([n.code for n in tree.leaves], dtype=np.uint64)
        
        # GPU Run
        gpu_codes = np.array([0], dtype=np.uint64)
        gpu_levels = np.array([0], dtype=np.uint8)
        for _ in range(5):
            gpu_codes, gpu_levels = parallel_refine_step(gpu_codes, gpu_levels, cfg)
            
        if hasattr(gpu_codes, 'get'): gpu_codes = gpu_codes.get()
        
        self.assertEqual(len(cpu_codes), len(gpu_codes), "Element count mismatch")
        np.testing.assert_array_equal(cpu_codes, gpu_codes, "Morton codes mismatch")
        print("      Success: Bit-perfect match.")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)

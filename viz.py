import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as mc
import numpy as np

class MeshVisualizer:
    @staticmethod
    def plot_2d_quadtree(tree, title="2D Adaptive Mesh"):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        rects = []
        limit = tree.domain_width
        
        for node in tree.leaves:
            coords, size = tree.get_geometry(node)
            rects.append(patches.Rectangle(coords, size, size))
            
        pc = mc.PatchCollection(rects, match_original=False, edgecolor='red', facecolor='none', linewidth=0.5)
        ax.add_collection(pc)
        ax.set_xlim(0, limit); ax.set_ylim(0, limit)
        ax.set_aspect('equal'); ax.set_title(f"{title}\nElements: {len(tree.leaves)}")
        plt.show()

    @staticmethod
    def plot_3d_octree(tree, title="3D Adaptive Mesh"):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        limit = tree.domain_width
        
        leaves = tree.leaves
        # Subsample for display if too large
        if len(leaves) > 50000:
            leaves = leaves[::int(len(leaves)/50000)]
            
        xs, ys, zs = [], [], []
        for node in leaves:
            if node.level >= 4:
                c, size = tree.get_geometry(node)
                xs.append(c[0] + size/2)
                ys.append(c[1] + size/2)
                zs.append(c[2] + size/2)
        
        ax.scatter(xs, ys, zs, s=1, c='tab:blue', alpha=0.5)
        ax.set_xlim(0, limit); ax.set_ylim(0, limit); ax.set_zlim(0, limit)
        ax.set_title(f"{title}\nOctants: {len(tree.leaves)}")
        plt.show()

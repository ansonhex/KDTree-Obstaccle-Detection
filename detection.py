"""
@Author:	Jiawei He(Anson)
@Date:		2024/07/31

This is a demo for obstacle detection using KDTree.
"""

import numpy as np
import pcd_utils
from KDTree import KDTree, Point

# Import PCD data
points = pcd_utils.load_pcd("data/demo.pcd")
# pcd_utils.visualize_pcd(points)

# Build KD-Tree
kd_tree = KDTree()
point_objects = [Point(x, y, z) for x, y, z in points]
kd_tree.build_tree(point_objects)

# Search for nearest points, at center 0, 0, 0
target_point = np.array([0, 0, 0])
nearest_points, _ = kd_tree.nearest_search(target_point, k=5000)

# Visualize the obstacle detected
obstacle_points = np.array([p.data for p in nearest_points])
pcd_utils.visualize_pcd(obstacle_points)

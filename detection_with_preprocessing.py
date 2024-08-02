"""
@Author:	Jiawei He(Anson)
@Date:		2024/07/31

This is a demo for obstacle detection using KDTree with preprocessing
such as voxel grid filter and plane segmentation.
"""

import numpy as np
from KDTree import KDTree, Point
import pcd_utils


def apply_voxel_grid_filter(pcd, voxel_size=0.2):
    """
    This function applies the voxel grid filter to the point cloud data.

    REF: http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Voxel-downsampling
    """
    return pcd.voxel_down_sample(voxel_size=voxel_size)


def segment_plane(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=100):
    """
    This function segments the plane from the point cloud data, such that the 
    ground and obstacles are separated using RANSAC algorithm.

    REF:
        https://en.wikipedia.org/wiki/Random_sample_consensus
        https://www.open3d.org/docs/latest/tutorial/geometry/pointcloud.html#plane-segmentation
    """

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n, 
        num_iterations=num_iterations)
    ground_cloud = pcd.select_by_index(inliers)
    obstacles_cloud = pcd.select_by_index(inliers, invert=True)
    return ground_cloud, obstacles_cloud


if __name__ == "__main__":
    pcd = pcd_utils.load_pcd("data/demo.pcd", pcd=True)
    filtered_pcd = apply_voxel_grid_filter(pcd)
    ground, obstacles = segment_plane(filtered_pcd)

    # Visualize the point cloud data
    # 1. Original point cloud data
    # 2. Voxeled point cloud data
    # 3. Obstacles detected using plane segmentation

    pcd_utils.visualize_pcd(np.asarray(pcd.points))
    pcd_utils.visualize_pcd(np.asarray(filtered_pcd.points))
    pcd_utils.visualize_pcd(np.asarray(obstacles.points))

    kd_tree = KDTree()
    obstacle_points = np.asarray(obstacles.points)
    point_objects = [Point(x, y, z) for x, y, z in obstacle_points]
    kd_tree.build_tree(point_objects)

    # Search for nearest points, at center 0, 0, 0
    target_point = np.array([0, 0, 0])
    nearest_points, _ = kd_tree.nearest_search(target_point, k=2000)

    # Visualize the obstacle detected
    obstacle_points = np.array([p.data for p in nearest_points])
    pcd_utils.visualize_pcd(obstacle_points)

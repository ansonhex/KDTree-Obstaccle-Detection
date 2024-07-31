"""
@Author:	Jiawei He(Anson)
@Date:		2024/07/31

This is utility functions for point cloud data using Open3D.
"""

import time
import numpy as np
import open3d as o3d


def load_pcd(file_path, pcd=False):
    """
    This function loads the point cloud data from the file path.
    """
    point_cloud = o3d.io.read_point_cloud(file_path)
    if pcd:
        return point_cloud
    return np.asarray(point_cloud.points)   # points


def visualize_pcd(points):
    """
    This function visualizes the point cloud data.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Visualization")
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    vis.run()
    vis.destroy_window()


def visualize_pcd_sequence(pcd_sequence, title="PCD Visualization", delay=0.5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    for pcd in pcd_sequence:
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(delay)   # Add delay in seconds between frames
        vis.clear_geometries()
    vis.destroy_window()

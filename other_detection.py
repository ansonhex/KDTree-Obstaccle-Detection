import numpy as np
import open3d as o3d
import pcd_utils
import detection_with_preprocessing as preproc


def grid_based_detection(pcd, target_point, grid_size=0.2, search_radius=5.0, max_points=5000):
    points = np.asarray(pcd.points)

    # Calculate the distance of each point to the target point
    distances = np.linalg.norm(points - target_point, axis=1)
    
    # Mask points within the search radius
    mask = distances <= search_radius
    filtered_points = points[mask]
    filtered_distances = distances[mask]

    if len(filtered_points) > max_points:
        if len(filtered_points) > max_points * 2:  # If we have significantly more points, downsample first
            downsampled_pcd = o3d.geometry.PointCloud()
            downsampled_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            downsampled_pcd = downsampled_pcd.voxel_down_sample(voxel_size=grid_size)
            downsampled_points = np.asarray(downsampled_pcd.points)
            downsampled_distances = np.linalg.norm(downsampled_points - target_point, axis=1)

            # Sort and select the closest points
            sorted_indices = np.argsort(downsampled_distances)
            selected_points = downsampled_points[sorted_indices[:max_points]]
        else:
            # If we don't have too many points, just sort and select without downsampling
            sorted_indices = np.argsort(filtered_distances)
            selected_points = filtered_points[sorted_indices[:max_points]]
    else:
        selected_points = filtered_points

    return selected_points


pcd = pcd_utils.load_pcd("data/demo.pcd", pcd=True)
filtered_pcd = preproc.apply_voxel_grid_filter(pcd)
ground, obstacles = preproc.segment_plane(filtered_pcd)
target_point = np.array([0, 0, 0])

detected = grid_based_detection(obstacles, target_point, max_points=2000)
pcd_utils.visualize_pcd(detected)

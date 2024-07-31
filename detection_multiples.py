import numpy as np
from KDTree import KDTree, Point
import open3d as o3d
import glob
import pcd_utils
import detection_with_preprocessing as preproc


def process_pcd_files(pcd_files):
    all_obstacle_pcds = []
    for file_path in pcd_files:
        print(f"Processing file: {file_path}")
        pcd = pcd_utils.load_pcd(file_path, pcd=True)
        filtered_pcd = preproc.apply_voxel_grid_filter(pcd)
        ground, obstacles = preproc.segment_plane(filtered_pcd)

        kd_tree = KDTree()
        obstacle_points = np.asarray(obstacles.points)
        point_objects = [Point(x, y, z) for x, y, z in obstacle_points]
        kd_tree.build_tree(point_objects)

        target_point = np.array([0, 0, 0])
        nearest_points, _ = kd_tree.nearest_search(target_point, k=1500)

        # Collect detected obstacles
        obstacle_points_np = np.array([p.data for p in nearest_points])
        obstacle_pcd = o3d.geometry.PointCloud()
        obstacle_pcd.points = o3d.utility.Vector3dVector(obstacle_points_np)
        all_obstacle_pcds.append(obstacle_pcd)

    # Visualize all detected obstacles sequentially
    pcd_utils.visualize_pcd_sequence(
        all_obstacle_pcds, "Detected Obstacles Sequence")


if __name__ == "__main__":
    pcd_files = list(filter(lambda x: "demo.pcd" not in x, sorted(glob.glob("data/*.pcd"))))
    process_pcd_files(pcd_files)

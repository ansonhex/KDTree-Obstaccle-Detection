"""
@Author:	Jiawei He(Anson)
@Date:		2024/07/31

This is a simple implementation of KDTree in Python.
"""

import numpy as np


class Point:
    """
    The Point class is used to store the data of the point.
    """
    def __init__(self, x, y, z):
        self.data = np.array([x, y, z])
        self.deleted = False    # lazy deletion


class KDNode:
    """
    The KDNode class is used to store the point and the left and right child nodes.
    """
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right


class KDTree:
    """
    The KDTree Implementation:
    - insert_point: insert a point into the KDTree
    - search_point: search a point in the KDTree
    - delete_point: delete a point in the KDTree
    - build_tree: build a KDTree from a list of points
    - nearest_search: search the nearest k points to the given point
    """
    def __init__(self):
        self.root = None
        self.tree_size = 0
        self.delete_size = 0

    def insert_point(self, point):
        """
        Insert a point into the KDTree.
        """
        new_point = Point(*point)  # Unpack the point data
        if self.root is None:
            self.root = KDNode(new_point)
            self.tree_size = 1
            return

        current = self.root
        dimension = 0

        while current:
            if new_point.data[dimension] < current.point.data[dimension]:
                if current.left is None:
                    current.left = KDNode(new_point)
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = KDNode(new_point)
                    break
                current = current.right
            dimension = (dimension + 1) % 3

        self.tree_size += 1

    def search_point(self, point):
        """
        Search a point in the KDTree.
        """
        current = self.root
        dimension = 0
        while current:
            # if equal then return the current node
            if np.array_equal(current.point.data, point):
                return current
            if point[dimension] < current.point.data[dimension]:
                current = current.left
            else:
                current = current.right
            dimension = (dimension + 1) % 3
        return None

    def delete_point(self, point):
        """
        Delete a point in the KDTree.
        """
        node = self.search_point(point)
        if node:
            node.point.deleted = True   # lazy deletion
            self.delete_size += 1

    def build_tree(self, points):
        """
        Build a KDTree from a list of points.
        """
        def build(points, depth=0):
            if not points:
                return None
            axis = depth % 3
            points.sort(key=lambda x: x.data[axis])
            median = len(points) // 2
            return KDNode(
                points[median],
                left=build(points[:median], depth + 1),
                right=build(points[median + 1:], depth + 1))

        self.root = build(points)
        self.tree_size = len(points)

    def nearest_search(self, point, k=1):
        """
        Search the nearest k points to the given point.
        """
        best_points = []
        best_distances = []

        def distance(p1, p2):
            return np.linalg.norm(p1 - p2)  # Euclidean distance

        def search(node, depth):
            if node is None:
                return

            axis = depth % 3
            next_branch = node.left if point[axis] < node.point.data[axis] else node.right
            other_branch = node.right if next_branch == node.left else node.left

            search(next_branch, depth + 1)

            dist = distance(point, node.point.data)
            # If the point is not deleted, add it to the best points list
            if not node.point.deleted:
                if len(best_points) < k:
                    best_points.append(node.point)
                    best_distances.append(dist)
                else:
                    if dist < max(best_distances):
                        max_index = best_distances.index(max(best_distances))
                        best_points[max_index] = node.point
                        best_distances[max_index] = dist

            # If the distance between the point and the node is less
            # than the maximum distance in the best distances list
            if abs(point[axis] - node.point.data[axis]) < max(best_distances, default=float('inf')):
                search(other_branch, depth + 1)

        search(self.root, 0)
        return best_points, best_distances

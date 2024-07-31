# KDTree Obstacle Detectcion


![detecion workflow](assets/workflow.gif)
This is an exploration using KDTree for obstacle detection with PCD (Point Cloud DAta) using Python. It includes a simple implementation of a KDTree class along with various preprocessing techniques and comparisons with other search methods.

## Requirement

* Python Version: `3.11.*`
* Install dependencies from `requreiments.txt`

```sh
pip install -r requirements.txt
```

## Explored

* Simple KDTree class implementation
* Obstacle detection using KDTree
* Preprocessing techniques:
  * Grid voxel filter
  * Plane segmentation
* Basic workflow of obstacle detection
* Comparison with Grid-based nearest search

## Reference

* Inspired by [ikd-Tree](https://github.com/hku-mars/ikd-Tree)
* LiDAR PCD Data & Preprocessing  Techniques from [Lidar-Obstacle-Detection](https://github.com/olpotkin/Lidar-Obstacle-Detection)

```
@article{cai2021ikd,
  title={ikd-Tree: An Incremental KD Tree for Robotic Applications},
  author={Cai, Yixi and Xu, Wei and Zhang, Fu},
  journal={arXiv preprint arXiv:2102.10808},
  year={2021}
}
```

## LICENSE

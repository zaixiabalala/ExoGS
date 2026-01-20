## Feature Overview

This project implements pose processing and post-processing functions for robotic manipulation, including:

- **Pose collection**: Collecting object poses from multiple cameras and averaging them
- **Coordinate transformation**: Converting poses from camera coordinate system to base coordinate system
- **Pose manipulation**: Overwriting, offsetting, rotating, freezing and fixing
- **Grasp detection**: Automatic detection of grasp and release frames based on gripper width
- **Pose visualization**: 3D visualization of object poses and robot states using PyBullet and Open3D

---

## Create Conda Environment

```bash
conda create -n poseprocess python=3.12 -y
conda activate poseprocess
conda install -c conda-forge numpy scipy matplotlib tqdm pip -y
pip install pybullet>=3.2.0 ruptures>=1.1.0 open3d>=0.17 typed-argument-parser
cd PoseProcess
pip install -e poseprocess/
```

---

## Pose Processing Pipeline

After `FoundationPose`, you can process poses following these steps:

1. Run `scripts/collect_obj_poses.py` 
    - Transform object poses from FoundationPose output to base coordinate system and average them to reduce errors
    - Output to `record_*/poses/object_*_org/` directory

2. Run `scripts/process_poses.py`
    - Process object poses and visualize the results
        - pose_overwrite: For fixed objects, overwrite all poses with the first frame pose
        - pose_offset: Apply translation offset to object poses in base coordinate system
        - pose_rotate: Rotate object poses
        - set_container_bbox & pose_freeze: Set container bounding box, freeze object poses when detected entering the container
        - pose_fix: Fix object poses to the end-effector
        - pose_visualize: Visualize processed object and end-effector pose sequences for quick debugging to verify pose sequences are correct
        - pose_writeback: Write processed object poses to `record_*/poses/object_*/`
    - Ensure object mesh files exist in `assets/meshes/` directory, and camera extrinsic parameters are correctly written in `assets/camera_constants.py` file

3. Run `scripts/check_poses_writen.py`
    - Check if processed object poses have been written for each record

Then you can proceed with Gaussian data generation using `GaussianSim`.
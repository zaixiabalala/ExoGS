## Feature Overview

This module uses SAPIEN to render multi-view images of objects and generate COLMAP datasets for Gaussian Splatting asset generation.
It provides an alternative to real-world video capture for creating training data for Gaussian models.

---

## Environment Setup

### 1. Create conda environment
```bash
conda create -n sapien python=3.10 -y
conda activate sapien
```

### 2. Install PyTorch
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 3. Install Python dependencies
```bash
pip install sapien tqdm psutil pillow scipy open3d
```

---

## Generate COLMAP Dataset

Ensure that the urdf file and corresponding mesh file of the object are placed in the `assets/` directory.

Run `generate_data_pipeline.py` to generate multi-view images and COLMAP datasets for specified objects.

Output Structure:
```
sapien_data/
└── {urdf_name}/
    ├── images/                    # Rendered multi-view images
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── sparse/
        └── 0/
            ├── cameras.bin        # COLMAP camera parameters
            ├── images.bin         # COLMAP image poses
            └── points3D.bin       # COLMAP 3D points
```

The generated COLMAP dataset can then be used with Gaussian Splatting to create 3D Gaussian model assets.

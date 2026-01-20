## Feature Overview

This project implements Gaussian image rendering functionality, capable of generating original color, depth, and alpha images, classification mask label maps, color augmentation images, etc.
It can be used for training data generation and augmentation for robot grasping, assembly, and other tasks.

---

## Environment Setup

### 1. Create conda environment
```bash
conda create -n gaussian_sim python=3.10 -y
conda activate gaussian_sim
```

### 2. Install system-level dependencies (must use conda)
```bash
conda install -c nvidia cuda-toolkit=12.8 -y
conda install -c conda-forge libstdcxx-ng -y
```

### 3. Install PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Compile diff-gaussian-rasterization
```bash     
cd submodules/
git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
pip install --no-build-isolation -e .
cd ../..
```

---

## Gaussian Image Rendering

This project requires object poses in the base coordinate system. Please ensure that the PoseProcess project has completely processed the data.

```
OriginData/
└── records_xxxx/
    └── record_xxxx/
        └── poses/
            └── object_1/          # Ensure object poses are correctly written
                ├── xxxx.txt
                └── xxxx.txt
```

Please generate the required Gaussian model assets in advance and place them in the `assets/` directory. For the Gaussian model generation process, refer to [GSAssetGen](../GSAssetGen/README.md).

Run `scripts/gaussian_render.py` to implement the following features:
- 1. Render original RGB, depth, and alpha images
- 2. Render classification mask images
- 3. Render color augmentation images

Two methods for scene rendering:
- 1. Generate scene Gaussian model assets and place them in the `assets/scenes/` directory, then select the corresponding Gaussian model during rendering
- 2. Directly use camera-captured scene photos as background for rendering. Place the captured images in the `assets/bg_images/` directory, and select `scene_none` for the scene Gaussian model during rendering

For specific parameter settings, see `get_config()` and the `setup_mask_color_config()` function.

Output structure:
```
GSData/
└── records_xxxx/
    ├── transforms.json              # Camera pose file
    └── record_xxxx/
        ├── angles/                  # Robot joint angle data
        ├── tcps/                    # Robot TCP pose data
        ├── cam_0/
        │   ├── rgbs_bg0/            # Original RGB files
        │   ├── ...
        │   ├── rgb_0_bg0/          # Color augmentation RGB files
        │   ├── ...
        │   ├── depths/              # Depth maps
        │   ├── alphas/              # Alpha channel
        │   ├── masks/               # Original masks
        │   ├── masks_clean/         # Cleaned masks
        │   ├── masks_clean_labels/  # Single-channel labels
        │   └── masks_clean_labels_3/ # 3-channel labels
        ├── cam_1/
        │   └── ...
        └── ...
```

## Convert to Train Data

项目采用了HDF5格式存储训练数据，转化脚本位于`GaussianSim/convert_to_train/`目录

详情参考[convert_to_train](convert_to_train/README.md)
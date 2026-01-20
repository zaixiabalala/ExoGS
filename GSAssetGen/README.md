## Feature Overview

This project implements Gaussian model asset generation functionality, which can generate Gaussian model assets for Gaussian image rendering.

Gaussian model asset generation consists of two stages:
- 1. Generate COLMAP dataset
- 2. Generate scene Gaussian model assets through the gaussian-splatting project

---

## Generate COLMAP Dataset

There are two ways to obtain COLMAP datasets:

### 1. Generate COLMAP Dataset Using Real Video with COLMAP

#### (1) Capture Multi-view Video Around Real Object Using Mobile Phone

- Ensure the object is stationary, the environment is bright, keep the subject centered in the frame, move the phone slowly, and avoid zooming and motion blur
- Capture from comprehensive angles, covering all possible shooting angles, and appropriately adjust the shooting distance (closer or farther)
- It is recommended to use iPhone devices for shooting, as they produce better results
- Save the video at ColmapData/xxxx/xxxx.MOV

#### (2) Convert Video to Image Sequence Using ffmpeg

- For ffmpeg installation, refer to the [ffmpeg official website](https://ffmpeg.org/download.html)

- Run the `scripts/ffmpeg_vedio_2_images.py` script to convert the video to an image sequence

#### (3) Perform Sparse Reconstruction Using COLMAP to Generate COLMAP Dataset
```bash
cd GSAssetGen/
git clone https://github.com/colmap/colmap.git
```

- For COLMAP installation, refer to the [COLMAP official website](https://colmap.github.io/)
- Sparse reconstruction workflow:
  - (1) Run `colmap gui` in the terminal to open the graphical interface
  - (2) "File" -> "New Project", create a Database in the ColmapData/xxxx/ directory and select the image sequence
  - (3) "Processing" -> "Feature extraction" to perform feature extraction
  - (4) "Processing" -> "Feature matching" to perform feature matching
  - (5) "Reconstruction" -> "Start reconstruction" to perform sparse reconstruction. You can observe the reconstruction results in the 3D view at this time
  - (6) After sparse reconstruction is complete, "File" -> "Export all models", and save the sparse reconstruction results in the following COLMAP dataset format

```
ColmapData/xxxx/sparse/0/
├── cameras.bin
├── images.bin
└── points3D.bin
```

### 2. Generate COLMAP Dataset Using SAPIEN

Use SAPIEN to render multi-view images and generate COLMAP datasets. For details, refer to [sapien](sapien/README.md).

---

## Generate Gaussian Model

Use the above COLMAP dataset to generate Gaussian models through the gaussian-splatting project or related projects. For details, refer to [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).

If training fails, the data may have distortion and needs distortion correction. After correction is complete, use output_path to replace the original path:
```bash
colmap image_undistorter \
    --image_path ColmapData/xxxx/images \
    --input_path ColmapData/xxxx/sparse/0 \
    --output_path ColmapData/xxxx/dense
```

The trained Gaussian model generally requires manual processing, such as separating the target object, scaling, alignment, and other operations to obtain the final Gaussian model asset.
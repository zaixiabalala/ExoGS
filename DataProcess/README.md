## Feature Overview

This project implements data processing and analysis functions for robotic manipulation, including:

- **Data processing**: Timestamp alignment, cropping initial video frames
- **Kinematics computation**: Conversion from joint angles to TCP pose, eye-in-hand extrinsic parameter calculation
- **Data organization**: Organizing data into the format required by foundationpose
- **Data annotation**: Using X-AnyLabeling tool for mask annotation

---

## Environment Setup

### Create Conda Environment

```bash
conda create -n Fkexo python=3.10 -y
conda activate Fkexo
cd FkExo
```

### Install Dependencies

```bash
pip install numpy opencv-python scipy pybullet
```

If you need to use kinematics computation features, install r3kit:

```bash
git clone https://github.com/dadadadawjb/r3kit.git
cd r3kit
pip install -e .
cd ..
```

---

## X-AnyLabeling Setup

Use X-AnyLabeling for mask annotation. For installation and usage details, refer to [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling).

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
```

This project also provides a brief guide for environment setup and usage of the annotation features required for the current task. For details, refer to [x_anylabeling_guide.md](./docs/x_anylabeling_guide.md).

---

## FoundationPose Setup

Use FoundationPose for 6D pose estimation and tracking. For installation, weights download and usage details, refer to [FoundationPose](https://github.com/NVlabs/FoundationPose).

We provide the `FoundationPose/tracking.py` script for batch pose estimation and tracking on this project's data.

```bash
git clone https://github.com/NVlabs/FoundationPose.git
```

---

## Data Processing Pipeline

1. Run `data_processor_1.py` to process the data:
    - Crop initial video frames to avoid camera exposure affecting annotation
    - Generate corresponding `organized` folders for each camera for FoundationPose inference
    - Generate corresponding `data_for_mask` folders for each camera for mask annotation

2. Run X-AnyLabeling, open the annotation software, and perform mask annotation for each object in the `data_for_mask` folders separately. Save the annotations to the `masks_i` folders in the `data_for_mask` directory (where i is the object index).

3. Run `data_processor_2.py` to backfill the mask files into the `masks_i` folders in the corresponding `organized` folders for each camera.

4. Run `FoundationPose/tracking.py` to perform 6D pose estimation and tracking using FoundationPose, generating the corresponding `tracking` folders for each camera and each object.

Then you can proceed with pose processing using `PoseProcess`.

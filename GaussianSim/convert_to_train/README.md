# Convert to HDF5

Convert rendered 3D Gaussian Splatting (3DGS) data to HDF5 format for robot learning.

## Installation

```bash
pip install h5py numpy Pillow
```

## Data Structure

Expected input directory structure:
```
trajectory_dir/
├── cam_0/
│   ├── rgbs/           # RGB images
│   └── masks_clean_labels_3/  # Segmentation masks
├── angles/             # Joint angles (.npy)
└── tcps/               # TCP poses (.npy)
```

Output HDF5 structure:
```
data/
├── demo_0/
│   ├── images   (N, H, W, 3)
│   ├── masks    (N, H, W, 3)
│   ├── joints   (N, num_joints)
│   └── tcps     (N, 10)
├── demo_1/
│   ...
```

## Usage

```python
from convert_to_hdf5 import convert_to_hdf5
from pathlib import Path

base_dir = Path("/path/to/records")
trajectory_dirs = sorted([str(p) for p in base_dir.iterdir() if p.is_dir()])

convert_to_hdf5(
    trajectory_dirs=trajectory_dirs,
    output_path="output.hdf5",
    image_processor_params={
        'crop': {'start_x': 100, 'end_x': 580, 'start_y': 0, 'end_y': 480},
        'resize': (224, 224),
    },
    mask_processor_params={
        'crop': {'start_x': 100, 'end_x': 580, 'start_y': 0, 'end_y': 480},
        'mask_resize': (224, 224),
    },
)
```

## Image Processing Options

- `crop`: Crop region `{start_x, end_x, start_y, end_y}`
- `resize`: Target size `(W, H)` using LANCZOS interpolation
- `mask_resize`: Target size for masks using NEAREST interpolation
- `center_crop_square`: Center crop to square
- `border`: Add border `{top, bottom, left, right}`
- `bgr2rgb`: Convert BGR to RGB



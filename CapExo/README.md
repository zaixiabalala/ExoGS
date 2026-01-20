# CapExo

Data collection and postprocessing tools for exoskeleton teleoperation with Intel RealSense D415 camera.

## Installation

```bash
conda create -n capexo python=3.10
conda activate capexo

git clone git@github.com:dadadadawjb/r3kit.git
cd r3kit
pip install -e .[comm,rs] --config-settings editable_mode=compat
cd ..

git clone git@github.com:dadadadawjb/CapExo.git
cd CapExo
pip install -r requirements.txt
```

## Usage

### Data Collection

Collect synchronized encoder and camera data:

```bash
python collect_data.py \
    --encoder_id /dev/ttyUSB0 \
    --camera_id <camera_serial> \
    --camera_depth \
    --save_path /path/to/save
```

### Postprocessing

Align encoder and camera data by timestamps:

```bash
python postprocess.py \
    --data_path /path/to/raw/data \
    --camera_name cam0 \
    --batch \
    --save_tcp_pose
```

## Data Format

Raw data structure:
```
<timestamp>/
├── encoder/
│   ├── angle.npy
│   └── timestamps.npy
└── camera/
    ├── color/
    ├── depth/
    ├── timestamps.npy
    └── intrinsics.txt
```

Processed data structure:
```
record_<timestamp>/
├── <camera_name>/
│   ├── color/
│   ├── depth/
│   └── intrinsics.txt
├── angles/
└── tcp_poses/
```

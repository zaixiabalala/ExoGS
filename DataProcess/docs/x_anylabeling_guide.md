## Create conda environment

```bash
conda create -n x-anylabeling python=3.10 -y
conda activate x-anylabeling
cd ./X-AnyLabeling
pip install -r requirements.txt
pip install requests packaging six
```

## Annotation

### 1. Launch annotation tool

```bash
conda activate x-anylabeling
python ./anylabeling/app.py
```

### 2. Load data and model

Click the `Open Dir` button in the left panel to open the image folder for annotation.

Click `Auto Labeling` in the left panel, then click `No Model` to select the model to use (the model will be automatically downloaded on first selection). This project uses the `Segment Anything 2.1 (base)` model.

### 3. Annotate and export

Annotate each image with the label set to `target`.

After completing all annotations, click the `Export` button in the top bar, select `Export MASK Annotations`, then choose the mask configuration JSON file. This project uses `DataProcess/assets/color_map.json`. Finally, export the annotation results.


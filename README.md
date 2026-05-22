# COMPASS

Official code release scaffold for **Text as a Compass: Semantic-Navigated 3D Bone Collapse Prediction via Orthogonal Micro-Volume Primitives**.

COMPASS predicts collapse risk in osteonecrosis of the femoral head (ONFH) by combining clinical-report semantics with tri-planar orthogonal micro-volume primitives (OMVPs). The implementation includes the semantic-navigated MIL model, counterfactual grounding loss, a 3D CNN baseline, and feature-map visualization utilities.

## Repository Layout

```text
COMPASS-GitHub/
  src/compass/              # reusable model and dataset modules
  scripts/crop_roi.py       # padded ROI extraction from 3D image/mask pairs
  scripts/resample_to_npy.py
  scripts/generate_omvp.py  # tri-planar OMVP generation
  scripts/train_stage1_alignment.py
  scripts/train_compass.py  # main COMPASS training/evaluation script
  scripts/train_baseline_3d.py
  scripts/visualize_feature_maps.py
  configs/compass_example.yaml
  data/README.md            # expected metadata columns
```

Large artifacts such as patient data, checkpoints, model weights, W&B logs, and generated feature maps are intentionally excluded.

## Installation

```bash
conda create -n compass python=3.10
conda activate compass
pip install -e .
```

Install the PyTorch build that matches your CUDA version if the default wheel is not appropriate.

## Preprocessing

The private clinical data are not redistributed, but the preprocessing scripts follow the pipeline described in the paper.

1. Crop the femoral-head ROI with a padded mask bounding box:

```bash
python scripts/crop_roi.py \
  --image-dir /path/to/raw/images \
  --mask-dir /path/to/raw/masks \
  --output-image-dir /path/to/cropped/images \
  --output-mask-dir /path/to/cropped/masks \
  --padding 2
```

2. Resample cropped volumes to `64 x 64 x 64` NumPy arrays:

```bash
python scripts/resample_to_npy.py \
  --image-dir /path/to/cropped/images \
  --mask-dir /path/to/cropped/masks \
  --output-image-dir /path/to/processed/images \
  --output-mask-dir /path/to/processed/masks \
  --size 64
```

3. Generate tri-planar OMVP bags and optional aligned masks:

```bash
python scripts/generate_omvp.py \
  --csv-path /path/to/metadata.csv \
  --output-dir /path/to/omvp_arrays \
  --image-col preprocessed_path \
  --mask-col preprocessed_mask_path \
  --num-points 256 \
  --slice-size 64 \
  --sampling grid \
  --seed 2024
```

## Training

Stage I semantic-space alignment:

```bash
python scripts/train_stage1_alignment.py \
  --csv_path /path/to/metadata_with_omvp.csv \
  --text_model_path ncbi/MedCPT-Query-Encoder \
  --checkpoint-dir ./checkpoints/stage1 \
  --epochs 100
```

Stage II COMPASS training with semantic navigation and counterfactual grounding:

```bash
python scripts/train_compass.py \
  --csv_path /path/to/metadata_with_omvp.csv \
  --text_model_path ncbi/MedCPT-Query-Encoder \
  --pretrain_path ./checkpoints/stage1/stage1_alignment_best.pth \
  --checkpoint-dir ./checkpoints/compass \
  --epochs 200 \
  --batch-size 64 \
  --num-samples 64 \
  --top_k 5
```

The 3D-CNN baseline can be launched with:

```bash
python scripts/train_baseline_3d.py \
  --csv_path /path/to/metadata_3d.csv \
  --checkpoint-dir ./checkpoints/baseline_3d
```

See [data/README.md](data/README.md) for the required metadata columns.

## Citation

If you find this repository useful, please cite our MICCAI paper:

```bibtex
@inproceedings{zeng2026compass,
  title     = {Text as a Compass: Semantic-Navigated 3D Bone Collapse Prediction via Orthogonal Micro-Volume Primitives},
  author    = {Zeng, Qingyuan and Guan, Zixin and Chen, Yusen and Wu, Zifeng and Li, Hao and Ma, Qian and Lu, Zixiao and He, Wei and Chen, Leilei and Zhou, Wu and Chen, Jintai},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year      = {2026}
}
```

The DOI, publisher, and page numbers will be added after the official MICCAI proceedings metadata is available.

## Release Notes

- Private clinical data are not redistributed.
- Select a final open-source license before making the repository public.
- Use `requirements.txt` or `pyproject.toml` for a clean install.

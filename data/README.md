# Data Format

Patient data and model weights are not included in this repository.

For `scripts/train_compass.py`, the metadata table should contain:

- `mri_omvp_path`: path to a NumPy array with shape `(num_primitives, 3, 64, 64)`.
- `mask_omvp_path`: optional path to aligned OMVP masks with the same first dimension.
- `generated_report`: clinical report text used by the text encoder.
- `tkr_incident_108`: binary target label.
- `mri_path`: optional original image path used for filtering and logging.

For `scripts/generate_omvp.py`, the metadata table should contain:

- `preprocessed_path`: path to a normalized `(64, 64, 64)` image NumPy array.
- `preprocessed_mask_path`: optional path to an aligned binary mask NumPy array.
- one patient identifier column, or pass `--id-col`.

For `scripts/train_baseline_3d.py`, the metadata table should contain:

- `preprocessed_path`: path to a preprocessed 3D NumPy volume.
- `report`: clinical report text.
- `label`: binary label. The loader also supports common Chinese clinical labels via Unicode aliases.
- `CT_path`: optional raw path used only for logging.

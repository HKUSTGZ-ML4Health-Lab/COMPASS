"""Generate tri-planar orthogonal micro-volume primitives (OMVPs)."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from skimage.transform import resize
from tqdm import tqdm


def read_table(path):
    if str(path).endswith(".xlsx"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="gbk")


def write_table(df, path):
    if str(path).endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def sample_coordinates(shape, num_points, rng):
    depth, height, width = shape
    return np.stack(
        [
            rng.randint(0, depth, num_points),
            rng.randint(0, height, num_points),
            rng.randint(0, width, num_points),
        ],
        axis=1,
    )


def grid_coordinates(shape, num_points):
    depth, height, width = shape
    points_per_axis = int(np.ceil(num_points ** (1 / 3)))
    z = np.linspace(0, depth - 1, points_per_axis).round().astype(int)
    y = np.linspace(0, height - 1, points_per_axis).round().astype(int)
    x = np.linspace(0, width - 1, points_per_axis).round().astype(int)
    grid = np.array(np.meshgrid(z, y, x, indexing="ij")).reshape(3, -1).T
    if len(grid) > num_points:
        keep = np.linspace(0, len(grid) - 1, num_points).round().astype(int)
        grid = grid[keep]
    return grid


def resize_slice(slice_array, target_size, is_mask=False):
    if slice_array.shape == target_size:
        return slice_array
    return resize(
        slice_array,
        target_size,
        order=0 if is_mask else 1,
        mode="reflect",
        anti_aliasing=not is_mask,
        preserve_range=True,
    )


def extract_omvp(volume, coordinates, target_size=(64, 64), is_mask=False):
    dtype = np.uint8 if is_mask else np.float32
    primitives = np.zeros((len(coordinates), 3, target_size[0], target_size[1]), dtype=dtype)

    for i, (z, y, x) in enumerate(coordinates):
        axial = volume[z, :, :]
        coronal = volume[:, y, :]
        sagittal = volume[:, :, x]
        planes = [resize_slice(p, target_size, is_mask=is_mask) for p in (axial, coronal, sagittal)]
        primitives[i] = np.stack(planes, axis=0).astype(dtype)

    return primitives


def main():
    parser = argparse.ArgumentParser(description="Build OMVP arrays and update the metadata table.")
    parser.add_argument("--csv-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-col", default="preprocessed_path")
    parser.add_argument("--mask-col", default=None, help="Optional mask column for aligned OMVP masks.")
    parser.add_argument("--id-col", default=None, help="Optional patient id column. Defaults to the first column.")
    parser.add_argument("--num-points", default=256, type=int)
    parser.add_argument("--slice-size", default=64, type=int)
    parser.add_argument("--sampling", choices=["grid", "random"], default="grid")
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--output-csv", default=None, type=Path)
    args = parser.parse_args()

    df = read_table(args.csv_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    mask_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating OMVP"):
        patient_id = str(row[args.id_col]) if args.id_col else str(row.iloc[0])
        image_path = Path(row[args.image_col])
        volume = np.load(image_path)
        if args.sampling == "grid":
            coordinates = grid_coordinates(volume.shape, args.num_points)
        else:
            rng = np.random.RandomState(args.seed)
            coordinates = sample_coordinates(volume.shape, args.num_points, rng)
        omvp = extract_omvp(volume, coordinates, (args.slice_size, args.slice_size), is_mask=False)

        image_save_path = args.output_dir / f"{patient_id}_omvp.npy"
        np.save(image_save_path, omvp)
        image_paths.append(str(image_save_path))

        if args.mask_col:
            mask_volume = np.load(row[args.mask_col])
            mask_omvp = extract_omvp(mask_volume, coordinates, (args.slice_size, args.slice_size), is_mask=True)
            mask_save_path = args.output_dir / f"{patient_id}_mask_omvp.npy"
            np.save(mask_save_path, mask_omvp)
            mask_paths.append(str(mask_save_path))

    df["mri_omvp_path"] = image_paths
    if args.mask_col:
        df["mask_omvp_path"] = mask_paths

    output_csv = args.output_csv or args.csv_path.with_name(args.csv_path.stem + "_with_omvp.csv")
    write_table(df, output_csv)
    print(f"Saved updated metadata to {output_csv}")


if __name__ == "__main__":
    main()

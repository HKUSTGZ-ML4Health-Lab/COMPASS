"""Resample cropped 3D image/mask pairs to fixed-size NumPy arrays."""

from pathlib import Path
import argparse

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def resample_image(image, new_size=(64, 64, 64), is_label=False):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [
        original_size[i] * original_spacing[i] / new_size[i]
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(v) for v in new_size])
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def min_max_normalize(array):
    array = array.astype(np.float32)
    value_range = array.max() - array.min()
    if value_range == 0:
        return np.zeros_like(array, dtype=np.float32)
    return (array - array.min()) / value_range


def find_image_for_mask(mask_name, image_dir):
    candidates = [
        mask_name.replace(".nii.gz", "_0000.nii.gz"),
        mask_name,
    ]
    for candidate in candidates:
        path = image_dir / candidate
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser(description="Resample cropped NIfTI volumes to .npy arrays.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-image-dir", required=True, type=Path)
    parser.add_argument("--output-mask-dir", required=True, type=Path)
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--label", default=1, type=int)
    args = parser.parse_args()

    args.output_image_dir.mkdir(parents=True, exist_ok=True)
    args.output_mask_dir.mkdir(parents=True, exist_ok=True)
    target_size = (args.size, args.size, args.size)

    for mask_path in tqdm(sorted(args.mask_dir.glob("*.nii.gz")), desc="Resampling"):
        image_path = find_image_for_mask(mask_path.name, args.image_dir)
        if image_path is None:
            print(f"Skip {mask_path.name}: matching image was not found.")
            continue

        image = resample_image(sitk.ReadImage(str(image_path)), target_size, is_label=False)
        mask = resample_image(sitk.ReadImage(str(mask_path)), target_size, is_label=True)

        image_array = min_max_normalize(sitk.GetArrayFromImage(image))
        mask_array = (sitk.GetArrayFromImage(mask) == args.label).astype(np.float32)

        stem = mask_path.name.replace(".nii.gz", "")
        np.save(args.output_image_dir / f"{stem}.npy", image_array)
        np.save(args.output_mask_dir / f"{stem}.npy", mask_array)


if __name__ == "__main__":
    main()

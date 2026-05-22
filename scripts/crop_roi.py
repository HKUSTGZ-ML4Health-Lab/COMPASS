"""Crop 3D image/mask pairs with the minimal mask bounding box."""

from pathlib import Path
import argparse

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm


def crop_roi(image_path, mask_path, output_image_path, output_mask_path, label=1, padding=2):
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    mask_array = sitk.GetArrayFromImage(mask)

    coords = np.argwhere(mask_array == label)
    if coords.size == 0:
        raise ValueError(f"Label {label} was not found in {mask_path}")

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    size_x, size_y, size_z = image.GetSize()

    x0 = max(0, int(x_min) - padding)
    y0 = max(0, int(y_min) - padding)
    z0 = max(0, int(z_min) - padding)
    x1 = min(size_x, int(x_max) + padding + 1)
    y1 = min(size_y, int(y_max) + padding + 1)
    z1 = min(size_z, int(z_max) + padding + 1)

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([x0, y0, z0])
    roi.SetSize([x1 - x0, y1 - y0, z1 - z0])

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(roi.Execute(image), str(output_image_path))
    sitk.WriteImage(roi.Execute(mask), str(output_mask_path))


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
    parser = argparse.ArgumentParser(description="Crop volumes with a padded ROI around the label mask.")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-image-dir", required=True, type=Path)
    parser.add_argument("--output-mask-dir", required=True, type=Path)
    parser.add_argument("--label", default=1, type=int)
    parser.add_argument("--padding", default=2, type=int)
    args = parser.parse_args()

    mask_files = sorted(args.mask_dir.glob("*.nii.gz"))
    for mask_path in tqdm(mask_files, desc="Cropping ROI"):
        image_path = find_image_for_mask(mask_path.name, args.image_dir)
        if image_path is None:
            print(f"Skip {mask_path.name}: matching image was not found.")
            continue

        crop_roi(
            image_path=image_path,
            mask_path=mask_path,
            output_image_path=args.output_image_dir / image_path.name,
            output_mask_path=args.output_mask_dir / mask_path.name,
            label=args.label,
            padding=args.padding,
        )


if __name__ == "__main__":
    main()

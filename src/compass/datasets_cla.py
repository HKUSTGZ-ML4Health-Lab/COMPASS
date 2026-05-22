import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import ndimage
import random


# ==========================================
# ==========================================
class Ortho_CT_Sample_Dataset(Dataset):
    def __init__(self, text_csv, target_size=(64, 64, 64), transform=None, augment=False, **args):
        """
        Initialize the 3D CT dataset.
        :param text_csv: DataFrame with patient metadata and paths.
        :param target_size: target 3D size as (Depth, Height, Width).
        :param augment: whether to enable 3D augmentation.
        """
        self.text_csv = text_csv
        self.target_size = target_size
        self.transform = transform
        self.augment = augment
        self.mode = args.get('train_test', 'train')
        self.resize_mode = 'anisotropic'

        self.slices_per_patient = self.target_size[0]

    def __len__(self):
        return self.text_csv.shape[0]

    def _apply_augmentations(self, image: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            axes = random.choice([(1, 2), (0, 1), (0, 2)])
            angle = random.uniform(-15, 15)
            image = ndimage.rotate(image, angle, axes=axes, reshape=False, order=1,
                                   mode='constant', cval=image.min())
        if random.random() > 0.5:
            axis = random.choice([0, 1, 2, (0, 1), (1, 2)])
            image = np.flip(image, axis=axis).copy()
        if random.random() > 0.5:
            sigma = random.uniform(0.01, 0.03)
            noise = np.random.normal(0, sigma, image.shape)
            image = np.clip(image + noise, 0.0, 1.0)
        return image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.text_csv.iloc[idx]

        ct_path = row['preprocessed_path']
        report = str(row['report']).lower()
        target_val = row['label']
        raw_path = str(row['CT_path'])

        if target_val == '\u584c\u9677':
            label = 1.0
        elif target_val == '\u672a\u584c\u9677':
            label = 0.0
        else:
            try:
                label = float(target_val)
            except (ValueError, TypeError):
                label = 0.0
        target_tensor = torch.tensor([label]).float()

        try:
            ct_processed = np.load(ct_path)
        except Exception as e:
            print(f"\n[ERROR] Failed to load volume. index={idx}, path={ct_path}, error={e}")
            ct_processed = np.zeros(self.target_size)

        if self.augment and self.mode == 'train':
            ct_processed = self._apply_augmentations(ct_processed)

        ct_tensor = torch.from_numpy(ct_processed).float().unsqueeze(0)

        sample = {
            'ct': ct_tensor,
            'target': target_tensor,
            'raw_text': report,
            'path': raw_path
        }
        return sample


# ==========================================
# ==========================================
class Ortho_CT_TEXT_Dataset_Manager:
    def __init__(self, csv_path, target_size=(64, 64, 64)):
        self.target_size = target_size
        print(f'Loading Ortho dataset from {csv_path}...')
        try:
            self.full_csv = pd.read_csv(csv_path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            self.full_csv = pd.read_csv(csv_path, low_memory=False, encoding='gbk')

        if 'omvp_path' in self.full_csv.columns:
            self.full_csv = self.full_csv.dropna(subset=['omvp_path'])

        if 'preprocessed_path' in self.full_csv.columns:
            self.full_csv = self.full_csv.dropna(subset=['preprocessed_path'])

        self.train_csv, self.val_csv = train_test_split(
            self.full_csv, test_size=0.3, random_state=42
        )

        self.train_csv.reset_index(inplace=True, drop=True)
        self.val_csv.reset_index(inplace=True, drop=True)

        print(f'Train patients: {self.train_csv.shape[0]}')
        print(f'Val patients: {self.val_csv.shape[0]}')

    def get_dataset(self, train_test, T=None):
        if train_test == 'train':
            print('Apply Train-stage Data Loader (Augmentation Enabled)...')
            misc_args = {'train_test': 'train', 'text_csv': self.train_csv}
            augment = True
        else:
            print('Apply Val-stage Data Loader (Augmentation Disabled)...')
            misc_args = {'train_test': 'val', 'text_csv': self.val_csv}
            augment = False

        dataset = Ortho_CT_Sample_Dataset(
            target_size=self.target_size,
            transform=T,
            augment=augment,
            **misc_args
        )
        print(f'{train_test} dataset total volumes (3D): ', len(dataset))
        return dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test the 3D CT baseline dataset loader.")
    parser.add_argument("--csv-path", required=True)
    args = parser.parse_args()

    manager = Ortho_CT_TEXT_Dataset_Manager(args.csv_path)
    val_patient_names = manager.val_csv.iloc[:, 0].tolist()

    print("\n--- Baseline validation patients ---")
    print(val_patient_names)
    print(f"Number of validation patients: {len(val_patient_names)}")


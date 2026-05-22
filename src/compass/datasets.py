import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import ast
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
from sklearn.model_selection import train_test_split


class Ortho_OMVP_Dataset(Dataset):
    def __init__(self, df, num_samples=64, train=True, target_size=(64, 64)):
        self.df = df.reset_index(drop=True)
        self.num_samples = num_samples
        self.train = train
        self.target_size = target_size

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.df)

    def add_noise(self, img_tensor):
        """Add random Gaussian noise."""
        noise = torch.randn_like(img_tensor) * 0.05
        return img_tensor + noise

    def synchronized_transform(self, img, mask):
        """
        Apply the same spatial transforms to an OMVP image and mask.
        """
        img = TF.resize(img, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.target_size, interpolation=transforms.InterpolationMode.NEAREST)

        if self.train:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            if random.random() > 0.2:
                jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
                img = jitter(img)

        img_tensor = TF.to_tensor(img)
        mask_tensor = TF.to_tensor(mask)

        if self.train and random.random() > 0.5:
            img_tensor = self.add_noise(img_tensor)
            img_tensor = torch.clamp(img_tensor, 0, 1)

        img_tensor = TF.normalize(img_tensor, mean=self.mean, std=self.std)

        return img_tensor, mask_tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        omvp_path = row['mri_omvp_path']
        mask_path = row.get('mask_omvp_path', None)

        try:
            omvp_pool = np.load(omvp_path)
            total_omvps = omvp_pool.shape[0]
        except Exception as e:
            print(f"Error loading {omvp_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.df))

        has_mask = False
        if mask_path and pd.notna(mask_path):
            try:
                mask_pool = np.load(mask_path)
                if mask_pool.shape[0] == total_omvps:
                    has_mask = True
            except:
                has_mask = False

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        if self.train:
            indices = np.random.choice(total_omvps, self.num_samples, replace=(total_omvps < self.num_samples))
        else:
            rng = np.random.RandomState(42)
            indices = rng.choice(total_omvps, self.num_samples, replace=(total_omvps < self.num_samples))

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        bag_images, bag_masks = [], []
        for i in indices:
            img_arr = omvp_pool[i]

            img_pil = Image.fromarray((img_arr.transpose(1, 2, 0) * 255).astype(np.uint8)).convert('RGB')

            if has_mask:
                mask_arr = mask_pool[i]
                mask_pil = Image.fromarray((mask_arr.transpose(1, 2, 0) * 255).astype(np.uint8)).convert('RGB')
            else:
                mask_pil = Image.new('RGB', (self.target_size))

            img_t, mask_t = self.synchronized_transform(img_pil, mask_pil)
            bag_images.append(img_t)
            bag_masks.append(mask_t)

        # -----------------------------------------------------------
        # -----------------------------------------------------------
        target_val = row['tkr_incident_108']
        label = float(target_val)

        target_tensor = torch.tensor([label]).float()

        return {
            'image': torch.stack(bag_images),
            'mask': torch.stack(bag_masks),
            'has_mask': torch.tensor([float(has_mask)]),
            'text': str(row.get('generated_report', "")).lower(),
            'target': target_tensor,
            'image_path': str(row.get('mri_path', ''))
        }


class Ortho_OMVP_Manager:
    def __init__(self, csv_path, num_samples=64, target_size=(64, 64)):
        self.csv_path = csv_path
        self.num_samples = num_samples
        self.target_size = target_size

        print(f'Loading Ortho OMVP dataset from {csv_path}...')
        if csv_path.endswith('.xlsx'):
            self.full_csv = pd.read_excel(csv_path)
        else:
            try:
                self.full_csv = pd.read_csv(csv_path, low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                self.full_csv = pd.read_csv(csv_path, low_memory=False, encoding='gbk')

        if 'mri_path' in self.full_csv.columns:
            print(f"Original size: {len(self.full_csv)}")
            self.full_csv = self.full_csv.dropna(subset=['mri_path'])
            print(f"After dropping missing paths: {len(self.full_csv)}")
        else:
            print("[Warning] 'mri_path' column not found in dataset!")

        self.train_df, self.val_df = train_test_split(
            self.full_csv, test_size=0.3, random_state=42
        )
        print(f">>> Dataset Loaded: {len(self.train_df)} Train, {len(self.val_df)} Val")

    def get_dataset(self, split='train'):
        is_train = (split == 'train')
        return Ortho_OMVP_Dataset(
            self.train_df if is_train else self.val_df,
            num_samples=self.num_samples,
            train=is_train,
            target_size=self.target_size
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test the OMVP dataset loader.")
    parser.add_argument("--csv-path", required=True)
    args = parser.parse_args()

    manager = Ortho_OMVP_Manager(args.csv_path)
    val_patient_names = manager.val_df.iloc[:, 0].tolist()

    train_dataset = manager.get_dataset('train')
    val_dataset = manager.get_dataset('val')

    first_data = train_dataset[0]

    print("Dataset loading succeeded. Image shape:", first_data['image'].shape)

    print("\n--- OMVP validation patients ---")
    print(val_patient_names)
    print(f"Number of validation patients: {len(val_patient_names)}")

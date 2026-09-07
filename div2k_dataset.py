import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):
    def __init__(
        self,
        image_dir,
        patch_size=128,
        patches_per_image=5,
        noise_std=25
    ):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.noise_std = noise_std

        self.image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        print(f"Total images found: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files) * self.patches_per_image

    def __getitem__(self, index):
        image_index = index // self.patches_per_image

        image_path = self.image_files[image_index]

        image = Image.open(image_path).convert("RGB")

        clean = TF.to_tensor(image)

        _, height, width = clean.shape

        top = random.randint(
            0,
            height - self.patch_size
        )

        left = random.randint(
            0,
            width - self.patch_size
        )

        clean = clean[
            :,
            top:top + self.patch_size,
            left:left + self.patch_size
        ]

        if random.random() < 0.5:
            clean = torch.flip(clean, dims=[2])

        if random.random() < 0.5:
            clean = torch.flip(clean, dims=[1])

        noise = torch.randn_like(clean) * (
            self.noise_std / 255.0
        )

        noisy = torch.clamp(
            clean + noise,
            0.0,
            1.0
        )

        return noisy, clean
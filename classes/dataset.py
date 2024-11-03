# data/dataset.py

from torch.utils.data import Dataset
from PIL import Image
import torch
import os

class HAM10000Dataset(Dataset):
    def __init__(self, df, transform=None, use_additional_features=False):
        self.df = df
        self.transform = transform
        self.use_additional_features = use_additional_features

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        # Convert additional features if needed
        if self.use_additional_features:
            age = torch.tensor(self.df.iloc[idx]['age'], dtype=torch.float32)
            sex = torch.tensor(self.df.iloc[idx]['sex'], dtype=torch.long)
            localization = torch.tensor(self.df.iloc[idx]['localization'], dtype=torch.long)
            return (image, age, sex, localization), label
        else:
            return image, label

    def __len__(self):
        return len(self.df)

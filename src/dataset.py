import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string, optional): Directory with all the images (ignored if CSV has full paths).
            transform (callable, optional): transforms (resize, normalize) to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        # 1. Load Image
        img_path = row['image_path']
        if self.img_dir and not os.path.exists(img_path) and not os.path.isabs(img_path):
             img_path = os.path.join(self.img_dir, img_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 2. Load Label
        label = int(row['label'])

        # Apply image transformations (Resize, Normalize)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label
        }
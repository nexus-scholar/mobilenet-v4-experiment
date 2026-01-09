import os
import pandas as pd
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPTokenizer


class PlantDiseaseDataset(Dataset):
    def __init__(self, csv_file, img_dir=None, mask_dir=None, transform=None, tokenizer=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string, optional): Directory with all the images (ignored if CSV has full paths).
            mask_dir (string, optional): Directory with segmentation masks (ignored if CSV has full paths).
            transform (callable, optional): transforms (resize, normalize) to be applied on a sample.
            tokenizer (callable, optional): CLIP tokenizer for text descriptions.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.tokenizer = tokenizer

        # Load prompts
        self.prompts = {}
        prompts_path = os.path.join("data", "PlantWild", "plantwild_prompts.json")
        if os.path.exists(prompts_path):
            with open(prompts_path, 'r') as f:
                self.prompts = json.load(f)
            print(f"Loaded {len(self.prompts)} classes from prompt dictionary.")

        # Max length for CLIP text inputs (standard is 77)
        self.max_length = 77

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

        # --- FIX 1: Strip whitespace and handle potential float/NaN issues ---
        class_name = str(row['class_name']).strip()

        # 3. Process Text (for Phase 2: Distillation)
        # --- FIX 2: Robust Lookup (Try exact match, then lowercase) ---
        # This fixes the "apple black rot " vs "apple black rot" mismatch
        if class_name in self.prompts:
            input_prompts = self.prompts[class_name]
        elif class_name.lower() in self.prompts:
            input_prompts = self.prompts[class_name.lower()]
        else:
            # Fallback only if absolutely necessary
            input_prompts = [f"A photo of {class_name}"]

        # Ensure we have a list
        if not isinstance(input_prompts, list):
            input_prompts = [str(input_prompts)]

        text_description = random.choice(input_prompts)
        text_tokens = torch.zeros(self.max_length, dtype=torch.long)

        if self.tokenizer:
            # Tokenize the description for the CLIP Teacher
            inputs = self.tokenizer(
                text_description,
                padding='max_length',
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            text_tokens = inputs.input_ids.squeeze(0)

        # 4. Load Mask (for Phase 3: Segmentation)
        mask = torch.zeros((1, 224, 224))  # Default empty

        mask_path = row.get('mask_path')
        if pd.notna(mask_path) and isinstance(mask_path, str) and len(mask_path) > 0 and os.path.exists(mask_path):
             try:
                mask_img = Image.open(mask_path).convert('L') # Grayscale
                if self.transform:
                    mask_img = mask_img.resize((224, 224), Image.NEAREST)
                    mask_tensor = torch.from_numpy(np.array(mask_img)).float().unsqueeze(0) / 255.0
                    mask = mask_tensor
             except:
                 pass

        # Apply image transformations (Resize, Normalize)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
            'text_tokens': text_tokens,
            'mask': mask
        }
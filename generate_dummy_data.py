import os
import pandas as pd
import numpy as np
from PIL import Image

# Configuration
NUM_SAMPLES = 10
IMG_DIR = "data/PlantWild/images"
CSV_PATH = "data/PlantWild/metadata.csv"

# Ensure directory exists
os.makedirs(IMG_DIR, exist_ok=True)

data = []

print(f"Generating {NUM_SAMPLES} dummy samples...")

for i in range(NUM_SAMPLES):
    filename = f"leaf_{i}.jpg"

    # 1. Create a random noise image (224x224 RGB)
    # This simulates a plant leaf image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(os.path.join(IMG_DIR, filename))

    # 2. Create dummy metadata
    # We simulate 3 disease classes (0, 1, 2)
    label = i % 3

    # We add text descriptions as required for CLIP distillation
    descriptions = [
        "healthy leaf with smooth texture",
        "leaf with yellow halos and brown spots",  # e.g., Early Blight
        "leaf covered in white powdery substance"  # e.g., Powdery Mildew
    ]

    data.append({
        "filename": filename,
        "label": label,
        "description": descriptions[label],
        "mask_filename": f"mask_{i}.png"  # Placeholder for future segmentation
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(CSV_PATH, index=False)

print(f"Done! Dummy data saved to {CSV_PATH}")
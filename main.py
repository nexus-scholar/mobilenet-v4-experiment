import os

# Define the structure
folders = [
    "data/PlantWild/images",
    "data/PlantSeg/images",
    "data/PlantSeg/masks",
    "src",
    "notebooks",       # For your experiments and analysis
    "models/checkpoints", # To save your trained model weights
    "logs"             # For training logs
]

files = [
    "src/__init__.py",
    "src/dataset.py",
    "src/model.py",
    "src/train.py",
    "requirements.txt",
    "README.md"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created directory: {folder}")

# Create empty files
for file in files:
    if not os.path.exists(file):
        with open(file, 'w') as f:
            pass
        print(f"Created file: {file}")

print("\nProject structure is ready!")
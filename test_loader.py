import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import PlantDiseaseDataset


def test_pipeline():
    # 1. Setup Paths
    csv_file = "data/unified_dataset.csv"
    img_dir = "data/PlantWild/images" # Optional, not used for samples with full paths

    # 2. Define Transforms (CRITICAL FIX)
    # We must convert the raw PIL Image to a PyTorch Tensor.
    # We also resize to ensure all images are 224x224.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts [0-255] image to [0.0-1.0] Tensor
    ])

    # 3. Initialize Dataset WITH Transform
    print("Initializing Dataset...")
    try:
        dataset = PlantDiseaseDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=transform  # <--- PASS THE TRANSFORM HERE
        )
    except FileNotFoundError as e:
        print(f"Dataset CSV not found: {e}")
        return

    # 4. Test loading a single batch
    # We use batch_size=2 to test if the "stacking" works
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    try:
        batch = next(iter(loader))
        print("\n--- Success! Data Loading Verified ---")
        print(f"Image Batch Shape: {batch['image'].shape}")  # Should be [2, 3, 224, 224]
        print(f"Label Batch:       {batch['label']}")  # Should be tensor([x, y])
        print("--------------------------------------")
    except Exception as e:
        print(f"\nFAILED: {e}")


if __name__ == "__main__":
    test_pipeline()
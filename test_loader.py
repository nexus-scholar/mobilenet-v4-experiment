import torch
from torch.utils.data import DataLoader
from torchvision import transforms  # <--- NEW IMPORT
from transformers import CLIPTokenizer
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

    # 3. Initialize Tokenizer
    print("Loading CLIP Tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 4. Initialize Dataset WITH Transform
    print("Initializing Dataset...")
    dataset = PlantDiseaseDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        tokenizer=tokenizer,
        transform=transform  # <--- PASS THE TRANSFORM HERE
    )

    # 5. Test loading a single batch
    # We use batch_size=2 to test if the "stacking" works
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    try:
        batch = next(iter(loader))
        print("\n--- Success! Data Loading Verified ---")
        print(f"Image Batch Shape: {batch['image'].shape}")  # Should be [2, 3, 224, 224]
        print(f"Label Batch:       {batch['label']}")  # Should be tensor([x, y])
        print(f"Text Tokens Shape: {batch['text_tokens'].shape}")  # Should be [2, 77]
        print("--------------------------------------")
    except Exception as e:
        print(f"\nFAILED: {e}")


if __name__ == "__main__":
    test_pipeline()
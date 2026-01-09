import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from src.model import DistilledMobileNet
from src.dataset import PlantDiseaseDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Config
MODEL_PATH = "models/mobilenetv4_tomato_domain_shift_best_model.pth"
CSV_PATH = "data/unified_dataset.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_id_to_name_map(csv_path):
    """Builds a dictionary mapping label_id -> class_name from the CSV."""
    df = pd.read_csv(csv_path)
    # Drop the -1 labels (Out-of-Distribution)
    df = df[df['label'] != -1]
    # Create map: label -> class_name
    # Using groupby to get unique mapping (in case there are duplicates)
    id_to_name = {}
    for _, row in df.iterrows():
        label = int(row['label'])
        class_name = str(row['class_name'])
        if label not in id_to_name:
            id_to_name[label] = class_name
    return id_to_name


def analyze():
    print(f"Loading model from {MODEL_PATH}...")
    # Load Model structure
    model = DistilledMobileNet(num_classes=89)

    # Load Weights
    checkpoint = torch.load(MODEL_PATH)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    # --- FIX: ADD IMAGENET NORMALIZATION ---
    # This matches the normalization used during training in train.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    dataset = PlantDiseaseDataset(CSV_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Get Class ID to Name Mapping
    id_to_name = get_id_to_name_map(CSV_PATH)
    print(f"Loaded {len(id_to_name)} class names from {CSV_PATH}")

    print("Running inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(DEVICE)
            labels = batch['label']

            # Forward pass
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu()

            # Filter out -1 labels (Out-of-Distribution)
            valid_mask = labels != -1
            all_preds.extend(preds[valid_mask].numpy())
            all_labels.extend(labels[valid_mask].numpy())

    # Calculate Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"Overall Accuracy (excluding OOD samples): {acc * 100:.2f}%")
    print(f"{'='*60}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plotting
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, cmap='Blues', annot=False)
    plt.title(f"Confusion Matrix (Accuracy: {acc * 100:.1f}%)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # Top Confused Pairs
    np.fill_diagonal(cm, 0)  # Remove correct predictions

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((i, j, cm[i, j]))

    # Sort by count (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 Most Confused Pairs (Disease Names):")
    print("-" * 80)
    for idx, (i, j, count) in enumerate(pairs[:10], 1):
        name_i = id_to_name.get(i, f"Unknown_ID_{i}")
        name_j = id_to_name.get(j, f"Unknown_ID_{j}")
        print(f"{idx:2d}. '{name_i}' confused as '{name_j}': {int(count)} times")
    print("-" * 80)


if __name__ == "__main__":
    analyze()

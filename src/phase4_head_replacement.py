import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
from tqdm import tqdm
import os
import random

# Project Imports
from src.model import DistilledMobileNet
from src.data_loader import get_domain_dataloaders

# CONFIG
GHOST_CLASSES = [7, 8]  # Spider Mites, Target Spot
SHARED_CLASSES_COUNT = 8
LR = 1e-3
EPOCHS = 15
BATCH_SIZE = 32

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TargetDomainAdapter(nn.Module):
    """Wrapper to handle head replacement."""
    def __init__(self, backbone, num_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features).flatten(1)
        logits = self.classifier(pooled)
        return logits

def get_shared_class_mapping():
    """Maps original 10-class indices to new 8-class indices."""
    # Original: 0,1,2,3,4,5,6, (7), (8), 9
    # New:      0,1,2,3,4,5,6,       7
    mapping = {}
    new_idx = 0
    for old_idx in range(10):
        if old_idx not in GHOST_CLASSES:
            mapping[old_idx] = new_idx
            new_idx += 1
    return mapping

def remap_labels(dataset, mapping):
    """
    Wraps a dataset to remap labels on the fly and filter out ghost classes.
    """
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, label_mapping):
            self.base_dataset = base_dataset
            self.mapping = label_mapping
            
            # Filter indices that belong to ghost classes
            self.valid_indices = []
            for i in range(len(base_dataset)):
                _, label = base_dataset[i]
                if label in self.mapping:
                    self.valid_indices.append(i)
            print(f"Filtered Dataset: {len(self.valid_indices)} samples remaining out of {len(base_dataset)}.")

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            img, label = self.base_dataset[self.valid_indices[idx]]
            return img, self.mapping[label]

    return RemappedDataset(dataset, mapping)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Source model path')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    # 1. Load Data
    print("Loading Target Data...")
    loaders, _ = get_domain_dataloaders(mapping_csv='data/mappings/tomato_class_mapping_fixed.csv', root_dir='data')
    full_target = loaders['target_test'].dataset
    
    mapping = get_shared_class_mapping()
    remapped_target = remap_labels(full_target, mapping)
    
    # Split into Train (500 samples) and Test (Rest)
    # The user mentioned 500 samples for learning.
    train_size = 500
    test_size = len(remapped_target) - train_size
    train_set, test_set = torch.utils.data.random_split(remapped_target, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Prepare Model (Surgery)
    print("Performing Surgery: Replacing 10-class head with fresh 8-class head...")
    base_model = DistilledMobileNet(num_classes=10).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    base_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    
    # Extract backbone and features
    backbone = base_model.backbone
    num_features = base_model.num_features
    
    # Create Adapted Model
    model = TargetDomainAdapter(backbone, num_features, SHARED_CLASSES_COUNT).to(device)
    
    # 3. Fine-Tune
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Fine-tuning for {EPOCHS} epochs on {train_size} field samples...")
    
    best_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "models/mobilenetv4_phase4_adapted.pth")

    print(f"\n==========================================")
    print(f"Surgery Complete.")
    print(f"Final Target Accuracy (8-class): {best_acc:.4f}")
    print(f"Model saved to models/mobilenetv4_phase4_adapted.pth")
    print(f"==========================================")

if __name__ == '__main__':
    main()

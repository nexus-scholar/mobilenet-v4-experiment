import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import csv
import random

# Project Imports
from src.model import DistilledMobileNet
from src.data_loader import get_domain_dataloaders

# CONFIG
ROUNDS = 10
BUDGET_PER_ROUND = 50  # Number of images to "label" per round
EPOCHS_PER_ROUND = 5   # Fine-tuning epochs per round
LR = 1e-4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_uncertainties(model, loader, device):
    """Calculate Entropy for the entire unlabeled pool."""
    model.eval()
    uncertainties = []
    indices = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(loader, desc="Scanning Pool")):
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Entropy: -sum(p * log(p))
            probs = torch.softmax(outputs, dim=1)
            log_probs = torch.log_softmax(outputs, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            
            uncertainties.extend(entropy.cpu().numpy())
            # We need to track which original dataset index this corresponds to
            # This logic assumes the loader is sequential and not shuffled for scanning
            start_idx = batch_idx * loader.batch_size
            batch_indices = list(range(start_idx, start_idx + inputs.size(0)))
            indices.extend(batch_indices)
            
    return np.array(indices), np.array(uncertainties)

def train_round(model, train_loader, device, epochs):
    """Fine-tune the model on the newly labeled set."""
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return model

def evaluate(model, test_loader, device):
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
    return correct / total

def run_strategy(strategy_name, full_dataset, test_loader, device, initial_model_state):
    print(f"\n--- Running Strategy: {strategy_name} ---")
    
    # Reset Model
    model = DistilledMobileNet(num_classes=10).to(device)
    model.load_state_dict(initial_model_state)
    
    # Available Pool (Indices of the full dataset)
    pool_indices = np.arange(len(full_dataset))
    labeled_indices = []
    
    accuracies = []
    sample_counts = []
    
    # Initial Evaluation (0 labels)
    acc = evaluate(model, test_loader, device)
    accuracies.append(acc)
    sample_counts.append(0)
    print(f"Round 0 (0 samples): Acc = {acc:.4f}")
    
    for r in range(1, ROUNDS + 1):
        # 1. Select Samples
        if strategy_name == "Random":
            # Ensure we don't select more than available
            n_select = min(BUDGET_PER_ROUND, len(pool_indices))
            if n_select == 0:
                break
            selected = np.random.choice(pool_indices, size=n_select, replace=False)
        
        elif strategy_name == "Uncertainty":
            # Create a loader for the current pool to scan it
            pool_subset = Subset(full_dataset, pool_indices)
            # Important: shuffle=False to map indices back correctly
            pool_loader = DataLoader(pool_subset, batch_size=32, shuffle=False, num_workers=0)
            
            # Get entropy
            relative_indices, uncertainties = calculate_uncertainties(model, pool_loader, device)
            
            # Get top K entropy indices (relative to the subset)
            n_select = min(BUDGET_PER_ROUND, len(pool_indices))
            if n_select == 0:
                break
            
            # argsort is ascending, so we take the last K for highest entropy
            top_k_relative = np.argsort(uncertainties)[-n_select:]
            
            # Map back to global pool indices
            # The pool_subset indices are 0..len(pool)-1, which correspond to pool_indices[i]
            # So if top_k_relative is [0, 5], we want pool_indices[0] and pool_indices[5]
            selected = pool_indices[top_k_relative]

        # 2. Update Pools
        labeled_indices.extend(selected)
        # Remove selected from pool
        pool_indices = np.setdiff1d(pool_indices, selected)
        
        # 3. Train
        train_subset = Subset(full_dataset, labeled_indices)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=0)
        
        model = train_round(model, train_loader, device, EPOCHS_PER_ROUND)
        
        # 4. Evaluate
        acc = evaluate(model, test_loader, device)
        accuracies.append(acc)
        sample_counts.append(len(labeled_indices))
        
        print(f"Round {r} ({len(labeled_indices)} samples): Acc = {acc:.4f}")
        
    return sample_counts, accuracies

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Source model path')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(42)
    
    # 1. Load Data
    print("Loading Data...")
    # We use 'target_train' as our Unlabeled Pool (simulating field data we collect)
    # We use 'target_test' for evaluation
    loaders, _ = get_domain_dataloaders(mapping_csv='data/mappings/tomato_class_mapping_fixed.csv', root_dir='data')
    
    # NOTE: Your loader function likely splits Target into Train/Test. 
    # If not, you might need to split it here.
    if 'target_train' not in loaders:
        print("Splitting Target Data into Pool (50%) and Test (50%)...")
        full_target = loaders['target_test'].dataset
        total_size = len(full_target)
        train_size = int(0.5 * total_size)
        test_size = total_size - train_size
        
        # Use generator for reproducibility
        generator = torch.Generator().manual_seed(42)
        target_pool_set, target_test_set = torch.utils.data.random_split(full_target, [train_size, test_size], generator=generator)
        
        pool_dataset = target_pool_set
        test_loader = DataLoader(target_test_set, batch_size=32, shuffle=False, num_workers=0)
    else:
        pool_dataset = loaders['target_train'].dataset
        test_loader = loaders['target_test']

    print(f"Pool Size: {len(pool_dataset)}")
    print(f"Test Size: {len(test_loader.dataset)}")
    
    # 2. Load Initial Model State
    print(f"Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 3. Run Experiments
    x_rand, y_rand = run_strategy("Random", pool_dataset, test_loader, device, state_dict)
    x_unc, y_unc = run_strategy("Uncertainty", pool_dataset, test_loader, device, state_dict)
    
    # 4. Save Results
    results_df = pd.DataFrame({
        'Samples': x_rand,
        'Random_Acc': y_rand,
        'Uncertainty_Acc': y_unc
    })
    results_df.to_csv('active_learning_results.csv', index=False)
    print("\nResults saved to active_learning_results.csv")
    
    # Plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(x_rand, y_rand, marker='o', label='Random Sampling')
        plt.plot(x_unc, y_unc, marker='s', label='Uncertainty Sampling')
        plt.title('Active Learning: Lab-to-Field Adaptation')
        plt.xlabel('Number of Field Labels')
        plt.ylabel('Target Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig('active_learning_curve.png')
        print("Plot saved to active_learning_curve.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == '__main__':
    main()

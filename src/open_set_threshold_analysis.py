import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Project Imports
from src.model import DistilledMobileNet
from src.data_loader import get_domain_dataloaders

# CONFIG
GHOST_CLASSES = [7, 8]  # Spider Mites, Target Spot (Source Only)
THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def analyze_thresholds(model, loader, device, model_name):
    model.eval()
    results = []
    
    print(f"\nScanning {model_name}...")
    
    # Store all logits and labels first
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get Max Probability and Prediction
            max_probs, preds = torch.max(probs, dim=1)
            
            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Analyze per threshold
    for t in THRESHOLDS:
        # 1. Reject samples below threshold
        accepted_mask = all_probs >= t
        rejected_count = (~accepted_mask).sum()
        
        # 2. Filter predictions
        valid_preds = all_preds[accepted_mask]
        valid_labels = all_labels[accepted_mask]
        
        if len(valid_labels) == 0:
            acc = 0.0
        else:
            acc = (valid_preds == valid_labels).sum() / len(valid_labels)
            
        # 3. Count Ghost Class Predictions (How many times did it predict Mites/Target Spot?)
        # We look at ALL predictions (before rejection) to see the bias
        # But specifically, let's look at how many ghosts survive the threshold
        ghost_mask = np.isin(valid_preds, GHOST_CLASSES)
        ghost_count = ghost_mask.sum()
        ghost_ratio = ghost_count / len(valid_preds) if len(valid_preds) > 0 else 0
        
        print(f"Thresh {t}: Acc {acc:.4f} | Rejected {rejected_count} ({rejected_count/len(all_labels)*100:.1f}%) | Ghost Preds {ghost_ratio*100:.1f}%")
        
        results.append({
            "Model": model_name,
            "Threshold": t,
            "Accuracy": acc,
            "Rejection_Rate": rejected_count / len(all_labels),
            "Ghost_Prediction_Rate": ghost_ratio
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, help='Path to Source-Only Baseline .pth')
    # Optional: Compare with your active learning model if you saved it
    # parser.add_argument('--active', type=str, help='Path to Active Learning Best Model .pth')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    # 1. Load Data
    # Added root_dir='data' to fix the error encountered in previous tasks
    loaders, _ = get_domain_dataloaders(mapping_csv='data/mappings/tomato_class_mapping_fixed.csv', root_dir='data')
    test_loader = loaders['target_test']
    
    all_results = []
    
    # 2. Analyze Baseline
    model = DistilledMobileNet(num_classes=10).to(device)
    print(f"Loading Baseline: {args.baseline}")
    ckpt = torch.load(args.baseline, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    
    res_baseline = analyze_thresholds(model, test_loader, device, "Baseline (Source Only)")
    all_results.extend(res_baseline)
    
    # 3. Save & Plot
    df = pd.DataFrame(all_results)
    print("\nResults Summary:")
    print(df)
    df.to_csv("threshold_analysis.csv", index=False)
    
    # Simple Plot
    try:
        plt.figure(figsize=(10,5))
        subset = df[df['Model'] == "Baseline (Source Only)"]
        plt.plot(subset['Threshold'], subset['Accuracy'], marker='o', label='Accuracy (on Accepted)')
        plt.plot(subset['Threshold'], subset['Ghost_Prediction_Rate'], marker='x', linestyle='--', label='Ghost Prediction Rate')
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Rate")
        plt.title("Impact of Rejection Threshold on Accuracy & Hallucinations")
        plt.legend()
        plt.grid(True)
        plt.savefig("threshold_analysis.png")
        print("\nSaved plot to threshold_analysis.png")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == '__main__':
    main()

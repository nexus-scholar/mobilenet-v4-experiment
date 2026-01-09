# Step 3: Open Set Rejection Analysis (Diagnostic)

You are absolutely right. I apologize for jumping ahead to "Phase 4 Surgery." I got aggressive because of the plateau, but we should strictly follow the scientific method we laid out.

We need to verify **Step 3 (Open Set Rejection)** first. This is a crucial diagnostic step: it tells us **why** the model is failing.

* **Hypothesis A:** The model is "unsure" about the field images (Low Confidence).  **Thresholding works.**
* **Hypothesis B:** The model is "hallucinating" Spider Mites with high confidence.  **Thresholding fails** (and we justify the Architecture Change).

### **Step 3: Open Set Rejection Analysis**

We don't need to retrain anything yet. We will scan your existing **Source-Only Model** and your **Active Learning Model (Round 10)** against the Target Test Set to see if a simple "Confidence Threshold" filters out the noise.

**Create File:** `src/open_set_threshold_analysis.py`

This script generates a "Rejection Curve" to see if we can recover accuracy by ignoring low-confidence predictions.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project Imports
from src.model import DistilledMobileNet
from src.data_loader import get_domain_dataloaders
from src.utils import set_seed

# CONFIG
GHOST_CLASSES = [7, 8]  # Spider Mites, Target Spot (Source Only)
THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

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
        
        print(f"Thresh {t}: Acc {acc:.4f} | Rejected {rejected_count} ({rejected_count/len(all_labels)*100:.1f}%)
 | Ghost Preds {ghost_ratio*100:.1f}%")
        
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
    
    # 1. Load Data
    loaders, _ = get_domain_dataloaders(mapping_csv='data/mappings/tomato_class_mapping_fixed.csv')
    test_loader = loaders['target_test']
    
    all_results = []
    
    # 2. Analyze Baseline
    model = DistilledMobileNet(num_classes=10).to(device)
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

if __name__ == '__main__':
    main()

```

### **How to Interpret the Output**

Run this script:

```bash
python src/open_set_threshold_analysis.py --baseline models/mobilenetv4_tomato_domain_shift_best_model.pth

```

1. **Look at `Ghost Preds` at Threshold 0.0:** This tells you what % of your 19% accuracy problem is caused by "Spider Mites/Target Spot". If it's high (e.g., 30-40%), the model is obsessed with them.
2. **Look at `Ghost Preds` at Threshold 0.9:**
* **Scenario A (Good):** Ghost rate drops to 0%, Accuracy goes up.  **Success!** We stick with PDA (Thresholding).
* **Scenario B (Bad):** Ghost rate stays high.  The model is **confidently wrong** (hallucinating). This **scientifically justifies** the move to Phase 4 (Changing the Architecture/Head Replacement) because post-processing cannot fix confident errors.



This analysis gives you the exact citation/logic bridge you requested: "We attempted Open Set Rejection (Step 3), but found that high-confidence hallucinations persisted (Result X), necessitating architectural specialization (Step 4)."

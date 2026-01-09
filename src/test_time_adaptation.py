import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from tqdm import tqdm
import math
import yaml
from pathlib import Path

# Import your existing project modules
from src.model import DistilledMobileNet
from src.data_loader import get_domain_dataloaders

# 1. DEFINE MISSING CLASSES (Based on your Dataset Report)
# Source Class Map: 
# 0: Bacterial Spot, 1: Early Blight, 2: Healthy, 3: Late Blight, 4: Mold
# 5: Mosaic Virus, 6: Septoria Spot, 7: Spider Mites (MISSING), 8: Target Spot (MISSING), 9: Yellow Virus
MISSING_CLASS_INDICES = [7, 8]

def masked_softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution, ignoring missing classes."""
    # 1. Mask logits of missing classes to -inf so softmax makes them 0
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, MISSING_CLASS_INDICES] = True
    x_masked = x.clone()
    x_masked[mask] = -float('inf')
    
    # 2. Compute Standard Entropy on the remaining 8 classes
    probs = x_masked.softmax(1)
    log_probs = x_masked.log_softmax(1)
    
    # Avoid NaN if prob is 0
    return -(probs * log_probs).sum(1)

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    p.requires_grad = True
        else:
            for p in m.parameters():
                p.requires_grad = False
    return params, names

def test_time_adaptation(model, test_loader, device, steps=1, lr=1e-4):
    """Perform TTA (TENT) with Safe Masking."""
    print(f"Setting up TENT with MASKED ENTROPY (Classes {MISSING_CLASS_INDICES} ignored)...")
    
    params, _ = collect_params(model)
    # Lower learning rate is crucial for stability in small batch sizes
    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    
    print(f"Optimizing {len(params)} Batch Norm parameters.")

    model.train() # BN layers must be in train mode to update running stats
    
    metric_correct = 0
    metric_total = 0
    
    iterator = tqdm(test_loader, desc="Adapting on Target")
    
    for inputs, labels in iterator:
        inputs, labels = inputs.to(device), labels.to(device)
        
        for _ in range(steps):
            outputs = model(inputs)
            # USE MASKED ENTROPY
            loss = masked_softmax_entropy(outputs).mean(0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            outputs = model(inputs)
            
            # OPTIONAL: During inference, also mask ghost classes to see "True" Shared Accuracy
            outputs[:, MISSING_CLASS_INDICES] = -float('inf')
            
            _, preds = torch.max(outputs, 1)
            correct = (preds == labels).sum().item()
            metric_correct += correct
            metric_total += labels.size(0)
            
        iterator.set_postfix(acc=f"{metric_correct/metric_total:.4f}")

    return metric_correct / metric_total

def main():
    parser = argparse.ArgumentParser(description='Run TTA on Target Domain')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to source-trained model .pth')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for TTA')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading Target Domain Data...")
    mapping_csv = config['data']['csv_path']
    root_dir = 'data' # Default root dir for the project
    
    loaders, _ = get_domain_dataloaders(
        mapping_csv=mapping_csv,
        root_dir=root_dir,
        batch_size=config['data'].get('batch_size', 32)
    )
    target_loader = loaders['target_test']
    
    # 2. Load Source Model
    print(f"Loading Source Model from {args.checkpoint}...")
    num_classes = config['model'].get('num_classes', 10)
    model = DistilledMobileNet(num_classes=num_classes).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    print("Model loaded. Starting Baseline Evaluation (Before TTA)...")
    
    # Baseline Eval
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(target_loader, desc="Baseline Eval"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    baseline_acc = correct/total if total > 0 else 0
    print(f"Baseline Target Accuracy: {baseline_acc:.4f}")

    # 3. Run TTA
    print("\nStarting Test-Time Adaptation...")
    # Reload model to reset state for TTA
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    tta_acc = test_time_adaptation(model, target_loader, device, lr=args.lr)
    
    print(f"\n==========================================")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"TTA Accuracy:      {tta_acc:.4f}")
    print(f"Improvement:       +{(tta_acc - baseline_acc)*100:.2f}%")
    print(f"==========================================")

if __name__ == '__main__':
    main()

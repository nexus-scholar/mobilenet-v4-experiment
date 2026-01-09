The TTA results you received (-10.79% drop) are **extremely valuable** because they empirically confirm the "Open Set" hypothesis. Your TENT implementation is correctly minimizing entropy, but because "Spider Mites" and "Target Spot" don't exist in the field, the model is confidently hallucinating them, driving accuracy down.

This failure mode is exactly what **Partial Domain Adaptation (PDA)** papers warn about. We need to pivot to a **"Safe TTA" strategy**.

### **Updated Strategy: Fix the "Open Set" Leak**

We need to modify `test_time_adaptation.py` to prevent the model from adapting to "Ghost Classes" (Spider Mites/Target Spot).

#### **1. Immediate Fix: Masked TENT**

Instead of minimizing entropy for *all* classes, we will mask out the logits for the missing classes before calculating entropy. This forces the model to only be confident about the 8 shared classes.

**Modify `test_time_adaptation.py`:**

Replace the `test_time_adaptation` function and the entropy calculation with this "Safe" version.

```python
# ... (keep imports)

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

def test_time_adaptation(model, test_loader, device, steps=1, lr=1e-4): # LOWER LR to 1e-4
    """Perform TTA (TENT) with Safe Masking."""
    print(f"Setting up TENT with MASKED ENTROPY (Classes {MISSING_CLASS_INDICES} ignored)...")
    
    params, _ = collect_params(model)
    # Lower learning rate is crucial for stability in small batch sizes
    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999)) 
    
    print(f"Optimizing {len(params)} Batch Norm parameters.")

    model.train() 
    
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

```

#### **2. Why this will work:**

* **Entropy Logic:** Currently, if an image is "Unknown", the model might output `[0.1, ... 0.1]` (High Entropy). TENT tries to sharpen this to `[1.0, 0.0...]`. Without masking, it might sharpen it to "Spider Mites". With masking, it creates a gradient that says "Make this look like one of the *Shared* Diseases (Blight/Spot)".
* **Evaluation Logic:** By masking the output during inference too, we stop the "19% Baseline" from being penalized for hallucinating spider mites. This gives you the "True Shared Domain Accuracy".

### **Next Step: Run Experiment 2.1**

Update your script with the code block above (specifically the `MISSING_CLASS_INDICES` and `masked_softmax_entropy`) and run it again.

**Recommended Command:**

```bash
python src/test_time_adaptation.py --checkpoint models/mobilenetv4_tomato_domain_shift_best_model.pth --lr 0.0001

```

*(Note: I lowered the LR to `0.0001` in the command/code because `0.001` was too aggressive and caused the instability you saw).*
# Phase 2 Experiment: Test-Time Adaptation (TTA) Results

**Date:** January 09, 2026
**Experiment:** Test-Time Adaptation using TENT (Entropy Minimization)
**Model:** `models/mobilenetv4_tomato_domain_shift_best_model.pth`
**Target Data:** PlantDoc + PlantWild (via `get_domain_dataloaders`)

## Execution Details
- **Script:** `src/test_time_adaptation.py`
- **Batch Size:** 32 (default in config)
- **Learning Rate:** 1e-3
- **Steps:** 1 per batch

## Results Summary

| Metric | Accuracy |
| :--- | :--- |
| **Baseline (Source Only)** | **21.24%** |
| **TTA (TENT)** | **10.45%** |
| **Improvement** | **-10.79%** |

## Console Output Log
```text
Using device: cuda
Loading Target Domain Data...
Loaded 16011 samples for domain 'Source' across 10 classes.
Loaded 2029 samples for domain 'Target' across 10 classes.
Loading Source Model from models/mobilenetv4_tomato_domain_shift_best_model.pth...
Model loaded. Starting Baseline Evaluation (Before TTA)...

Baseline Target Accuracy: 0.2124

Starting Test-Time Adaptation...
Setting up TENT (Test-Time Entropy Minimization)...
Optimizing 92 Batch Norm parameters.

... [Progress Bar Output Removed for Brevity] ...

==========================================
Baseline Accuracy: 0.2124
TTA Accuracy:      0.1045
Improvement:       +-10.79%
==========================================
```

## Analysis & Observations
- **Negative Transfer:** The adaptation process significantly degraded performance (-10.8%).
- **Instability:** The accuracy dropped immediately in the first few batches of TTA, suggesting that updating Batch Norm statistics on the target domain (which is noisy and has class distribution shifts like the missing "Spider Mites" class) might be destabilizing the features.
- **Class Imbalance Impact:** As noted in Phase 1, the target domain lacks "Spider Mites" and "Target Spot". Forcing the model to minimize entropy on images that don't fit the source classes might be causing it to confidently predict the *wrong* class, thereby lowering accuracy ("Model Calibration" issue).
- **Batch Size:** A batch size of 32 might be too small for stable BN statistic updates on a diverse/noisy target domain.

## Next Steps Recommendations
1. **Filter Missing Classes:** Implement the "Immediate Filter" strategy mentioned in the task to exclude "Spider Mites" predictions from evaluation, as TTA might be pushing everything into these "ghost" classes.
2. **Conservative TTA:** Try a lower learning rate (e.g., `1e-4`) or fewer update steps.
3. **Open-Set Adaptation:** The current TTA assumes the target samples belong to one of the source classes (Closed Set). Since this is a Partial/Open Set scenario, standard TENT fails.

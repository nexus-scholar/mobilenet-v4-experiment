# Phase 2.1 Experiment: Safe Test-Time Adaptation (Masked) Results

**Date:** January 09, 2026
**Strategy:** "Safe TTA" - Masking "Ghost Classes" (Spider Mites, Target Spot) during Entropy Minimization.
**Rationale:** Preventing the model from adapting towards classes that do not exist in the target domain to reduce hallucination.

## Execution Details
- **Script:** `src/test_time_adaptation.py` (Modified for Masking)
- **Masked Classes:** `[7, 8]` (Spider Mites, Target Spot)
- **Learning Rate:** `1e-4` (Lowered from 1e-3)
- **Batch Size:** 32

## Results Summary

| Metric | Accuracy | Notes |
| :--- | :--- | :--- |
| **Baseline** | **21.24%** | Standard Source-Only Model |
| **TTA (Standard)** | **10.45%** | Phase 2.0 Result (Unstable) |
| **Safe TTA (Masked)** | **14.24%** | **+3.79%** vs Standard TTA, but still **-7.0%** vs Baseline |

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
Setting up TENT with MASKED ENTROPY (Classes [7, 8] ignored)...
Optimizing 92 Batch Norm parameters.

... [Progress Bar Output Removed] ...

==========================================
Baseline Accuracy: 0.2124
TTA Accuracy:      0.1424
Improvement:       +-7.00%
==========================================
```

## Analysis & Conclusion

1.  **Masking Helped Stability:** The accuracy improved from 10.45% (Phase 2.0) to 14.24% (Phase 2.1) by masking the ghost classes and lowering the learning rate. This confirms that "Negative Transfer" into the missing classes was part of the problem.
2.  **Still Degrading:** Despite the safety measures, TTA still hurts performance (-7% vs baseline). This indicates that the **Batch Norm statistics of the target domain are fundamentally mismatched** with the source-trained weights, or the batch size (32) is insufficient to estimate robust statistics for adaptation.
3.  **Strategic Pivot:** "Blind" adaptation (TTA) is not working for this specific Lab-to-Field shift. The model needs supervision to bridge the gap. We must move to **Phase 3: Active Learning** to explicitly teach the model the features of the target domain.

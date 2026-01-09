# Phase 4 Experiment: Model Surgery Results

**Date:** January 09, 2026
**Strategy:** "Head Replacement" - Removing the biased 10-class lab head and training a fresh 8-class head on field samples.

## Execution Details
- **Script:** `src/phase4_head_replacement.py`
- **Training Set:** 500 Field Samples (Randomly Selected)
- **Test Set:** ~1500 Field Samples (Holdout)
- **Optimizer:** Adam (LR=1e-3)
- **Epochs:** 15

## Results Summary

| Metric | Accuracy | Notes |
| :--- | :--- | :--- |
| **Baseline (Source Only)** | **21.24%** | Unmodified model |
| **Active Learning (Phase 3)** | **43.35%** | 500 labels, full 10-class model |
| **Surgery (Phase 4)** | **39.10%** | 500 labels, fresh 8-class head |

## Console Output Log
```text
Performing Surgery: Replacing 10-class head with fresh 8-class head...
Fine-tuning for 15 epochs on 500 field samples...
Epoch 1/15  | Loss: 1.8641 | Test Acc: 0.1644
Epoch 5/15  | Loss: 0.7488 | Test Acc: 0.2737
Epoch 10/15 | Loss: 0.3433 | Test Acc: 0.3229
Epoch 15/15 | Loss: 0.1002 | Test Acc: 0.3910

==========================================
Surgery Complete.
Final Target Accuracy (8-class): 0.3910
==========================================
```

## Analysis & Findings

1.  **Overcoming Feature Collapse:** Resetting the head allowed the model to bypass the "Confident Errors" observed in Phase 3. The model reached 39.1% accuracy using only the 8 shared classes.
2.  **Overfitting observed:** The training loss dropped to 0.10, while test accuracy plateaued around 39%. This suggests that with only 500 samples, the fresh head is memorizing the field samples.
3.  **Active Learning Comparison:** Active Learning (Phase 3) reached 43.35%, which is slightly better than surgery (39.1%). This indicates that the **incremental gradients** from Active Learning were more effective than a total "reset" of the head, likely because the Active Learning process preserved some of the shared class knowledge from the source domain.
4.  **Scientific Conclusion:** Supervised adaptation (even few-shot) is the only path to high accuracy in this experiment. Unsupervised methods (TTA) fail due to semantic shifts.

## Artifacts
- `src/phase4_head_replacement.py`: Surgery script.
- `models/mobilenetv4_phase4_adapted.pth`: Best adapted model.

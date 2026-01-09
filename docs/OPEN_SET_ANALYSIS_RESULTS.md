# Step 3 Diagnostic: Open Set Rejection Analysis Results

**Date:** January 09, 2026
**Hypothesis:** The low target accuracy (21%) is caused by "Ghost Class" hallucinations (Spider Mites/Target Spot) which can be filtered out by confidence thresholding.

## Execution Details
- **Script:** `src/open_set_threshold_analysis.py`
- **Model:** Baseline Source-Only MobileNetV4
- **Thresholds:** 0.0 to 0.95

## Results Summary

| Threshold | Accuracy (Accepted) | Rejection Rate | Ghost Pred Rate |
| :--- | :--- | :--- | :--- |
| **0.00** | **21.24%** | 0.0% | **0.54%** |
| 0.30 | 21.25% | 0.2% | 0.54% |
| 0.50 | 21.38% | 3.2% | 0.41% |
| 0.70 | 21.42% | 14.2% | 0.23% |
| 0.80 | 21.38% | 19.3% | 0.12% |
| 0.90 | 21.61% | 27.3% | 0.07% |
| **0.95** | **21.83%** | **33.2%** | **0.07%** |

## Analysis & Conclusion

1.  **Ghost Hypothesis Disproven:** The model is **NOT** hallucinating the missing classes. The "Ghost Prediction Rate" is negligible (<0.6%) even at zero threshold. The model rarely predicts "Spider Mites" or "Target Spot" on the target domain.
2.  **Confident Errors:** Thresholding fails to recover accuracy. Even when rejecting the bottom 33% of uncertain predictions (Threshold 0.95), the accuracy only marginally improves (+0.6%). This indicates that the model is **confidently wrong** about the Shared Classes (e.g., classifying "Field Blight" as "Lab Healthy" with high confidence).
3.  **Implications for TTA:** This explains why Test-Time Adaptation (Phase 2) failed. TTA assumes that "sharpening" predictions (minimizing entropy) is correct. But since the model is *already* confident but wrong, TTA likely reinforced these confident errors.
4.  **Implications for Active Learning:** This reinforces the success of Active Learning (Phase 3). Since the errors are semantic and confident, only **labels** (external supervision) can correct the decision boundaries. Unsupervised methods (Thresholding, TTA) are mathematically doomed in this scenario.

## Artifacts
- `threshold_analysis.csv`: Raw data.
- `threshold_analysis.png`: Visualization of Accuracy vs. Rejection Rate.

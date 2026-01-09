# Phase 4 Strategy: Model Surgery (Head Replacement)

## The "Plot Twist": It’s Not Hallucination, It’s Feature Collapse

The Open Set Analysis (Step 3) has revealed a critical insight:
- **Ghost Prediction Rate:** **0.54%** (Negligible).
- **Effect of Thresholding:** Rejecting 33% of the data (Threshold 0.95) only raised accuracy by **+0.6%**.

### **Scientific Conclusion**

1.  **The Model is "Confidently Wrong":** The low accuracy (21%) is **not** because the model is "confused" or hallucinating Spider Mites. It is because the model is predicting the *wrong* shared class with *high confidence* (e.g., classifying "Field Blight" as "Lab Healthy" with 99% certainty).
2.  **Why TTA Failed:** TENT works by sharpening predictions. Since the model was already confidently wrong, **TTA aggressively reinforced the wrong answers**, causing the performance crash.
3.  **The Root Cause:** The domain shift caused a **Semantic Feature Collapse**. The convolutional filters learned in the lab are being misinterpreted by the classifier head when exposed to field backgrounds.

---

## **The Solution: Phase 4 Surgery**

We must break the "Feature Collapse" by disconnecting the old logic and forcing a re-learning of the class mapping.

### **Mechanism**
1.  **Backbone Preservation:** Keep the MobileNetV4 backbone (which still has good edge/texture detectors).
2.  **Head Replacement:** Discard the 10-class head. Initialize a **fresh 8-class head** (random weights).
3.  **Target Fine-Tuning:** Train this new head purely on "Field" samples. This forces the model to map field features to labels without the interference of the "Ghost Class" neurons or the biased lab-trained weights.

### **Expected Outcome**
By treating this as a **Few-Shot Transfer Learning** task, we expect accuracy to jump significantly (target: 60-70%) as the model finally "understands" the field domain.

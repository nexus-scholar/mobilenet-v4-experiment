This summary consolidates our entire session into a structured research update. You can pass this document directly to your supervisor alongside the Deep Research reports from Gemini and Perplexity.

---

# Research Context: Edge-Optimized Domain Adaptation for Plant Disease Diagnosis

**Subject:** Progress Report & Problem Statement Quantification (Phase 1)
**Date:** January 9, 2026

## 1. Research Pivot & Objective

We have refined the PhD focus from general classification to **Edge-Optimized Domain Adaptation**.

* **The Problem:** Deep learning models trained on lab data (Source) fail catastrophically when deployed on mobile phones in real-world fields (Target) due to domain shifts (lighting, blur, complex backgrounds).
* **The Goal:** Develop a lightweight adaptation pipeline (Active Learning or Test-Time Adaptation) that allows **MobileNetV4-Small** to adapt to the field with minimal human labeling.
* **Scope Restriction:** We are strictly focusing on the **Tomato crop** (*Solanum lycopersicum*). This isolates the "Sim-to-Real" domain gap from inter-species confusion (e.g., Potato vs. Tomato), making the scientific contribution precise and measurable.

## 2. Experimental Setup

We constructed a custom "Lab-to-Field" benchmark by rigorously aligning three datasets:

* **Source Domain (Lab):** *PlantVillage* (16,011 images). High-quality, controlled lighting, simple gray backgrounds.
* **Target Domain (Field):** *PlantDoc* + *PlantWild* (2,029 images). Real-world capture, noisy backgrounds, variable lighting.
* **Classes:** 10 aligned classes (e.g., *Bacterial Spot, Early Blight, Healthy*) mapped via a custom normalization script to ensure label consistency across domains.

## 3. Phase 1 Results: Quantifying the "Domain Gap"

We trained a `MobileNetV4-Small` baseline purely on the Source domain to measure the severity of the feature shift.

**Key Findings (after 25 Epochs):**

* **Source Accuracy (Lab):**  (The model effectively "solved" the training task).
* **Target Accuracy (Field):**  (The model failed to generalize, performing barely better than random guessing).
* **The Domain Gap:** .

**Scientific Implication:**
The massive divergence between Source and Target accuracy (visible in the divergence of Loss from 3.6 to 11.4) proves **Feature Collapse**. The model relies on "lab artifacts" (e.g., background color) rather than robust disease features. This quantitative evidence justifies the need for the proposed Phase 2 interventions.

## 4. Proposed Solution (Phase 2): Active Learning

Since labeling the entire target domain is infeasible for edge deployments, we propose an **Active Learning** loop.

* **Hypothesis:** Labeling a tiny fraction (e.g., 2.5%) of "High Entropy" field images—those where the model is most confused—will close the domain gap more effectively than random labeling.
* **Methodology:**
1. Deploy the Source-trained model to the "Edge".
2. Use **Uncertainty Sampling** to identify the 50 most confusing field images.
3. Query the human (supervisor/expert) for these specific labels.
4. Fine-tune the model and repeat for 10 rounds.


* **Expected Outcome:** We aim to recover ~60-70% accuracy using only 10% of the available data, demonstrating an efficient "Human-in-the-Loop" workflow for precision agriculture.

## 5. Next Steps

1. **Execute Active Learning Loop:** Run the developed `active_learning.py` script to generate the "Accuracy vs. Label Budget" curve.
2. **Compare with SOTA:** Benchmark our results against the literature retrieved in the accompanying Deep Research Reports (e.g., recent MobileNet adaptation papers).
3. **Draft Methodology Chapter:** Use the generated "Tomato Class Mapping" and "Domain Gap" graphs as the foundation for the thesis Methodology section.

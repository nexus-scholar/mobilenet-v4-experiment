# Dataset Report: Tomato Domain Shift Experiment

**Date:** January 09, 2026  
**Project:** MobileNetV4 Tomato Domain Shift  
**Goal:** Domain Adaptation from Lab (Source) to Field (Target)

---

## 1. Dataset Overview

This experiment utilizes three datasets focused on Tomato leaf diseases. **PlantVillage** serves as the label-rich **Source Domain** (lab setting), while **PlantDoc** and **PlantWild** serve as the **Target Domains** (field/wild settings).

| Feature | PlantVillage (Source) | PlantDoc (Target) | PlantWild (Target) |
| :--- | :--- | :--- | :--- |
| **Source URL** | [Kaggle / Penn State](https://www.kaggle.com/datasets/emmarex/plantdisease) | [GitHub](https://github.com/pratikkayal/PlantDoc-Dataset) | [Kaggle / Research](https://www.kaggle.com/datasets) |
| **Environment** | Controlled Lab (Uniform BG) | Field / In-the-wild | Field / In-the-wild |
| **Total Size** | 0.24 GB | 0.22 GB | 0.26 GB |
| **Image Count** | 16,011 | 746 | 1,966 |
| **File Formats** | JPG, JPEG | JPG, PNG | JPG |
| **Resolution** | 256x256 (Fixed) | Variable (Mode: 800x600) | Variable (Mode: 690x518) |
| **License** | Public Domain / CC BY-SA | MIT (Check Repo) | Public Research |

---

## 2. Class Distribution & Statistics

### 2.1 PlantVillage (Source)
*High-quality, centered leaves on grey/uniform backgrounds.*

*   **Total Images:** 16,011
*   **Class Count:** 10
*   **Distribution:** Imbalanced. "Yellow Leaf Curl Virus" is dominant.

```text
Tomato_YellowLeaf_Curl_Virus (3209) |####################
Tomato_Bacterial_spot        (2127) |#############
Tomato_Late_blight           (1909) |###########
Tomato_Septoria_leaf_spot    (1771) |###########
Tomato_Spider_mites          (1676) |##########
Tomato_healthy               (1591) |#########
Tomato_Target_Spot           (1404) |########
Tomato_Early_blight          (1000) |######
Tomato_Leaf_Mold             (952)  |#####
Tomato_mosaic_virus          (373)  |##
```

### 2.2 PlantDoc (Target)
*Real-world images, often multiple leaves, complex backgrounds.*

*   **Total Images:** 746
*   **Class Count:** 9 (Mapped)
*   **Distribution:** Sparse. "Spider mites" is critically under-represented.

```text
Tomato Septoria leaf spot    (151) |####################
Tomato leaf late blight      (111) |##############
Tomato leaf bacterial spot   (110) |##############
Tomato mold leaf             (91)  |############
Tomato Early blight leaf     (88)  |###########
Tomato leaf yellow virus     (76)  |##########
Tomato leaf (Healthy?)       (63)  |########
Tomato leaf mosaic virus     (54)  |#######
Tomato two spotted mites     (2)   |
```

### 2.3 PlantWild (Target)
*Wild images, variable lighting and zoom.*

*   **Total Images:** 1,966
*   **Class Count:** 8 (Mapped)
*   **Distribution:** Relatively balanced compared to PlantDoc.

```text
tomato early blight          (346) |####################
tomato late blight           (295) |#################
tomato bacterial leaf spot   (280) |################
tomato leaf mold             (239) |#############
tomato leaf (Healthy?)       (226) |#############
tomato septoria leaf spot    (220) |############
tomato mosaic virus          (189) |##########
tomato yellow leaf curl      (171) |#########
```

---

## 3. Data Processing & Experiment Setup

### 3.1 Splits and Adaptation Protocol
*   **Source (PlantVillage):**
    *   **Train:** 80% (Standard Supervised Training)
    *   **Validation:** 20% (Model Selection)
*   **Target (PlantDoc/Wild):**
    *   **Test/Adaptation:** 100% used for evaluation or unsupervised adaptation.
    *   *Note:* The `get_domain_dataloaders` function creates a `target_test` loader containing all target images.

### 3.2 Preprocessing Pipeline
All images undergo the following transformations for the **MobileNetV4** model:

1.  **Resize:** All inputs rescaled to `256x256` pixels.
2.  **Normalization:** Standard ImageNet statistics.
    *   Mean: `[0.485, 0.456, 0.406]`
    *   Std: `[0.229, 0.224, 0.225]`
3.  **Augmentation (Train only):**
    *   Random Horizontal Flip
    *   Color Jitter (Brightness, Contrast, Saturation: 0.2)

### 3.3 Class Alignment (Mapping)
A custom mapping file (`tomato_class_mapping.csv`) aligns diverse folder names to universal scientific labels.

| Universal Label | PlantVillage (Source) | PlantDoc (Target) | PlantWild (Target) |
| :--- | :--- | :--- | :--- |
| **bacterial spot** | `Tomato_Bacterial_spot` | `Tomato leaf bacterial spot` | `tomato bacterial leaf spot` |
| **early blight** | `Tomato_Early_blight` | `Tomato Early blight leaf` | `tomato early blight` |
| **late blight** | `Tomato_Late_blight` | `Tomato leaf late blight` | `tomato late blight` |
| **mold** | `Tomato_Leaf_Mold` | `Tomato mold leaf` | `tomato leaf mold` |
| **septoria spot** | `Tomato_Septoria_leaf_spot` | `Tomato Septoria leaf spot` | `tomato septoria leaf spot` |
| **mites** | `Tomato_Spider_mites...` | `Tomato two spotted...` | *Absent* |
| **target spot** | `Tomato_Target_Spot` | *Absent* | *Absent* |
| **mosaic virus** | `Tomato_mosaic_virus` | `Tomato leaf mosaic virus` | `tomato mosaic virus` |
| **yellow virus** | `Tomato_YellowLeaf_...` | `Tomato leaf yellow virus` | `tomato yellow leaf curl...` |
| **healthy** | `Tomato_healthy` | `Tomato leaf` (Ambiguous) | `tomato leaf` (Ambiguous) |

---

## 4. Known Issues & Limitations

1.  **Class Imbalance:** PlantVillage has significantly more samples than target domains.
2.  **Missing Classes:**
    *   **PlantWild** completely lacks `Spider Mites` and `Target Spot`.
    *   **PlantDoc** has only 2 images for `Spider Mites`, making it effectively absent for evaluation.
3.  **Label Ambiguity:** The folder named `tomato leaf` in Target datasets is mapped to `healthy`, but visual inspection is recommended to ensure it doesn't contain diseased leaves.
4.  **Resolution Gap:** Target images (esp. PlantDoc) vary wildly in resolution (thumbnails to 4K), potentially introducing artifacts during the 256x256 resize.

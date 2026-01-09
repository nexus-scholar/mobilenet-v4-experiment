# MobilenetV4 Tomato Domain-Shift Experiment

## 1. Goal & Scope
- Focus exclusively on the tomato crop (Solanum lycopersicum) to isolate **domain shift** (lab vs. field) from **species classification** noise.
- Treat PlantVillage as the labeled source domain and PlantDoc/PlantWild as unlabeled or sparsely labeled target domains, forming a **partial domain adaptation** setup.
- Resize every image to the MobileNetV4 input resolution you plan to deploy (e.g., `256×256`) so accuracy numbers are reproducible on edge hardware.

## 2. Scientific Justification
1. **Isolation of variables:** Removing non-tomato species ensures observed performance drops are tied to lighting/background/weather changes rather than cross-species confusion.
2. **Partial domain alignment:** Tomato classes differ across datasets (missing labels, merged categories); documenting these mismatches strengthens the methodology section in the thesis.
3. **Edge-optimized iteration speed:** Narrowing to tomatoes cuts epoch time drastically, enabling rapid ablation studies (active learning loops, lightweight distillation, etc.).

## 3. Methodology Checklist
1. **Explicit selection rule** – document the exact filters used to extract tomato folders from PlantVillage, PlantDoc, and PlantWild.
2. **Class mapping table** – publish the mapping between dataset-specific folder names and the standardized scientific class names, marking source-only/target-only rows.
3. **Dataset statistics** – report per-split counts (images per class per domain). Example wording:
   > "After filtering, the source domain contains **S** images across **k** tomato diseases, while the combined target domain contains **T** images across **m** overlapping diseases."
4. **Preprocessing statement** – record the resize/cropping pipeline and normalization constants used for MobileNetV4.

## 4. Dataset Alignment Script
Use the script below to auto-generate the tomato class mapping. Paths assume the repo root (`mobile-v4-experiment`).

```python
import os
import pandas as pd
import re

PATHS = {
    "PlantVillage": "mobile-v4-experiment/data/PlantVillage_processed",
    "PlantDoc": "mobile-v4-experiment/data/PlantDoc_processed",
    "PlantWild": "mobile-v4-experiment/data/PlantWild/images",
}

def normalize_name(folder_name: str) -> str:
    name = folder_name.lower()
    name = name.replace("tomato", "")
    name = name.replace("_", " ")
    name = name.replace("leaf", "")
    name = name.replace("two spotted", "")
    name = name.replace("spider mites", "mites")
    name = name.replace("curl", "")
    name = re.sub(" +", " ", name).strip()
    return name

def generate_mapping_table():
    data = {}
    for dataset, path in PATHS.items():
        if not os.path.exists(path):
            print(f"Warning: Path not found: {path}")
            continue
        folders = [
            f for f in os.listdir(path)
            if "tomato" in f.lower() and os.path.isdir(os.path.join(path, f))
        ]
        for folder in folders:
            key = normalize_name(folder)
            data.setdefault(key, {"Universal_Name": key})[dataset] = folder

    df = pd.DataFrame(data.values())
    cols = ["Universal_Name", "PlantVillage", "PlantDoc", "PlantWild"]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df[cols].sort_values("Universal_Name")
    output = "tomato_class_mapping.csv"
    df.to_csv(output, index=False)
    print(f"Saved mapping to {output}")
    print("Intersection preview:\n", df.dropna())
    return df

if __name__ == "__main__":
    generate_mapping_table()
```

## 5. Post-Script Actions
1. **Manual review** – open the CSV, merge near-duplicates (e.g., `yellow_virus` vs. `yellow_curl_virus`) and flag missing targets.
2. **Final selection table** – publish a clean Markdown/LaTeX table listing standardized disease names and their presence in each dataset. Example snippet:

| Disease (Std.) | PlantVillage | PlantDoc | PlantWild |
| --- | --- | --- | --- |
| Bacterial Spot | `tomato_bacterial_spot` | `tomato_bacterial_spot` | `tomato bacterial leaf spot` |
| Early Blight | `tomato_early_blight` | `tomato_early_blight` | `tomato early blight` |
| Mosaic Virus | `tomato_mosaic_virus` | `tomato_mosaic_virus` | `tomato mosaic virus` |
| Spider Mites | `tomato_spider_mites_two_spotted_spider_mite` | `tomato_two_spotted_spider_mites` | *Absent* |

3. **Document exclusions** – explicitly note any source-only classes dropped before training (e.g., `tomato_leaf_mold`).
4. **Record preprocessing** – log resize/crop/normalization values in the experiment tracker so MobileNetV4 runs remain reproducible.

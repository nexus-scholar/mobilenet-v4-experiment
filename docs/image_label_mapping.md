# Image-Label Mapping Utility

This document explains how `src/build_image_label_maps.py` consolidates tomato-only samples from the three source datasets (PlantVillage, PlantDoc, PlantWild) into CSV mapping tables.

## Overview
- **Goal:** Produce reproducible lists of image paths and their associated class labels for each dataset after removing non-tomato categories.
- **Output:** Three CSV files under `data/mappings/` plus a `summary.json` with per-dataset counts.

## Workflow
1. **PlantVillage parsing (`gather_plantvillage`)**
   - Scans all folders that start with `Tomato` inside `data/PlantVillage/`.
   - Emits one row per image with `dataset`, `class_name`, and `image_path` fields.

2. **PlantDoc parsing (`gather_plantdoc`)**
   - Walks both `train/` and `test/` splits in `data/PlantDoc-Dataset-master/`.
   - Keeps folders whose names start with `Tomato`, recording the split, class folder, and image path.

3. **PlantWild parsing (`gather_plantwild`)**
   - Reads `classes.txt` to map numeric IDs to class names.
   - Iterates over `trainval.txt` entries, keeping only rows where the resolved class name contains "tomato".
   - Writes dataset name, numeric ID, resolved class name, and absolute image path.

4. **CSV writing (`write_csv`)**
   - Determines all fieldnames present in the collected records.
   - Saves per-dataset CSVs in `data/mappings/` and prints the row counts.

5. **Summary generation**
   - Stores a JSON file listing the number of tomato images per dataset; useful for audits and experiment logs.

## Usage
```bash
python src/build_image_label_maps.py
```
The command populates/refreshes:
- `data/mappings/plantvillage_image_labels.csv`
- `data/mappings/plantdoc_image_labels.csv`
- `data/mappings/plantwild_image_labels.csv`
- `data/mappings/summary.json`

Each run re-scans the dataset folders, so ensure any preprocessing (e.g., removing non-tomato folders) is complete beforehand.

## Notes
- The script assumes datasets live under the repository's `data/` directory with the default folder names from the supervisor's instructions.
- For PlantWild, inspect the generated CSV to manually merge near-duplicate disease names (e.g., spider-mite variations) as part of the domain-alignment process described in `NEW_EXPERIMENT.md`.


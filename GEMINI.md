# MobileNetV4 Tomato Domain Shift Experiment

## Project Overview
This project is a research experiment focused on **domain adaptation** for tomato disease classification. The goal is to train a MobileNetV4 model on lab-controlled images (PlantVillage) and evaluate its performance on "in-the-wild" field images (PlantDoc, PlantWild).

The core hypothesis is that focusing exclusively on a single crop (tomato) isolates the domain shift variable (lighting, background, weather) from species classification noise.

## Directory Structure
*   **`src/`**: Source code for data processing, loading, model definition, and training.
    *   `dataset.py`: Generic PyTorch Dataset implementation reading from a simple CSV.
    *   `data_loader.py`: Specialized dataset/loader (`TomatoDomainDataset`) for the tomato domain shift experiment, handling source/target splits and class mappings.
    *   `train.py`: Main training script (currently configured for the generic dataset).
    *   `model.py`: Model definition (DistilledMobileNet).
    *   `generate_mapping.py` & `fix_and_test_loader.py`: Scripts to generate and validate class mappings between datasets.
*   **`data/`**: Storage for datasets and mapping files.
    *   `PlantVillage`, `PlantDoc-Dataset-master`, `PlantWild`: Raw image datasets.
    *   `mappings/`: generated CSVs mapping classes across datasets.
*   **`config.yaml`**: Configuration file for the `train.py` script.
*   **`docs/`**: Documentation files (`workflow.md`, `image_label_mapping.md`).
*   **`NEW_EXPERIMENT.md`**: Detailed rationale and checklist for the tomato domain shift experiment.

## Workflows

### 1. Environment Setup
The project uses Python. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation (Tomato Domain Shift)
To align the datasets and prepare for the domain shift experiment:

1.  **Build Inventories:** Scan datasets.
    ```bash
    python src/build_image_label_maps.py
    ```
2.  **Generate Mappings:** Create `tomato_class_mapping.csv` to align class names across PlantVillage, PlantDoc, and PlantWild.
    ```bash
    python src/generate_mapping.py
    ```
3.  **Validate Loaders:** specific fix script to merge naming issues and verify the loaders work.
    ```bash
    python src/fix_and_test_loader.py
    ```

### 3. Training
The training script `src/train.py` uses `config.yaml` for parameters.

```bash
python src/train.py --config config.yaml
```

**Note:** `src/train.py` currently uses `src.dataset.PlantDiseaseDataset` which expects a standard CSV input (defined in `config.yaml` as `data/csv_path`). The new domain shift workflow (`src/data_loader.py`) is designed to work with the generated class mappings directly. Ensure your `config.yaml` points to the correct CSV or update `train.py` to utilize `get_domain_dataloaders` if running the specific domain shift experiment.

## Key Configurations (`config.yaml`)
*   **`experiment_name`**: Identifier for the run.
*   **`data`**: Paths and batch size.
*   **`model`**: Model parameters (e.g., `num_classes`).
*   **`training`**: Hyperparameters (epochs, lr, etc.).

## Development Conventions
*   **Framework:** PyTorch (`torch`, `torchvision`).
*   **Model:** MobileNetV4 (via `timm` or custom implementation).
*   **Type Hinting:** Python type hints are used throughout the new loader code (`src/data_loader.py`).
*   **Formatting:** Code appears to follow standard PEP 8 conventions.

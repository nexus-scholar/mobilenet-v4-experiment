# MobilenetV4 Tomato Domain Shift

Clean baseline focused on tomato-only domain adaptation (PlantVillage → PlantDoc/Wild) using MobileNetV4.

## Repository Highlights
- `src/generate_mapping.py` scans raw tomato folders and emits `data/mappings/tomato_class_mapping.csv`.
- `src/fix_and_test_loader.py` merges spider-mite/healthy naming issues, saves `tomato_class_mapping_fixed.csv`, and validates loaders.
- `src/data_loader.py` provides `get_domain_dataloaders` for aligned source/target PyTorch loaders.
- `src/train.py` runs the simplified MobileNetV4 baseline defined by `config.yaml`.

## Workflow
```bash
# 1. Build dataset inventories
python src/build_image_label_maps.py

# 2. Generate/clean alignment table and test loaders
python src/generate_mapping.py
python src/fix_and_test_loader.py

# 3. Launch training once mappings are ready
python src/train.py --config config.yaml
```

Additional notes live in `docs/image_label_mapping.md` and `docs/workflow.md`.

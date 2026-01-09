# Experiment Workflow

1. **Dataset Alignment**
   - Run `python src/generate_mapping.py` to scan tomato folders and produce `data/mappings/tomato_class_mapping.csv`.
   - Run `python src/fix_and_test_loader.py` to fix edge cases (spider mites, tomato leaf) and validate loaders. This writes `data/mappings/tomato_class_mapping_fixed.csv`.

2. **Image-Label Maps**
   - `python src/build_image_label_maps.py` creates per-dataset CSV inventories in `data/mappings/` for auditing.

3. **Training Prep**
   - Use `src/data_loader.py/get_domain_dataloaders` with the fixed mapping to obtain aligned source/target dataloaders.
   - Configure experiments via `config.yaml` before launching `python src/train.py --config config.yaml`.

4. **Next Steps**
   - With mappings verified and directories cleaned, proceed to model training and logging on fresh checkpoints/logs directories.


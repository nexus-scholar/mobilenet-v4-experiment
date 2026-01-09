# MobilenetV4 Tomato Domain Shift

Lightweight baseline to study lab-to-field domain shift on tomato diseases using MobileNetV4.

## Quickstart
```bash
python src/preprocess_data.py
python src/train.py --config config.yaml
```

## Dataset Mapping Utility
- See `docs/image_label_mapping.md` for details on generating the tomato-only image/label CSVs via `python src/build_image_label_maps.py`.
- Use `src/data_loader.py` to align classes across domains. Example:
```bash
python - <<'PY'
from src.data_loader import get_domain_dataloaders
loaders, class_map = get_domain_dataloaders(
    mapping_csv="data/mappings/tomato_class_mapping.csv",
    root_dir="data"
)
print(class_map)
for name, loader in loaders.items():
    print(name, len(loader))
PY
```

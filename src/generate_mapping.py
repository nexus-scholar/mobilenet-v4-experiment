import os
import pandas as pd
import re
from pathlib import Path

DATASETS = {
    "PlantVillage": {
        "base": Path("data/PlantVillage"),
        "subfolders": None,
    },
    "PlantDoc": {
        "base": Path("data/PlantDoc-Dataset-master"),
        "subfolders": ["train", "test"],
    },
    "PlantWild": {
        "base": Path("data/PlantWild"),
        "trainval": Path("data/PlantWild/trainval.txt"),
        "images": Path("data/PlantWild/images"),
        "classes": Path("data/PlantWild/classes.txt"),
    },
}

OUTPUT_DIR = Path("data/mappings")
OUTPUT_FILE = OUTPUT_DIR / "tomato_class_mapping.csv"

def normalize_name(folder_name):
    name = folder_name.lower()
    name = name.replace('tomato', '')
    name = name.replace('_', ' ')
    name = name.replace('leaf', '')
    name = name.replace('two spotted', '')
    name = name.replace('spider mites', 'mites')
    name = name.replace('curl', '')
    name = re.sub(' +', ' ', name).strip()
    return name

def _iter_tomato_folders(base: Path, subfolders):
    if subfolders:
        for sub in subfolders:
            sub_path = base / sub
            if not sub_path.exists():
                continue
            for item in sub_path.iterdir():
                if item.is_dir() and 'tomato' in item.name.lower():
                    yield item
    else:
        if not base.exists():
            return
        for item in base.iterdir():
            if item.is_dir() and 'tomato' in item.name.lower():
                yield item

def _iter_plantwild_entries(cfg):
    classes = {}
    if cfg["classes"].exists():
        with open(cfg["classes"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    classes[int(parts[0])] = parts[1]
    if not cfg["trainval"].exists():
        return
    seen = set()
    with open(cfg["trainval"], "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            image_rel, class_id_str, *_ = line.split('=')
            class_id = int(class_id_str)
            class_name = classes.get(class_id, f"class_{class_id}")
            if 'tomato' not in class_name.lower():
                continue
            folder = Path(image_rel).parent.name
            if folder not in seen:
                seen.add(folder)
                yield folder

def generate_mapping_table():
    data = {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Scanning directories...")
    for dataset, cfg in DATASETS.items():
        if dataset == "PlantWild":
            for folder in _iter_plantwild_entries(cfg):
                universal_key = normalize_name(folder)
                data.setdefault(universal_key, {"Universal_Name": universal_key})[dataset] = folder
            continue

        base = cfg["base"]
        subfolders = cfg.get("subfolders")
        if not base.exists():
            print(f"Warning: Path not found: {base}")
            continue
        for folder in _iter_tomato_folders(base, subfolders):
            universal_key = normalize_name(folder.name)
            data.setdefault(universal_key, {"Universal_Name": universal_key})[dataset] = _relative_folder(folder, base, subfolders)
    df = pd.DataFrame(data.values())
    cols = ["Universal_Name", "PlantVillage", "PlantDoc", "PlantWild"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values("Universal_Name")
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Mapping generated! Saved to {OUTPUT_FILE}")
    print(df)

def _relative_folder(folder: Path, base: Path, subfolders):
    if subfolders:
        return str(folder.relative_to(base))
    return folder.name

if __name__ == "__main__":
    generate_mapping_table()

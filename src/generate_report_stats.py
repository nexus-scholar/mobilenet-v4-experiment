import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
import numpy as np

DATA_DIR = Path('data')
MAPPINGS_DIR = DATA_DIR / 'mappings'

DATASETS = {
    'PlantVillage': DATA_DIR / 'PlantVillage',
    'PlantDoc': DATA_DIR / 'PlantDoc-Dataset-master',
    'PlantWild': DATA_DIR / 'PlantWild'
}

def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024 * 1024) # GB

def get_image_stats(path, sample_size=100):
    resolutions = []
    formats = set()
    image_count = 0
    
    all_images = []
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                all_images.append(os.path.join(dirpath, f))
                formats.add(Path(f).suffix.lower())
    
    image_count = len(all_images)
    
    if image_count > 0:
        # Sample images for resolution
        indices = np.random.choice(len(all_images), min(sample_size, len(all_images)), replace=False)
        for i in indices:
            try:
                with Image.open(all_images[i]) as img:
                    resolutions.append(img.size)
            except Exception:
                pass

    return {
        'count': image_count,
        'resolutions': resolutions,
        'formats': list(formats)
    }

def analyze_mappings():
    report = {}
    
    # Try loading summary.json
    try:
        with open(MAPPINGS_DIR / 'summary.json', 'r') as f:
            summary = json.load(f)
            report['summary_json'] = summary
    except Exception as e:
        report['summary_json'] = str(e)

    # Analyze class distributions from CSVs
    dataset_csvs = {
        'PlantVillage': 'plantvillage_image_labels.csv',
        'PlantDoc': 'plantdoc_image_labels.csv',
        'PlantWild': 'plantwild_image_labels.csv'
    }
    
    class_dists = {}
    
    for ds_name, csv_name in dataset_csvs.items():
        try:
            df = pd.read_csv(MAPPINGS_DIR / csv_name)
            # Assuming 'class_name' or similar column exists. 
            # Based on build_image_label_maps.py it emits: dataset, class_name, image_path
            if 'class_name' in df.columns:
                class_dists[ds_name] = df['class_name'].value_counts().to_dict()
            elif 'resolved_name' in df.columns: # PlantWild might have this
                 class_dists[ds_name] = df['resolved_name'].value_counts().to_dict()
            else:
                 # Fallback to checking columns
                 class_dists[ds_name] = f"Columns: {list(df.columns)}"
        except Exception as e:
            class_dists[ds_name] = f"Error: {str(e)}"
            
    report['class_distributions'] = class_dists
    
    # Load Main Mapping
    try:
        df_map = pd.read_csv(MAPPINGS_DIR / 'tomato_class_mapping.csv')
        report['universal_classes'] = df_map['Universal_Name'].tolist()
        report['mapping_preview'] = df_map.head().to_dict()
    except Exception as e:
        report['universal_classes'] = str(e)
        
    return report

def main():
    print("--- START REPORT GENERATION ---")
    
    final_stats = {}
    
    for name, path in DATASETS.items():
        if not path.exists():
            final_stats[name] = {"error": "Path not found"}
            continue
            
        print(f"Analyzing {name}...")
        size_gb = get_dir_size(path)
        img_stats = get_image_stats(path)
        
        final_stats[name] = {
            "size_gb": f"{size_gb:.4f}",
            "image_count": img_stats['count'],
            "formats": img_stats['formats'],
            "resolutions_sample": img_stats['resolutions'][:5], # just show a few
            "resolution_mode": max(set(img_stats['resolutions']), key=img_stats['resolutions'].count) if img_stats['resolutions'] else "N/A"
        }

    mapping_info = analyze_mappings()
    
    print("\n--- PHYSICAL STATS ---")
    print(json.dumps(final_stats, indent=2))
    
    print("\n--- MAPPING INFO ---")
    print(json.dumps(mapping_info, indent=2))

if __name__ == "__main__":
    main()

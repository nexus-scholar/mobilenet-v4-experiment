import os
import pandas as pd
import json
import re

# Paths
PLANTWILD_ROOT = "data/PlantWild"
PLANTSEG_ROOT = "data/PlantSeg"
OUTPUT_CSV = "data/unified_dataset.csv"

def load_plantwild_classes():
    classes_file = os.path.join(PLANTWILD_ROOT, "classes.txt")
    class_map = {}
    name_to_id = {}
    with open(classes_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                cid = int(parts[0])
                cname = parts[1].lower().strip()
                class_map[cid] = cname
                name_to_id[cname] = cid

    # Restore missing classes (IDs 0 and 1) likely dropped from classes.txt
    if 0 not in class_map:
        class_map[0] = "apple black rot"
        name_to_id["apple black rot"] = 0
    if 1 not in class_map:
        class_map[1] = "apple leaf"
        name_to_id["apple leaf"] = 1

    return class_map, name_to_id

def process_plantwild(name_to_id):
    print("Processing PlantWild...")
    trainval_file = os.path.join(PLANTWILD_ROOT, "trainval.txt")
    data = []

    with open(trainval_file, 'r') as f:
        for line in f:
            # format: apple black rot/google_0082.jpg=0=1
            parts = line.strip().split('=')
            if len(parts) >= 2:
                rel_path = parts[0]
                label_id = int(parts[1])

                # Check mapping consistency
                # We trust the ID from the file, but we get the name for the CSV
                class_name = None
                for name, lid in name_to_id.items():
                    if lid == label_id:
                        class_name = name
                        break

                full_path = os.path.join(PLANTWILD_ROOT, "images", rel_path)
                # Verify existence? (Skipping for speed, assume dataset integrity)

                data.append({
                    "image_path": full_path,
                    "mask_path": "",
                    "label": label_id,
                    "class_name": class_name,
                    "dataset_source": "PlantWild"
                })
    print(f"  Found {len(data)} samples.")
    return data

def clean_and_match_class_name(dirty_name, name_to_id):
    """
    Cleans noisy class names from PlantSeg and attempts to match them to PlantWild classes.

    Strategies (in order):
    1. Exact match on original name
    2. Strip Suffixes: Remove " google", " bing", " baidu", " - copy" (repeatedly)
    3. Handle Parentheses: Remove content in parentheses
    4. Prefix Matching: Check if dirty_name starts with any known class name
    5. Substring Matching: Check if any known class name is contained in the dirty_name

    Returns:
        tuple: (label_id, matched_class_name)
    """
    # Noise suffixes to strip (order matters - strip " - copy" first as it may appear with others)
    NOISE_SUFFIXES = [" - copy", " google", " bing", " baidu"]

    # 1. Exact match on original
    if dirty_name in name_to_id:
        return name_to_id[dirty_name], dirty_name

    # 2. Strip all noise suffixes (repeatedly until no more changes)
    cleaned = dirty_name
    changed = True
    while changed:
        changed = False
        for noise in NOISE_SUFFIXES:
            if cleaned.endswith(noise):
                cleaned = cleaned[:-len(noise)].strip()
                changed = True
            # Also handle noise appearing in the middle (e.g., "name google - copy")
            elif noise in cleaned:
                cleaned = cleaned.replace(noise, " ").strip()
                # Normalize multiple spaces
                cleaned = re.sub(r'\s+', ' ', cleaned)
                changed = True

    if cleaned in name_to_id:
        return name_to_id[cleaned], cleaned

    # 3. Handle Parentheses
    # e.g. "wheat bacterial leaf streak (black chaff)" -> "wheat bacterial leaf streak"
    no_parens = re.sub(r'\s*\(.*?\)', '', cleaned).strip()
    no_parens = re.sub(r'\s+', ' ', no_parens)  # Normalize spaces
    if no_parens in name_to_id:
        return name_to_id[no_parens], no_parens

    # 4. Prefix Matching
    # Sort known names by length descending to match longest possible prefix first
    # e.g. "banana anthracnose google" should match "banana anthracnose" not "banana"
    known_names = sorted(name_to_id.keys(), key=len, reverse=True)

    # Check against all variations: original, cleaned, and no_parens
    candidates = [dirty_name, cleaned, no_parens]

    for known in known_names:
        for candidate in candidates:
            if candidate.startswith(known + " ") or candidate == known:
                return name_to_id[known], known
            # Also check if known class is a prefix (without requiring space after)
            if candidate.startswith(known):
                return name_to_id[known], known

    # 5. Substring Matching (last resort)
    # Check if any known class name appears anywhere in the dirty name
    for known in known_names:
        if known in dirty_name or known in cleaned or known in no_parens:
            return name_to_id[known], known

    # No match found after all attempts
    print(f"  Warning: Unknown class in PlantSeg: '{dirty_name}' (cleaned: '{cleaned}')")
    return -1, cleaned

def process_plantseg(name_to_id):
    print("Processing PlantSeg...")
    metadata_file = os.path.join(PLANTSEG_ROOT, "Metadata.csv")
    if not os.path.exists(metadata_file):
        print("  PlantSeg metadata not found.")
        return []

    try:
        df = pd.read_csv(metadata_file)
    except:
        print("  Could not read PlantSeg metadata (maybe too large or corrupt).")
        return []

    data = []

    # Iterate rows
    for idx, row in df.iterrows():
        # Columns: Name, Plant, Disease, Label file, Training/Test
        img_name = row['Name']
        disease_name = str(row['Disease']).lower().strip()
        mask_name = row['Label file']
        split = row['Training/Test']

        # Determine path based on split (assuming folder structure matches main.py output or common convention)
        subfolder = "train" if split == "Training" else "test"

        image_path = os.path.join(PLANTSEG_ROOT, "images", subfolder, img_name)
        mask_path = os.path.join(PLANTSEG_ROOT, "annotations", subfolder, mask_name)

        # Map label using new helper function
        label_id, final_name = clean_and_match_class_name(disease_name, name_to_id)

        data.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "label": label_id,
            "class_name": final_name,
            "dataset_source": "PlantSeg"
        })

    print(f"  Found {len(data)} samples.")
    return data

def main():
    # 1. Load Classes
    id_to_name, name_to_id = load_plantwild_classes()
    print(f"Loaded {len(id_to_name)} classes from PlantWild.")

    # 2. Process Datasets
    wild_data = process_plantwild(name_to_id)
    seg_data = process_plantseg(name_to_id)

    # 3. Combine
    all_data = wild_data + seg_data
    df = pd.DataFrame(all_data)

    # 4. Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"------------------------------------------------")
    print(f"Saved merged dataset to {OUTPUT_CSV}")
    print(f"Total samples: {len(df)}")
    print(f"Breakdown:\n{df['dataset_source'].value_counts()}")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()


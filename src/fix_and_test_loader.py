import pandas as pd
from pathlib import Path

from src.data_loader import get_domain_dataloaders

MAPPING_FILE = Path("data/mappings/tomato_class_mapping.csv")
FIXED_FILE = Path("data/mappings/tomato_class_mapping_fixed.csv")
DATA_ROOT = "data"


def fix_mapping_issues():
    print(f"Loading {MAPPING_FILE}...")
    df = pd.read_csv(MAPPING_FILE)

    # Fix 1: merge spider mite variants
    row_complex = df[df['Universal_Name'].str.contains('mites spider mite', na=False, case=False)].index
    row_simple = df[df['Universal_Name'].str.fullmatch('mites', case=False, na=False)].index

    if len(row_complex) > 0 and len(row_simple) > 0:
        print("  [Fix] Merging spider mite rows")
        target_idx = row_simple[0]
        source_idx = row_complex[0]

        for col in ['PlantVillage', 'PlantDoc', 'PlantWild']:
            if pd.isna(df.at[target_idx, col]):
                df.at[target_idx, col] = df.at[source_idx, col]
        df.at[target_idx, 'Universal_Name'] = 'spider mites'
        df.drop(source_idx, inplace=True)

    # Fix 2: map PlantWild "tomato leaf" into healthy
    wild_leaf_rows = df[df['PlantWild'].str.contains('tomato leaf', na=False, case=False)]
    healthy_rows = df[df['Universal_Name'].str.contains('healthy', na=False, case=False)]

    if not wild_leaf_rows.empty:
        wild_idx = wild_leaf_rows.index[0]
        if healthy_rows.empty:
            print("  [Fix] Renaming tomato leaf row to healthy")
            df.at[wild_idx, 'Universal_Name'] = 'healthy'
        else:
            print("  [Fix] Merging tomato leaf row into healthy")
            healthy_idx = healthy_rows.index[0]
            df.at[healthy_idx, 'PlantWild'] = df.at[wild_idx, 'PlantWild']
            if wild_idx != healthy_idx:
                df.drop(wild_idx, inplace=True)

    df.sort_values('Universal_Name', inplace=True)
    df.to_csv(FIXED_FILE, index=False)
    print(f"✅ Fixed mapping saved to {FIXED_FILE}")
    print(df[['Universal_Name', 'PlantVillage', 'PlantDoc', 'PlantWild']].to_string())
    return FIXED_FILE


def test_loader(csv_path: Path):
    print("\n🚀 Starting loader test...")
    loaders, class_map = get_domain_dataloaders(
        mapping_csv=str(csv_path),
        root_dir=DATA_ROOT,
        batch_size=4,
        input_size=224,
    )

    print("\n✅ Loader initialization succeeded")
    print(f"   Classes: {len(class_map)} -> {class_map}")
    print(f"   Source train batches: {len(loaders['source_train'])}")
    print(f"   Source val batches:   {len(loaders['source_val'])}")
    print(f"   Target test batches:  {len(loaders['target_test'])}")

    images, labels = next(iter(loaders['source_train']))
    print("\nShape check:")
    print(f"   Images: {tuple(images.shape)}")
    print(f"   Labels: {labels}")


def main():
    fixed_csv = fix_mapping_issues()
    test_loader(fixed_csv)


if __name__ == "__main__":
    main()


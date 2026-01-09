import csv
import json
from pathlib import Path

from typing import Iterable


def gather_plantvillage(root: Path) -> Iterable[dict]:
    for class_dir in sorted(root.glob('Tomato*')):
        for img_path in class_dir.glob('*'):
            if img_path.is_file():
                yield {
                    'dataset': 'PlantVillage',
                    'class_name': class_dir.name,
                    'image_path': str(img_path)
                }


def gather_plantdoc(root: Path) -> Iterable[dict]:
    for split in ('train', 'test'):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.glob('Tomato*')):
            for img_path in class_dir.glob('*'):
                if img_path.is_file():
                    yield {
                        'dataset': 'PlantDoc',
                        'split': split,
                        'class_name': class_dir.name,
                        'image_path': str(img_path)
                    }


def gather_plantwild(root: Path) -> Iterable[dict]:
    classes_file = root / 'classes.txt'
    class_map = {}
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    class_map[int(parts[0])] = parts[1]

    trainval = root / 'trainval.txt'
    if not trainval.exists():
        return
    with open(trainval, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('=')
            if len(parts) < 2:
                continue
            rel_path = parts[0]
            class_id = int(parts[1])
            class_name = class_map.get(class_id, f'id_{class_id}')
            if 'tomato' not in class_name.lower():
                continue
            yield {
                'dataset': 'PlantWild',
                'class_id': class_id,
                'class_name': class_name,
                'image_path': str(root / 'images' / rel_path)
            }


def write_csv(records, output_file: Path):
    if not records:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record.keys()})
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f'Saved {len(records)} rows to {output_file}')


def main():
    root = Path('data')
    pv_records = list(gather_plantvillage(root / 'PlantVillage'))
    pd_records = list(gather_plantdoc(root / 'PlantDoc-Dataset-master'))
    pw_records = list(gather_plantwild(root / 'PlantWild'))

    write_csv(pv_records, root / 'mappings' / 'plantvillage_image_labels.csv')
    write_csv(pd_records, root / 'mappings' / 'plantdoc_image_labels.csv')
    write_csv(pw_records, root / 'mappings' / 'plantwild_image_labels.csv')

    summary = {
        'PlantVillage': len(pv_records),
        'PlantDoc': len(pd_records),
        'PlantWild': len(pw_records)
    }
    print('Summary:', summary)
    with open(root / 'mappings' / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()


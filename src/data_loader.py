"""Unified tomato domain dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_FOLDER_MAP = {
    "PlantVillage": "PlantVillage",
    "PlantDoc": "PlantDoc-Dataset-master",
    "PlantWild": "PlantWild",
}

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Sample:
    """Represents a single image/label pair."""

    path: str
    label: int


class TomatoDomainDataset(Dataset):
    """Dataset that aligns tomato classes across PlantVillage, PlantDoc, and PlantWild."""

    def __init__(
        self,
        mapping_csv: Optional[str],
        root_dir: str,
        domain_type: Optional[str] = None,
        transform=None,
        return_path: bool = False,
        folder_map: Optional[Dict[str, str]] = None,
        allowed_extensions: Sequence[str] = DEFAULT_EXTENSIONS,
        samples: Optional[List[Sample]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        idx_to_class: Optional[Dict[int, str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_path = return_path
        self.folder_map = folder_map or DEFAULT_FOLDER_MAP
        self.allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)

        if samples is not None:
            if class_to_idx is None or idx_to_class is None:
                raise ValueError("class_to_idx and idx_to_class must be provided when supplying custom samples")
            self.samples = samples
            self.class_to_idx = class_to_idx
            self.idx_to_class = idx_to_class
            return

        if mapping_csv is None:
            raise ValueError("mapping_csv is required when samples are not provided")
        if domain_type not in {"Source", "Target", "All"}:
            raise ValueError("domain_type must be 'Source', 'Target', or 'All'")

        df = pd.read_csv(mapping_csv)
        universal_names = sorted(df["Universal_Name"].dropna().unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(universal_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}

        self.samples: List[Sample] = []
        for _, row in df.iterrows():
            label_idx = self.class_to_idx[row["Universal_Name"]]
            if domain_type in {"Source", "All"}:
                self._add_images(row.get("PlantVillage"), label_idx, dataset_key="PlantVillage")
            if domain_type in {"Target", "All"}:
                self._add_images(row.get("PlantDoc"), label_idx, dataset_key="PlantDoc")
                self._add_images(row.get("PlantWild"), label_idx, dataset_key="PlantWild", subfolder="images")

        print(
            f"Loaded {len(self.samples)} samples for domain '{domain_type}' across {len(self.class_to_idx)} classes."
        )

    def _add_images(
        self,
        folder_name: Optional[str],
        label_idx: int,
        dataset_key: str,
        subfolder: Optional[str] = None,
    ) -> None:
        if not isinstance(folder_name, str) or not folder_name.strip():
            return
        base_dir = self.root_dir / self.folder_map[dataset_key]
        if subfolder:
            base_dir = base_dir / subfolder
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: folder not found -> {folder_path}")
            return
        for path in folder_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.allowed_extensions:
                self.samples.append(Sample(str(path), label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, sample.label, sample.path
        return image, sample.label


class _SampleDataset(Dataset):
    """Lightweight wrapper used for train/val splits with distinct transforms."""

    def __init__(self, samples: List[Sample], transform=None, return_path: bool = False):
        self.samples = samples
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, sample.label, sample.path
        return image, sample.label


def _split_samples(
    samples: Sequence[Sample],
    labels: Sequence[int],
    val_split: float,
    random_state: int,
) -> Tuple[List[Sample], List[Sample]]:
    indices = list(range(len(samples)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=random_state,
    )
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    return train_samples, val_samples


def get_domain_dataloaders(
    mapping_csv: str,
    root_dir: str,
    batch_size: int = 32,
    input_size: int = 256,
    val_split: float = 0.2,
    num_workers: int = 4,
    random_state: int = 42,
    return_paths: bool = False,
):
    """Create PyTorch dataloaders for tomato domain adaptation experiments."""

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    base_source = TomatoDomainDataset(
        mapping_csv=mapping_csv,
        root_dir=root_dir,
        domain_type="Source",
        transform=None,
        return_path=return_paths,
    )
    labels = [sample.label for sample in base_source.samples]
    train_samples, val_samples = _split_samples(
        base_source.samples,
        labels,
        val_split,
        random_state,
    )

    source_train = _SampleDataset(train_samples, transform=data_transforms["train"], return_path=return_paths)
    source_val = _SampleDataset(val_samples, transform=data_transforms["val"], return_path=return_paths)

    target_dataset = TomatoDomainDataset(
        mapping_csv=mapping_csv,
        root_dir=root_dir,
        domain_type="Target",
        transform=data_transforms["val"],
        return_path=return_paths,
    )

    dataloaders = {
        "source_train": DataLoader(
            source_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "source_val": DataLoader(
            source_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "target_test": DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    return dataloaders, base_source.class_to_idx


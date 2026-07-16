"""
Utility functions package
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional


def detect_dataset_structure(dataset_path: Union[str, Path]) -> Dict:
    """Detect dataset structure."""
    dataset_path = Path(dataset_path)

    structure = {
        "train": None,
        "val": None,
        "test": None,
        "images": [],
        "labels": [],
        "yaml_file": None
    }

    if not dataset_path.exists():
        return structure

    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        if split_dir.exists():
            structure[split] = str(split_dir)

    for yaml_file in dataset_path.glob("*.yaml"):
        structure["yaml_file"] = str(yaml_file)
        break

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in image_extensions:
        structure["images"].extend(list(dataset_path.rglob(f"*{ext}")))

    structure["labels"].extend(list(dataset_path.rglob("*.txt")))

    return structure


def detect_class_names(dataset_path: Union[str, Path]) -> List[str]:
    """Detect class names from dataset."""
    dataset_path = Path(dataset_path)

    for yaml_file in dataset_path.glob("*.yaml"):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], dict):
                        return list(data['names'].values())
                    elif isinstance(data['names'], list):
                        return data['names']
        except Exception as e:
            print(f"Warning: Failed to read {yaml_file}: {e}")

    return ["microsphere"]


def split_dataset(dataset_path: Union[str, Path],
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1) -> Dict:
    """Split dataset into train/val/test sets."""
    import random

    dataset_path = Path(dataset_path)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    all_images = []

    for ext in image_extensions:
        all_images.extend(list(dataset_path.rglob(f"*{ext}")))

    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]

    return {
        "train": [str(img) for img in train_images],
        "val": [str(img) for img in val_images],
        "test": [str(img) for img in test_images],
        "total": total,
        "train_count": len(train_images),
        "val_count": len(val_images),
        "test_count": len(test_images)
    }

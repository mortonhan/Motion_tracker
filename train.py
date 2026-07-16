#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced microsphere detection model training script.
Integrates color attention mechanism and scale-aware FPN module.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from utils import detect_dataset_structure, detect_class_names

from models.enhanced_yolo import EnhancedMicrosphereYOLO
from utils.color_attention import ColorAttention, HSVAttention
from utils.scale_fpn import ScaleAwareFPN, SizeLoss
from utils.data_augmentation import MicrosphereAugmentation
from config import TRAIN, MODEL_ENHANCEMENT, DATA_AUGMENTATION
from utils.general import ensure_dir

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced microsphere detection model training script')
    parser.add_argument('--dataset', type=str, default=str(CONFIG["DATASET_PATH"]),
                        help='Dataset directory path')
    parser.add_argument('--weights', type=str, default=None,
                        help='Initial weights file path. Uses pretrained YOLOv8 if not specified.')
    parser.add_argument('--enhanced-weights', type=str, default=None,
                        help='Enhanced model weights file path, for loading or saving')
    parser.add_argument('--epochs', type=int, default=TRAIN["epochs"],
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=TRAIN["batch_size"],
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=TRAIN["img_size"],
                        help='Input image size')
    parser.add_argument('--device', default=TRAIN["device"],
                        help='Training device, e.g. 0 or cpu')
    parser.add_argument('--workers', type=int, default=TRAIN["workers"],
                        help='Number of data loading workers')
    parser.add_argument('--project', type=str, default=TRAIN["project"],
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default=TRAIN["name"],
                        help='Experiment name')
    parser.add_argument('--optimizer', type=str, default=TRAIN["optimizer"],
                        help='Optimizer type: SGD, Adam, AdamW')
    parser.add_argument('--lr', type=float, default=TRAIN["lr0"],
                        help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=TRAIN["lrf"],
                        help='Final learning rate = lr0 * lrf')
    parser.add_argument('--momentum', type=float, default=TRAIN["momentum"],
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=TRAIN["weight_decay"],
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=TRAIN["warmup_epochs"],
                        help='Warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=TRAIN["warmup_momentum"],
                        help='Warmup momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=TRAIN["warmup_bias_lr"],
                        help='Warmup bias learning rate')
    parser.add_argument('--save-period', type=int, default=TRAIN["save_period"],
                        help='Checkpoint save interval (-1 means save last only)')
    parser.add_argument('--seed', type=int, default=TRAIN["seed"],
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')

    parser.add_argument('--multi-scale', action='store_true', default=TRAIN.get("multi_scale", False),
                        help='Enable multi-scale training')
    parser.add_argument('--rect', action='store_true', default=TRAIN.get("rect", False),
                        help='Enable rectangular training (batch by aspect ratio)')
    parser.add_argument('--min-img-size', type=int, default=TRAIN.get("min_img_size", 320),
                        help='Minimum image size for multi-scale training')
    parser.add_argument('--max-img-size', type=int, default=TRAIN.get("max_img_size", 1024),
                        help='Maximum image size for multi-scale training')

    parser.add_argument('--use-enhanced-model', action='store_true',
                        help='Use enhanced model for training')
    parser.add_argument('--no-enhanced-model', action='store_false', dest='use_enhanced_model',
                        help='Use base YOLOv8 model instead')
    parser.add_argument('--use-color-attention', action='store_true',
                        help='Use color attention mechanism')
    parser.add_argument('--no-color-attention', action='store_false', dest='use_color_attention',
                        help='Disable color attention mechanism')
    parser.add_argument('--use-scale-fpn', action='store_true',
                        help='Use scale-aware FPN')
    parser.add_argument('--no-scale-fpn', action='store_false', dest='use_scale_fpn',
                        help='Disable scale-aware FPN')

    parser.set_defaults(
        use_enhanced_model=CONFIG["MODEL_ENHANCEMENT"]["use_enhanced_model"],
        use_color_attention=CONFIG["MODEL_ENHANCEMENT"]["use_color_attention"],
        use_scale_fpn=CONFIG["MODEL_ENHANCEMENT"]["use_scale_fpn"]
    )

    return parser.parse_args()

def create_temp_data_yaml(dataset_path):
    """Create a temporary data config YAML file from the dataset path."""
    dataset_path = Path(dataset_path).resolve()

    dataset_structure = detect_dataset_structure(dataset_path, auto_split=True)
    if dataset_structure is None:
        raise ValueError(f"Cannot detect valid dataset structure: {dataset_path}")

    orig_data_yaml = dataset_path / "data.yaml"
    class_names = None

    if orig_data_yaml.exists():
        try:
            with open(orig_data_yaml, 'r') as f:
                orig_data = yaml.safe_load(f)
                if 'names' in orig_data:
                    class_names = orig_data['names']
                    print(f"Loaded class names from original data.yaml")

                    if isinstance(class_names, dict):
                        sorted_names = []
                        for i in range(len(class_names)):
                            if str(i) in class_names:
                                sorted_names.append(class_names[str(i)])
                            elif i in class_names:
                                sorted_names.append(class_names[i])

                        if len(sorted_names) == len(class_names):
                            class_names = sorted_names
                            print(f"Converted class names from dict to list: {class_names}")
        except Exception as e:
            print(f"Error reading original data.yaml: {e}")

    if class_names is None:
        class_names = detect_class_names(dataset_path)

    temp_yaml_path = dataset_path / "temp_data.yaml"

    yaml_content = {
        "path": str(dataset_path),
        "train": str(Path(dataset_structure["train"]).resolve()),
        "val": str(Path(dataset_structure["val"]).resolve()),
        "test": str(Path(dataset_structure["test"]).resolve()) if dataset_structure["test"] else "",
        "nc": len(class_names),
        "names": class_names
    }

    with open(temp_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"Created temp data config: {temp_yaml_path}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Train path: {yaml_content['train']}")
    print(f"Val path: {yaml_content['val']}")

    return temp_yaml_path

def load_dataset(data_yaml, img_size, batch_size, workers):
    """Load dataset and return train/val dataloaders."""
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    nc = len(data_cfg['names'])
    names = data_cfg['names']

    print(f"Loading dataset: {data_yaml}")
    print(f"Number of classes: {nc}")
    print(f"Class names: {names}")

    try:
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionValidator, DetectionTrainer
        from ultralytics.data.build import build_dataloader
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")

    print("Creating training dataloader...")
    train_loader = build_dataloader(
        path=data_cfg['train'],
        imgsz=img_size,
        batch_size=batch_size,
        stride=32,
        hyp=None,
        augment=True,
        cache=False,
        pad=0.0,
        rect=False,
        workers=workers,
        prefix=f'train: ',
        shuffle=True,
        seed=0,
    )

    print("Creating validation dataloader...")
    val_loader = build_dataloader(
        path=data_cfg['val'],
        imgsz=img_size,
        batch_size=batch_size,
        stride=32,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.5,
        rect=True,
        workers=workers,
        prefix=f'val: ',
        shuffle=False,
        seed=0,
    )

    print("Dataloaders created successfully")
    return train_loader, val_loader, nc, names

class EnhancedDetectionLoss(nn.Module):
    """Enhanced detection loss combining object detection, color classification, and size regression losses."""
    def __init__(self, num_classes=1, num_colors=4, color_weight=1.0, size_weight=1.0):
        super(EnhancedDetectionLoss, self).__init__()
        from ultralytics.utils.loss import v8DetectionLoss
        self.det_loss = v8DetectionLoss(nc=num_classes)

        self.color_loss = nn.CrossEntropyLoss()
        self.color_weight = color_weight

        self.size_loss = SizeLoss()
        self.size_weight = size_weight

    def forward(self, preds, targets, color_preds=None, color_targets=None, size_preds=None, size_targets=None):
        """Compute total loss from detection, color, and size components."""
        det_loss, loss_items = self.det_loss(preds, targets)

        color_loss = torch.tensor(0.0, device=det_loss.device)
        size_loss = torch.tensor(0.0, device=det_loss.device)

        if color_preds is not None and color_targets is not None:
            color_loss = self.color_loss(color_preds, color_targets) * self.color_weight

        if size_preds is not None and size_targets is not None:
            size_loss = self.size_loss(size_preds, size_targets) * self.size_weight

        total_loss = det_loss + color_loss + size_loss

        return total_loss, (det_loss, color_loss, size_loss)

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """Train for one epoch, returning average losses."""
    model.train()
    total_loss = 0
    total_det_loss = 0
    total_color_loss = 0
    total_size_loss = 0

    from tqdm import tqdm
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

    for i, (imgs, targets, paths, _) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)

        det_preds, color_preds, size_preds = model(imgs)

        # targets format: [img_idx, class_id, x, y, w, h, color_id, size]
        if targets.shape[1] > 6:
            color_targets = targets[:, 6].long()
            size_targets = targets[:, 7].float().unsqueeze(1)
        else:
            color_targets = None
            size_targets = None

        loss, (det_loss, color_loss, size_loss) = loss_fn(
            det_preds, targets, color_preds, color_targets, size_preds, size_targets
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_det_loss += det_loss.item()
        total_color_loss += color_loss.item()
        total_size_loss += size_loss.item()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'det_loss': f"{det_loss.item():.4f}",
            'color_loss': f"{color_loss.item():.4f}",
            'size_loss': f"{size_loss.item():.4f}"
        })

    avg_loss = total_loss / len(train_loader)
    avg_det_loss = total_det_loss / len(train_loader)
    avg_color_loss = total_color_loss / len(train_loader)
    avg_size_loss = total_size_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{args.epochs} - "
          f"Avg Loss: {avg_loss:.4f}, "
          f"Det Loss: {avg_det_loss:.4f}, "
          f"Color Loss: {avg_color_loss:.4f}, "
          f"Size Loss: {avg_size_loss:.4f}")

    return avg_loss, avg_det_loss, avg_color_loss, avg_size_loss

def validate(model, val_loader, loss_fn, device):
    """Validate the model, returning average loss and metrics."""
    model.eval()
    total_loss = 0
    total_det_loss = 0
    total_color_loss = 0
    total_size_loss = 0

    from ultralytics.utils.metrics import DetMetrics
    metrics = DetMetrics()

    with torch.no_grad():
        for imgs, targets, paths, shapes in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            det_preds, color_preds, size_preds = model(imgs)

            if targets.shape[1] > 6:
                color_targets = targets[:, 6].long()
                size_targets = targets[:, 7].float().unsqueeze(1)
            else:
                color_targets = None
                size_targets = None

            loss, (det_loss, color_loss, size_loss) = loss_fn(
                det_preds, targets, color_preds, color_targets, size_preds, size_targets
            )

            total_loss += loss.item()
            total_det_loss += det_loss.item()
            total_color_loss += color_loss.item()
            total_size_loss += size_loss.item()

            metrics.update(det_preds, targets)

    avg_loss = total_loss / len(val_loader)
    avg_det_loss = total_det_loss / len(val_loader)
    avg_color_loss = total_color_loss / len(val_loader)
    avg_size_loss = total_size_loss / len(val_loader)

    results = metrics.results_dict

    print(f"Validation - "
          f"Avg Loss: {avg_loss:.4f}, "
          f"Det Loss: {avg_det_loss:.4f}, "
          f"Color Loss: {avg_color_loss:.4f}, "
          f"Size Loss: {avg_size_loss:.4f}, "
          f"mAP50: {results['metrics/mAP50(B)']:.4f}, "
          f"mAP50-95: {results['metrics/mAP50-95(B)']:.4f}")

    return avg_loss, results

def print_multi_scale_info(multi_scale, rect, min_img_size, max_img_size, img_size):
    """Print multi-scale training configuration info."""
    print("\n===== Multi-Scale Training Config =====")
    if multi_scale:
        print(f"Multi-scale training enabled")
        print(f"Size range: {min_img_size} - {max_img_size} pixels")
        print(f"Image sizes will be randomly selected from this range during training")
    else:
        print(f"Fixed size training: {img_size} pixels")

    if rect:
        print(f"Rectangular training enabled (batch by aspect ratio)")
        print(f"This reduces padding and improves efficiency for images with varying aspect ratios")
    else:
        print(f"Rectangular training disabled (square training)")

    print("Tips for large images:")
    print("1. Enable multi-scale training to handle varied image sizes")
    print("2. Enable rectangular training for images with diverse aspect ratios")
    print("3. Increase max image size for higher accuracy if memory permits")
    print("=======================================\n")

def main():
    """Main entry point."""
    args = parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return

    try:
        data_yaml = create_temp_data_yaml(dataset_path)
    except Exception as e:
        print(f"Failed to create data config: {e}")
        import traceback
        traceback.print_exc()
        return

    device = args.device
    cuda_available = torch.cuda.is_available()
    if device != "cpu" and not cuda_available:
        print("Warning: CUDA unavailable, falling back to CPU")
        device = "cpu"
    print(f"Using device: {device}")

    try:
        CONFIG["MODEL_ENHANCEMENT"]["use_enhanced_model"] = args.use_enhanced_model
        CONFIG["MODEL_ENHANCEMENT"]["use_color_attention"] = args.use_color_attention
        CONFIG["MODEL_ENHANCEMENT"]["use_scale_fpn"] = args.use_scale_fpn

        CONFIG["TRAIN"]["multi_scale"] = args.multi_scale
        CONFIG["TRAIN"]["rect"] = args.rect
        CONFIG["TRAIN"]["min_img_size"] = args.min_img_size
        CONFIG["TRAIN"]["max_img_size"] = args.max_img_size

        print("\n===== Enhanced Model Config =====")
        print(f"Enhanced model: {args.use_enhanced_model}")
        if args.use_enhanced_model:
            print(f"  - Color attention: {args.use_color_attention}")
            print(f"  - Scale-aware FPN: {args.use_scale_fpn}")
        print("==================================\n")

        if args.multi_scale:
            print("\n===== Multi-Scale Training Config =====")
            print(f"Multi-scale training enabled: size range {args.min_img_size} - {args.max_img_size}")
            if args.rect:
                print("Rectangular training enabled (batch by aspect ratio)")
            print("=======================================\n")

        print("Initializing microsphere data augmenter...")
        data_augmenter = MicrosphereAugmentation(
            config=DATA_AUGMENTATION
        )

        model = EnhancedMicrosphereYOLO(
            model_path=args.weights,
            config=CONFIG,
            enhanced_weights_path=args.enhanced_weights
        )

        print(f"Model config:")
        print(f"  - Base model: {model.model_path}")
        print(f"  - Enhanced weights: {model.enhanced_weights_path if model.enhanced_weights_path else 'Not specified'}")
        print(f"  - Enhanced model enabled: {model.use_enhanced_model}")
        if model.use_enhanced_model:
            print(f"  - Color attention: {model.use_color_attention}")
            print(f"  - Scale-aware FPN: {model.use_scale_fpn}")

        print("Starting training...")

        if args.multi_scale:
            img_size = [args.min_img_size, args.max_img_size]
            print(f"Using multi-scale training, size range: {args.min_img_size} - {args.max_img_size}")
        else:
            img_size = args.img_size
            print(f"Using fixed size training: {img_size}")

        results = model.train(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=img_size,
            device=device,
            project=args.project,
            name=args.name,
            optimizer=args.optimizer,
            lr0=args.lr,
            lrf=args.lrf,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            warmup_momentum=args.warmup_momentum,
            warmup_bias_lr=args.warmup_bias_lr,
            save_period=args.save_period,
            seed=args.seed,
            resume=args.resume,
            augment=not args.no_augment,
            multi_scale=args.multi_scale,
            rect=args.rect,
            min_img_size=args.min_img_size,
            max_img_size=args.max_img_size,
            enhanced_model_save_path=args.enhanced_weights
        )

        if results:
            print("\n===== Training Results =====")
            if isinstance(results, dict):
                if 'save_dir' in results:
                    print(f"Results saved to: {results['save_dir']}")
                if 'enhanced_weights_path' in results:
                    print(f"Enhanced model saved to: {results['enhanced_weights_path']}")
            print("=============================\n")

    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
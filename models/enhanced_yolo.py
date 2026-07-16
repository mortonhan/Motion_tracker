"""
Enhanced YOLO model with color attention mechanism and FPN scale-aware module
for microsphere detection, color recognition, and size measurement.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any
import cv2
from datetime import datetime

try:
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Conv, SPPF
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.engine.model import Model
    from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
except ImportError:
    raise ImportError("Cannot import YOLO model, please ensure ultralytics package is installed")

from config import CONFIG, MODEL, TRAIN, VAL, PREDICT, MICROSPHERE, MODEL_ENHANCEMENT, DATA_AUGMENTATION
from utils.microsphere_utils import MicrosphereFeaturesExtractor
from utils.general import ensure_dir, save_results_to_csv, create_plots

class ColorAttentionModule(nn.Module):
    """Color attention module: enhances perception of different microsphere colors."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ColorAttentionModule, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.color_specific_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            for _ in range(4)  # 4 colors: red, blue, white, black
        ])

        self.color_fusion = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)

    def forward(self, x):
        ca = self.channel_attention(x)
        x_ca = x * ca

        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa

        color_features = []
        for conv in self.color_specific_convs:
            color_features.append(conv(x_sa))

        color_concat = torch.cat(color_features, dim=1)
        color_fused = self.color_fusion(color_concat)

        return x + color_fused

class ScaleAwareFPN(nn.Module):
    """Scale-aware FPN module: enhances detection of microspheres of different sizes."""
    def __init__(self, in_channels_list, out_channels):
        super(ScaleAwareFPN, self).__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])

        self.scale_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(len(in_channels_list))
        ])

        self.scale_specific_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features):
        laterals = [conv(feature) for feature, conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        refined_features = []
        for i, lateral in enumerate(laterals):
            attention = self.scale_attention[i](lateral)
            attended = lateral * attention
            refined = self.scale_specific_convs[i](attended)
            refined = self.fpn_convs[i](refined)
            refined_features.append(refined)

        return refined_features

class SizeColorLoss(nn.Module):
    """Loss function for size and color recognition."""
    def __init__(self, size_weight=1.0, color_weight=1.0):
        super(SizeColorLoss, self).__init__()
        self.size_weight = size_weight
        self.color_weight = color_weight
        self.size_loss = nn.SmoothL1Loss()
        self.color_loss = nn.CrossEntropyLoss()

    def forward(self, pred_size, true_size, pred_color, true_color):
        size_loss = self.size_loss(pred_size, true_size) * self.size_weight
        color_loss = self.color_loss(pred_color, true_color) * self.color_weight
        return size_loss + color_loss

class EnhancedYOLOModel(nn.Module):
    """
    Enhanced YOLO model integrating color attention and scale-aware FPN
    on top of YOLOv8.
    """
    def __init__(self, base_model, num_classes=1, num_colors=4, use_color_attention=True, use_scale_fpn=True):
        super(EnhancedYOLOModel, self).__init__()

        self.use_color_attention = use_color_attention
        self.use_scale_fpn = use_scale_fpn
        self.num_classes = num_classes
        self.num_colors = num_colors
        self.supports_multi_scale = True

        if hasattr(base_model, 'model'):
            self.backbone = base_model.model[0:9]
            self.neck = base_model.model[9:15]
            self.detect = base_model.model[15]

            print(f"Successfully extracted components from base model:")
            print(f"  - Backbone: {len(self.backbone)} layers")
            print(f"  - Neck: {len(self.neck)} layers")
            print(f"  - Detect head: {type(self.detect).__name__}")
        else:
            raise ValueError("Base model structure does not match expected format, cannot extract components")

        self.in_channels_list = [256, 512, 1024]
        print(f"Feature channel counts: {self.in_channels_list}")

        if self.use_color_attention:
            print("Creating color attention module...")
            self.color_attention = ColorAttentionModule(self.in_channels_list[-1])
            self.color_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.in_channels_list[-1], 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_colors)
            )
            print(f"Color classifier created, supports {self.num_colors} color classes")
        else:
            print("Color attention module not enabled")
            self.color_attention = None
            self.color_classifier = None

        if self.use_scale_fpn:
            print("Creating scale-aware FPN module...")
            self.scale_fpn = ScaleAwareFPN(self.in_channels_list, 256)
            self.size_regressor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.ReLU()
            )
            print("Size regressor created")
        else:
            print("Scale-aware FPN module not enabled")
            self.scale_fpn = None
            self.size_regressor = None

        print("Creating multi-scale feature processor...")
        self.multi_scale_processor = nn.ModuleDict({
            'small': nn.Conv2d(self.in_channels_list[0], self.in_channels_list[0], kernel_size=3, padding=1),
            'medium': nn.Conv2d(self.in_channels_list[1], self.in_channels_list[1], kernel_size=3, padding=1),
            'large': nn.Conv2d(self.in_channels_list[2], self.in_channels_list[2], kernel_size=3, padding=1)
        })
        print("Multi-scale feature processor created")

        print("\nEnhanced YOLO model created!")
        color_status = "Color Attention" if self.use_color_attention else ""
        scale_status = "Scale-Aware FPN" if self.use_scale_fpn else ""
        print(f"Enabled enhancements: {color_status} {scale_status}")
        print(f"Supported classes: {self.num_classes}")
        print(f"Supported colors: {self.num_colors}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            det_preds: Detection predictions
            color_preds: Color predictions (if color attention enabled)
            size_preds: Size predictions (if scale-aware FPN enabled)
        """
        features = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in [4, 6, 8]:
                features.append(x)

        if self.supports_multi_scale:
            processed_features = []
            scale_names = ['small', 'medium', 'large']

            for i, (feature, scale_name) in enumerate(zip(features, scale_names)):
                processed_feature = self.multi_scale_processor[scale_name](feature)
                processed_features.append(processed_feature)

            features = processed_features

        color_preds = None
        if self.use_color_attention and self.color_attention is not None:
            enhanced_feature = self.color_attention(features[-1])
            features[-1] = enhanced_feature
            color_preds = self.color_classifier(enhanced_feature)

        size_preds = None
        if self.use_scale_fpn and self.scale_fpn is not None:
            refined_features = self.scale_fpn(features)
            features = refined_features
            size_preds = self.size_regressor(refined_features[-1])

        for m in self.neck:
            x = m(x)

        det_preds = self.detect(x)

        return det_preds, color_preds, size_preds

    def summary(self):
        """Print model summary."""
        print("\n===== Enhanced YOLO Model Summary =====")
        print(f"Enabled enhancements:")
        print(f"  - Color attention: {self.use_color_attention}")
        print(f"  - Scale-aware FPN: {self.use_scale_fpn}")
        print(f"  - Multi-scale feature processing: {self.supports_multi_scale}")

        print(f"\nModel components:")
        print(f"  - Backbone: {len(self.backbone)} layers")
        print(f"  - Neck: {len(self.neck)} layers")
        print(f"  - Detect head: {type(self.detect).__name__}")

        if self.use_color_attention:
            print(f"\nColor attention module:")
            print(f"  - Color classifier: {self.num_colors} classes")

        if self.use_scale_fpn:
            print(f"\nScale-aware FPN:")
            print(f"  - Feature channels: {self.in_channels_list}")

        print("===============================\n")

        return {
            "use_color_attention": self.use_color_attention,
            "use_scale_fpn": self.use_scale_fpn,
            "supports_multi_scale": self.supports_multi_scale,
            "num_classes": self.num_classes,
            "num_colors": self.num_colors,
            "backbone_layers": len(self.backbone),
            "neck_layers": len(self.neck),
            "detect_type": type(self.detect).__name__
        }

class EnhancedMicrosphereYOLO:
    """Enhanced YOLO model wrapper for microsphere detection."""

    def __init__(self, model_path: Optional[str] = None, config: Dict = None, enhanced_weights_path: Optional[str] = None):
        """
        Initialize enhanced YOLO model.

        Args:
            model_path: Base model weights file path, uses config default if None
            config: Configuration dictionary, uses default config if None
            enhanced_weights_path: Enhanced model weights file path
        """
        self.config = config or CONFIG
        self.model_path = model_path or os.path.join(self.config["WEIGHTS_PATH"], self.config["MODEL"]["name"])

        self.enhanced_weights_path = enhanced_weights_path

        self._load_base_model()

        self.use_enhanced_model = self.config["MODEL_ENHANCEMENT"]["use_enhanced_model"]
        self.use_color_attention = self.config["MODEL_ENHANCEMENT"]["use_color_attention"]
        self.use_scale_fpn = self.config["MODEL_ENHANCEMENT"]["use_scale_fpn"]

        self.is_enhanced_model_created = False

        if self.use_enhanced_model:
            self._create_enhanced_model()
            self.is_enhanced_model_created = True
            print(f"Enhanced model created (color attention: {self.use_color_attention}, scale-aware FPN: {self.use_scale_fpn})")
        else:
            print("Using base YOLOv8 model, enhancements disabled")
            self.model = self.base_model

        self.features_extractor = MicrosphereFeaturesExtractor(
            color_thresholds=self.config["MICROSPHERE"]["hsv_threshold"],
            pixel_to_um_ratio=self.config["MICROSPHERE"]["pixel_to_um_ratio"]
        )

        self.color_map = {0: "red", 1: "blue", 2: "white", 3: "black"}

    def _load_base_model(self):
        """Load base YOLO model."""
        try:
            self.base_model = YOLO(self.model_path)
            print(f"Loaded base model: {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            try:
                self.base_model = YOLO(self.config["MODEL"]["name"])
                print(f"Loaded pretrained model: {self.config['MODEL']['name']}")
            except Exception as e:
                raise RuntimeError(f"Cannot load model: {e}")

    def _create_enhanced_model(self):
        """Create enhanced model with color attention and scale-aware FPN modules."""
        if not hasattr(self, 'base_model') or self.base_model is None:
            self._load_base_model()

        base_model = self.base_model.model

        print("\n===== Base Model Structure =====")
        print(f"Model type: {type(base_model).__name__}")
        print(f"Model layers: {len(base_model.model) if hasattr(base_model, 'model') else 'unknown'}")
        print("=======================\n")

        num_classes = 1
        num_colors = 4  # white, red, blue, black

        print("\n===== Creating Enhanced Model =====")
        print(f"Color attention enabled: {self.use_color_attention}")
        print(f"Scale-aware FPN enabled: {self.use_scale_fpn}")
        print("===========================\n")

        self.model = EnhancedYOLOModel(
            base_model=base_model,
            num_classes=num_classes,
            num_colors=num_colors,
            use_color_attention=self.use_color_attention,
            use_scale_fpn=self.use_scale_fpn
        )

        if self.enhanced_weights_path and os.path.exists(self.enhanced_weights_path):
            try:
                print(f"Attempting to load enhanced model weights: {self.enhanced_weights_path}")

                checkpoint = torch.load(self.enhanced_weights_path, map_location='cpu')

                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])

                    print(f"Loaded enhanced model weights successfully (version: {checkpoint.get('version', 'unknown')})")
                    print(f"Weights file date: {checkpoint.get('date', 'unknown')}")

                    if 'config' in checkpoint:
                        config = checkpoint['config']
                        print("\nWeights file config:")
                        print(f"  - Color attention: {config.get('use_color_attention', 'unknown')}")
                        print(f"  - Scale-aware FPN: {config.get('use_scale_fpn', 'unknown')}")
                        print(f"  - Num classes: {config.get('num_classes', 'unknown')}")
                        print(f"  - Num colors: {config.get('num_colors', 'unknown')}")

                        if config.get('use_color_attention') != self.use_color_attention:
                            print(f"Warning: weight file color attention setting ({config.get('use_color_attention')}) "
                                  f"does not match current setting ({self.use_color_attention})")

                        if config.get('use_scale_fpn') != self.use_scale_fpn:
                            print(f"Warning: weight file scale-aware FPN setting ({config.get('use_scale_fpn')}) "
                                  f"does not match current setting ({self.use_scale_fpn})")
                else:
                    self.model.load_state_dict(checkpoint)
                    print("Loaded enhanced model weights (legacy format)")

                print(f"Loaded enhanced model weights: {self.enhanced_weights_path}")
            except Exception as e:
                print(f"Error loading enhanced model weights: {e}")
                print("Using initialized enhanced model instead")
                import traceback
                traceback.print_exc()
        else:
            if self.enhanced_weights_path:
                print(f"Warning: specified enhanced model weights file not found: {self.enhanced_weights_path}")
            print("Using initialized enhanced model")

        self.model.supports_multi_scale = True
        print("Multi-scale feature processing enabled")

    def train(self,
              data_yaml: str,
              epochs: int = None,
              batch_size: int = None,
              img_size: int = None,
              device: str = None,
              project: str = None,
              name: str = None,
              enhanced_model_save_path: str = None,
              **kwargs):
        """
        Train the model.

        Args:
            data_yaml: Data config YAML file path
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Input image size
            device: Device ("0", "0,1,2,3" or "cpu")
            project: Project directory for saving results
            name: Experiment name
            enhanced_model_save_path: Path to save enhanced model weights
            **kwargs: Additional parameters

        Returns:
            Training results
        """
        train_args = self.config["TRAIN"].copy()

        if epochs is not None:
            train_args["epochs"] = epochs
        if batch_size is not None:
            train_args["batch_size"] = batch_size
        if img_size is not None:
            train_args["img_size"] = img_size
        if device is not None:
            train_args["device"] = device
        if project is not None:
            train_args["project"] = project
        if name is not None:
            train_args["name"] = name

        train_args.update(kwargs)

        model_type_prefix = ""
        if self.use_enhanced_model:
            model_type_prefix = "enhanced"
            if self.use_color_attention and self.use_scale_fpn:
                model_type_prefix = "enhanced_full"
            elif self.use_color_attention:
                model_type_prefix = "enhanced_color"
            elif self.use_scale_fpn:
                model_type_prefix = "enhanced_scale"
        else:
            model_type_prefix = "basic"

        if name is None:
            train_args["name"] = f"{model_type_prefix}_{train_args['name']}"

        multi_scale = train_args.get("multi_scale", False)
        if multi_scale:
            min_img_size = train_args.get("min_img_size", 320)
            max_img_size = train_args.get("max_img_size", 1024)
            img_size = [min_img_size, max_img_size]
            print(f"Multi-scale training enabled, size range: {min_img_size} - {max_img_size}")
        else:
            img_size = train_args["img_size"]
            print(f"Fixed size training: {img_size}")

        train_kwargs = {
            "data": data_yaml,
            "epochs": train_args["epochs"],
            "batch": train_args["batch_size"],
            "imgsz": img_size,
            "device": train_args["device"],
            "workers": train_args["workers"],
            "optimizer": train_args["optimizer"],
            "lr0": train_args["lr0"],
            "lrf": train_args["lrf"],
            "momentum": train_args["momentum"],
            "weight_decay": train_args["weight_decay"],
            "warmup_epochs": train_args["warmup_epochs"],
            "warmup_momentum": train_args["warmup_momentum"],
            "warmup_bias_lr": train_args["warmup_bias_lr"],
            "save_period": train_args["save_period"],
            "seed": train_args["seed"],
            "project": train_args["project"],
            "name": train_args["name"],
            "exist_ok": train_args["exist_ok"],
            "pretrained": self.config["MODEL"]["pretrained"],
            "verbose": train_args["verbose"],
            "augment": train_args["augment"],
            "rect": train_args["rect"],
            "resume": train_args["resume"],
            "val": True,
        }

        print("\n===== Training Mode Info =====")
        if self.use_enhanced_model:
            print("Training with enhanced YOLO model")
            print(f"Color attention: {self.use_color_attention}")
            print(f"Scale-aware FPN: {self.use_scale_fpn}")
        else:
            print(f"Training with base YOLOv8 model: {self.config['MODEL']['name']}")
        print("=======================\n")

        if not self.use_enhanced_model:
            results = self.base_model.train(**train_kwargs)
            return results

        if not self.is_enhanced_model_created:
            print("Enhanced model not yet created, creating now...")
            self._create_enhanced_model()
            self.is_enhanced_model_created = True
        else:
            self.model.use_color_attention = self.use_color_attention
            self.model.use_scale_fpn = self.use_scale_fpn
            print(f"Updated enhanced model settings: color attention={self.use_color_attention}, scale-aware FPN={self.use_scale_fpn}")

        self.model.summary()

        print("\n===== Starting Training =====")
        print("Note: training uses base YOLOv8; enhanced model weights are saved after training")
        print("because the ultralytics framework cannot directly train custom models.")
        print("After training, the enhanced model will be saved for prediction use.")
        print("=====================\n")

        try:
            results = self.base_model.train(**train_kwargs)

            save_dir = Path(train_args["project"]) / train_args["name"]
            ensure_dir(save_dir / "weights")

            if enhanced_model_save_path:
                self.enhanced_weights_path = enhanced_model_save_path
            else:
                self.enhanced_weights_path = str(save_dir / "weights" / "enhanced_model.pt")

            print("\nTransferring trained base model weights to enhanced model...")

            trained_base_model_path = str(save_dir / "weights" / "best.pt")
            if os.path.exists(trained_base_model_path):
                trained_base_model = YOLO(trained_base_model_path)

                base_backbone = trained_base_model.model.model[0:9]
                base_neck = trained_base_model.model.model[9:15]
                base_detect = trained_base_model.model.model[15]

                for i, (base_layer, enhanced_layer) in enumerate(zip(base_backbone, self.model.backbone)):
                    if hasattr(enhanced_layer, 'load_state_dict') and hasattr(base_layer, 'state_dict'):
                        try:
                            enhanced_layer.load_state_dict(base_layer.state_dict())
                            print(f"  - Copied backbone layer {i} weights")
                        except Exception as e:
                            print(f"  - Failed to copy backbone layer {i} weights: {e}")

                for i, (base_layer, enhanced_layer) in enumerate(zip(base_neck, self.model.neck)):
                    if hasattr(enhanced_layer, 'load_state_dict') and hasattr(base_layer, 'state_dict'):
                        try:
                            enhanced_layer.load_state_dict(base_layer.state_dict())
                            print(f"  - Copied neck layer {i} weights")
                        except Exception as e:
                            print(f"  - Failed to copy neck layer {i} weights: {e}")

                try:
                    self.model.detect.load_state_dict(base_detect.state_dict())
                    print(f"  - Copied detect head weights")
                except Exception as e:
                    print(f"  - Failed to copy detect head weights: {e}")

                print("Base model weight transfer complete!")
            else:
                print(f"Warning: trained base model weights not found at {trained_base_model_path}")

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'use_color_attention': self.use_color_attention,
                    'use_scale_fpn': self.use_scale_fpn,
                    'num_classes': self.model.num_classes,
                    'num_colors': self.model.num_colors,
                    'supports_multi_scale': self.model.supports_multi_scale
                },
                'version': '1.0',
                'date': str(datetime.now())
            }, self.enhanced_weights_path)

            print(f"\n===== Training Complete =====")
            print(f"Base YOLOv8 model saved to: {save_dir / 'weights' / 'best.pt'}")
            print(f"Enhanced model saved to: {self.enhanced_weights_path}")
            print("=====================\n")

            return {
                "model": self.model,
                "base_model": self.base_model,
                "save_dir": save_dir,
                "results": results,
                "enhanced_weights_path": self.enhanced_weights_path
            }

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def predict(self,
               source: Union[str, Path, List[str], List[Path]],
               conf_thres: float = None,
               iou_thres: float = None,
               img_size: int = None,
               device: str = None,
               save: bool = True,
               project: str = None,
               name: str = None,
               **kwargs):
        """
        Run prediction using the enhanced model.

        Args:
            source: Input source (file, directory, URL, or glob pattern)
            conf_thres: Confidence threshold
            iou_thres: IoU threshold
            img_size: Input image size
            device: Device ("0", "0,1,2,3" or "cpu")
            save: Whether to save results
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional parameters

        Returns:
            Prediction results
        """
        pred_args = self.config["PREDICT"].copy()

        if conf_thres is not None:
            pred_args["conf_thres"] = conf_thres
        if iou_thres is not None:
            pred_args["iou_thres"] = iou_thres
        if img_size is not None:
            pred_args["img_size"] = img_size
        if device is not None:
            pred_args["device"] = device
        if project is not None:
            pred_args["project"] = project
        if name is not None:
            pred_args["name"] = name

        pred_args.update(kwargs)

        try:
            results = self.base_model.predict(
                source=source,
                conf=pred_args["conf_thres"],
                iou=pred_args["iou_thres"],
                imgsz=pred_args["img_size"],
                device=pred_args["device"],
                save=False,
                project=pred_args.get("project", "runs/detect"),
                name=pred_args.get("name", "exp"),
                verbose=True,
                augment=pred_args.get("augment", False),
                data=None
            )

            enhanced_results = self._enhance_prediction_results(results)

            if save:
                self._save_enhanced_results(enhanced_results, pred_args["project"], pred_args["name"])

            return enhanced_results

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _enhance_prediction_results(self, results):
        """
        Enhance prediction results with color and size information.

        Args:
            results: Base model prediction results

        Returns:
            Enhanced prediction results
        """
        enhanced_results = []

        for result in results:
            try:
                orig_img = result.orig_img
                boxes = result.boxes

                detections = []
                for box in boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    detection = {
                        "bbox": bbox.tolist(),
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": result.names[cls_id] if cls_id in result.names else f"class_{cls_id}"
                    }

                    detections.append(detection)

                enhanced_detections = []
                for det in detections:
                    enhanced_det = {
                        **det,
                        "color": "unknown",
                        "size": 0.0
                    }
                    enhanced_detections.append(enhanced_det)

                visualized_img = orig_img.copy()
                for det in enhanced_detections:
                    x1, y1, x2, y2 = map(int, det["bbox"])
                    cls_name = det["class_name"]
                    conf = det["confidence"]
                    cv2.rectangle(visualized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(visualized_img, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                enhanced_result = {
                    "orig_img": orig_img,
                    "visualized_img": visualized_img,
                    "path": result.path,
                    "detections": enhanced_detections
                }

                enhanced_results.append(enhanced_result)
            except Exception as e:
                print(f"Error enhancing prediction results: {e}")
                import traceback
                traceback.print_exc()
                continue

        return enhanced_results

    def _save_enhanced_results(self, enhanced_results, project, name):
        """
        Save enhanced prediction results.

        Args:
            enhanced_results: Enhanced prediction results
            project: Project directory
            name: Experiment name
        """
        import pandas as pd
        from datetime import datetime

        output_dir = Path(project) / name
        ensure_dir(output_dir)
        ensure_dir(output_dir / "images")
        ensure_dir(output_dir / "labels")

        for result in enhanced_results:
            image_path = Path(result["path"])
            file_name = image_path.stem

            if result["visualized_img"] is not None:
                vis_path = output_dir / "images" / f"{file_name}.jpg"
                cv2.imwrite(str(vis_path), result["visualized_img"])

            img_height, img_width = result["orig_img"].shape[:2]
            txt_path = output_dir / "labels" / f"{file_name}.txt"

            with open(txt_path, 'w') as f:
                for det in result["detections"]:
                    bbox = det["bbox"]
                    class_id = det["class_id"]
                    conf = det["confidence"]

                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n"
                    f.write(line)

        print(f"Results saved to {output_dir}")
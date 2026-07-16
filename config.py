"""
Configuration file: defines parameters for model training, validation and prediction.
"""

import os
from pathlib import Path

from utils import detect_dataset_structure, detect_class_names, split_dataset

# Base path configuration
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = ROOT / "data"
OUTPUT_PATH = ROOT / "runs"
WEIGHTS_PATH = ROOT / "weights"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(WEIGHTS_PATH, exist_ok=True)

DEFAULT_CLASS_NAMES = ["microsphere"]

# =============================================================================
# Core configuration options
# =============================================================================

MODEL_ENHANCEMENT = {
    "use_enhanced_model": True,
    "use_color_attention": True,
    "use_scale_fpn": True,
}

DATA_AUGMENTATION = {
    "enabled": False,
    "preserve_color": True,
    "preserve_size": True,
    "rotation_range": [-10, 10],
    "brightness_range": [0.8, 1.2],
    "contrast_range": [0.8, 1.2],
    "blur_probability": 0.2,
    "noise_probability": 0.2,
    "flip_probability": 0.5,
}

MODEL = {
    "name": "yolov8l.pt",
    "pretrained": True,
}

TRAIN = {
    "batch_size": 16,
    "epochs": 200,
    "img_size": 512,
    "device": "0",
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "save_period": -1,
    "seed": 0,
    "augment": True,
    "rect": True,
    "resume": False,
    "verbose": True,
    "project": "runs/train",
    "name": "yolov8_nomal",
    "exist_ok": False,
    "multi_scale": True,
    "scale_range": [0.5, 1.5],
    "min_img_size": 320,
    "max_img_size": 640,
}

# =============================================================================
# Advanced configuration options
# =============================================================================

VAL = {
    "batch_size": 8,
    "img_size": 512,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "max_det": 300,
    "task": "val",
    "device": "0",
    "workers": 4,
    "verbose": True,
    "project": "runs/val",
    "name": "exp",
    "exist_ok": False,
    "half": False,
}

PREDICT = {
    "source": "track_data/5type_antibiotics/0_1.mp4",
    "weights": "runs/train/enhanced8/weights/best.pt",
    "project": "prediction_results/4",
    "name": "labels",
    "exist_ok": True,
    "save_mot_format": True,
    "model_img_size": 1920,
    "original_size": True,
    "conf_thres": 0.3,
    "iou_thres": 0.25,
    "max_det": 200,
    "device": "0",
    "half": False,
    "dnn": False,
    "save_img": True,
    "save_txt": True,
    "save_csv": False,
    "save_conf": True,
    "save_crop": False,
    "nosave": False,
    "view_img": False,
    "line_thickness": 2,
    "hide_labels": False,
    "hide_conf": False,
    "classes": None,
    "agnostic_nms": False,
    "augment": True,
    "visualize": False,
    "generate_report": False,
    "sliding_window": {
        "enabled": False,
        "window_size": 512,
        "overlap": 0.2,
        "batch_size": 16,
        "nms_threshold": 0.45,
    },
}

MICROSPHERE = {
    "color_enhance": True,
    "hsv_threshold": {
        "white": [[0, 0, 200], [180, 30, 255]],
        "red": [[0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]],
        "blue": [[100, 100, 100], [140, 255, 255]],
        "black": [[0, 0, 0], [180, 255, 50]],
        "others": []
    },
    "size_measurement": True,
    "pixel_to_um_ratio": 0.5,
    "min_confidence": 0.15,
}

POSTPROCESS = {
    "csv_format": ["frame", "id", "class", "confidence", "x1", "y1", "x2", "y2", "color", "size"],
    "visualization": True,
    "plot_confusion_matrix": True,
    "plot_pr_curve": True,
    "plot_size_distribution": True,
    "plot_color_distribution": True,
}


CONFIG = {
    "ROOT": ROOT,
    "DATASET_PATH": DATASET_PATH,
    "OUTPUT_PATH": OUTPUT_PATH,
    "WEIGHTS_PATH": WEIGHTS_PATH,
    "MODEL": MODEL,
    "TRAIN": TRAIN,
    "VAL": VAL,
    "PREDICT": PREDICT,
    "MICROSPHERE": MICROSPHERE,
    "POSTPROCESS": POSTPROCESS,
    "MODEL_ENHANCEMENT": MODEL_ENHANCEMENT,
    "DATA_AUGMENTATION": DATA_AUGMENTATION,
}

"""
CLI argument parsing utilities
"""

import argparse
from pathlib import Path
import torch
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Microsphere detection prediction tool")

    # Basic parameters
    parser.add_argument('--weights', type=str, default='weights/yolov8n.pt', help='Model weights file path')
    parser.add_argument('--source', type=str, default='dataset/predict', help='Input source (image file or directory)')
    parser.add_argument('--project', type=str, default='prediction_results/4', help='Output directory')
    parser.add_argument('--exist-ok', action='store_true', help='Allow overwriting existing output directory')

    # Detection parameters
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='Compute device (cuda device, e.g. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--augment', action='store_true', help='Use test-time augmentation')

    # Output parameters
    parser.add_argument('--save-txt', action='store_true', help='Save text results')
    parser.add_argument('--save-conf', action='store_true', help='Save confidence in text results')
    parser.add_argument('--save-img', action='store_true', default=True, help='Save detection result images')
    parser.add_argument('--save-mot', action='store_true', default=True, help='Save MOT format annotation files')
    parser.add_argument('--hide-labels', action='store_true', help='Hide labels')
    parser.add_argument('--hide-conf', action='store_true', help='Hide confidence')
    parser.add_argument('--line-thickness', type=int, default=3, help='Bounding box line thickness')

    # Sliding window parameters
    parser.add_argument('--sliding-window', action='store_true', help='Use sliding window detection')
    parser.add_argument('--window-size', type=int, default=640, help='Sliding window size')
    parser.add_argument('--overlap', type=float, default=0.2, help='Sliding window overlap ratio')

    return parser.parse_args()


def get_device(device_str: str = '') -> torch.device:
    """Get compute device

    Args:
        device_str: Device string, e.g. '0', '0,1', 'cpu'

    Returns:
        torch.device object
    """
    if not device_str:
        device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if device_str.lower() != 'cpu' and ',' in device_str:
        device_str = f'cuda:{device_str.split(",")[0]}'

    device = torch.device(device_str)

    return device


def print_args(args: argparse.Namespace) -> None:
    """Print argument information"""
    print("Running parameters:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argument namespace to dictionary"""
    return {k: v for k, v in vars(args).items()}


def dict_to_args(args_dict: Dict[str, Any]) -> argparse.Namespace:
    """Convert dictionary to argument namespace"""
    args = argparse.Namespace()
    for k, v in args_dict.items():
        setattr(args, k, v)
    return args


def save_args(args: argparse.Namespace, output_path: str) -> None:
    """Save arguments to a JSON file"""
    import json

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = args_to_dict(args)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)


def load_args(input_path: str) -> argparse.Namespace:
    """Load arguments from a JSON file"""
    import json

    with open(input_path, 'r', encoding='utf-8') as f:
        args_dict = json.load(f)

    return dict_to_args(args_dict)
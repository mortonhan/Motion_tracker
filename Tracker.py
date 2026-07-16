#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microsphere tracker executor.
Unified tracker parameter management for easy adjustment and execution.
"""

import argparse
import os
import sys
from pathlib import Path
import torch

from Motion_tracker.tracker import YOLOTrackPipeline
from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Microsphere motion tracker',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--video', type=str, default='track_data/11111-2.mp4',
                       help='Input video path')
    parser.add_argument('--weights', type=str, default='runs/train/enhanced8/weights/best.pt',
                       help='YOLO model weights file path')
    parser.add_argument('--output', type=str, default='tracking_results',
                       help='Output directory')

    parser.add_argument('--conf-thres', type=float, default=0.3,
                       help='YOLO detection confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3,
                       help='YOLO detection IoU threshold')
    parser.add_argument('--img-size', type=int, default=1920,
                       help='YOLO input image size. '
                            'Use 640 for training, 1280-1920 for large images (e.g. 3000x1024) '
                            'to maintain accuracy at the cost of speed.')

    parser.add_argument('--disappear-zone-width', type=int, default=250,
                       help='Disappear zone width (pixels)')
    parser.add_argument('--appear-zone-width', type=int, default=250,
                       help='Appear zone width (pixels)')

    parser.add_argument('--initial-match-range', type=int, default=180,
                       help='Initial match range (pixels)')
    parser.add_argument('--max-match-range', type=int, default=300,
                       help='Maximum match range (pixels)')
    parser.add_argument('--initial-match-angle', type=float, default=10.0,
                       help='Initial match angle range (degrees)')

    parser.add_argument('--max-lost-frames', type=int, default=4,
                       help='Maximum consecutive lost frames')
    parser.add_argument('--first-miss-range-extend', type=int, default=150,
                       help='Range extension on first miss (pixels)')
    parser.add_argument('--later-miss-range-extend', type=int, default=200,
                       help='Range extension on subsequent misses (pixels)')
    parser.add_argument('--miss-angle-extend', type=float, default=0.0,
                       help='Angle extension on miss (degrees)')

    parser.add_argument('--static-neighborhood', type=int, default=5,
                       help='Static detection neighborhood (pixels)')

    parser.add_argument('--max-history-len', type=int, default=30,
                       help='Maximum history length')

    parser.add_argument('--save-video', action='store_true', default=True,
                       help='Save tracking result video')
    parser.add_argument('--show-trajectory', action='store_true', default=True,
                       help='Show trajectories in video')
    parser.add_argument('--line-thickness', type=int, default=1,
                       help='Line thickness')

    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, cuda, cuda:0, mps (Mac GPU)')

    parser.add_argument('--tracker', type=str, default='motion',
                       choices=['motion', 'bytetrack'],
                       help='Tracker type: motion (custom), bytetrack')

    parser.add_argument('--track-thresh', type=float, default=0.25,
                       help='ByteTrack tracking confidence threshold')
    parser.add_argument('--track-buffer', type=int, default=30,
                       help='ByteTrack tracking buffer size')
    parser.add_argument('--match-thresh', type=float, default=0.8,
                       help='ByteTrack matching threshold')
    parser.add_argument('--aspect-ratio-thresh', type=float, default=3.0,
                       help='ByteTrack aspect ratio threshold')
    parser.add_argument('--min-box-area', type=float, default=1.0,
                       help='ByteTrack minimum bounding box area')
    parser.add_argument('--mot20', action='store_true', default=False,
                       help='Use ByteTrack MOT20 dataset settings')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Model weights file not found: {weights_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Microsphere Motion Tracker")
    print("=" * 60)
    print(f"Input video: {video_path}")
    print(f"Model weights: {weights_path}")
    print(f"Output directory: {output_dir}")
    print()

    print("Loading YOLO model...")
    try:
        try:
            os.environ['TORCH_WEIGHTS_ONLY'] = '0'
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = patched_torch_load

            model = YOLO(str(weights_path))
            print(f"Model loaded successfully: {weights_path}")

            torch.load = original_torch_load
        except Exception as e:
            print(f"Warning: Model loading issue: {e}")
            model = YOLO(str(weights_path))
    except Exception as e:
        print(f"Error: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    device = args.device.lower()

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            print("CUDA detected, using GPU acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Mac GPU (MPS) detected, using GPU acceleration")
            print("Note: Model will run on Mac GPU even if trained on CUDA")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
            print("Note: Model will run on CPU even if trained on CUDA")
    elif device == 'cpu':
        device = 'cpu'
        print("Using CPU")
    elif device.startswith('cuda'):
        if torch.cuda.is_available():
            if ':' in device:
                device_id = device.split(':')[1]
                device = f'cuda:{device_id}'
            else:
                device = 'cuda:0'
            print(f"Using CUDA device: {device}")
        else:
            print("Warning: CUDA unavailable, switching to CPU")
            device = 'cpu'
    elif device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Using Mac GPU (MPS)")
        else:
            print("Warning: MPS unavailable, switching to CPU")
            device = 'cpu'
    else:
        try:
            device_id = int(device)
            if torch.cuda.is_available():
                device = f'cuda:{device_id}'
                print(f"Using CUDA device: {device}")
            else:
                print("Warning: CUDA unavailable, switching to CPU")
                device = 'cpu'
        except ValueError:
            print(f"Warning: Unrecognized device '{args.device}', using CPU")
            device = 'cpu'

    print(f"Final device: {device}")
    print()

    print("Initializing tracker...")
    print(f"Tracker type: {args.tracker}")
    print(f"YOLO detection params:")
    print(f"  - Input image size: {args.img_size}")
    print(f"  - Confidence threshold: {args.conf_thres}")
    print(f"  - IoU threshold: {args.iou_thres}")

    if args.tracker == 'bytetrack':
        print(f"ByteTrack params:")
        print(f"  - Track confidence threshold: {args.track_thresh}")
        print(f"  - Track buffer: {args.track_buffer}")
        print(f"  - Match threshold: {args.match_thresh}")
        print(f"  - Aspect ratio threshold: {args.aspect_ratio_thresh}")
        print(f"  - Min bounding box area: {args.min_box_area}")
        print(f"  - MOT20 settings: {args.mot20}")

        tracker_kwargs = {
            'track_thresh': args.track_thresh,
            'track_buffer': args.track_buffer,
            'match_thresh': args.match_thresh,
            'aspect_ratio_thresh': args.aspect_ratio_thresh,
            'min_box_area': args.min_box_area,
            'mot20': args.mot20,
            'disappear_zone_width': args.disappear_zone_width,
            'appear_zone_width': args.appear_zone_width,
        }
    else:
        print(f"Motion tracker params:")
        print(f"  - Disappear zone width: {args.disappear_zone_width} px")
        print(f"  - Appear zone width: {args.appear_zone_width} px")
        print(f"  - Initial match range: {args.initial_match_range} px")
        print(f"  - Max match range: {args.max_match_range} px")
        print(f"  - Initial match angle: {args.initial_match_angle} deg")
        print(f"  - Max lost frames: {args.max_lost_frames}")
        print(f"  - Static neighborhood: {args.static_neighborhood} px")

        tracker_kwargs = {
            'disappear_zone_width': args.disappear_zone_width,
            'appear_zone_width': args.appear_zone_width,
            'static_neighborhood': args.static_neighborhood,
            'initial_match_range': args.initial_match_range,
            'max_match_range': args.max_match_range,
            'max_lost_frames': args.max_lost_frames,
            'max_history_len': args.max_history_len,
            'initial_match_angle': args.initial_match_angle,
            'first_miss_range_extend': args.first_miss_range_extend,
            'later_miss_range_extend': args.later_miss_range_extend,
            'miss_angle_extend': args.miss_angle_extend
        }
    print()

    pipeline = YOLOTrackPipeline(
        yolo_model=model,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        device=device,
        tracker_type=args.tracker,
        **tracker_kwargs
    )

    print("Starting video processing...")
    print()

    try:
        pipeline.process_video(
            video_path=str(video_path),
            output_dir=str(output_dir),
            save_video=args.save_video,
            show_trajectory=args.show_trajectory,
            line_thickness=args.line_thickness
        )

        print()
        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
        print(f"Output files:")
        video_filename = video_path.stem
        print(f"  - Tracked video: {output_dir / f'{video_filename}_tracked.mp4'}")
        print(f"  - Matching areas video: {output_dir / f'{video_filename}_matching_areas.mp4'}")
        print(f"  - Results CSV: {output_dir / f'{video_filename}_results.csv'}")
        print(f"  - Trajectory plot: {output_dir / f'{video_filename}_trajectories.png'}")

    except Exception as e:
        print(f"Error: Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
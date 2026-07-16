#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch video tracking processing script.
Supports batch processing of multiple video files.
Each video is processed independently to ensure tracking accuracy.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from tqdm import tqdm
import time

from Motion_tracker.tracker import YOLOTrackPipeline
from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch microsphere motion tracker',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, default='track_data/5',
                       help='Input video path or directory (supports batch processing)')
    parser.add_argument('--weights', type=str, default='runs/train/enhanced8/weights/best.pt',
                       help='YOLO model weights file path')
    parser.add_argument('--output', type=str, default='tracking_results/5',
                       help='Output directory')
    parser.add_argument('--recursive', action='store_true', default=False,
                       help='Recursively search for video files in subdirectories')
    parser.add_argument('--pattern', type=str, default='*.mp4',
                       help='Video file matching pattern (default: *.mp4)')
    
    parser.add_argument('--conf-thres', type=float, default=0.3,
                       help='YOLO detection confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3,
                       help='YOLO detection IoU threshold')
    parser.add_argument('--img-size', type=int, default=1920,
                       help='YOLO input image size. Use 640 for training, '
                            '1280-1920 for large images to maintain accuracy at the cost of speed.')
    
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
    
    parser.add_argument('--skip-existing', action='store_true', default=False,
                       help='Skip existing output files')
    parser.add_argument('--max-workers', type=int, default=1,
                       help='Maximum parallel workers (currently only supports 1)')
    
    return parser.parse_args()


def find_video_files(input_path: Path, pattern: str = '*.mp4', recursive: bool = False) -> list:
    """Find video files."""
    video_files = []
    
    if input_path.is_file():
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
            video_files.append(input_path)
    elif input_path.is_dir():
        if recursive:
            video_files = list(input_path.rglob(pattern))
        else:
            video_files = list(input_path.glob(pattern))
        video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    return sorted(video_files)


def setup_device(device_arg: str) -> str:
    """Setup compute device."""
    device = device_arg.lower()
    
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
            print("CUDA detected, using GPU acceleration")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            print("Mac GPU (MPS) detected, using GPU acceleration")
        else:
            device = 'cpu'
            print("No GPU detected, using CPU")
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
            print(f"Warning: Unrecognized device '{device_arg}', using CPU")
            device = 'cpu'
    
    return device


def load_yolo_model(weights_path: Path) -> YOLO:
    """Load YOLO model."""
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
        raise
    
    return model


def create_pipeline(model: YOLO, args, device: str) -> YOLOTrackPipeline:
    """Create tracking pipeline."""
    if args.tracker == 'bytetrack':
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
    
    pipeline = YOLOTrackPipeline(
        yolo_model=model,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        img_size=args.img_size,
        device=device,
        tracker_type=args.tracker,
        **tracker_kwargs
    )
    
    return pipeline


def process_single_video(video_path: Path, output_dir: Path, pipeline: YOLOTrackPipeline, 
                        args, skip_existing: bool = False) -> dict:
    """Process a single video file."""
    video_filename = video_path.stem
    result = {
        'video': str(video_path),
        'success': False,
        'error': None,
        'output_files': []
    }
    
    if skip_existing:
        expected_outputs = []
        if args.save_video:
            expected_outputs.extend([
                output_dir / f'{video_filename}_tracked.mp4',
                output_dir / f'{video_filename}_matching_areas.mp4'
            ])
        expected_outputs.extend([
            output_dir / f'{video_filename}_results.csv',
            output_dir / f'{video_filename}_trajectories.png'
        ])
        
        if all(f.exists() for f in expected_outputs):
            result['success'] = True
            result['skipped'] = True
            result['output_files'] = [str(f) for f in expected_outputs if f.exists()]
            return result
    
    try:
        pipeline.process_video(
            video_path=str(video_path),
            output_dir=str(output_dir),
            save_video=args.save_video,
            show_trajectory=args.show_trajectory,
            line_thickness=args.line_thickness
        )
        
        if args.save_video:
            result['output_files'] = [
                str(output_dir / f'{video_filename}_tracked.mp4'),
                str(output_dir / f'{video_filename}_matching_areas.mp4'),
                str(output_dir / f'{video_filename}_results.csv'),
                str(output_dir / f'{video_filename}_trajectories.png')
            ]
        else:
            result['output_files'] = [
                str(output_dir / f'{video_filename}_results.csv'),
                str(output_dir / f'{video_filename}_trajectories.png')
            ]
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def main():
    args = parse_args()
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Model weights file not found: {weights_path}")
        sys.exit(1)
    
    input_path = Path(args.input)
    video_files = find_video_files(input_path, args.pattern, args.recursive)
    
    if not video_files:
        print(f"Error: No video files found: {input_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Batch Microsphere Motion Tracker")
    print("=" * 60)
    print(f"Found {len(video_files)} video files")
    print(f"Model weights: {weights_path}")
    print(f"Output directory: {args.output}")
    print()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_yolo_model(weights_path)
    
    device = setup_device(args.device)
    print(f"Final device: {device}")
    print()
    
    print("Initializing tracker...")
    print(f"Tracker type: {args.tracker}")
    print(f"YOLO detection params:")
    print(f"  - Input image size: {args.img_size}")
    print(f"  - Confidence threshold: {args.conf_thres}")
    print(f"  - IoU threshold: {args.iou_thres}")
    print()
    
    results = []
    start_time = time.time()
    
    print("Starting batch video processing...")
    print()
    
    for i, video_path in enumerate(tqdm(video_files, desc="Progress", unit="video")):
        print(f"\n[{i+1}/{len(video_files)}] Processing: {video_path.name}")
        
        pipeline = create_pipeline(model, args, device)
        
        result = process_single_video(
            video_path=video_path,
            output_dir=output_dir,
            pipeline=pipeline,
            args=args,
            skip_existing=args.skip_existing
        )
        
        results.append(result)
        
        if result['success']:
            if result.get('skipped', False):
                print(f"  ✓ Skipped (output files already exist)")
            else:
                print(f"  ✓ Processing successful")
        else:
            print(f"  ✗ Processing failed: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    success_count = sum(1 for r in results if r['success'] and not r.get('skipped', False))
    skipped_count = sum(1 for r in results if r.get('skipped', False))
    failed_count = sum(1 for r in results if not r['success'])
    
    print()
    print("=" * 60)
    print("Batch processing complete")
    print("=" * 60)
    print(f"Total videos: {len(video_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped files: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total time: {total_time:.2f} seconds")
    if len(video_files) > 0:
        print(f"Average time: {total_time / len(video_files):.2f} seconds/video")
    print()
    
    if failed_count > 0:
        print("Failed files:")
        for result in results:
            if not result['success']:
                print(f"  - {result['video']}")
                print(f"    Error: {result.get('error', 'Unknown error')}")
        print()
        
        error_log_path = output_dir / 'batch_processing_errors.log'
        with open(error_log_path, 'w', encoding='utf-8') as f:
            f.write("Batch Processing Error Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Processing time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total videos: {len(video_files)}\n")
            f.write(f"Failed count: {failed_count}\n")
            f.write("\n" + "=" * 60 + "\n\n")
            
            for result in results:
                if not result['success']:
                    f.write(f"Video: {result['video']}\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    if 'traceback' in result:
                        f.write("\nDetailed traceback:\n")
                        f.write(result['traceback'])
                    f.write("\n" + "-" * 60 + "\n\n")
        
        print(f"Error log saved to: {error_log_path}")
    
    summary_path = output_dir / 'batch_processing_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Batch Processing Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Processing time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input path: {args.input}\n")
        f.write(f"Output directory: {args.output}\n")
        f.write(f"Model weights: {args.weights}\n")
        f.write(f"Tracker type: {args.tracker}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total videos: {len(video_files)}\n")
        f.write(f"Successfully processed: {success_count}\n")
        f.write(f"Skipped files: {skipped_count}\n")
        f.write(f"Failed: {failed_count}\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        if len(video_files) > 0:
            f.write(f"Average time: {total_time / len(video_files):.2f} seconds/video\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("Processed files:\n")
        for i, result in enumerate(results, 1):
            status = "Success" if result['success'] else ("Skipped" if result.get('skipped', False) else "Failed")
            f.write(f"{i}. [{status}] {result['video']}\n")
            if result['success'] and result.get('output_files'):
                for output_file in result['output_files']:
                    f.write(f"   - {output_file}\n")
    
    print(f"Processing summary saved to: {summary_path}")
    print()
    
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

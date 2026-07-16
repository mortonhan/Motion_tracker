#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microsphere detection prediction script.
Uses YOLO model for microsphere detection.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import torch
import sys
import gc
from tqdm import tqdm
import argparse

from utils.general import ensure_dir
from utils.file_utils import get_file_name
from utils.cli_utils import parse_args
from utils.process_utils import process_image, process_image_sliding_window, process_video
from config import PREDICT


def is_video_file(file_path: Path) -> bool:
    """Check if file is a video file."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.3gp', '.webm'}
    return file_path.suffix.lower() in video_extensions


def main():
    args = parse_args()

    # Apply config defaults when CLI args use default values
    if args.weights == 'weights/yolov8n.pt':
        args.weights = PREDICT.get('weights', args.weights)
    if args.source == 'dataset/predict':
        args.source = PREDICT.get('source', args.source)
    if args.project == 'prediction_results/4':
        args.project = PREDICT.get('project', args.project)
    if args.conf_thres == 0.25:
        args.conf_thres = PREDICT.get('conf_thres', args.conf_thres)
    if args.iou_thres == 0.45:
        args.iou_thres = PREDICT.get('iou_thres', args.iou_thres)
    if args.img_size == 640:
        args.img_size = PREDICT.get('model_img_size', args.img_size)
    if not args.augment:
        args.augment = PREDICT.get('augment', args.augment)
    if args.line_thickness == 3:
        args.line_thickness = PREDICT.get('line_thickness', args.line_thickness)
    if not args.save_conf:
        args.save_conf = PREDICT.get('save_conf', args.save_conf)
    if not args.exist_ok:
        args.exist_ok = PREDICT.get('exist_ok', args.exist_ok)
    
    # MOT format output
    if hasattr(args, 'save_mot') and not args.save_mot:
        args.save_mot = PREDICT.get('save_mot_format', True)
    elif not hasattr(args, 'save_mot'):
        args.save_mot = PREDICT.get('save_mot_format', True)

    # Sliding window config
    sliding_window_config = PREDICT.get('sliding_window', {})
    if not args.sliding_window:
        args.sliding_window = sliding_window_config.get('enabled', args.sliding_window)
    if args.window_size == 640:
        args.window_size = sliding_window_config.get('window_size', args.window_size)
    if args.overlap == 0.2:
        args.overlap = sliding_window_config.get('overlap', args.overlap)

    output_dir = Path(args.project)
    if output_dir.exists() and not args.exist_ok:
        raise ValueError(f"Output directory exists and exist-ok is False: {output_dir}")
    ensure_dir(output_dir)

    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    ensure_dir(labels_dir)
    ensure_dir(images_dir)

    print(f"Output directory: {output_dir}")

    # Load YOLO model
    try:
        from ultralytics import YOLO
        
        try:
            os.environ['TORCH_WEIGHTS_ONLY'] = '0'
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            torch.load = patched_torch_load
            
            model = YOLO(args.weights)
            print(f"Model loaded: {args.weights}")
            
            torch.load = original_torch_load
        except Exception as e:
            print(f"Patch loading failed: {e}")
            weights_path = Path(args.weights)
            if weights_path.exists():
                print(f"Model file exists: {weights_path}")
                print("Please try reloading model with older torch and ultralytics versions")
                return
            else:
                print(f"Model file not found: {weights_path}")
                return
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    if args.sliding_window:
        print(f"Sliding window detection enabled - window size: {args.window_size}x{args.window_size}, overlap: {args.overlap}")
    else:
        print("Sliding window detection disabled - using normal detection mode")

    source = Path(args.source).resolve()
    if not source.exists():
        print(f"Input source not found: {source}")
        return

    if source.is_file():
        if is_video_file(source):
            print(f"Video file detected, processing: {source}")
            process_video(
                model=model,
                video_path=source,
                output_dir=output_dir,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                img_size=args.img_size,
                device=args.device,
                save_img=args.save_img,
                save_txt=args.save_txt,
                save_conf=args.save_conf,
                line_thickness=args.line_thickness,
                hide_labels=args.hide_labels,
                hide_conf=args.hide_conf,
                augment=args.augment,
                save_video=True,
                save_mot_format=getattr(args, 'save_mot', True)
            )
        else:
            if args.sliding_window:
                print(f"Processing image with sliding window: {source}")
                process_image_sliding_window(
                    model=model,
                    image_path=source,
                    output_dir=output_dir,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    img_size=args.img_size,
                    device=args.device,
                    save_img=args.save_img,
                    save_txt=args.save_txt,
                    save_conf=args.save_conf,
                    line_thickness=args.line_thickness,
                    hide_labels=args.hide_labels,
                    hide_conf=args.hide_conf,
                    augment=args.augment,
                    window_size=args.window_size,
                    overlap=args.overlap
                )
            else:
                print(f"Processing image with normal detection: {source}")
                process_image(
                    model=model,
                    image_path=source,
                    output_dir=output_dir,
                    conf_thres=args.conf_thres,
                    iou_thres=args.iou_thres,
                    img_size=args.img_size,
                    device=args.device,
                    save_img=args.save_img,
                    save_txt=args.save_txt,
                    save_conf=args.save_conf,
                    line_thickness=args.line_thickness,
                    hide_labels=args.hide_labels,
                    hide_conf=args.hide_conf,
                    augment=args.augment
                )
    elif source.is_dir():
        media_files = []

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            media_files.extend(list(source.glob(f"**/*{ext}")))

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.3gp', '.webm']
        for ext in video_extensions:
            media_files.extend(list(source.glob(f"**/*{ext}")))

        if not media_files:
            print(f"No image or video files in directory: {source}")
            return

        media_files.sort()

        image_files = [f for f in media_files if f.suffix.lower() in image_extensions]
        video_files = [f for f in media_files if f.suffix.lower() in video_extensions]

        print(f"Found {len(image_files)} image files and {len(video_files)} video files")

        if args.sliding_window:
            print(f"Processing all images with sliding window - window size: {args.window_size}x{args.window_size}, overlap: {args.overlap}")
        else:
            print("Processing all images with normal detection")

        checkpoint_file = output_dir / "checkpoint.txt"
        last_processed = -1

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    last_processed = int(f.read().strip())
                print(f"Checkpoint found, resuming from file {last_processed + 1}")
            except:
                print("Checkpoint file corrupted, starting from beginning")
                last_processed = -1

        processed_count = 0
        error_count = 0

        for i, file_path in enumerate(tqdm(media_files, desc="Processing files")):
            if i <= last_processed:
                processed_count += 1
                continue

            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if is_video_file(file_path):
                    print(f"Processing video: {file_path}")
                    process_video(
                        model=model,
                        video_path=file_path,
                        output_dir=output_dir,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        img_size=args.img_size,
                        device=args.device,
                        save_img=args.save_img,
                        save_txt=args.save_txt,
                        save_conf=args.save_conf,
                        line_thickness=args.line_thickness,
                        hide_labels=args.hide_labels,
                        hide_conf=args.hide_conf,
                        augment=args.augment,
                        save_video=True,
                        save_mot_format=getattr(args, 'save_mot', True)
                    )
                else:
                    if args.sliding_window:
                        process_image_sliding_window(
                            model=model,
                            image_path=file_path,
                            output_dir=output_dir,
                            conf_thres=args.conf_thres,
                            iou_thres=args.iou_thres,
                            img_size=args.img_size,
                            device=args.device,
                            save_img=args.save_img,
                            save_txt=args.save_txt,
                            save_conf=args.save_conf,
                            line_thickness=args.line_thickness,
                            hide_labels=args.hide_labels,
                            hide_conf=args.hide_conf,
                            augment=args.augment,
                            window_size=args.window_size,
                            overlap=args.overlap
                        )
                    else:
                        process_image(
                            model=model,
                            image_path=file_path,
                            output_dir=output_dir,
                            conf_thres=args.conf_thres,
                            iou_thres=args.iou_thres,
                            img_size=args.img_size,
                            device=args.device,
                            save_img=args.save_img,
                            save_txt=args.save_txt,
                            save_conf=args.save_conf,
                            line_thickness=args.line_thickness,
                            hide_labels=args.hide_labels,
                            hide_conf=args.hide_conf,
                            augment=args.augment
                        )
                processed_count += 1
                error_count = 0

                with open(checkpoint_file, 'w') as f:
                    f.write(str(i))

                if processed_count % 5 == 0:
                    print(f"Processed {processed_count}/{len(media_files)} files, cleaning memory...")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if processed_count % 20 == 0 and processed_count > 0:
                    print(f"Processed {processed_count} files, reloading model to prevent memory leak...")
                    try:
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        model = YOLO(args.weights)
                        print("Model reloaded successfully")
                    except Exception as e:
                        print(f"Model reload failed: {e}")
                        print("Continuing with original model...")

            except KeyboardInterrupt:
                print("Processing interrupted by user")
                with open(checkpoint_file, 'w') as f:
                    f.write(str(i-1))
                print(f"Checkpoint saved, will resume from file {i} next run")
                break

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                import traceback
                traceback.print_exc()
                error_count += 1

                if error_count > 3:
                    print("Multiple consecutive errors, attempting model reinitialization...")
                    try:
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        model = YOLO(args.weights)
                        print("Model reloaded successfully")
                        error_count = 0
                    except Exception as e:
                        print(f"Model reload failed: {e}")
                        print("Continuing with original model...")

        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print("Processing complete, checkpoint file removed")
            except:
                print("Could not remove checkpoint file")

        print(f"Successfully processed {processed_count}/{len(media_files)} files")
    else:
        print(f"Unsupported input source type: {source}")

    print(f"Prediction complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up resources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Program finished")

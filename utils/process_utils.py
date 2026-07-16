"""
Image processing utilities
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any, Optional
import os

from utils.file_utils import get_file_name, create_output_filename
from utils.general import ensure_dir


def process_image(model: Any,
                 image_path: Union[str, Path],
                 output_dir: Union[str, Path],
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 img_size: int = 640,
                 device: str = '',
                 save_img: bool = True,
                 save_txt: bool = False,
                 save_conf: bool = False,
                 line_thickness: int = 3,
                 hide_labels: bool = False,
                 hide_conf: bool = False,
                 augment: bool = False) -> Dict:
    """Process a single image

    Args:
        model: YOLO model
        image_path: Image path
        output_dir: Output directory
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
        img_size: Input image size
        device: Compute device
        save_img: Save detection result images
        save_txt: Save text results
        save_conf: Save confidence in text results
        line_thickness: Bounding box line thickness
        hide_labels: Hide labels
        hide_conf: Hide confidence
        augment: Use test-time augmentation

    Returns:
        Detection result dictionary
    """
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    ensure_dir(labels_dir)
    ensure_dir(images_dir)

    filename = get_file_name(image_path)

    results = model(image_path,
                   imgsz=img_size,
                   conf=conf_thres,
                   iou=iou_thres,
                   augment=augment)

    if results and len(results) > 0:
        result = results[0]

        if save_txt:
            txt_path = labels_dir / f"{filename}.txt"
            with open(txt_path, 'w') as f:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        img_h, img_w = result.orig_shape
                        x_center = (x1 + x2) / 2 / img_w
                        y_center = (y1 + y2) / 2 / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        cls = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        line = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        if save_conf:
                            line += f" {conf:.6f}"
                        f.write(line + '\n')

        if save_img:
            img_path = images_dir / f"{filename}.jpg"
            annotated_img = result.plot(line_width=line_thickness,
                                      labels=not hide_labels,
                                      conf=not hide_conf)
            import cv2
            cv2.imwrite(str(img_path), annotated_img)

    return {
        "filename": filename,
        "results": results,
        "txt_path": labels_dir / f"{filename}.txt" if save_txt else None,
        "img_path": images_dir / f"{filename}.jpg" if save_img else None
    }


def process_image_sliding_window(model: Any,
                                image_path: Union[str, Path],
                                output_dir: Union[str, Path],
                                conf_thres: float = 0.25,
                                iou_thres: float = 0.45,
                                img_size: int = 640,
                                device: str = '',
                                save_img: bool = True,
                                save_txt: bool = False,
                                save_conf: bool = False,
                                line_thickness: int = 3,
                                hide_labels: bool = False,
                                hide_conf: bool = False,
                                augment: bool = False,
                                window_size: int = 640,
                                overlap: float = 0.2) -> Dict:
    """Process an image using sliding window detection

    Args:
        model: YOLO model
        image_path: Image path
        output_dir: Output directory
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
        img_size: Input image size
        device: Compute device
        save_img: Save detection result images
        save_txt: Save text results
        save_conf: Save confidence in text results
        line_thickness: Bounding box line thickness
        hide_labels: Hide labels
        hide_conf: Hide confidence
        augment: Use test-time augmentation
        window_size: Sliding window size
        overlap: Sliding window overlap ratio

    Returns:
        Detection result dictionary
    """
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    ensure_dir(labels_dir)
    ensure_dir(images_dir)

    filename = get_file_name(image_path)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    height, width = image.shape[:2]

    stride = int(window_size * (1 - overlap))

    num_windows_x = max(1, (width - window_size + stride) // stride)
    num_windows_y = max(1, (height - window_size + stride) // stride)

    all_detections = []

    for y in range(num_windows_y):
        for x in range(num_windows_x):
            x1 = x * stride
            y1 = y * stride
            x2 = min(x1 + window_size, width)
            y2 = min(y1 + window_size, height)

            if x2 - x1 < window_size:
                x1 = max(0, x2 - window_size)
            if y2 - y1 < window_size:
                y1 = max(0, y2 - window_size)

            window = image[y1:y2, x1:x2]

            results = model(window,
                           imgsz=img_size,
                           conf=conf_thres,
                           iou=iou_thres,
                           augment=augment)

            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue

                for box in boxes:
                    x1_box, y1_box, x2_box, y2_box = box.xyxy[0].tolist()

                    x1_orig = x1 + x1_box
                    y1_orig = y1 + y1_box
                    x2_orig = x1 + x2_box
                    y2_orig = y1 + y2_box

                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    all_detections.append([x1_orig, y1_orig, x2_orig, y2_orig, conf, cls])

    if all_detections:
        all_detections = torch.tensor(all_detections)

        from torchvision.ops import nms
        boxes = all_detections[:, :4]
        scores = all_detections[:, 4]
        keep = nms(boxes, scores, iou_thres)
        all_detections = all_detections[keep].tolist()

    result_image = image.copy()

    if save_txt:
        txt_path = labels_dir / f"{filename}.txt"
        with open(txt_path, 'w') as f:
            for det in all_detections:
                x1, y1, x2, y2, conf, cls = det

                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                line = f"{int(cls)} {x_center} {y_center} {w} {h}"
                if save_conf:
                    line += f" {conf}"
                f.write(line + '\n')

    if save_img:
        for det in all_detections:
            x1, y1, x2, y2, conf, cls = det

            color = (0, 255, 0)
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness)

            if not hide_labels:
                label = f"Class {int(cls)}"
                if not hide_conf:
                    label += f" {conf:.2f}"

                t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=max(line_thickness - 1, 1))[0]
                c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3

                cv2.rectangle(result_image, (int(x1), int(y1) - t_size[1] - 3), c2, color, -1, cv2.LINE_AA)

                cv2.putText(result_image, label, (int(x1), int(y1) - 2), 0, line_thickness / 3,
                           (225, 255, 255), thickness=max(line_thickness - 1, 1), lineType=cv2.LINE_AA)

        img_path = images_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), result_image)

    return {
        "filename": filename,
        "detections": all_detections,
        "txt_path": labels_dir / f"{filename}.txt" if save_txt else None,
        "img_path": images_dir / f"{filename}.jpg" if save_img else None
    }


def process_video(model: Any,
                 video_path: Union[str, Path],
                 output_dir: Union[str, Path],
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 img_size: int = 640,
                 device: str = '',
                 save_img: bool = True,
                 save_txt: bool = False,
                 save_conf: bool = False,
                 line_thickness: int = 3,
                 hide_labels: bool = False,
                 hide_conf: bool = False,
                 augment: bool = False,
                 save_video: bool = True,
                 fps: Optional[int] = None,
                 save_mot_format: bool = True) -> Dict:
    """Process a video file

    Args:
        model: YOLO model
        video_path: Video path
        output_dir: Output directory
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
        img_size: Input image size
        device: Compute device
        save_img: Save detection result frame images
        save_txt: Save text results
        save_conf: Save confidence in text results
        line_thickness: Bounding box line thickness
        hide_labels: Hide labels
        hide_conf: Hide confidence
        augment: Use test-time augmentation
        save_video: Save detection result video
        fps: Output video FPS (uses original if None)
        save_mot_format: Save MOT format annotation files

    Returns:
        Detection result dictionary
    """
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    videos_dir = output_dir / "videos"
    mot_dir = output_dir / "mot_labels"
    ensure_dir(labels_dir)
    ensure_dir(images_dir)
    ensure_dir(videos_dir)
    if save_mot_format:
        ensure_dir(mot_dir)

    filename = get_file_name(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = fps if fps is not None else original_fps
    video_writer = None

    if save_video:
        output_video_path = videos_dir / f"{filename}_detected.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, output_fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {original_fps:.2f}, Resolution: {width}x{height}")

    frame_count = 0
    all_results = []

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Processing video frames")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = model(frame,
                           imgsz=img_size,
                           conf=conf_thres,
                           iou=iou_thres,
                           augment=augment)

            if results and len(results) > 0:
                result = results[0]

                if save_txt:
                    txt_path = labels_dir / f"{filename}_frame_{frame_count:06d}.txt"
                    with open(txt_path, 'w') as f:
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                x_center = (x1 + x2) / 2 / width
                                y_center = (y1 + y2) / 2 / height
                                w = (x2 - x1) / width
                                h = (y2 - y1) / height
                                cls = int(box.cls[0].item())
                                conf = float(box.conf[0].item())

                                line = f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                                if save_conf:
                                    line += f" {conf:.6f}"
                                f.write(line + '\n')

                if save_mot_format:
                    mot_txt_path = mot_dir / f"frame_{frame_count:06d}.txt"
                    with open(mot_txt_path, 'w') as f:
                        if result.boxes is not None and len(result.boxes) > 0:
                            for idx, box in enumerate(result.boxes):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                bb_width = x2 - x1
                                bb_height = y2 - y1
                                cls = int(box.cls[0].item())
                                conf = float(box.conf[0].item())

                                # MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>
                                object_id = idx + 1
                                line = f"{frame_count},{object_id},{x1:.2f},{y1:.2f},{bb_width:.2f},{bb_height:.2f},{conf:.6f},{cls}"
                                f.write(line + '\n')

                annotated_frame = result.plot(line_width=line_thickness,
                                            labels=not hide_labels,
                                            conf=not hide_conf)

                if save_img:
                    img_path = images_dir / f"{filename}_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(img_path), annotated_frame)

                if save_video and video_writer is not None:
                    video_writer.write(annotated_frame)

                frame_results = {
                    "frame": frame_count,
                    "detections": []
                }

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        frame_results["detections"].append({
                            "bbox": [x1, y1, x2, y2],
                            "class": cls,
                            "confidence": conf
                        })

                all_results.append(frame_results)

            else:
                if save_video and video_writer is not None:
                    video_writer.write(frame)

                all_results.append({
                    "frame": frame_count,
                    "detections": []
                })

            pbar.update(1)

        pbar.close()

    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    print(f"Video processing complete! Processed {frame_count} frames")

    return {
        "filename": filename,
        "total_frames": frame_count,
        "results": all_results,
        "output_video": videos_dir / f"{filename}_detected.mp4" if save_video else None,
        "labels_dir": labels_dir if save_txt else None,
        "images_dir": images_dir if save_img else None,
        "mot_labels_dir": mot_dir if save_mot_format else None
    }
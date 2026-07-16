#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microsphere motion tracker.
Implements continuous trajectory tracking based on position and class.
"""

import numpy as np
import cv2
import math
import colorsys
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
import json
import os
from pathlib import Path

from .track import Track
from .matching import MatchingStrategy
from .motion_model import MotionModel, AdaptiveMotionModel
from .data_association import DataAssociation
from .utils import calculate_distance, calculate_angle, is_bbox_in_matching_region


class MicroBeadTracker:
    """Microsphere motion tracker."""

    def __init__(self,
                disappear_zone_width: int = 100,
                appear_zone_width: int = 100,
                static_neighborhood: int = 5,
                initial_match_range: int = 250,
                max_match_range: int = 500,
                max_lost_frames: int = 5,
                max_history_len: int = 30,
                initial_match_angle: float = 15.0,
                first_miss_range_extend: int = 200,
                later_miss_range_extend: int = 150,
                miss_angle_extend: float = 15.0):
        """
        Initialize tracker.

        Args:
            disappear_zone_width: Disappear zone width
            appear_zone_width: Appear zone width
            static_neighborhood: Static detection neighborhood (pixels)
            initial_match_range: Initial match range
            max_match_range: Maximum match range
            max_lost_frames: Max consecutive lost frames
            max_history_len: Max history length
            initial_match_angle: Initial match angle
            first_miss_range_extend: Range extension on first miss
            later_miss_range_extend: Range extension on later misses
            miss_angle_extend: Angle extension on miss
        """
        self.next_id = 1
        self.tracks = []
        self.inactive_tracks = []

        self.disappear_zone_width = disappear_zone_width
        self.appear_zone_width = appear_zone_width
        self.static_neighborhood = static_neighborhood
        self.initial_match_range = initial_match_range
        self.max_match_range = max_match_range
        self.max_lost_frames = max_lost_frames
        self.max_history_len = max_history_len

        self.initial_match_angle = initial_match_angle
        self.first_miss_range_extend = first_miss_range_extend
        self.later_miss_range_extend = later_miss_range_extend
        self.miss_angle_extend = miss_angle_extend

        self.matching_strategy = MatchingStrategy(
            disappear_zone_width=disappear_zone_width,
            appear_zone_width=appear_zone_width,
            static_threshold=static_neighborhood,
            max_match_range=max_match_range,
            initial_match_range=initial_match_range,
            initial_match_angle=initial_match_angle,
            miss_angle_extend=miss_angle_extend
        )

        self.motion_model = AdaptiveMotionModel(
            static_threshold=static_neighborhood,
            velocity_smooth_factor=0.7,
            direction_smooth_factor=0.3,
            acceleration_threshold=0.5
        )

        self.data_association = DataAssociation()

        self.track_history = []
        self.recovery_log = []

        self.frame_width = None
        self.frame_height = None
        self.current_frame_id = 0

        self.detection_count = 0
        self.total_matched_count = 0
        self.total_unmatched_count = 0

        self.prev_detections_summary: List[Dict[str, Any]] = []

    def initialize_with_first_frame(self, detections: List[Dict], frame_width: int, frame_height: int) -> None:
        """Initialize tracker with first frame."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.current_frame_id = 1
        self.detection_count += len(detections)

        if not detections:
            return

        sorted_detections = sorted(detections, key=lambda det: (det['bbox'][0] + det['bbox'][2]) / 2)

        for det in sorted_detections:
            bbox = det['bbox']
            class_id = det['class_id']
            conf = det['confidence']

            center_x = (bbox[0] + bbox[2]) / 2
            left_edge = bbox[0]

            if left_edge <= self.disappear_zone_width:
                continue

            track = Track(
                self.next_id,
                bbox,
                class_id,
                self.current_frame_id,
                conf,
                max_history_len=self.max_history_len,
                initial_match_range=self.initial_match_range,
                initial_match_angle=self.initial_match_angle,
                first_miss_range_extend=self.first_miss_range_extend,
                later_miss_range_extend=self.later_miss_range_extend
            )

            track.update_zone_status(self.disappear_zone_width, self.appear_zone_width, self.frame_width)
            self.tracks.append(track)
            self.track_history.append(track.get_state_dict())
            self.next_id += 1

    def update(self, detections: List[Dict], frame_id: int) -> List[Dict]:
        """
        Update tracker state.

        Args:
            detections: Current frame detections
            frame_id: Current frame ID

        Returns:
            Track states for current frame
        """
        self.current_frame_id = frame_id
        self.detection_count += len(detections)

        if not self.tracks:
            if self.frame_width is None:
                raise ValueError("Frame size not initialized, call initialize_with_first_frame first")
            self.initialize_with_first_frame(detections, self.frame_width, self.frame_height)
            return self.get_current_tracks_state()

        filtered_detections = self.matching_strategy.filter_disappear_zone(detections, self.frame_width)

        def _center_of(b):
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

        def _is_static_det(curr_cx, curr_cy, cls_id) -> bool:
            for prev in self.prev_detections_summary:
                if prev['class_id'] != cls_id:
                    continue
                if max(abs(curr_cx - prev['center_x']), abs(curr_cy - prev['center_y'])) <= float(self.static_neighborhood):
                    return True
            return False

        for det in filtered_detections:
            cx, cy = _center_of(det['bbox'])
            det['center_x'] = cx
            det['center_y'] = cy
            det['is_static_detection'] = _is_static_det(cx, cy, det['class_id'])

        used_det_indices: Set[int] = set()
        pre_matches: Dict[int, int] = {}
        track_id_to_track = {t.id: t for t in self.tracks}

        for det_idx, det in enumerate(filtered_detections):
            if not det.get('is_static_detection', False):
                continue
            cx, cy = det['center_x'], det['center_y']
            cls_id = det['class_id']
            best_track = None
            best_dist = 1e9
            for t in self.tracks:
                if t.class_id != cls_id:
                    continue
                d = max(abs(cx - t.center_x), abs(cy - t.center_y))
                if d <= float(self.static_neighborhood) and d < best_dist:
                    best_dist = d
                    best_track = t
            if best_track is not None:
                pre_matches[best_track.id] = det_idx
                used_det_indices.add(det_idx)

        if self.tracks:
            tracks_to_match = [t for t in self.tracks if t.id not in pre_matches]
            base_states = [t.get_state_dict() for t in tracks_to_match]

            dets_for_matching = []
            dets_map_indices = []
            for idx, det in enumerate(filtered_detections):
                if idx in used_det_indices:
                    continue
                if det.get('is_static_detection', False):
                    continue
                dets_for_matching.append(det)
                dets_map_indices.append(idx)

            matches: Dict[int, int] = {}

            round1_states = []
            for ts in base_states:
                if ts.get('is_lost', False):
                    continue
                ts1 = dict(ts)
                ts1['match_range'] = ts.get('initial_match_range', self.initial_match_range)
                ts1['match_angle'] = ts.get('initial_match_angle', self.initial_match_angle)
                round1_states.append(ts1)

            used_rel_det_round1 = set()
            if round1_states and dets_for_matching:
                r1_matches_rel, r1_unmatched_track_idx, r1_unmatched_det_rel_idx = self.matching_strategy.find_best_matches(
                    round1_states, dets_for_matching
                )
                for track_id, rel_did in r1_matches_rel.items():
                    det_idx_global = dets_map_indices[rel_did]
                    matches[track_id] = det_idx_global
                    used_rel_det_round1.add(rel_did)

            matched_ids_round1 = set(matches.keys())
            tracks_round2 = [t for t in tracks_to_match if t.id not in matched_ids_round1]
            lost_states_round2 = [t.get_state_dict() for t in tracks_round2 if t.is_lost]

            dets_for_round2 = [d for i, d in enumerate(dets_for_matching) if i not in used_rel_det_round1]
            dets_for_round2_map = [dets_map_indices[i] for i in range(len(dets_for_matching)) if i not in used_rel_det_round1]

            if lost_states_round2 and dets_for_round2:
                r2_matches_rel, r2_unmatched_track_idx, r2_unmatched_det_rel_idx = self.matching_strategy.find_best_matches(
                    lost_states_round2, dets_for_round2
                )
                for track_id, rel_did in r2_matches_rel.items():
                    det_idx_global = dets_for_round2_map[rel_did]
                    matches[track_id] = det_idx_global

            for tid, didx in pre_matches.items():
                matches[tid] = didx

            for track_id in matches.keys():
                track_obj = self._get_track_by_id(track_id)
                track_state = track_obj.get_state_dict() if track_obj is not None else None
                if track_state and track_state.get('is_lost', False):
                    det_idx = matches[track_id]
                    det = filtered_detections[det_idx]
                    det_center = ((det['bbox'][0] + det['bbox'][2])/2, (det['bbox'][1] + det['bbox'][3])/2)
                    track_center = (track_state['center_x'], track_state['center_y'])
                    recovery_info = {
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'missed_frames': track_state.get('missed_frames', 0),
                        'original_class': track_state['class_id'],
                        'matched_class': det['class_id'],
                        'match_distance': calculate_distance(track_center, det_center),
                        'class_match': track_state['class_id'] == det['class_id']
                    }
                    self.recovery_log.append(recovery_info)

            self.total_matched_count += len(matches)
            unmatched_ids = set(t.id for t in tracks_to_match) - set(matches.keys())
            self.total_unmatched_count += len(unmatched_ids)

            tracks_to_remove = []
            for track_idx, det_idx in matches.items():
                track = self._get_track_by_id(track_idx)
                det = filtered_detections[det_idx]
                if track:
                    track.update(det['bbox'], det['class_id'], frame_id, det['confidence'])
                    track.update_zone_status(self.disappear_zone_width, self.appear_zone_width, self.frame_width)
                    if track.should_terminate_in_disappear_zone():
                        tracks_to_remove.append(track)
                        track_state = track.get_state_dict()
                        track_state['is_terminated'] = True
                        track_state['termination_reason'] = 'disappear_zone_timeout'
                        self.track_history.append(track_state)
                        continue
                    self.track_history.append(track.get_state_dict())

            for track in tracks_to_remove:
                if track in self.tracks:
                    self.tracks.remove(track)
                    self.inactive_tracks.append(track)

            tracks_to_remove_unmatched = []
            for track in tracks_to_match:
                if track.id in matches:
                    continue
                track.update_zone_status(self.disappear_zone_width, self.appear_zone_width, self.frame_width)
                if track.should_terminate_in_disappear_zone():
                    tracks_to_remove_unmatched.append(track)
                    track_state = track.get_state_dict()
                    track_state['is_terminated'] = True
                    track_state['termination_reason'] = 'disappear_zone_timeout'
                    self.track_history.append(track_state)
                    continue

                track.mark_missed()
                if track.missed_frames > self.max_lost_frames:
                    tracks_to_remove_unmatched.append(track)
                    track_state = track.get_state_dict()
                    track_state['is_terminated'] = True
                    track_state['termination_reason'] = 'max_lost_frames'
                    self.track_history.append(track_state)
                else:
                    self.track_history.append(track.get_state_dict())

            for track in tracks_to_remove_unmatched:
                if track in self.tracks:
                    self.tracks.remove(track)
                    self.inactive_tracks.append(track)

            unmatched_detections = []
            used_all_detection_indices = set(matches.values()) if 'matches' in locals() else set()
            for i in range(len(filtered_detections)):
                if i not in used_all_detection_indices:
                    unmatched_detections.append(filtered_detections[i])
        else:
            unmatched_detections = filtered_detections

        new_track_detections = self.matching_strategy.find_new_tracks_in_appear_zone(
            unmatched_detections, set(), self.frame_width
        )

        for det in new_track_detections:
            bbox = det['bbox']
            class_id = det['class_id']
            conf = det['confidence']
            left_edge = bbox[0]
            if left_edge <= self.disappear_zone_width:
                continue

            track = Track(
                self.next_id, bbox, class_id, frame_id, conf,
                max_history_len=self.max_history_len,
                initial_match_range=self.initial_match_range,
                initial_match_angle=self.initial_match_angle,
                first_miss_range_extend=self.first_miss_range_extend,
                later_miss_range_extend=self.later_miss_range_extend
            )
            track.update_zone_status(self.disappear_zone_width, self.appear_zone_width, self.frame_width)
            self.tracks.append(track)
            self.track_history.append(track.get_state_dict())
            self.next_id += 1

        if len(self.tracks) > 0:
            high_conf_detections = self.matching_strategy.find_unmatched_high_confidence_detections(
                unmatched_detections,
                {detections.index(det) for det in new_track_detections},
                min_confidence=0.6
            )

            isolated_high_conf_detections = []
            for det in high_conf_detections:
                det_center = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)
                is_isolated = True
                for track in self.tracks:
                    track_pos = (track.center_x, track.center_y)
                    if calculate_distance(track_pos, det_center) < self.initial_match_range * 1.5:
                        is_isolated = False
                        break
                if is_isolated:
                    isolated_high_conf_detections.append(det)

            for det in isolated_high_conf_detections:
                bbox = det['bbox']
                class_id = det['class_id']
                conf = det['confidence']
                left_edge = bbox[0]
                if left_edge <= self.disappear_zone_width:
                    continue

                track = Track(
                    self.next_id, bbox, class_id, frame_id, conf,
                    max_history_len=self.max_history_len,
                    initial_match_range=self.initial_match_range,
                    initial_match_angle=self.initial_match_angle,
                    first_miss_range_extend=self.first_miss_range_extend,
                    later_miss_range_extend=self.later_miss_range_extend
                )
                track.update_zone_status(self.disappear_zone_width, self.appear_zone_width, self.frame_width)
                self.tracks.append(track)
                self.track_history.append(track.get_state_dict())
                self.next_id += 1

        self.prev_detections_summary = [
            {
                'center_x': ((det['bbox'][0] + det['bbox'][2]) / 2.0),
                'center_y': ((det['bbox'][1] + det['bbox'][3]) / 2.0),
                'class_id': det['class_id']
            }
            for det in filtered_detections
        ]

        return self.get_current_tracks_state()

    def _get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        for track in self.tracks:
            if track.id == track_id:
                return track
        return None

    def get_current_tracks_state(self) -> List[Dict]:
        """Get current active track states."""
        results = []
        for track in self.tracks:
            state = track.get_state_dict()
            state['frame_id'] = self.current_frame_id
            results.append(state)
        return results

    def get_tracking_statistics(self) -> Dict:
        """Get tracking statistics."""
        track_lengths = {}
        for track_state in self.track_history:
            track_id = track_state['track_id']
            track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

        avg_length = sum(track_lengths.values()) / len(track_lengths) if track_lengths else 0
        active_count = len(self.tracks)
        inactive_count = len(self.inactive_tracks)
        static_tracks = sum(1 for track in self.tracks if track.is_static)

        return {
            'total_frames': self.current_frame_id,
            'total_detections': self.detection_count,
            'total_matched': self.total_matched_count,
            'total_unmatched': self.total_unmatched_count,
            'active_tracks': active_count,
            'inactive_tracks': inactive_count,
            'total_tracks': active_count + inactive_count,
            'avg_track_length': avg_length,
            'static_tracks': static_tracks,
            'max_track_length': max(track_lengths.values()) if track_lengths else 0,
            'min_track_length': min(track_lengths.values()) if track_lengths else 0
        }

    def save_results(self, output_dir: str, filename_prefix: str) -> None:
        """Save tracking results to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        tracks_data = {}
        for track_state in self.track_history:
            track_id = track_state['track_id']
            if track_id not in tracks_data:
                tracks_data[track_id] = []
            tracks_data[track_id].append(track_state)

        csv_path = os.path.join(output_dir, f"{filename_prefix}_results.csv")
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write("ID,Class,Duration(frames),Horizontal Displacement(pixels),Vertical Displacement(pixels),Instant Velocity at Disappearance(pixels/frame),Final Vertical Velocity(pixels/frame)\n")

            for track_id, track_frames in tracks_data.items():
                if not track_frames:
                    continue

                track_frames.sort(key=lambda x: x['frame_id'])
                start_frame = track_frames[0]['frame_id']
                end_frame = track_frames[-1]['frame_id']
                duration = end_frame - start_frame + 1

                class_counts = {}
                for frame in track_frames:
                    class_id = frame['class_id']
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]

                start_pos = (track_frames[0]['center_x'], track_frames[0]['center_y'])
                end_pos = (track_frames[-1]['center_x'], track_frames[-1]['center_y'])
                displacement_x = end_pos[0] - start_pos[0]
                displacement_y = end_pos[1] - start_pos[1]

                if len(track_frames) >= 3:
                    last_three = track_frames[-3:]
                    first_pos = (last_three[0]['center_x'], last_three[0]['center_y'])
                    last_pos = (last_three[-1]['center_x'], last_three[-1]['center_y'])
                    frame_diff = last_three[-1]['frame_id'] - last_three[0]['frame_id']
                    if frame_diff > 0:
                        vel_x = (last_pos[0] - first_pos[0]) / frame_diff
                        vel_y = (last_pos[1] - first_pos[1]) / frame_diff
                        velocity = math.sqrt(vel_x**2 + vel_y**2)
                    else:
                        vel_x = vel_y = velocity = 0.0
                elif len(track_frames) >= 2:
                    first_pos = (track_frames[0]['center_x'], track_frames[0]['center_y'])
                    last_pos = (track_frames[-1]['center_x'], track_frames[-1]['center_y'])
                    frame_diff = track_frames[-1]['frame_id'] - track_frames[0]['frame_id']
                    if frame_diff > 0:
                        vel_x = (last_pos[0] - first_pos[0]) / frame_diff
                        vel_y = (last_pos[1] - first_pos[1]) / frame_diff
                        velocity = math.sqrt(vel_x**2 + vel_y**2)
                    else:
                        vel_x = vel_y = velocity = 0.0
                else:
                    vel_x = vel_y = velocity = 0.0

                f.write(f"{track_id},{most_common_class},{duration},{displacement_x:.2f},{displacement_y:.2f},{velocity:.2f},{vel_y:.2f}\n")

        print(f"Tracking results saved to: {csv_path}")


class YOLOTrackPipeline:
    """YOLO detection and tracking pipeline."""

    def __init__(self,
                yolo_model: Any,
                conf_thres: float = 0.3,
                iou_thres: float = 0.4,
                img_size: int = 640,
                device: str = 'auto',
                tracker_type: str = 'motion',
                **tracker_kwargs):
        """
        Initialize pipeline.

        Args:
            yolo_model: YOLO model
            conf_thres: Confidence threshold
            iou_thres: NMS IoU threshold
            img_size: Input image size
            device: Compute device
            tracker_type: Tracker type ('motion' or 'bytetrack')
            tracker_kwargs: Tracker parameters
        """
        self.yolo_model = yolo_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.device = device
        self.tracker_type = tracker_type

        if tracker_type.lower() == 'bytetrack':
            try:
                from .bytetrack_tracker import ByteTrackWrapper
                bytetrack_params = {
                    'track_thresh': tracker_kwargs.pop('track_thresh', 0.25),
                    'track_buffer': tracker_kwargs.pop('track_buffer', 30),
                    'match_thresh': tracker_kwargs.pop('match_thresh', 0.8),
                    'aspect_ratio_thresh': tracker_kwargs.pop('aspect_ratio_thresh', 3.0),
                    'min_box_area': tracker_kwargs.pop('min_box_area', 1.0),
                    'mot20': tracker_kwargs.pop('mot20', False),
                }
                bytetrack_params['disappear_zone_width'] = tracker_kwargs.get('disappear_zone_width', 100)
                bytetrack_params['appear_zone_width'] = tracker_kwargs.get('appear_zone_width', 100)
                self.tracker = ByteTrackWrapper(**bytetrack_params)
                print("Using ByteTrack tracker")
            except ImportError as e:
                print(f"Warning: Cannot import ByteTrack, falling back to Motion tracker: {e}")
                self.tracker = MicroBeadTracker(**tracker_kwargs)
                self.tracker_type = 'motion'
        else:
            self.tracker = MicroBeadTracker(**tracker_kwargs)
            print("Using Motion tracker")

        self.frame_count = 0
        self.detection_count = 0
        self.track_count = 0

    def convert_yolo_results_to_detections(self, results: Any) -> List[Dict]:
        """Convert YOLO results to detection format."""
        detections = []
        if results is None or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())

            if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                detection = {
                    'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    'confidence': confidence,
                    'class_id': class_id,
                }
                detections.append(detection)

        detections.sort(key=lambda x: x['confidence'], reverse=True)
        self.detection_count += len(detections)
        return detections

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process single frame."""
        self.frame_count += 1
        height, width = frame.shape[:2]

        if self.frame_count == 1:
            self.tracker.frame_width = width
            self.tracker.frame_height = height

        device_param = self.device
        if device_param == 'mps':
            try:
                yolo_results = self.yolo_model(frame,
                                            imgsz=self.img_size,
                                            conf=self.conf_thres,
                                            iou=self.iou_thres,
                                            augment=False,
                                            device='mps')
            except Exception:
                yolo_results = self.yolo_model(frame,
                                            imgsz=self.img_size,
                                            conf=self.conf_thres,
                                            iou=self.iou_thres,
                                            augment=False,
                                            device='cpu')
        else:
            yolo_results = self.yolo_model(frame,
                                        imgsz=self.img_size,
                                        conf=self.conf_thres,
                                        iou=self.iou_thres,
                                        augment=False,
                                        device=device_param)

        detections = self.convert_yolo_results_to_detections(yolo_results)
        track_results = self.tracker.update(detections, self.frame_count)
        self.track_count = len(self.tracker.tracks)

        return frame, track_results

    def process_video(self,
                     video_path: str,
                     output_dir: str,
                     save_video: bool = True,
                     show_trajectory: bool = True,
                     line_thickness: int = 2) -> None:
        """Process video file."""
        from .visualization import TrackVisualizer
        from .vis_match_area import MatchingAreaVisualizer

        os.makedirs(output_dir, exist_ok=True)
        video_filename = Path(video_path).stem

        visualizer = TrackVisualizer(
            max_trajectory_points=30,
            disappear_zone_width=self.tracker.disappear_zone_width,
            appear_zone_width=self.tracker.appear_zone_width
        )
        matching_area_visualizer = MatchingAreaVisualizer(max_trajectory_points=30)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_writer = None
        matching_area_video_writer = None

        if save_video:
            output_video_path = os.path.join(output_dir, f"{video_filename}_tracked.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            matching_area_video_path = os.path.join(output_dir, f"{video_filename}_matching_areas.mp4")
            matching_area_video_writer = cv2.VideoWriter(matching_area_video_path, fourcc, fps, (width, height))

        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Processing video")

        frame_count = 0
        all_track_results = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count == 1:
                    self.tracker.frame_width = width
                    self.tracker.frame_height = height

                device_param = self.device
                if device_param == 'mps':
                    try:
                        yolo_results = self.yolo_model(frame,
                                                    imgsz=self.img_size,
                                                    conf=self.conf_thres,
                                                    iou=self.iou_thres,
                                                    augment=False,
                                                    device='mps')
                    except Exception:
                        yolo_results = self.yolo_model(frame,
                                                    imgsz=self.img_size,
                                                    conf=self.conf_thres,
                                                    iou=self.iou_thres,
                                                    augment=False,
                                                    device='cpu')
                else:
                    yolo_results = self.yolo_model(frame,
                                                imgsz=self.img_size,
                                                conf=self.conf_thres,
                                                iou=self.iou_thres,
                                                augment=False,
                                                device=device_param)

                detections = self.convert_yolo_results_to_detections(yolo_results)

                if self.tracker_type == 'bytetrack' and hasattr(self.tracker, 'update'):
                    track_results = self.tracker.update(detections, frame_count, yolo_results=yolo_results[0] if yolo_results else None)
                else:
                    track_results = self.tracker.update(detections, frame_count)
                all_track_results.extend(track_results)

                frame_with_tracks = visualizer.draw_tracks(
                    frame.copy(), track_results, frame_count,
                    draw_matching_region=False,
                    draw_trajectory=show_trajectory,
                    line_thickness=line_thickness
                )

                info_text = f"Frame: {frame_count}/{total_frames} | Tracks: {len(track_results)}"
                cv2.putText(frame_with_tracks, info_text,
                          (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if video_writer is not None:
                    video_writer.write(frame_with_tracks)

                if matching_area_video_writer is not None:
                    all_track_states = []
                    for track in self.tracker.tracks:
                        all_track_states.append(track.get_state_dict())

                    matching_area_frame = frame.copy()

                    for det in detections:
                        bbox = det['bbox']
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        cv2.rectangle(matching_area_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.circle(matching_area_frame, (center_x, center_y), 3, (255, 0, 0), -1)
                        class_id = det['class_id']
                        conf = det['confidence']
                        label = f"C{class_id}:{conf:.2f}"
                        cv2.putText(matching_area_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                    matching_area_frame = matching_area_visualizer.visualize_all_matching_areas(
                        matching_area_frame, all_track_states,
                        alpha=0.3, show_trajectory=True, line_thickness=line_thickness
                    )

                    for track_state in all_track_states:
                        track_id = track_state['track_id']
                        center_x = int(track_state['center_x'])
                        center_y = int(track_state['center_y'])
                        track_color = matching_area_visualizer.get_track_color(track_id)
                        cv2.circle(matching_area_frame, (center_x, center_y), 5, track_color, -1)

                        is_lost = track_state.get('is_lost', False)
                        missed_frames = track_state.get('missed_frames', 0)
                        match_range = track_state.get('match_range', 150)
                        match_angle = track_state.get('match_angle', 15)

                        status_text = f"ID:{track_id}"
                        if is_lost:
                            status_text += f" Lost:{missed_frames}"
                        status_text += f" R:{int(match_range)} A:{int(match_angle)}"
                        cv2.putText(matching_area_frame, status_text,
                                   (center_x - 30, center_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)

                    cv2.putText(matching_area_frame, f"Frame: {frame_count}/{total_frames}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(matching_area_frame, "Blue boxes: Detections | Colored areas: Matching regions",
                               (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    matching_area_video_writer.write(matching_area_frame)

                pbar.update(1)

        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
            if matching_area_video_writer is not None:
                matching_area_video_writer.release()
            pbar.close()

        self.tracker.save_results(output_dir, video_filename)

        trajectory_plot_path = os.path.join(output_dir, f"{video_filename}_trajectories.png")
        visualizer.plot_trajectories(all_track_results, trajectory_plot_path)

        stats = self.tracker.get_tracking_statistics()
        print(f"Processing complete! Processed {frame_count} frames")
        print(f"Detections: {stats['total_detections']}, Matches: {stats['total_matched']}")
        print(f"Active tracks: {stats['active_tracks']}, Inactive tracks: {stats['inactive_tracks']}")
        print(f"Avg track length: {stats['avg_track_length']:.2f} frames")

        if save_video:
            print(f"Tracked video saved to: {output_video_path}")
            matching_area_video_path = os.path.join(output_dir, f"{video_filename}_matching_areas.mp4")
            print(f"Matching area video saved to: {matching_area_video_path}")
        print(f"Trajectory plot saved to: {trajectory_plot_path}")

    def draw_detections(self, frame: np.ndarray, detections: List[Dict], colors: Dict,
                        line_thickness: int = 1) -> np.ndarray:
        """Draw YOLO detections on frame."""
        frame_with_detections = frame.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            class_id = det['class_id']
            conf = det['confidence']
            color = colors.get(class_id, (0, 255, 0))

            cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, line_thickness)
            label = f"C{class_id}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame_with_detections, (x1, y1 - text_size[1] - 5),
                          (x1 + text_size[0], y1), color, -1)
            cv2.putText(frame_with_detections, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame_with_detections

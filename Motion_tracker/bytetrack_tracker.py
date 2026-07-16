#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ByteTrack wrapper with compatible interface for microsphere tracking.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any
import os

try:
    try:
        from ultralytics.trackers import BYTETracker
    except ImportError:
        try:
            from ultralytics import BYTETracker
        except ImportError:
            from ultralytics.trackers.track import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("Warning: ultralytics BYTETracker not available, install ultralytics>=8.0.0")


class ByteTrackWrapper:
    """ByteTrack wrapper with compatible interface."""

    def __init__(self,
                track_thresh: float = 0.25,
                track_buffer: int = 30,
                match_thresh: float = 0.8,
                aspect_ratio_thresh: float = 3.0,
                min_box_area: float = 1.0,
                mot20: bool = False,
                disappear_zone_width: int = 100,
                appear_zone_width: int = 100):
        """
        Initialize ByteTrack.

        Args:
            track_thresh: Detection confidence threshold
            track_buffer: Track buffer size
            match_thresh: Matching threshold
            aspect_ratio_thresh: Aspect ratio threshold
            min_box_area: Minimum box area
            mot20: Use MOT20 settings
            disappear_zone_width: Disappear zone width (for compat)
            appear_zone_width: Appear zone width (for compat)
        """
        if not BYTETRACK_AVAILABLE:
            raise ImportError("ByteTrack not available, install ultralytics>=8.0.0")

        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.aspect_ratio_thresh = aspect_ratio_thresh
        self.min_box_area = min_box_area
        self.mot20 = mot20

        self.bytetracker = None

        # Try multiple initialization methods
        try:
            self.bytetracker = BYTETracker()
            print("ByteTrack initialized (default params)")
        except Exception as e1:
            try:
                from types import SimpleNamespace
                args = SimpleNamespace()
                args.track_thresh = track_thresh
                args.track_high_thresh = track_thresh
                args.track_low_thresh = track_thresh * 0.5
                args.track_buffer = track_buffer
                args.match_thresh = match_thresh
                args.aspect_ratio_thresh = aspect_ratio_thresh
                args.min_box_area = min_box_area
                args.mot20 = mot20
                args.frame_rate = 30
                args.proximity_thresh = 0.5
                args.appearance_thresh = 0.25

                self.bytetracker = BYTETracker(args=args)
                print("ByteTrack initialized (args object)")
            except Exception as e2:
                import inspect
                try:
                    sig = inspect.signature(BYTETracker.__init__)
                    params = [p for p in sig.parameters.keys() if p != 'self']

                    init_kwargs = {}
                    param_map = {
                        'track_thresh': track_thresh,
                        'track_buffer': track_buffer,
                        'match_thresh': match_thresh,
                        'aspect_ratio_thresh': aspect_ratio_thresh,
                        'min_box_area': min_box_area,
                        'mot20': mot20
                    }

                    for param_name, param_value in param_map.items():
                        if param_name in params:
                            init_kwargs[param_name] = param_value

                    if init_kwargs:
                        self.bytetracker = BYTETracker(**init_kwargs)
                        print(f"ByteTrack initialized (params: {list(init_kwargs.keys())})")
                    else:
                        raise ValueError("No matching parameters found")
                except Exception as e3:
                    raise RuntimeError(
                        f"Failed to initialize BYTETracker.\n"
                        f"Method 1 error: {e1}\n"
                        f"Method 2 error: {e2}\n"
                        f"Method 3 error: {e3}\n"
                        f"Check ultralytics version."
                    )

        self.disappear_zone_width = disappear_zone_width
        self.appear_zone_width = appear_zone_width

        self.frame_width = None
        self.frame_height = None
        self.current_frame_id = 0

        self.track_history = []
        self.detection_count = 0
        self.total_matched_count = 0
        self.total_unmatched_count = 0

        self.tracks = []
        self.inactive_tracks = []
        self.track_id_map = {}

    def _create_mock_results(self, detections: List[Dict]) -> Any:
        """Create mock YOLO Results object for ByteTrack."""
        class MockResults:
            def __init__(self, xyxy, conf, cls, orig_shape):
                self.boxes = MockBoxes(xyxy, conf, cls)
                self.conf = conf
                self.cls = cls
                self.orig_shape = orig_shape
                self.orig_img = None
                self.names = {}

        class MockBoxes:
            def __init__(self, xyxy, conf, cls):
                self.xyxy = xyxy
                self.conf = conf
                self.cls = cls
                self._len = len(xyxy) if xyxy is not None and len(xyxy) > 0 else 0

            def __len__(self):
                return self._len

        if not detections:
            empty_xyxy = np.empty((0, 4), dtype=np.float32)
            empty_conf = np.empty((0,), dtype=np.float32)
            empty_cls = np.empty((0,), dtype=np.int32)
            orig_shape = (self.frame_height, self.frame_width) if self.frame_width else (480, 640)
            return MockResults(empty_xyxy, empty_conf, empty_cls, orig_shape)

        xyxy_list = []
        conf_list = []
        cls_list = []

        for det in detections:
            bbox = det['bbox']
            xyxy_list.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            conf_list.append(det['confidence'])
            cls_list.append(det['class_id'])

        xyxy = np.array(xyxy_list, dtype=np.float32)
        conf = np.array(conf_list, dtype=np.float32)
        cls = np.array(cls_list, dtype=np.int32)

        orig_shape = (self.frame_height, self.frame_width) if self.frame_width else (480, 640)
        return MockResults(xyxy, conf, cls, orig_shape)

    def initialize_with_first_frame(self, detections: List[Dict], frame_width: int, frame_height: int) -> None:
        """Initialize with first frame."""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.current_frame_id = 1
        self.detection_count += len(detections)

    def update(self, detections: List[Dict], frame_id: int, yolo_results: Any = None) -> List[Dict]:
        """
        Update tracker state.

        Args:
            detections: Current frame detections
            frame_id: Current frame ID
            yolo_results: Raw YOLO results (optional)

        Returns:
            Track states for current frame
        """
        self.current_frame_id = frame_id
        self.detection_count += len(detections)

        if self.frame_width is None and detections:
            bbox = detections[0]['bbox']
            self.frame_width = int(max(bbox[2] * 1.1, 640))
            self.frame_height = int(max(bbox[3] * 1.1, 480))

        if yolo_results is not None:
            if not hasattr(yolo_results, 'conf') and hasattr(yolo_results, 'boxes') and yolo_results.boxes is not None:
                if hasattr(yolo_results.boxes, 'conf'):
                    conf_tensor = yolo_results.boxes.conf
                    if hasattr(conf_tensor, 'cpu'):
                        yolo_results.conf = conf_tensor.cpu().numpy()
                    elif hasattr(conf_tensor, 'numpy'):
                        yolo_results.conf = conf_tensor.numpy()
                    else:
                        yolo_results.conf = np.array(conf_tensor, dtype=np.float32)

            try:
                online_targets = self.bytetracker.update(yolo_results)
            except Exception as e:
                print(f"Warning: YOLO results failed, trying mock: {e}")
                mock_results = self._create_mock_results(detections)
                try:
                    online_targets = self.bytetracker.update(mock_results)
                except Exception as e2:
                    print(f"Warning: Mock results also failed: {e2}")
                    online_targets = []
        else:
            mock_results = self._create_mock_results(detections)
            try:
                online_targets = self.bytetracker.update(mock_results)
            except Exception as e:
                print(f"Warning: ByteTrack update failed: {e}")
                online_targets = []

        track_results = []
        current_track_ids = set()

        for track in online_targets:
            try:
                track_id = int(track.track_id) if hasattr(track, 'track_id') else int(track.id)
            except:
                track_id = int(getattr(track, 'id', 0))

            try:
                if hasattr(track, 'tlbr'):
                    bbox = track.tlbr.tolist() if hasattr(track.tlbr, 'tolist') else list(track.tlbr)
                elif hasattr(track, 'bbox'):
                    bbox = track.bbox.tolist() if hasattr(track.bbox, 'tolist') else list(track.bbox)
                else:
                    bbox = track.xyxy.tolist() if hasattr(track, 'xyxy') else [0, 0, 0, 0]
            except:
                bbox = [0, 0, 0, 0]

            try:
                class_id = int(track.cls) if hasattr(track, 'cls') else int(getattr(track, 'class_id', 0))
            except:
                class_id = 0

            try:
                conf = float(track.conf) if hasattr(track, 'conf') else float(getattr(track, 'confidence', 0.0))
            except:
                conf = 0.0

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            velocity_x = 0.0
            velocity_y = 0.0
            if track_id in self.track_id_map:
                prev_pos = self.track_id_map[track_id].get('last_position')
                if prev_pos:
                    frame_diff = frame_id - self.track_id_map[track_id].get('last_frame', frame_id - 1)
                    if frame_diff > 0:
                        velocity_x = (center_x - prev_pos[0]) / frame_diff
                        velocity_y = (center_y - prev_pos[1]) / frame_diff

            if track_id not in self.track_id_map:
                self.track_id_map[track_id] = {
                    'start_frame': frame_id,
                    'class_id': class_id,
                    'positions': []
                }

            self.track_id_map[track_id]['last_frame'] = frame_id
            self.track_id_map[track_id]['last_position'] = (center_x, center_y)
            self.track_id_map[track_id]['positions'].append((frame_id, center_x, center_y, class_id, conf))

            track_state = {
                'track_id': track_id,
                'class_id': class_id,
                'bbox': bbox,
                'center_x': center_x,
                'center_y': center_y,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'direction': math.degrees(math.atan2(velocity_y, velocity_x)) if (abs(velocity_x) > 1e-6 or abs(velocity_y) > 1e-6) else 0.0,
                'confidence': conf,
                'is_static': False,
                'is_lost': False,
                'missed_frames': 0,
                'frame_id': frame_id
            }

            track_results.append(track_state)
            current_track_ids.add(track_id)
            self.track_history.append(track_state)

        self.total_matched_count += len(track_results)
        self.total_unmatched_count += max(0, len(detections) - len(track_results))
        self.tracks = track_results

        return track_results

    def get_current_tracks_state(self) -> List[Dict]:
        """Get current active track states."""
        return self.tracks

    def get_tracking_statistics(self) -> Dict:
        """Get tracking statistics."""
        track_lengths = {}
        for track_state in self.track_history:
            track_id = track_state['track_id']
            track_lengths[track_id] = track_lengths.get(track_id, 0) + 1

        avg_length = sum(track_lengths.values()) / len(track_lengths) if track_lengths else 0

        active_count = len(self.tracks)
        all_track_ids = set(self.track_id_map.keys())
        inactive_count = len(all_track_ids - {t['track_id'] for t in self.tracks})

        return {
            'total_frames': self.current_frame_id,
            'total_detections': self.detection_count,
            'total_matched': self.total_matched_count,
            'total_unmatched': self.total_unmatched_count,
            'active_tracks': active_count,
            'inactive_tracks': inactive_count,
            'total_tracks': len(all_track_ids),
            'avg_track_length': avg_length,
            'static_tracks': 0,
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
                most_common_class = max(class_counts.items(), key=lambda x: x[1])[0] if class_counts else 0

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microbead track class
Represents a single microbead's trajectory including position, class, and motion state
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import math


class Track:
    """
    Microbead track class
    """
    def __init__(self, track_id: int, bbox: List[float], class_id: int,
                 frame_id: int, conf: float, max_history_len: int = 30,
                 initial_match_range: int = 150,
                 initial_match_angle: float = 15.0,
                 first_miss_range_extend: int = 200,
                 later_miss_range_extend: int = 150):
        """
        Initialize microbead track

        Args:
            track_id: Track ID
            bbox: Bounding box [x1, y1, x2, y2]
            class_id: Class ID
            frame_id: Frame ID
            conf: Confidence score
            max_history_len: Max history length
            initial_match_range: Initial matching range
            initial_match_angle: Initial matching angle range
            first_miss_range_extend: Range extension on first miss
            later_miss_range_extend: Range extension on subsequent misses
        """
        self.id = track_id
        self.class_id = class_id
        self.original_class_id = class_id
        self.confidence = conf

        self.is_active = True
        self.is_lost = False
        self.is_static = False

        self.bbox = bbox
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]

        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.direction = 180.0  # Initial direction: leftward (typical microbead motion)
        self.avg_velocity = 0.0

        self.direction_smooth_factor = 0.3
        self.direction_history = deque(maxlen=5)
        self.max_direction_change = 30.0

        self.initial_match_range = initial_match_range
        self.initial_match_angle = initial_match_angle
        self.first_miss_range_extend = first_miss_range_extend
        self.later_miss_range_extend = later_miss_range_extend

        self.match_range = initial_match_range
        self.match_angle = initial_match_angle

        self.history = []
        self.max_history_len = max_history_len

        self.last_frame_id = frame_id
        self.total_tracked_frames = 1

        self.class_history = {class_id: (1, frame_id)}

        self.in_disappear_zone = False
        self.in_appear_zone = False
        self.disappear_zone_count = 0
        self.appear_zone_count = 0

        self.missed_frames = 0

        self.has_sub_ids = False
        self.sub_ids = []

        self.class_mismatch_count = 0

        self.static_since_frame = None
        self.last_movement_frame = frame_id
        self.static_verification_passed = False

    def update(self, bbox: List[float], class_id: int, frame_id: int, conf: float) -> None:
        """
        Update track with new detection
        """
        time_diff = frame_id - self.last_frame_id

        old_x, old_y = self.center_x, self.center_y
        old_width, old_height = self.width, self.height
        old_class = self.class_id
        old_conf = self.confidence

        self.bbox = bbox
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]

        if time_diff > 0:
            pass

        if time_diff > 0:
            vx = (self.center_x - old_x) / time_diff
            vy = (self.center_y - old_y) / time_diff

            self.velocity_x = vx
            self.velocity_y = vy
            self.avg_velocity = math.sqrt(self.velocity_x**2 + self.velocity_y**2)

            if abs(self.velocity_x) > 1e-6 or abs(self.velocity_y) > 1e-6:
                new_direction = math.degrees(math.atan2(self.velocity_y, self.velocity_x))
                new_direction = (new_direction + 360) % 360

                if self.total_tracked_frames < 5 or self.avg_velocity < 0.5:
                    self.direction = 180.0
                else:
                    old_direction = self.direction
                    angle_diff = ((new_direction - old_direction + 180) % 360) - 180

                    if abs(angle_diff) > self.max_direction_change:
                        if len(self.direction_history) >= 3:
                            recent_directions = list(self.direction_history)[-3:]
                            avg_recent = sum(recent_directions) / len(recent_directions)
                            variance = sum((d - avg_recent) ** 2 for d in recent_directions) / len(recent_directions)
                            std_dev = math.sqrt(variance)

                            if std_dev < 20.0 and abs(((new_direction - avg_recent + 180) % 360) - 180) > 45.0:
                                self.direction = (old_direction + self.direction_smooth_factor * angle_diff) % 360
                            else:
                                self.direction = (old_direction + self.direction_smooth_factor * angle_diff) % 360
                        else:
                            self.direction = (old_direction + self.direction_smooth_factor * angle_diff) % 360
                    else:
                        self.direction = (old_direction + self.direction_smooth_factor * angle_diff) % 360

                    self.direction_history.append(self.direction)
            else:
                if len(self.direction_history) > 0:
                    self.direction = self.direction_history[-1]
                else:
                    self.direction = 180.0

        self._update_class_history(class_id, frame_id)
        self.class_id = self._get_best_class()

        if class_id != self.class_id:
            self.class_mismatch_count += 1
        else:
            self.class_mismatch_count = 0

        self.confidence = conf
        self.last_frame_id = frame_id
        self.total_tracked_frames += 1

        self.is_lost = False
        self.missed_frames = 0
        self.match_range = self.initial_match_range
        self.match_angle = self.initial_match_angle

        self._update_match_range()

    def _update_class_history(self, class_id: int, frame_id: int) -> None:
        """
        Update class history with decay mechanism

        Args:
            class_id: Class ID
            frame_id: Current frame ID
        """
        decay_factor = 0.95

        for cls, (count, last_frame) in list(self.class_history.items()):
            frames_diff = frame_id - last_frame
            decayed_count = count * (decay_factor ** frames_diff)

            if decayed_count < 0.1:
                del self.class_history[cls]
            else:
                self.class_history[cls] = (decayed_count, last_frame)

        if class_id in self.class_history:
            count, _ = self.class_history[class_id]
            self.class_history[class_id] = (count + 1, frame_id)
        else:
            self.class_history[class_id] = (1, frame_id)

    def _get_best_class(self) -> int:
        """
        Get best class based on weighted class history

        Returns:
            Best class ID
        """
        if not self.class_history:
            return self.class_id

        best_class = max(self.class_history.items(), key=lambda x: x[1][0])
        return best_class[0]

    def _update_sub_ids(self, frame_id: int) -> None:
        """
        Update sub-ID state

        Args:
            frame_id: Current frame ID
        """
        for sub_id, (start_frame, position, cls, age) in list(self.sub_ids.items()):
            new_age = age + 1
            self.sub_ids[sub_id] = (start_frame, position, cls, new_age)

            if new_age > self.sub_id_timeout:
                del self.sub_ids[sub_id]
                if not self.sub_ids:
                    self.has_sub_ids = False

    def predict_next_position(self) -> Tuple[float, float]:
        """
        Predict next frame position using linear extrapolation

        Returns:
            Predicted center coordinates (x, y)
        """
        pred_x = self.center_x + self.velocity_x
        pred_y = self.center_y + self.velocity_y
        return pred_x, pred_y

    def mark_missed(self) -> None:
        """
        Mark track as missed and expand matching range
        """
        if self.is_static:
            return

        self.missed_frames += 1
        self.is_lost = True
        self._update_match_range()

        if self.in_disappear_zone:
            self.disappear_zone_count += 1

    def _update_match_range(self) -> None:
        """
        Update matching range based on track state
        """
        if self.is_static:
            self.match_range = self.initial_match_range
            self.match_angle = self.initial_match_angle
        elif self.is_lost:
            if self.missed_frames == 1:
                self.match_range = self.initial_match_range + self.first_miss_range_extend
                self.match_angle = self.initial_match_angle * 1.5
            else:
                self.match_range = self.initial_match_range + self.later_miss_range_extend
                self.match_angle = self.initial_match_angle * 1.2

            max_range = self.initial_match_range * 3
            if self.match_range > max_range:
                self.match_range = max_range
        else:
            self.match_range = self.initial_match_range
            self.match_angle = self.initial_match_angle

    def get_matching_region_corners(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box corners of the matching region

        Returns:
            Top-left and bottom-right coordinates (x1, y1, x2, y2)
        """
        angle1_rad = math.radians(180 - self.match_angle)
        angle2_rad = math.radians(180 + self.match_angle)

        apex_x, apex_y = self.center_x, self.center_y

        p1_x = self.center_x + self.match_range * math.cos(angle1_rad)
        p1_y = self.center_y + self.match_range * math.sin(angle1_rad)

        p2_x = self.center_x + self.match_range * math.cos(angle2_rad)
        p2_y = self.center_y + self.match_range * math.sin(angle2_rad)

        min_x = min(apex_x, p1_x, p2_x)
        min_y = min(apex_y, p1_y, p2_y)
        max_x = max(apex_x, p1_x, p2_x)
        max_y = max(apex_y, p1_y, p2_y)

        return (min_x, min_y, max_x, max_y)

    def get_state_dict(self) -> Dict:
        """
        Get track state dictionary for saving or transmission

        Returns:
            Track state dictionary
        """
        return {
            'track_id': self.id,
            'class_id': self.class_id,
            'original_class_id': self.original_class_id,
            'bbox': self.bbox,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y,
            'direction': self.direction,
            'is_static': False,
            'confidence': self.confidence,
            'missed_frames': self.missed_frames,
            'is_lost': self.is_lost,
            'total_tracked_frames': self.total_tracked_frames,
            'match_range': self.match_range,
            'initial_match_range': self.initial_match_range,
            'match_angle': self.match_angle,
            'initial_match_angle': self.initial_match_angle,
            'in_disappear_zone': self.in_disappear_zone,
            'in_appear_zone': self.in_appear_zone,
            'class_mismatch_count': self.class_mismatch_count,
            'has_sub_ids': self.has_sub_ids,
            'frame_id': self.last_frame_id
        }

    def update_zone_status(self, disappear_zone_width: int, appear_zone_width: int, frame_width: int) -> None:
        """
        Update track status within special zones

        Args:
            disappear_zone_width: Disappear zone width
            appear_zone_width: Appear zone width
            frame_width: Frame width
        """
        old_in_disappear = self.in_disappear_zone
        self.in_disappear_zone = self.center_x < disappear_zone_width

        if self.in_disappear_zone and not old_in_disappear:
            self.disappear_zone_count = 0

        self.in_appear_zone = frame_width - self.center_x < appear_zone_width

    def should_terminate_in_disappear_zone(self) -> bool:
        """
        Check if track should terminate in the disappear zone

        Returns:
            True if in disappear zone and persisted beyond threshold
        """
        if self.in_disappear_zone:
            self.disappear_zone_count += 1
            if self.disappear_zone_count > 1:
                return True
        else:
            self.disappear_zone_count = 0

        return False

    def is_in_disappear_zone(self, zone_width: int = 100) -> bool:
        """
        Check if track is in the disappear zone

        Args:
            zone_width: Disappear zone width (default 100 pixels)

        Returns:
            True if in disappear zone
        """
        self.in_disappear_zone = self.center_x < zone_width
        return self.in_disappear_zone

    def is_in_appear_zone(self, frame_width: int, zone_width: int = 100) -> bool:
        """
        Check if track is in the appear zone

        Args:
            frame_width: Frame width
            zone_width: Appear zone width (default 100 pixels)

        Returns:
            True if in appear zone
        """
        self.in_appear_zone = frame_width - self.center_x < zone_width
        return self.in_appear_zone

    def should_skip_matching(self) -> bool:
        """
        Check if matching should be skipped for this track

        Returns:
            True if matching should be skipped
        """
        if self.in_disappear_zone and self.disappear_zone_count >= 2:
            return True
        return False

    def is_really_static(self) -> bool:
        """
        Check if the track is genuinely static

        Returns:
            True if genuinely static
        """
        if not self.static_verification_passed:
            return False

        if len(self.position_history) < 3:
            return False

        recent_positions = list(self.position_history)[-3:]
        total_displacement = 0

        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            total_displacement += math.sqrt(dx*dx + dy*dy)

        return total_displacement <= self.static_threshold * 2

    def get_protected_match_range(self) -> float:
        """
        Get match range under protection level

        Returns:
            Protected match range
        """
        if self.static_protection_level == 0:
            return self.match_range

        multiplier = self.static_protection_range_multiplier[self.static_protection_level]
        return self.match_range * multiplier

    def get_protected_match_angle(self) -> float:
        """
        Get match angle under protection level

        Returns:
            Protected match angle
        """
        if self.static_protection_level == 0:
            return self.match_angle

        multiplier = self.static_protection_angle_multiplier[self.static_protection_level]
        return self.match_angle * multiplier

    def get_protected_class_weight(self) -> float:
        """
        Get class weight under protection level

        Returns:
            Protected class weight
        """
        if self.static_protection_level == 0:
            return 1.0

        return self.static_protection_class_weight[self.static_protection_level]
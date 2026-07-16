#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microbead motion model
Implements prediction and update of microbead motion state
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from collections import deque


class MotionModel:
    """
    Microbead motion model
    Implements prediction and update of motion state
    """
    def __init__(self,
                static_threshold: float = 5.0,
                velocity_smooth_factor: float = 0.7,
                direction_smooth_factor: float = 0.3,
                acceleration_threshold: float = 0.5):
        """
        Initialize motion model

        Args:
            static_threshold: Static judgment threshold (pixels)
            velocity_smooth_factor: Velocity smoothing factor
            direction_smooth_factor: Direction smoothing factor
            acceleration_threshold: Acceleration change threshold
        """
        self.static_threshold = static_threshold
        self.velocity_smooth_factor = velocity_smooth_factor
        self.direction_smooth_factor = direction_smooth_factor
        self.acceleration_threshold = acceleration_threshold

    def predict_position(self,
                       center_x: float,
                       center_y: float,
                       velocity_x: float,
                       velocity_y: float,
                       time_delta: int = 1) -> Tuple[float, float]:
        """
        Predict next position using linear extrapolation

        Args:
            center_x: Current x coordinate
            center_y: Current y coordinate
            velocity_x: X-direction velocity
            velocity_y: Y-direction velocity
            time_delta: Time interval

        Returns:
            Predicted position (x, y)
        """
        pred_x = center_x + velocity_x * time_delta
        pred_y = center_y + velocity_y * time_delta
        return pred_x, pred_y

    def update_motion_state(self,
                          old_pos: Tuple[float, float],
                          new_pos: Tuple[float, float],
                          old_velocity: Tuple[float, float],
                          old_direction: float,
                          time_delta: int = 1) -> Tuple[Tuple[float, float], float, bool]:
        """
        Update motion state based on position change

        Args:
            old_pos: Old position (x, y)
            new_pos: New position (x, y)
            old_velocity: Old velocity (vx, vy)
            old_direction: Old direction (degrees)
            time_delta: Time interval

        Returns:
            New velocity, new direction, and static flag
        """
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]

        displacement = math.sqrt(dx * dx + dy * dy)
        is_static = displacement <= self.static_threshold

        if is_static:
            return old_velocity, old_direction, True

        if time_delta > 0:
            new_vx = dx / time_delta
            new_vy = dy / time_delta
        else:
            new_vx, new_vy = 0.0, 0.0

        smooth_vx = self.velocity_smooth_factor * old_velocity[0] + (1 - self.velocity_smooth_factor) * new_vx
        smooth_vy = self.velocity_smooth_factor * old_velocity[1] + (1 - self.velocity_smooth_factor) * new_vy

        if abs(smooth_vx) > 1e-6 or abs(smooth_vy) > 1e-6:
            new_direction = math.degrees(math.atan2(smooth_vy, smooth_vx))
            new_direction = (new_direction + 360) % 360

            if old_direction != 0.0:
                angle_diff = ((new_direction - old_direction + 180) % 360) - 180
                smooth_direction = (old_direction + self.direction_smooth_factor * angle_diff) % 360
            else:
                smooth_direction = new_direction
        else:
            smooth_direction = old_direction

        return (smooth_vx, smooth_vy), smooth_direction, False

    def check_speed_change(self,
                         new_speed: float,
                         speed_history: List[float]) -> bool:
        """
        Check if speed change exceeds threshold

        Args:
            new_speed: New speed
            speed_history: Speed history

        Returns:
            True if speed change is significant
        """
        if not speed_history:
            return False

        avg_speed = sum(speed_history) / len(speed_history)

        if avg_speed > 1e-6:
            speed_change_rate = abs(new_speed - avg_speed) / avg_speed
            return speed_change_rate > self.acceleration_threshold
        else:
            return new_speed > self.static_threshold

    def check_direction_change(self,
                             new_direction: float,
                             old_direction: float) -> bool:
        """
        Check if direction change is significant

        Args:
            new_direction: New direction (degrees)
            old_direction: Old direction (degrees)

        Returns:
            True if direction change is significant (>30 degrees)
        """
        angle_diff = abs(((new_direction - old_direction + 180) % 360) - 180)
        return angle_diff > 30


class AdaptiveMotionModel(MotionModel):
    """
    Adaptive motion model
    Dynamically adjusts parameters based on motion state
    """
    def __init__(self,
                static_threshold: float = 5.0,
                velocity_smooth_factor: float = 0.7,
                direction_smooth_factor: float = 0.3,
                acceleration_threshold: float = 0.5):
        """
        Initialize adaptive motion model
        """
        super().__init__(
            static_threshold=static_threshold,
            velocity_smooth_factor=velocity_smooth_factor,
            direction_smooth_factor=direction_smooth_factor,
            acceleration_threshold=acceleration_threshold
        )
        self.history_weight = 0.8
        self.max_static_count = 3

    def predict_with_history(self,
                            track_history: List[Tuple[int, float, float, int, float]],
                            time_delta: int = 1) -> Tuple[float, float]:
        """
        Predict position based on historical trajectory

        Args:
            track_history: Track history [(frame_id, x, y, class_id, conf), ...]
            time_delta: Time interval

        Returns:
            Predicted position (x, y)
        """
        if len(track_history) < 2:
            return None

        frame_ids = [h[0] for h in track_history]
        positions = [(h[1], h[2]) for h in track_history]

        if len(frame_ids) >= 2 and frame_ids[-1] - frame_ids[-2] > 1:
            if len(positions) >= 3:
                dx_total = positions[-1][0] - positions[-3][0]
                dy_total = positions[-1][1] - positions[-3][1]
                time_total = frame_ids[-1] - frame_ids[-3]

                if time_total > 0:
                    avg_vx = dx_total / time_total
                    avg_vy = dy_total / time_total

                    pred_x = positions[-1][0] + avg_vx * time_delta
                    pred_y = positions[-1][1] + avg_vy * time_delta
                    return pred_x, pred_y

        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        dt = frame_ids[-1] - frame_ids[-2]

        if dt > 0:
            vx = dx / dt
            vy = dy / dt

            pred_x = positions[-1][0] + vx * time_delta
            pred_y = positions[-1][1] + vy * time_delta
            return pred_x, pred_y
        else:
            return positions[-1]

    def is_moving_left(self, track_history: List[Tuple[int, float, float, int, float]]) -> bool:
        """
        Determine if the trajectory is predominantly moving leftward

        Args:
            track_history: Track history

        Returns:
            True if moving leftward
        """
        if len(track_history) < 3:
            return False

        positions = [(h[1], h[2]) for h in track_history]
        total_dx = positions[-1][0] - positions[0][0]

        return total_dx < -5
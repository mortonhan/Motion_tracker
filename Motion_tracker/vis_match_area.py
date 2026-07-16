#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Matching area visualization for debugging track matching.
"""

import cv2
import numpy as np
import math
import colorsys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class MatchingAreaVisualizer:
    """Visualizes triangular matching areas for track debugging."""

    def __init__(self, max_trajectory_points: int = 30):
        """
        Initialize visualizer.

        Args:
            max_trajectory_points: Max trajectory history length
        """
        self.colors = {
            'normal': (0, 255, 0),
            'expanded': (255, 165, 0),
            'lost': (255, 0, 0)
        }

        self.prev_frame_tracks = []
        self.max_trajectory_points = max_trajectory_points
        self.track_colors = {}
        self.trajectory_points = defaultdict(list)

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID."""
        if track_id not in self.track_colors:
            hue = ((track_id * 0.618033988749895) % 1.0)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            self.track_colors[track_id] = (int(b*255), int(g*255), int(r*255))

        return self.track_colors[track_id]

    def draw_matching_area(self,
                          frame: np.ndarray,
                          track: Dict,
                          alpha: float = 0.3) -> np.ndarray:
        """Draw matching area on frame."""
        center_x, center_y = track['center_x'], track['center_y']
        match_range = track.get('match_range', 150.0)
        is_lost = track.get('is_lost', False)
        initial_match_range = track.get('initial_match_range', 150.0)
        track_id = track['track_id']

        track_color = self.get_track_color(track_id)

        if is_lost:
            area_color = self.colors['lost']
        elif match_range > initial_match_range:
            area_color = self.colors['expanded']
        else:
            area_color = self.colors['normal']

        overlay = frame.copy()

        vx = track.get('velocity_x', 0.0)
        vy = track.get('velocity_y', 0.0)
        speed = (vx*vx + vy*vy) ** 0.5
        tracked_frames = track.get('total_tracked_frames', 0)

        # Consistent with matching logic: early/slow tracks use fixed leftward 180°
        if tracked_frames < 5 or speed < 0.5:
            direction = 180.0
        else:
            direction = track.get('direction', 180.0)
            if direction is None or not (0 <= direction <= 360):
                direction = 180.0
        angle_range = track.get('match_angle', 15.0)

        angle1_rad = math.radians(direction - angle_range)
        angle2_rad = math.radians(direction + angle_range)

        p1_x = center_x + match_range * math.cos(angle1_rad)
        p1_y = center_y + match_range * math.sin(angle1_rad)

        p2_x = center_x + match_range * math.cos(angle2_rad)
        p2_y = center_y + match_range * math.sin(angle2_rad)

        triangle_points = np.array([
            [int(center_x), int(center_y)],
            [int(p1_x), int(p1_y)],
            [int(p2_x), int(p2_y)]
        ], np.int32)

        cv2.fillPoly(overlay, [triangle_points], area_color)
        cv2.polylines(overlay, [triangle_points], True, area_color, 2)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, f"ID:{track_id}",
                   (int(center_x) - 40, int(center_y) - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)

        if is_lost:
            cv2.putText(frame, f"Lost: {track.get('missed_frames', 0)}f",
                      (int(center_x) - 30, int(center_y) - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, area_color, 1)

        cv2.putText(frame, f"R:{int(match_range)}",
                   (int(center_x) - 20, int(center_y) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, area_color, 1)

        return frame

    def update_trajectory(self, tracks: List[Dict]) -> None:
        """Update trajectory history."""
        for track in tracks:
            track_id = track['track_id']
            self.trajectory_points[track_id].append((int(track['center_x']), int(track['center_y'])))

            if len(self.trajectory_points[track_id]) > self.max_trajectory_points:
                self.trajectory_points[track_id] = self.trajectory_points[track_id][-self.max_trajectory_points:]

    def draw_trajectories(self, frame: np.ndarray, line_thickness: int = 2) -> None:
        """Draw trajectory lines."""
        for track_id, points in self.trajectory_points.items():
            if len(points) > 1:
                color = self.get_track_color(track_id)
                path_points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [path_points], False, color, line_thickness)

    def visualize_all_matching_areas(self,
                                    frame: np.ndarray,
                                    current_tracks: List[Dict],
                                    alpha: float = 0.3,
                                    show_trajectory: bool = True,
                                    line_thickness: int = 2) -> np.ndarray:
        """Visualize all matching areas from previous frame."""
        result_frame = frame.copy()

        self.update_trajectory(current_tracks)

        if show_trajectory:
            self.draw_trajectories(result_frame, line_thickness)

        if self.prev_frame_tracks:
            lost_tracks = [t for t in self.prev_frame_tracks if t.get('is_lost', False)]
            normal_tracks = [t for t in self.prev_frame_tracks if not t.get('is_lost', False)]

            for track in normal_tracks:
                if not track.get('is_static', False):
                    self.draw_matching_area(result_frame, track, alpha)

            for track in lost_tracks:
                self.draw_matching_area(result_frame, track, alpha)

            self.add_legend(result_frame)

        self.prev_frame_tracks = current_tracks.copy()

        return result_frame

    def add_legend(self, frame: np.ndarray) -> None:
        """Add legend to frame."""
        h, w = frame.shape[:2]

        cv2.rectangle(frame, (w - 200, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - 200, 10), (w - 10, 100), (255, 255, 255), 1)

        cv2.putText(frame, "Matching Area Legend", (w - 190, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.rectangle(frame, (w - 190, 40), (w - 170, 55), self.colors['normal'], -1)
        cv2.putText(frame, "Normal Area", (w - 165, 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (w - 190, 60), (w - 170, 75), self.colors['expanded'], -1)
        cv2.putText(frame, "Expanded Area", (w - 165, 72),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(frame, (w - 190, 80), (w - 170, 95), self.colors['lost'], -1)
        cv2.putText(frame, "Lost Track Area", (w - 165, 92),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

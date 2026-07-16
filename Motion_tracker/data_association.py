#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microbead tracking data association module
Implements track-to-detection association logic
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import math

from .utils import calculate_distance, calculate_iou


class DataAssociation:
    """
    Microbead tracking data association
    Handles track-to-detection association
    """
    def __init__(self):
        pass

    def associate_detections_to_tracks(self,
                                     tracks: List[Dict],
                                     detections: List[Dict],
                                     max_distance: float = 150.0,
                                     iou_threshold: float = 0.3) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm

        Args:
            tracks: Track state list
            detections: Detection list
            max_distance: Maximum matching distance
            iou_threshold: IoU matching threshold

        Returns:
            Matched (track_idx, det_idx) pairs, unmatched track indices, unmatched detection indices
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        distance_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                track_center = (track['center_x'], track['center_y'])
                det_center = ((det['bbox'][0] + det['bbox'][2]) / 2,
                             (det['bbox'][1] + det['bbox'][3]) / 2)

                dist = calculate_distance(track_center, det_center)

                if dist > max_distance:
                    distance_matrix[i, j] = float('inf')
                else:
                    distance_matrix[i, j] = dist

        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(distance_matrix)

            matches = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_detections = list(range(len(detections)))

            for row, col in zip(row_indices, col_indices):
                if distance_matrix[row, col] < max_distance:
                    matches.append((row, col))
                    if row in unmatched_tracks:
                        unmatched_tracks.remove(row)
                    if col in unmatched_detections:
                        unmatched_detections.remove(col)

            return matches, unmatched_tracks, unmatched_detections

        except ImportError:
            return self._greedy_match(distance_matrix, max_distance)

    def _greedy_match(self,
                     distance_matrix: np.ndarray,
                     max_distance: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Greedy matching algorithm (fallback when scipy is unavailable)

        Args:
            distance_matrix: Distance matrix
            max_distance: Maximum matching distance

        Returns:
            Matched pairs, unmatched track indices, unmatched detection indices
        """
        num_tracks, num_detections = distance_matrix.shape

        matches = []
        unmatched_tracks = list(range(num_tracks))
        unmatched_detections = list(range(num_detections))

        while unmatched_tracks and unmatched_detections:
            min_dist = float('inf')
            min_row, min_col = -1, -1

            for i in unmatched_tracks:
                for j in unmatched_detections:
                    dist = distance_matrix[i, j]
                    if dist < min_dist:
                        min_dist = dist
                        min_row, min_col = i, j

            if min_dist > max_distance:
                break

            matches.append((min_row, min_col))
            unmatched_tracks.remove(min_row)
            unmatched_detections.remove(min_col)

        return matches, unmatched_tracks, unmatched_detections

    def associate_with_direction_constraints(self,
                                          tracks: List[Dict],
                                          detections: List[Dict],
                                          max_distance: float = 150.0,
                                          direction_threshold: float = 30.0) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks with direction constraints

        Args:
            tracks: Track state list
            detections: Detection list
            max_distance: Maximum matching distance
            direction_threshold: Direction angle threshold (degrees)

        Returns:
            Matched pairs, unmatched track indices, unmatched detection indices
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            track_center = (track['center_x'], track['center_y'])
            track_direction = track['direction']

            for j, det in enumerate(detections):
                det_center = ((det['bbox'][0] + det['bbox'][2]) / 2,
                             (det['bbox'][1] + det['bbox'][3]) / 2)

                dist = calculate_distance(track_center, det_center)

                if dist > max_distance:
                    cost_matrix[i, j] = float('inf')
                    continue

                dx = det_center[0] - track_center[0]
                dy = det_center[1] - track_center[1]
                move_direction = math.degrees(math.atan2(dy, dx))
                move_direction = (move_direction + 360) % 360

                angle_diff = abs(((move_direction - track_direction + 180) % 360) - 180)

                if angle_diff > direction_threshold and not track.get('is_static', False):
                    cost_matrix[i, j] = dist * (1.0 + angle_diff / 90.0)
                else:
                    cost_matrix[i, j] = dist

        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matches = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_detections = list(range(len(detections)))

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < float('inf'):
                    matches.append((row, col))
                    if row in unmatched_tracks:
                        unmatched_tracks.remove(row)
                    if col in unmatched_detections:
                        unmatched_detections.remove(col)

            return matches, unmatched_tracks, unmatched_detections

        except ImportError:
            return self._greedy_match(cost_matrix, float('inf'))

    def filter_matches_by_class(self,
                              matches: List[Tuple[int, int]],
                              tracks: List[Dict],
                              detections: List[Dict],
                              class_mismatch_threshold: int = 3) -> List[Tuple[int, int]]:
        """
        Filter matching results by class consistency

        Args:
            matches: Matched (track_idx, det_idx) pairs
            tracks: Track state list
            detections: Detection list
            class_mismatch_threshold: Class mismatch tolerance threshold

        Returns:
            Filtered matches
        """
        filtered_matches = []

        for track_idx, det_idx in matches:
            track_class = tracks[track_idx]['class_id']
            det_class = detections[det_idx]['class_id']

            class_mismatch_count = tracks[track_idx].get('class_mismatch_count', 0)

            if track_class == det_class:
                tracks[track_idx]['class_mismatch_count'] = 0
                filtered_matches.append((track_idx, det_idx))
            else:
                class_mismatch_count += 1
                tracks[track_idx]['class_mismatch_count'] = class_mismatch_count

                if class_mismatch_count < class_mismatch_threshold:
                    filtered_matches.append((track_idx, det_idx))

        return filtered_matches
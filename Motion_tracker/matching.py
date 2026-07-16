#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microbead tracking matching strategy module
Implements track-to-detection matching logic
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
import math

from .utils import (
    calculate_distance, calculate_angle, calculate_iou,
    is_bbox_in_matching_region, calculate_matching_score, is_point_in_triangle
)


class MatchingStrategy:
    """
    Microbead tracking matching strategy
    Implements matching logic for different track states
    """
    def __init__(self,
                disappear_zone_width: int = 100,
                appear_zone_width: int = 100,
                static_threshold: int = 5,
                max_match_range: int = 500,
                initial_match_range: int = 150,
                initial_match_angle: float = 15.0,
                miss_angle_extend: float = 15.0):
        """
        Initialize matching strategy

        Args:
            disappear_zone_width: Disappear zone width
            appear_zone_width: Appear zone width
            static_threshold: Static judgment threshold (pixels)
            max_match_range: Maximum matching range
            initial_match_range: Initial matching range
            initial_match_angle: Initial matching angle range
            miss_angle_extend: Angle extension on miss
        """
        self.disappear_zone_width = disappear_zone_width
        self.appear_zone_width = appear_zone_width
        self.static_threshold = static_threshold
        self.max_match_range = max_match_range
        self.initial_match_range = initial_match_range
        self.initial_match_angle = initial_match_angle
        self.miss_angle_extend = miss_angle_extend

        self.iou_threshold = 0.2
        self.direction_threshold = 30

        self.dense_region_threshold = 3
        self.dense_region_radius = 100

    def filter_disappear_zone(self, detections: List[Dict], frame_width: int) -> List[Dict]:
        """
        Filter detections in disappear zone (currently passes through all)

        Args:
            detections: Detection list
            frame_width: Frame width

        Returns:
            Filtered detection list
        """
        return detections

    def find_candidates_in_matching_region(self,
                                       track_state: Dict,
                                       detections: List[Dict],
                                       allow_all_classes: bool = False,
                                       consider_direction: bool = True,
                                       expand_angle: float = 0.0) -> List[Dict]:
        """
        Find candidate detections within matching region

        Args:
            track_state: Track state
            detections: Detection list
            allow_all_classes: Whether to allow all classes
            consider_direction: Whether to consider motion direction
            expand_angle: Angle expansion amount

        Returns:
            Candidate detection list
        """
        candidates = []

        match_range = track_state.get('match_range', self.initial_match_range)
        match_angle = track_state.get('match_angle', self.initial_match_angle)

        track_pos = (track_state['center_x'], track_state['center_y'])
        track_class = track_state['class_id']

        vx = track_state.get('velocity_x', 0.0)
        vy = track_state.get('velocity_y', 0.0)
        speed = math.sqrt(vx * vx + vy * vy)
        tracked_frames = track_state.get('total_tracked_frames', 0)
        if tracked_frames < 5 or speed < 0.5:
            effective_direction = 180.0
        else:
            effective_direction = track_state.get('direction', 180.0)
            if effective_direction is None or not (0 <= effective_direction <= 360):
                effective_direction = 180.0

        for det in detections:
            det_center = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)

            distance = calculate_distance(track_pos, det_center)
            if distance > match_range:
                continue

            if det_center[0] >= track_pos[0]:
                continue

            if not allow_all_classes and det['class_id'] != track_class:
                continue

            if consider_direction:
                angle = calculate_angle(track_pos, det_center)
                angle_diff = abs(((angle - effective_direction + 180) % 360) - 180)

                max_angle_diff = match_angle + expand_angle
                if angle_diff > max_angle_diff:
                    continue

            candidates.append(det)

        return candidates

    def _get_similar_classes(self, class_id: int) -> List[int]:
        """
        Get list of classes similar to the given class

        Args:
            class_id: Class ID

        Returns:
            List of similar class IDs
        """
        similar_class_groups = [
            [0, 3],
            [1, 5],
            [4, 6]
        ]

        for group in similar_class_groups:
            if class_id in group:
                return [cls for cls in group if cls != class_id]

        return []

    def select_best_match(self,
                       track_state: Dict,
                       candidates: List[Dict],
                       consider_direction: bool = True,
                       distance_weight: float = 0.4,
                       direction_weight: float = 0.3,
                       class_weight: float = 0.3) -> Optional[Dict]:
        """
        Select best match from candidates using weighted scoring

        Args:
            track_state: Track state
            candidates: Candidate detection list
            consider_direction: Whether to consider motion direction
            distance_weight, direction_weight, class_weight: Scoring weights

        Returns:
            Best matching detection, or None if no match
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        track_pos = (track_state['center_x'], track_state['center_y'])
        track_class = track_state['class_id']
        is_lost = track_state.get('is_lost', False)
        missed_frames = track_state.get('missed_frames', 0)

        best_score = -1
        best_candidate = None

        for det in candidates:
            det_center = ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2)

            distance = calculate_distance(track_pos, det_center)
            max_distance = track_state.get('match_range', self.initial_match_range)
            distance_score = 1.0 - min(distance / max_distance, 1.0)

            direction_score = 1.0
            if consider_direction and track_state.get('direction') is not None:
                angle = calculate_angle(track_pos, det_center)
                track_direction = track_state['direction']
                angle_diff = abs(((angle - track_direction + 180) % 360) - 180)
                max_angle = track_state.get('match_angle', self.initial_match_angle)
                direction_score = 1.0 - min(angle_diff / max_angle, 1.0)

            if det['class_id'] == track_class:
                class_score = 1.0
            else:
                class_score = 0.5

            total_score = (distance_weight * distance_score +
                          direction_weight * direction_score +
                          class_weight * class_score)

            if total_score > best_score:
                best_score = total_score
                best_candidate = det

        return best_candidate

    def is_dense_region(self, active_tracks: List[Dict], position: Tuple[float, float]) -> bool:
        """
        Check if a position is within a dense region

        Args:
            active_tracks: Active track list
            position: Position (x, y)

        Returns:
            True if in dense region
        """
        count = 0
        for track in active_tracks:
            track_pos = (track['center_x'], track['center_y'])
            if calculate_distance(track_pos, position) <= self.dense_region_radius:
                count += 1

        return count >= self.dense_region_threshold

    def find_best_matches(self,
                     active_tracks: List[Dict],
                     detections: List[Dict]) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Find best matching detections for active tracks

        Args:
            active_tracks: Active track list (already filtered)
            detections: Detection list

        Returns:
            Matches dict {track_id: detection_index}, unmatched track indices, unmatched detection indices
        """
        matches = {}
        used_detections = set()

        if not active_tracks or not detections:
            unmatched_track_indices = list(range(len(active_tracks)))
            unmatched_det_indices = list(range(len(detections)))
            return matches, unmatched_track_indices, unmatched_det_indices

        sorted_tracks = sorted(active_tracks, key=lambda x: x.get('matching_priority', 0))

        disappear_zone_track_indices = []
        for idx, track in enumerate(sorted_tracks):
            if track.get('in_disappear_zone', False):
                disappear_zone_track_indices.append(idx)

        normal_tracks = [t for t in sorted_tracks if not t.get('is_lost', False)]
        lost_tracks = [t for t in sorted_tracks if t.get('is_lost', False)]

        non_dense_tracks = []
        dense_region_tracks = []

        for track in normal_tracks:
            pos = (track['center_x'], track['center_y'])
            if self.is_dense_region(active_tracks, pos):
                dense_region_tracks.append(track)
            else:
                non_dense_tracks.append(track)

        # Phase 1: Non-dense normal tracks
        for track in non_dense_tracks:
            track_id = track['track_id']

            candidates = self.find_candidates_in_matching_region(
                track,
                [det for idx, det in enumerate(detections) if idx not in used_detections],
                consider_direction=True
            )

            if candidates:
                best_match = self.select_best_match(
                    track, candidates,
                    consider_direction=True
                )
                if best_match:
                    match_idx = detections.index(best_match)
                    matches[track_id] = match_idx
                    used_detections.add(match_idx)

        # Phase 2: Dense region normal tracks
        for track in dense_region_tracks:
            track_id = track['track_id']

            if track_id in matches:
                continue

            candidates = self.find_candidates_in_matching_region(
                track,
                [det for idx, det in enumerate(detections) if idx not in used_detections],
                consider_direction=True
            )

            if candidates:
                best_match = self.select_best_match(
                    track, candidates,
                    consider_direction=True
                )
                if best_match:
                    match_idx = detections.index(best_match)
                    matches[track_id] = match_idx
                    used_detections.add(match_idx)

        # Phase 3: Short-term lost tracks (1-2 frames, skip disappear zone tracks)
        short_lost_tracks = [t for t in lost_tracks if t.get('missed_frames', 0) < 3 and not t.get('in_disappear_zone', False)]
        for track in short_lost_tracks:
            track_id = track['track_id']
            candidates = self.find_candidates_in_matching_region(
                track,
                [det for idx, det in enumerate(detections) if idx not in used_detections],
                allow_all_classes=True,
                consider_direction=True,
                expand_angle=0
            )

            if candidates:
                best_match = self.select_best_match(
                    track, candidates,
                    consider_direction=True
                )
                if best_match:
                    match_idx = detections.index(best_match)
                    matches[track_id] = match_idx
                    used_detections.add(match_idx)

        # Phase 4: Long-lost tracks last-chance recovery
        long_lost_tracks = [t for t in lost_tracks if t.get('missed_frames', 0) >= 3 and not t.get('in_disappear_zone', False)]
        if long_lost_tracks:
            unmatched_dets = [det for i, det in enumerate(detections) if i not in used_detections]
            recovery_matches = self.last_chance_recovery(long_lost_tracks, unmatched_dets)

            for track_id, rel_det_idx in recovery_matches.items():
                det = unmatched_dets[rel_det_idx]
                original_det_idx = detections.index(det)
                matches[track_id] = original_det_idx
                used_detections.add(original_det_idx)

        matched_track_ids = set(matches.keys())
        unmatched_track_indices = []

        for i, track in enumerate(active_tracks):
            if track['track_id'] not in matched_track_ids:
                unmatched_track_indices.append(i)

        unmatched_det_indices = [i for i in range(len(detections)) if i not in used_detections]

        return matches, unmatched_track_indices, unmatched_det_indices

    def find_new_tracks_in_appear_zone(self,
                                detections: List[Dict],
                                used_detection_indices: Set[int],
                                frame_width: int) -> List[Dict]:
        """
        Find detections in appear zone that can create new tracks

        Args:
            detections: Detection list
            used_detection_indices: Set of used detection indices
            frame_width: Frame width

        Returns:
            Detections that can create new tracks
        """
        new_track_detections = []

        for i, det in enumerate(detections):
            if i in used_detection_indices:
                continue

            bbox = det['bbox']
            center_x = (bbox[0] + bbox[2]) / 2

            if frame_width - center_x <= self.appear_zone_width:
                new_track_detections.append(det)

        if new_track_detections:
            new_track_detections.sort(key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)

        return new_track_detections

    def find_unmatched_high_confidence_detections(self,
                                               detections: List[Dict],
                                               used_detection_indices: Set[int],
                                               min_confidence: float = 0.5) -> List[Dict]:
        """
        Find unmatched high-confidence detections

        Args:
            detections: Detection list
            used_detection_indices: Set of used detection indices
            min_confidence: Minimum confidence threshold

        Returns:
            High-confidence unmatched detections
        """
        high_conf_detections = []

        for i, det in enumerate(detections):
            if i in used_detection_indices:
                continue

            if det['confidence'] >= min_confidence:
                high_conf_detections.append(det)

        if high_conf_detections:
            high_conf_detections.sort(key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)

        return high_conf_detections

    def handle_multiple_matches(self,
                              track: Dict,
                              candidates: List[Dict]) -> Optional[Dict]:
        """
        Handle multiple candidate matches with three-level filtering

        Args:
            track: Track state
            candidates: Candidate detection list

        Returns:
            Best match after filtering
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Direction consistency filter
        if not track.get('is_static', False):
            direction_filtered = []
            track_position = (track['center_x'], track['center_y'])
            track_direction = track['direction']

            for det in candidates:
                bbox = det['bbox']
                det_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                movement_direction = calculate_angle(track_position, det_center)
                angle_diff = abs(((movement_direction - track_direction + 180) % 360) - 180)

                if angle_diff <= self.direction_threshold:
                    direction_filtered.append(det)

            if direction_filtered:
                candidates = direction_filtered

        # Distance priority filter
        track_position = (track['center_x'], track['center_y'])
        distances = []

        for det in candidates:
            bbox = det['bbox']
            det_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            distance = calculate_distance(track_position, det_center)
            distances.append(distance)

        min_distance = min(distances)
        closest_candidates = [
            det for det, dist in zip(candidates, distances)
            if dist <= min_distance * 1.2
        ]

        if closest_candidates:
            candidates = closest_candidates

        # Confidence filter
        if len(candidates) > 1:
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x['confidence'],
                reverse=True
            )
            return sorted_candidates[0]
        else:
            return candidates[0]

    def last_chance_recovery(self,
                      long_lost_tracks: List[Dict],
                      unmatched_detections: List[Dict]) -> Dict[int, int]:
        """
        Last-chance recovery for long-lost tracks (>=3 frames)
        Uses very relaxed conditions

        Args:
            long_lost_tracks: Long-lost track list
            unmatched_detections: Unmatched detection list

        Returns:
            Recovery matches {track_id: detection_index}
        """
        recovery_matches = {}

        if not long_lost_tracks or not unmatched_detections:
            return recovery_matches

        sorted_tracks = sorted(long_lost_tracks, key=lambda x: x.get('missed_frames', 0), reverse=True)
        used_detection_indices = set()

        for track in sorted_tracks:
            track_id = track['track_id']
            track_pos = (track['center_x'], track['center_y'])
            track_class = track['class_id']
            missed_frames = track.get('missed_frames', 0)

            extended_range = min(track['match_range'] * (1.0 + 0.2 * missed_frames), self.max_match_range)
            angle_range = min(15 + missed_frames * 5, 40)

            candidates = []
            for i, det in enumerate(unmatched_detections):
                if i in used_detection_indices:
                    continue

                det_bbox = det['bbox']
                det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
                det_class = det['class_id']

                if det_center[0] >= track_pos[0]:
                    continue

                distance = calculate_distance(track_pos, det_center)
                if distance > extended_range:
                    continue

                angle = calculate_angle(track_pos, det_center)
                center_angle = 180
                angle_diff = abs(((angle - center_angle + 180) % 360) - 180)

                if angle_diff > angle_range:
                    continue

                class_match_score = 0
                if det_class == track_class:
                    class_match_score = 1.0
                else:
                    similar_classes = self._get_similar_classes(track_class)
                    if det_class in similar_classes:
                        class_match_score = 0.5

                recovery_score = (
                    0.5 * (1.0 - min(distance / extended_range, 1.0)) +
                    0.3 * class_match_score +
                    0.2 * (1.0 - min(angle_diff / angle_range, 1.0))
                )

                if recovery_score > 0.4:
                    candidates.append((i, det, recovery_score))

            if candidates:
                candidates.sort(key=lambda x: x[2], reverse=True)
                best_idx, best_det, _ = candidates[0]
                recovery_matches[track_id] = unmatched_detections.index(best_det)
                used_detection_indices.add(best_idx)

        return recovery_matches
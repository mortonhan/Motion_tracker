#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microbead tracking utility functions
Provides distance, angle, IoU, and other common calculations
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Union


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points

    Args:
        point1: Point 1 (x, y)
        point2: Point 2 (x, y)

    Returns:
        Distance between the two points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate angle of the line from point1 to point2 relative to the x-axis

    Args:
        point1: Start point (x, y)
        point2: End point (x, y)

    Returns:
        Angle in degrees (0-360)
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return (angle + 360) % 360


def calculate_angle_difference(angle1: float, angle2: float) -> float:
    """
    Calculate the smallest difference between two angles (handles wrap-around)

    Args:
        angle1: Angle 1 (degrees)
        angle2: Angle 2 (degrees)

    Returns:
        Smallest angle difference (-180 to 180 degrees)
    """
    diff = (angle2 - angle1 + 180) % 360 - 180
    return diff


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes

    Args:
        bbox1: Bounding box 1 [x1, y1, x2, y2]
        bbox2: Bounding box 2 [x1, y1, x2, y2]

    Returns:
        IoU value [0, 1]
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def is_point_in_triangle(point: Tuple[float, float],
                         apex: Tuple[float, float],
                         p1: Tuple[float, float],
                         p2: Tuple[float, float]) -> bool:
    """
    Check if a point is inside a triangle

    Args:
        point: Point to check (x, y)
        apex: Triangle vertex (x, y)
        p1: Triangle vertex 1 (x, y)
        p2: Triangle vertex 2 (x, y)

    Returns:
        True if point is inside the triangle
    """
    v0 = [p2[0] - apex[0], p2[1] - apex[1]]
    v1 = [p1[0] - apex[0], p1[1] - apex[1]]
    v2 = [point[0] - apex[0], point[1] - apex[1]]

    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


def is_bbox_center_in_region(bbox: List[float], region: Tuple[float, float, float, float]) -> bool:
    """
    Check if a bounding box center is within a specified region

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        region: Region (left, top, right, bottom)

    Returns:
        True if center is within the region
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    return (region[0] <= center_x <= region[2] and
            region[1] <= center_y <= region[3])


def is_bbox_in_matching_region(bbox: List[float], track_position: Tuple[float, float],
                              direction: float, match_range: float) -> bool:
    """
    Check if a detection box is within the track's matching region

    Args:
        bbox: Detection box [x1, y1, x2, y2]
        track_position: Track position (x, y)
        direction: Motion direction angle (unused, always uses 180 degrees leftward)
        match_range: Matching region range

    Returns:
        True if inside the triangular matching region
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    apex = track_position

    angle1_rad = math.radians(165)
    angle2_rad = math.radians(195)

    p1_x = track_position[0] + match_range * math.cos(angle1_rad)
    p1_y = track_position[1] + match_range * math.sin(angle1_rad)

    p2_x = track_position[0] + match_range * math.cos(angle2_rad)
    p2_y = track_position[1] + match_range * math.sin(angle2_rad)

    is_in_triangle = is_point_in_triangle((center_x, center_y), apex, (p1_x, p1_y), (p2_x, p2_y))
    is_on_left = center_x < track_position[0]

    return is_in_triangle and is_on_left


def calculate_matching_score(track: Dict, detection: Dict,
                          distance_weight: float = 0.6,
                          direction_weight: float = 0.2,
                          class_weight: float = 0.2) -> float:
    """
    Calculate matching score between a track and a detection
    Note: This function is kept for backward compatibility.
    The new multi-level filtering strategy no longer uses weighted scoring.

    Args:
        track: Track info dictionary
        detection: Detection info dictionary
        distance_weight: Distance weight (for backward compat)
        direction_weight: Direction weight (for backward compat)
        class_weight: Class weight (for backward compat)

    Returns:
        Matching score [0, 1], higher is better
    """
    track_center = (track['center_x'], track['center_y'])
    det_center = ((detection['bbox'][0] + detection['bbox'][2]) / 2,
                 (detection['bbox'][1] + detection['bbox'][3]) / 2)

    distance = calculate_distance(track_center, det_center)
    max_distance = track['match_range']
    distance_score = 1.0 - min(distance / max_distance, 1.0)

    position_score = 0.0
    if det_center[0] < track_center[0]:
        position_score = 1.0
    else:
        position_score = 0.0

    if track['class_id'] == detection['class_id']:
        class_score = 1.0
    else:
        similar_classes = _get_similar_classes(track['class_id'])
        if detection['class_id'] in similar_classes:
            class_score = 0.4
        else:
            class_score = 0.0

    total_score = (distance_weight * distance_score +
                  direction_weight * position_score +
                  class_weight * class_score)

    return total_score

def _get_similar_classes(class_id: int) -> List[int]:
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
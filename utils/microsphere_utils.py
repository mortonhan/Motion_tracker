"""
Microsphere size and color recognition utilities.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union


class ColorClassifier:
    """HSV-based color classifier."""

    def __init__(self, thresholds: Dict[str, List] = None):
        """
        Initialize color classifier.

        Args:
            thresholds: Color thresholds dict
        """
        if thresholds is None:
            self.thresholds = {
                "white": [[0, 0, 200], [180, 30, 255]],
                "red": [[0, 100, 100], [10, 255, 255], [160, 100, 100], [180, 255, 255]],
                "blue": [[100, 100, 100], [140, 255, 255]],
                "black": [[0, 0, 0], [180, 255, 50]],
            }
        else:
            self.thresholds = thresholds

    def classify(self, image: np.ndarray, mask: np.ndarray = None) -> str:
        """
        Classify color of image region.

        Args:
            image: BGR image
            mask: Optional mask

        Returns:
            Color name
        """
        if mask is None:
            mask = np.ones(image.shape[:2], dtype=np.uint8) * 255

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)

        color_scores = {}
        for color_name, ranges in self.thresholds.items():
            if color_name == "others":
                continue

            score = 0
            for i in range(0, len(ranges), 2):
                lower = np.array(ranges[i])
                upper = np.array(ranges[i + 1])
                color_mask = cv2.inRange(masked_hsv, lower, upper)
                overlap = cv2.bitwise_and(color_mask, mask)
                score += cv2.countNonZero(overlap)

            color_scores[color_name] = score

        if max(color_scores.values()) < 10:
            return "unknown"

        return max(color_scores, key=color_scores.get)

    def enhance_color(self, image: np.ndarray, color: str) -> np.ndarray:
        """
        Enhance specific color in image.

        Args:
            image: BGR image
            color: Color name

        Returns:
            Enhanced image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if color not in self.thresholds or color == "others":
            return image

        ranges = self.thresholds[color]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for i in range(0, len(ranges), 2):
            lower = np.array(ranges[i])
            upper = np.array(ranges[i + 1])
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, color_mask)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        result = image.copy()
        result[mask > 0] = cv2.addWeighted(image[mask > 0], 0.5, enhanced[mask > 0], 0.5, 0)

        return result


class SizeMeasurer:
    """Microsphere size measurement."""

    def __init__(self, pixel_to_um_ratio: float = 0.5):
        """
        Initialize size measurer.

        Args:
            pixel_to_um_ratio: Pixel to micrometer ratio
        """
        self.pixel_to_um_ratio = pixel_to_um_ratio

    def measure_diameter(self, bbox: List[float]) -> float:
        """
        Measure microsphere diameter from bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Diameter in micrometers
        """
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        diameter_pixels = (width + height) / 2
        return diameter_pixels * self.pixel_to_um_ratio

    def measure_area(self, mask: np.ndarray) -> float:
        """
        Measure microsphere area from mask.

        Args:
            mask: Binary mask

        Returns:
            Area in square micrometers
        """
        pixel_count = cv2.countNonZero(mask)
        area_pixels = pixel_count
        return area_pixels * (self.pixel_to_um_ratio ** 2)

    def extract_contour_and_radius(self, image: np.ndarray, bbox: List[float]) -> Tuple[np.ndarray, float]:
        """
        Extract contour and equivalent radius.

        Args:
            image: Grayscale image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Contour and equivalent radius (micrometers)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = image[y1:y2, x1:x2]

        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        equivalent_radius_pixels = np.sqrt(area / np.pi)
        equivalent_radius_um = equivalent_radius_pixels * self.pixel_to_um_ratio

        largest_contour[:, :, 0] += x1
        largest_contour[:, :, 1] += y1

        return largest_contour, equivalent_radius_um


class MicrosphereFeaturesExtractor:
    """Combined color and size feature extractor."""

    def __init__(self, color_thresholds: Dict = None, pixel_to_um_ratio: float = 0.5):
        """
        Initialize feature extractor.

        Args:
            color_thresholds: Color thresholds
            pixel_to_um_ratio: Pixel to micrometer ratio
        """
        self.color_classifier = ColorClassifier(color_thresholds)
        self.size_measurer = SizeMeasurer(pixel_to_um_ratio)

    def extract_features(self, image: np.ndarray, bbox: List[float]) -> Dict:
        """
        Extract microsphere features.

        Args:
            image: BGR image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Feature dict with color and size
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return {"color": "unknown", "size": 0.0}

        color = self.color_classifier.classify(roi)
        size = self.size_measurer.measure_diameter(bbox)

        return {"color": color, "size": size}

    def process_detections(self, image: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        Process detections and add color/size info.

        Args:
            image: BGR image
            detections: Detection list

        Returns:
            Enhanced detections
        """
        enhanced_detections = []

        for det in detections:
            bbox = det["bbox"]
            features = self.extract_features(image, bbox)

            enhanced_det = {**det, **features}
            enhanced_detections.append(enhanced_det)

        return enhanced_detections

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Visualize detections with color and size info.

        Args:
            image: BGR image
            detections: Detection list

        Returns:
            Visualized image
        """
        vis_img = image.copy()

        color_map = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "unknown": (0, 255, 0)
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color_name = det.get("color", "unknown")
            size = det.get("size", 0.0)
            confidence = det.get("confidence", 0.0)

            box_color = color_map.get(color_name, (0, 255, 0))
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, 2)

            label = f"{color_name} {size:.1f}um {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - label_h - 5), (x1 + label_w, y1), box_color, -1)

            text_color = (255, 255, 255) if color_name != "white" else (0, 0, 0)
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return vis_img


def plot_size_distribution(sizes: List[float], colors: List[str], output_path: str):
    """Plot microsphere size distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Size (μm)')
    plt.ylabel('Frequency')
    plt.title('Microsphere Size Distribution')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_color_distribution(colors: List[str], output_path: str):
    """Plot microsphere color distribution."""
    from collections import Counter
    color_counts = Counter(colors)

    plt.figure(figsize=(8, 6))
    plt.bar(color_counts.keys(), color_counts.values(), edgecolor='black')
    plt.xlabel('Color')
    plt.ylabel('Count')
    plt.title('Microsphere Color Distribution')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

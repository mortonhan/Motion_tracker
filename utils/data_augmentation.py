"""
Data augmentation for microsphere detection.
Preserves color and size features critical for microsphere analysis.
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path
import os
from typing import Dict, List, Tuple, Union, Optional, Any


class MicrosphereAugmentation:
    """Data augmentation for microsphere detection."""

    def __init__(self, config: Dict = None):
        """
        Initialize augmentation.

        Args:
            config: Augmentation config dict
        """
        self.default_config = {
            "enabled": True,
            "preserve_color": True,
            "preserve_size": True,
            "rotation_range": [-10, 10],
            "brightness_range": [0.8, 1.2],
            "contrast_range": [0.8, 1.2],
            "blur_probability": 0.2,
            "noise_probability": 0.2,
            "flip_probability": 0.5,
        }
        self.config = config if config is not None else self.default_config

    def augment(self, image: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment image and labels.

        Args:
            image: BGR image
            labels: YOLO format labels [class_id, x_center, y_center, width, height]

        Returns:
            Augmented image and labels
        """
        if not self.config["enabled"] or image is None:
            return image, labels

        img = image.copy()
        labs = labels.copy() if labels is not None else None

        if random.random() < 0.5:
            img = self._adjust_brightness_contrast(img)

        if random.random() < self.config["blur_probability"]:
            img = self._apply_blur(img)

        if random.random() < self.config["noise_probability"]:
            img = self._add_noise(img)

        if random.random() < self.config["flip_probability"]:
            img, labs = self._horizontal_flip(img, labs)

        if not self.config["preserve_size"] and random.random() < 0.3:
            img, labs = self._rotate_image(img, labs)

        return img, labs

    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Adjust brightness and contrast."""
        alpha = random.uniform(self.config["contrast_range"][0], self.config["contrast_range"][1])
        beta = random.uniform(self.config["brightness_range"][0], self.config["brightness_range"][1])

        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

        if beta > 1:
            brightness_increase = int((beta - 1) * 255 * 0.1)
            adjusted = cv2.add(adjusted, np.array([brightness_increase, brightness_increase, brightness_increase]))
        elif beta < 1:
            brightness_decrease = int((1 - beta) * 255 * 0.1)
            adjusted = cv2.subtract(adjusted, np.array([brightness_decrease, brightness_decrease, brightness_decrease]))

        return adjusted

    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply blur."""
        blur_type = random.choice(["gaussian", "median"])
        kernel_size = random.choice([3, 5])

        if blur_type == "gaussian":
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            return cv2.medianBlur(image, kernel_size)

    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add noise."""
        noise_type = random.choice(["gaussian", "salt_pepper"])
        noisy = image.copy()

        if noise_type == "gaussian":
            row, col, ch = image.shape
            sigma = random.uniform(5, 15)
            gauss = np.random.normal(0, sigma, (row, col, ch))
            noisy = cv2.add(noisy, gauss.astype(np.uint8))
        else:
            s_vs_p = 0.5
            amount = random.uniform(0.01, 0.05)
            out = image.copy()

            num_salt = int(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
            out[coords[0], coords[1], :] = 255

            num_pepper = int(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
            out[coords[0], coords[1], :] = 0

            noisy = out

        return noisy

    def _horizontal_flip(self, image: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Horizontal flip."""
        flipped_img = cv2.flip(image, 1)

        if labels is None:
            return flipped_img, None

        flipped_labels = labels.copy()
        if len(flipped_labels) > 0:
            flipped_labels[:, 1] = 1.0 - flipped_labels[:, 1]

        return flipped_img, flipped_labels

    def _rotate_image(self, image: np.ndarray, labels: np.ndarray = None, max_angle: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and labels."""
        if max_angle is None:
            angle_range = self.config["rotation_range"]
            angle = random.uniform(angle_range[0], angle_range[1])
        else:
            angle = random.uniform(-max_angle, max_angle)

        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        rotation_matrix[0, 2] += new_w / 2 - center[0]
        rotation_matrix[1, 2] += new_h / 2 - center[1]

        rotated_img = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

        if labels is None or len(labels) == 0:
            return rotated_img, labels

        rotated_labels = []
        for label in labels:
            class_id, x_center, y_center, width, height = label

            x_pixel = x_center * w
            y_pixel = y_center * h

            new_x = rotation_matrix[0, 0] * x_pixel + rotation_matrix[0, 1] * y_pixel + rotation_matrix[0, 2]
            new_y = rotation_matrix[1, 0] * x_pixel + rotation_matrix[1, 1] * y_pixel + rotation_matrix[1, 2]

            rad_angle = math.radians(angle)
            new_width = abs(width * math.cos(rad_angle)) + abs(height * math.sin(rad_angle))
            new_height = abs(width * math.sin(rad_angle)) + abs(height * math.cos(rad_angle))

            new_x_center = new_x / new_w
            new_y_center = new_y / new_h

            if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
                rotated_labels.append([class_id, new_x_center, new_y_center, new_width, new_height])

        return rotated_img, np.array(rotated_labels) if rotated_labels else np.array([])

    def batch_augment(self, images: List[np.ndarray], labels_list: List[np.ndarray] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Augment a batch of images."""
        if not self.config["enabled"]:
            return images, labels_list

        if labels_list is None:
            labels_list = [None] * len(images)

        augmented_images = []
        augmented_labels = []

        for img, labs in zip(images, labels_list):
            aug_img, aug_labs = self.augment(img, labs)
            augmented_images.append(aug_img)
            augmented_labels.append(aug_labs)

        return augmented_images, augmented_labels

    def apply_to_dataset(self, dataset_path: Union[str, Path], output_path: Union[str, Path] = None, augmentation_factor: int = 3):
        """Apply augmentation to entire dataset."""
        dataset_path = Path(dataset_path)
        if output_path is None:
            output_path = dataset_path.parent / f"{dataset_path.name}_augmented"
        else:
            output_path = Path(output_path)

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_path / "images", exist_ok=True)
        os.makedirs(output_path / "labels", exist_ok=True)

        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(dataset_path.glob(f"**/*{ext}")))

        if not image_files:
            raise ValueError(f"No image files found in: {dataset_path}")

        print(f"Found {len(image_files)} image files")

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Failed to read image: {img_file}")
                continue

            label_file = dataset_path / "labels" / f"{img_file.stem}.txt"
            if not label_file.exists():
                label_file = img_file.parent / f"{img_file.stem}.txt"

            labels = None
            if label_file.exists():
                try:
                    labels = np.loadtxt(label_file, ndmin=2)
                except Exception as e:
                    print(f"Error reading label file: {label_file}, {e}")

            orig_img_file = output_path / "images" / img_file.name
            cv2.imwrite(str(orig_img_file), img)

            if labels is not None and label_file.exists():
                orig_label_file = output_path / "labels" / f"{img_file.stem}.txt"
                np.savetxt(orig_label_file, labels, fmt="%g")

            for i in range(augmentation_factor):
                aug_img, aug_labels = self.augment(img, labels)

                aug_img_file = output_path / "images" / f"{img_file.stem}_aug{i}{img_file.suffix}"
                cv2.imwrite(str(aug_img_file), aug_img)

                if aug_labels is not None:
                    aug_label_file = output_path / "labels" / f"{img_file.stem}_aug{i}.txt"
                    np.savetxt(aug_label_file, aug_labels, fmt="%g")

        print(f"Augmentation complete, saved to: {output_path}")
        return output_path


if __name__ == "__main__":
    test_img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.circle(test_img, (150, 150), 50, (0, 0, 255), -1)

    test_labels = np.array([[0, 0.5, 0.5, 0.3, 0.3]])

    augmentor = MicrosphereAugmentation()
    aug_img, aug_labels = augmentor.augment(test_img, test_labels)

    print("Original labels:", test_labels)
    print("Augmented labels:", aug_labels)

    cv2.imwrite("original.jpg", test_img)
    cv2.imwrite("augmented.jpg", aug_img)

    print("Augmentation test complete")

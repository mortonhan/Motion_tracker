"""
General utility functions
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

def ensure_dir(path):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def load_image(image_path):
    """Load an image file"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return img

def save_image(image, output_path):
    """Save image to file"""
    ensure_dir(os.path.dirname(output_path))
    cv2.imwrite(str(output_path), image)

def save_results_to_csv(results, output_path, columns=None):
    """Save results to CSV file"""
    ensure_dir(os.path.dirname(output_path))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
    return df

def create_plots(data, output_dir, prefix="plot"):
    """Create visualization charts from detection results"""
    ensure_dir(output_dir)

    # Size distribution
    if "size" in data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(data["size"], bins=20)
        plt.xlabel('Size (μm)')
        plt.ylabel('Frequency')
        plt.title('Microsphere Size Distribution')
        plt.savefig(os.path.join(output_dir, f"{prefix}_size_distribution.png"))
        plt.close()

        if "color" in data.columns:
            plt.figure(figsize=(10, 6))
            for color in data["color"].unique():
                color_sizes = data[data["color"] == color]["size"]
                plt.hist(color_sizes, bins=20, alpha=0.7, label=color)
            plt.xlabel('Size (μm)')
            plt.ylabel('Frequency')
            plt.title('Size Distribution by Color')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{prefix}_size_by_color.png"))
            plt.close()

    # Color distribution
    if "color" in data.columns:
        plt.figure(figsize=(10, 6))
        color_counts = data["color"].value_counts()
        plt.bar(color_counts.index, color_counts.values)
        plt.xlabel('Color')
        plt.ylabel('Count')
        plt.title('Microsphere Color Distribution')
        plt.savefig(os.path.join(output_dir, f"{prefix}_color_distribution.png"))
        plt.close()

    # Confidence distribution
    if "confidence" in data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(data["confidence"], bins=20)
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Detection Confidence Distribution')
        plt.savefig(os.path.join(output_dir, f"{prefix}_confidence_distribution.png"))
        plt.close()

        if "class" in data.columns:
            plt.figure(figsize=(10, 6))
            for cls in data["class"].unique():
                class_conf = data[data["class"] == cls]["confidence"]
                plt.hist(class_conf, bins=20, alpha=0.7, label=f"Class {cls}")
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution by Class')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{prefix}_confidence_by_class.png"))
            plt.close()

def create_confusion_matrix(true_labels, pred_labels, class_names, output_path=None):
    """Create and save a confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def calculate_metrics(true_labels, pred_labels, class_names=None):
    """Calculate evaluation metrics (accuracy, precision, recall, F1)"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
        "f1": f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    }

    if class_names:
        class_precision = precision_score(true_labels, pred_labels, average=None, zero_division=0)
        class_recall = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        class_f1 = f1_score(true_labels, pred_labels, average=None, zero_division=0)

        for i, class_name in enumerate(class_names):
            if i < len(class_precision):
                metrics[f"{class_name}_precision"] = class_precision[i]
                metrics[f"{class_name}_recall"] = class_recall[i]
                metrics[f"{class_name}_f1"] = class_f1[i]

    return metrics

def plot_pr_curve(true_labels, pred_scores, output_path=None):
    """Plot Precision-Recall curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if isinstance(true_labels[0], list):
        true_labels = [np.argmax(label) for label in true_labels]

    if isinstance(pred_scores[0], list):
        n_classes = len(pred_scores[0])

        plt.figure(figsize=(10, 8))

        for i in range(n_classes):
            binary_true = [1 if label == i else 0 for label in true_labels]
            class_scores = [score[i] for score in pred_scores]

            precision, recall, _ = precision_recall_curve(binary_true, class_scores)
            ap = average_precision_score(binary_true, class_scores)

            plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {ap:.2f})')
    else:
        precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
        ap = average_precision_score(true_labels, pred_scores)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, lw=2, label=f'AP = {ap:.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path)
    else:
        plt.show()

    plt.close()

def load_yaml(file_path):
    """Load a YAML file"""
    import yaml
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_yaml(data, file_path):
    """Save data to a YAML file"""
    import yaml
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
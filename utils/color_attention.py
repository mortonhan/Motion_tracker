"""
Color attention mechanism module
Provides color attention for enhanced microsphere color recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional

class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class ColorAttention(nn.Module):
    """Color attention module combining channel and spatial attention for color features"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ColorAttention, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

        # Color-specific convolution kernels
        self.red_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.blue_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.white_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.black_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # Color feature fusion
        self.fusion = nn.Conv2d(in_channels * 4, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        ca_weight = self.ca(x)
        ca_out = x * ca_weight

        sa_weight = self.sa(ca_out)
        sa_out = ca_out * sa_weight

        red_feat = self.red_conv(sa_out)
        blue_feat = self.blue_conv(sa_out)
        white_feat = self.white_conv(sa_out)
        black_feat = self.black_conv(sa_out)

        color_feats = torch.cat([red_feat, blue_feat, white_feat, black_feat], dim=1)
        fused = self.fusion(color_feats)
        fused = self.norm(fused)
        fused = self.relu(fused)

        return x + fused

class HSVAttention(nn.Module):
    """HSV color space attention module"""
    def __init__(self, in_channels):
        super(HSVAttention, self).__init__()
        self.h_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.s_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.v_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.hsv_fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h_feat = self.h_attention(x) * x
        s_feat = self.s_attention(x) * x
        v_feat = self.v_attention(x) * x

        hsv_concat = torch.cat([h_feat, s_feat, v_feat], dim=1)
        hsv_fused = self.hsv_fusion(hsv_concat)
        hsv_fused = self.norm(hsv_fused)
        hsv_fused = self.relu(hsv_fused)

        return x + hsv_fused

class ColorEnhancedFeatureExtractor(nn.Module):
    """Color-enhanced feature extractor"""
    def __init__(self, in_channels, out_channels):
        super(ColorEnhancedFeatureExtractor, self).__init__()
        self.color_attention = ColorAttention(in_channels)
        self.hsv_attention = HSVAttention(in_channels)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.color_attention(x)
        x = self.hsv_attention(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class ColorClassifier(nn.Module):
    """Deep learning based color classifier"""
    def __init__(self, in_channels, num_colors=4):
        super(ColorClassifier, self).__init__()
        self.color_names = ["red", "blue", "white", "black"]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_colors)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

    def predict_color(self, x):
        """Predict color class

        Args:
            x: Input image tensor

        Returns:
            Color name
        """
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            color_idx = predicted.item()
            return self.color_names[color_idx]

def preprocess_image_for_color_attention(image):
    """Preprocess BGR image for color attention module

    Args:
        image: BGR format image

    Returns:
        Preprocessed image tensor
    """
    resized = cv2.resize(image, (224, 224))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    return tensor

def apply_color_attention(model, image):
    """Apply color attention model to an image

    Args:
        model: Color attention model
        image: BGR format image

    Returns:
        Processed image and predicted color
    """
    tensor = preprocess_image_for_color_attention(image)

    with torch.no_grad():
        enhanced_tensor = model(tensor)

    enhanced_np = enhanced_tensor.squeeze(0).permute(1, 2, 0).numpy()
    enhanced_np = (enhanced_np * 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

    return enhanced_bgr
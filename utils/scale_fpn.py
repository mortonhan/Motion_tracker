"""
Scale-aware FPN module for multi-scale microsphere detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class ScaleAttention(nn.Module):
    """Scale attention module for adaptive feature weighting."""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ScaleAttention, self).__init__()
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


class ScaleConvBlock(nn.Module):
    """Scale-specific convolution block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ScaleConvBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ScaleAwareFPN(nn.Module):
    """Scale-aware Feature Pyramid Network."""
    def __init__(self, in_channels_list, out_channels=256):
        super(ScaleAwareFPN, self).__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])

        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(len(in_channels_list))
        ])

        self.scale_attentions = nn.ModuleList([
            ScaleAttention(out_channels)
            for _ in range(len(in_channels_list))
        ])

        self.scale_fusion = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Multi-scale features [P3, P4, P5]

        Returns:
            Enhanced features and fused feature
        """
        laterals = [conv(f) for f, conv in zip(features, self.lateral_convs)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled

        fpn_features = []
        for i, lateral in enumerate(laterals):
            attended = lateral * self.scale_attentions[i](lateral)
            fpn_feat = self.fpn_convs[i](attended)
            fpn_features.append(fpn_feat)

        target_size = fpn_features[0].shape[-2:]
        upsampled_features = [fpn_features[0]]
        for i in range(1, len(fpn_features)):
            upsampled = F.interpolate(fpn_features[i], size=target_size, mode='nearest')
            upsampled_features.append(upsampled)

        fused_feature = torch.cat(upsampled_features, dim=1)
        fused_feature = self.scale_fusion[0](fused_feature[:, :fpn_features[0].shape[1]])

        return fpn_features, fused_feature


class SizeRegressionHead(nn.Module):
    """Microsphere size regression head."""
    def __init__(self, in_channels, hidden_channels=128):
        super(SizeRegressionHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.conv3 = nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return self.activation(x)


class MultiScaleSizePredictor(nn.Module):
    """Multi-scale microsphere size predictor."""
    def __init__(self, in_channels_list=[256, 512, 1024], out_channels=256):
        super(MultiScaleSizePredictor, self).__init__()

        self.scale_fpn = ScaleAwareFPN(in_channels_list, out_channels)

        self.global_size_head = SizeRegressionHead(out_channels)

        self.scale_specific_heads = nn.ModuleList([
            SizeRegressionHead(out_channels)
            for _ in range(len(in_channels_list))
        ])

        self.fusion_conv = nn.Conv2d(out_channels * (len(in_channels_list) + 1), out_channels, kernel_size=1)
        self.final_size_head = SizeRegressionHead(out_channels)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Multi-scale features

        Returns:
            Predicted microsphere size
        """
        fpn_features, fused_feature = self.scale_fpn(features)

        global_size = self.global_size_head(fused_feature)

        scale_sizes = []
        for feat, head in zip(fpn_features, self.scale_specific_heads):
            upsampled = F.interpolate(feat, size=fused_feature.shape[-2:], mode='nearest')
            scale_size = head(upsampled)
            scale_sizes.append(scale_size)

        all_sizes = [global_size] + scale_sizes
        concat_sizes = torch.cat(all_sizes, dim=1)
        fused = self.fusion_conv(concat_sizes)
        final_size = self.final_size_head(fused)

        return final_size


class MicrosphereSizeLoss(nn.Module):
    """Microsphere size regression loss."""
    def __init__(self, smooth_l1_beta=1.0, size_weight=1.0, relative_weight=0.5):
        super(MicrosphereSizeLoss, self).__init__()
        self.smooth_l1_beta = smooth_l1_beta
        self.size_weight = size_weight
        self.relative_weight = relative_weight

    def forward(self, pred, target):
        """
        Compute loss.

        Args:
            pred: Predicted size
            target: Target size

        Returns:
            Loss value
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        abs_diff = torch.abs(pred_flat - target_flat)
        smooth_l1_loss = torch.where(
            abs_diff < self.smooth_l1_beta,
            0.5 * abs_diff ** 2 / self.smooth_l1_beta,
            abs_diff - 0.5 * self.smooth_l1_beta
        ).mean()

        relative_error = torch.abs(pred_flat - target_flat) / (target_flat + 1e-6)
        relative_loss = relative_error.mean()

        total_loss = self.size_weight * smooth_l1_loss + self.relative_weight * relative_loss
        return total_loss


def create_multi_scale_features(feature_maps, target_sizes=None):
    """
    Create multi-scale features.

    Args:
        feature_maps: Feature map list
        target_sizes: Target sizes

    Returns:
        Multi-scale features
    """
    if target_sizes is None:
        return feature_maps

    assert len(feature_maps) == len(target_sizes), "Feature maps and target sizes count mismatch"

    resized_features = []
    for feat, size in zip(feature_maps, target_sizes):
        if feat.shape[-2:] != size:
            resized = F.interpolate(feat, size=size, mode='bilinear', align_corners=False)
            resized_features.append(resized)
        else:
            resized_features.append(feat)

    return resized_features


def extract_roi_features(feature_map, boxes, output_size=(7, 7)):
    """
    Extract ROI features from feature map.

    Args:
        feature_map: Feature map [B, C, H, W]
        boxes: Bounding boxes [N, 4] (x1, y1, x2, y2), normalized to [0, 1]
        output_size: ROI pooling output size

    Returns:
        ROI features [N, C, output_size[0], output_size[1]]
    """
    batch_size = feature_map.shape[0]
    device = feature_map.device

    batch_indices = torch.zeros(boxes.shape[0], dtype=torch.int64, device=device)
    rois = torch.cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)

    from torchvision.ops import roi_pool
    roi_features = roi_pool(feature_map, rois, output_size, spatial_scale=1.0)

    return roi_features

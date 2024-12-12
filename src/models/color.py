import os
from src.models.base import BaseModel
from src.models.experts import ColorExpertModel
import torch.nn as nn
import torch
import src.local_config as cfg

class WeightedColorModel(nn.Module):
    def __init__(self, num_bins, num_classes, device):
        super(WeightedColorModel, self).__init__(num_bins, num_classes, device)
        self.color_fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, locations, gray_images, lab_images):
        preds_semantics, preds_color_base, preds_color_expert = self.forward_base(locations, gray_images, lab_images)

        preds_color_combined = self.color_fusion_weight * preds_color_base + (1 - self.color_fusion_weight) * preds_color_expert
        return preds_semantics, preds_color_combined


class ChannelWeightedColorModel(nn.Module):
    def __init__(self, num_bins, num_classes, device):
        super(ChannelWeightedColorModel, self).__init__(num_bins, num_classes, device)
        self.color_fusion_channel_weights = nn.Parameter(torch.full((3,), 0.5))

    def forward(self, locations, gray_images, lab_images):
        preds_semantics, preds_color_base, preds_color_expert = self.forward_base(locations, gray_images, lab_images)

        weights = self.color_fusion_channel_weights.view(1, 3, 1)
        preds_color_combined = weights * preds_color_base + (1 - weights) * preds_color_expert
        return preds_semantics, preds_color_combined


class ColorModelSimple(nn.Module):
    def __init__(self, num_bins):
        super(ColorModelSimple, self).__init__()
        self.color_fusion_channel_bin_weights = nn.Parameter(torch.full((3, num_bins), 0.5))

    def forward(self, preds_semantics_base, preds_color_base, preds_color_expert):
        weights = self.color_fusion_channel_bin_weights.unsqueeze(0)
        preds_color_combined = weights * preds_color_base + (1 - weights) * preds_color_expert
        return preds_semantics_base, preds_color_combined


class ColorModelLinear(nn.Module):
    def __init__(self, num_bins):
        super(ColorModelLinear, self).__init__()
        self.color_fusion_fc1 = nn.Linear(num_bins * 2, num_bins)

    def forward(self, preds_semantics_base, preds_color_base, preds_color_expert):
        preds_color_combined = torch.cat([preds_color_base, preds_color_expert], dim=-1)  # (N, C, 2*num_bins)
        samples, channels, _ = preds_color_combined.shape
        preds_color_combined = preds_color_combined.view(samples * channels, -1)
        preds_color_combined = self.color_fusion_fc1(preds_color_combined)
        preds_color_combined = preds_color_combined.view(samples, channels, -1)
        return preds_semantics_base, preds_color_combined

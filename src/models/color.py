import torch.nn as nn
import torch


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


class ColorModelMLP(nn.Module):
    def __init__(self, num_bins):
        super(ColorModelMLP, self).__init__()
        self.color_fusion_fc1 = nn.Linear(num_bins * 2, num_bins)
        self.bn1 = nn.BatchNorm1d(num_bins)
        self.color_fusion_fc2 = nn.Linear(num_bins, num_bins)
        self.bn2 = nn.BatchNorm1d(num_bins)
        self.color_fusion_fc3 = nn.Linear(num_bins, num_bins)

        self.relu = nn.ReLU()


    def forward(self, preds_semantics_base, preds_color_base, preds_color_expert):
        preds_color_combined = torch.cat([preds_color_base, preds_color_expert], dim=-1)  # (N, C, 2*num_bins)
        samples, channels, _ = preds_color_combined.shape
        preds_color_combined = preds_color_combined.view(samples * channels, -1)
        preds_color_combined = self.relu(self.bn1(self.color_fusion_fc1(preds_color_combined)))
        preds_color_combined = self.relu(self.bn2(self.color_fusion_fc2(preds_color_combined)))
        preds_color_combined = self.color_fusion_fc3(preds_color_combined)
        preds_color_combined = preds_color_combined.view(samples, channels, -1)
        return preds_semantics_base, preds_color_combined

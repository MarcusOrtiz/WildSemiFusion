import torch.nn as nn
import torch


class SemanticsColorModelSimple(nn.Module):
    def __init__(self, num_bins, num_classes):
        super(SemanticsColorModelSimple, self).__init__()
        self.color_fusion_channel_bin_weights = nn.Parameter(torch.full((3, num_bins), 0.5))
        self.semantics_fusion_classes_weights = nn.Parameter(torch.full((num_classes,), 0.5))

    def forward(self, preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert):
        color_weights = self.color_fusion_channel_bin_weights.unsqueeze(0)
        preds_color_combined = color_weights * preds_color_base + (1 - color_weights) * preds_color_expert
        semantics_weights = self.semantics_fusion_classes_weights.unsqueeze(0)
        preds_semantics_combined = semantics_weights * preds_semantics_base + (1 - semantics_weights) * preds_semantics_expert
        return preds_semantics_combined, preds_color_combined


class SemanticsColorModelLinear(nn.Module):
    def __init__(self, num_bins, num_classes):
        super(SemanticsColorModelLinear, self).__init__()
        self.color_fusion_fc1 = nn.Linear(num_bins * 2, num_bins)
        self.semantics_fusion_fc1 = nn.Linear(num_classes * 2, num_classes)

    def forward(self, preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert):
        preds_color_combined = torch.cat([preds_color_base, preds_color_expert], dim=-1)  # (N, C, 2*num_bins)
        samples, channels, _ = preds_color_combined.shape
        preds_color_combined = preds_color_combined.view(samples * channels, -1)
        preds_color_combined = self.color_fusion_fc1(preds_color_combined)
        preds_color_combined = preds_color_combined.view(samples, channels, -1)

        preds_semantics_combined = torch.cat([preds_semantics_base, preds_semantics_expert], dim=-1)  # (N, 2*num_classes)
        preds_semantics_combined = self.semantics_fusion_fc1(preds_semantics_combined)
        return preds_semantics_combined, preds_color_combined


class SemanticsColorModelMLP(nn.Module):
    def __init__(self, num_bins, num_classes):
        super(SemanticsColorModelMLP, self).__init__()
        self.color_fusion_fc1 = nn.Linear(num_bins * 2, num_bins)
        self.color_fusion_bn1 = nn.BatchNorm1d(num_bins)
        self.color_fusion_drop1 = nn.Dropout(0.1)
        self.color_fusion_fc2 = nn.Linear(num_bins, num_bins)
        self.color_fusion_bn2 = nn.BatchNorm1d(num_bins)
        self.color_fusion_drop2 = nn.Dropout(0.2)
        self.color_fusion_fc3 = nn.Linear(num_bins, num_bins)

        self.semantics_fusion_fc1 = nn.Linear(num_classes * 2, num_classes)
        self.semantics_fusion_bn1 = nn.BatchNorm1d(num_classes)
        self.semantics_fusion_drop1 = nn.Dropout(0.1)
        self.semantics_fusion_fc2 = nn.Linear(num_classes, num_classes)
        self.semantics_fusion_bn2 = nn.BatchNorm1d(num_classes)
        self.semantics_fusion_drop2 = nn.Dropout(0.2)
        self.semantics_color_fusion_fc3 = nn.Linear(num_classes, num_classes)

        self.relu = nn.ReLU()


    def forward(self, preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert):
        preds_color_combined = torch.cat([preds_color_base, preds_color_expert], dim=-1)  # (N, C, 2*num_bins)
        samples, channels, _ = preds_color_combined.shape
        preds_color_combined = preds_color_combined.view(samples * channels, -1)

        preds_color_combined = self.relu(self.bn1(self.color_fusion_fc1(preds_color_combined)))
        preds_color_combined = self.dropout_1(preds_color_combined)
        preds_color_combined = self.relu(self.bn2(self.color_fusion_fc2(preds_color_combined)))
        preds_color_combined = self.dropout_2(preds_color_combined)

        preds_color_combined = self.color_fusion_fc3(preds_color_combined)
        preds_color_combined = preds_color_combined.view(samples, channels, -1)

        preds_semantics_combined = torch.cat([preds_semantics_base, preds_semantics_expert], dim=-1)
        preds_semantics_combined = self.relu(self.bn1(self.semantics_fusion_fc1(preds_semantics_combined)))
        preds_semantics_combined = self.dropout_1(preds_semantics_combined)
        preds_semantics_combined = self.relu(self.bn2(self.semantics_fusion_fc2(preds_semantics_combined)))
        preds_semantics_combined = self.dropout_2(preds_semantics_combined)

        preds_semantics_combined = self.semantics_color_fusion_fc3(preds_semantics_combined)

        return preds_semantics_combined, preds_color_combined

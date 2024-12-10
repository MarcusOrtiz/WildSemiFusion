import os
from src.models.base import WildFusionModel
from src.models.experts import ColorExpertModel, SemanticExpertModel
from src.models.color import ColorModel, WeightedColorModel
import torch.nn as nn
import torch
import src.local_config as cfg


class ColorSemanticsModel(nn.Module):
    def __init__(self, num_bins, num_classes, device):
        super(ColorSemanticsModel, self).__init__()
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.device = device

        self.base_model = WildFusionModel(num_bins, num_classes)
        self.color_expert = ColorExpertModel(num_bins)
        self.semantics_expert = SemanticExpertModel(num_classes)
        self.base_model.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR_BASE, "best_model.pth"), map_location=device))
        self.color_expert.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR_COLOR_EXPERT, "best_model.pth"), map_location=device))
        self.semantics_expert.load_state_dict(torch.load(os.path.join(cfg.SAVE_DIR_SEMANTICS_EXPERT, "best_model.pth"), map_location=device))
        for p in self.base_model.parameters():
            p.requires_grad = False
        for p in self.color_expert.parameters():
            p.requires_grad = False
        self.base_model.eval()
        self.color_expert.eval()
        for p in self.semantics_expert.parameters():
            p.requires_grad = False
        self.semantics_expert.eval()

    def forward_base(self, locations, gray_images, lab_images):
        preds_semantics_base, preds_color_base = self.base_model(locations, gray_images, lab_images)
        preds_color_expert = self.color_expert(locations, lab_images)
        preds_semantics_expert = self.semantics_expert(locations, gray_images)
        return preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert


class WeightedColorSemanticsModel(ColorSemanticsModel):
    def __init__(self, num_bins, num_classes, device):
        super(WeightedColorSemanticsModel, self).__init__(num_bins, num_classes, device)
        self.color_fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.semantics_fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, locations, gray_images, lab_images):
        preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert = self.forward_base(locations, gray_images, lab_images)

        preds_color_combined = self.color_fusion_weight * preds_color_base + (1 - self.color_fusion_weight) * preds_color_expert
        preds_semantics_combined = self.semantics_fusion_weight * preds_semantics_base + (1 - self.semantics_fusion_weight) * preds_semantics_expert
        return preds_semantics_combined, preds_color_combined


class ChannelWeightedColorSemanticsModel(ColorSemanticsModel):
    def __init__(self, num_bins, num_classes, device):
        super(ChannelWeightedColorSemanticsModel, self).__init__(num_bins, num_classes, device)
        self.color_fusion_channel_weights = nn.Parameter(torch.full((3,), 0.5))
        self.semantics_fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, locations, gray_images, lab_images):
        preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert = self.forward_base(locations, gray_images, lab_images)

        preds_color_combined = self.color_fusion_weights.view(1, 3, 1) * preds_color_base + (1 - self.color_fusion_weights.view(1, 3, 1)) * preds_color_expert
        preds_semantics_combined = self.semantics_fusion_weight * preds_semantics_base + (1 - self.semantics_fusion_weight) * preds_semantics_expert
        return preds_semantics_combined, preds_color_combined


class ChannelBinWeightedColorSemanticsModel(ColorSemanticsModel):
    def __init__(self, num_bins, num_classes, device):
        super(ChannelBinWeightedColorSemanticsModel, self).__init__(num_bins, num_classes, device)
        self.color_fusion_channel_bin_weights = nn.Parameter(torch.full((3, num_bins), 0.5))
        self.semantics_fusion_classes_weights = nn.Parameter(torch.full((num_classes,), 0.5))

    def forward(self, locations, gray_images, lab_images):
        preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert = self.forward_base(locations, gray_images, lab_images)

        preds_color_combined = (self.color_fusion_channel_bin_weights.unsqueeze(0) * preds_color_base +
                                (1 - self.color_fusion_channel_bin_weights.unsqueeze(0)) * preds_color_expert)
        preds_semantics_combined = (self.semantics_fusion_classes_weights.unsqueeze(0) * preds_semantics_base +
                                    (1 - self.semantics_fusion_classes_weights.unsqueeze(0)) * preds_semantics_expert)

        return preds_semantics_combined, preds_color_combined


class LinearSemanticsColorModel(ColorSemanticsModel):
    def __init__(self, num_bins, num_classes, device):
        super(LinearSemanticsColorModel, self).__init__(num_bins, num_classes, device)
        self.color_fusion_fc1 = nn.Linear(num_bins * 2, num_bins)
        self.semantics_fusion_fc1 = nn.Linear(num_classes * 2, num_classes)

    def forward(self, locations, gray_images, lab_images):
        preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert = self.forward_base(locations, gray_images, lab_images)

        preds_color_combined = torch.cat([preds_color_base, preds_color_expert], dim=-1)  # (batch_size * locations, C, 2*num_bins)
        samples, channels, _ = preds_color_combined.shape
        preds_color_combined = preds_color_combined.view(samples * channels, -1)
        preds_color_combined = self.color_fusion_fc1(preds_color_combined)
        preds_color_combined = preds_color_combined.view(samples, channels, -1)

        preds_semantics_combined = torch.cat([preds_semantics_base, preds_semantics_expert], dim=-1)
        preds_semantics_combined = self.semantics_fusion_fc1(preds_semantics_combined)

        return preds_semantics_combined, preds_color_combined

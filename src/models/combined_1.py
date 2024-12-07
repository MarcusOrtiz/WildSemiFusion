from src.models.model_1 import MultiModalNetwork
from src.models.experts import ColorExpert
import torch.nn as nn
import torch
import src.config as cfg


class WeightedColorModel(nn.Module):
    def __init__(self, num_bins, num_classes):
        super(WeightedColorModel, self).__init__()
        self.base_model = MultiModalNetwork(num_bins, num_classes)
        self.color_expert = ColorExpert(num_bins)
        self.base_model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH_BASE))
        self.color_expert.load_state_dict(torch.load(cfg.BEST_MODEL_PATH_COLOR))
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.color_expert.parameters():
            param.requires_grad = False

        self.weight = nn.Parameter(torch.tensor(0.5))

    # Output from these is a discretized color prediction in logits and a semnatic prediciton, we can disregard the semantic prediction and
    # use the logits to combine them in a mlp with a linear relationship
    def forward(self, locations, gray_images, lab_images):
        preds_semantics, preds_color_base = self.base_model(locations, gray_images, lab_images)
        preds_color_expert = self.color_expert(locations, lab_images)
        preds_color_combined = self.weight * preds_color_base + (1 - self.weight) * preds_color_expert
        return preds_semantics, preds_color_combined


class DimensionalWeightedColorModel(nn.Module):
    def __init__(self, num_bins, num_classes):
        super(DimensionalWeightedColorModel, self).__init__()
        self.base_model = MultiModalNetwork(num_bins, num_classes)
        self.color_expert = ColorExpert(num_bins)
        self.base_model.load_state_dict(torch.load(cfg.BEST_MODEL_PATH_BASE))
        self.color_expert.load_state_dict(torch.load(cfg.BEST_MODEL_PATH_COLOR))
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.color_expert.parameters():
            param.requires_grad = False

        self.weights = nn.Parameter(torch.full((3,), 0.5))

    def forward(self, locations, gray_images, lab_images):
        preds_semantics, preds_color_base = self.base_model(locations, gray_images, lab_images)
        preds_color_expert = self.color_expert(locations, lab_images)

        weights_space = self.weights.view(1, 3, 1)
        preds_color_combined = weights_space * preds_color_base + (1 - weights_space) * preds_color_expert
        return preds_semantics, preds_color_combined


class FusionColorModel():
    def __init__(self, num_bins, num_classes):
        super(FusionColorModel, self).__init__()
        self.base_model = MultiModalNetwork(num_bins, num_classes)
        self.color_expert = ColorExpert(num_bins)

    def forward(self, locations, gray_images, lab_images):
        semantics, raw_color_logits = self.base_model(locations, gray_images, lab_images)
        color_predictions = self.color_expert(locations, lab_images)
        return semantics, raw_color_logits, color_predictions


class CombinedSemanticModel():
    def __init__(self, num_bins, num_classes):
        super(CombinedSemanticModel, self).__init__()
        self.base_model = MultiModalNetwork(num_bins, num_classes)
        # Define a semantic expert

    def forward(self, locations, gray_images, lab_images):
        pass


class CombinedSemanticFusedModel():
    def __init__(self, num_bins, num_classes):
        super(CombinedSemanticFusedModel, self).__init__()
        self.base_model = MultiModalNetwork(num_bins, num_classes)
        # Define a semantic expert

    def forward(self, locations, gray_images, lab_images):
        pass
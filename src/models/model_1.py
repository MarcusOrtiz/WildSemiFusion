import torch
import torch.nn as nn
from src.models.common_models import FourierFeatureLayer, ResidualBlock, SemanticNet, ColorNet

class Model1(nn.Module):
    # TODO: Fix layer parameters for 2D
    def __init__(self, num_bins=313):
        super(Model1, self).__init__()
        self.fourier_layer = FourierFeatureLayer(2, 256)
        self.
        self.compression_layer = nn.Linear(1280, 512)
        self.semantic_fcn = SemanticNet(input_dim=512)
        self.color_fcn = ColorNet(in_features=512, hidden_dim=128, num_bins=num_bins)

    def forward(self, locations, gray_image, lab_image, scan_indices=None):
        batch_size, num_locations, _ = locations.shape

        if scan_indices is not None:
            unique_scans, inverse_indices = torch.unique(scan_indices, return_inverse=True)
            pc_features = unique_pc_features[inverse_indices]
            audio_features = unique_audio_features[inverse_indices]
        else:
            pc_features = self.pointnet_encoder(point_clouds.transpose(-1, -2))
            audio_features = self.audio_cnn(audio)

        location_features = self.fourier_layer(locations.view(-1, 3))

        num_queries = location_features.shape[0]
        pc_features = pc_features.repeat_interleave(num_locations // pc_features.shape[0], dim=0)
        audio_features = audio_features.repeat_interleave(num_locations // audio_features.shape[0], dim=0)
        distance_weights = self.compute_distance_weight(locations)
        weighted_pc_features = pc_features * distance_weights
        weighted_audio_features = audio_features * distance_weights

        if location_features.dim() == 2:
            location_features = location_features.unsqueeze(0)  # Adds batch dimension [1, 4096, 512]

        assert location_features.shape[1] == weighted_pc_features.shape[1] == weighted_audio_features.shape[
            1], "Query points mismatch"

        combining_features = torch.cat([location_features, weighted_pc_features, weighted_audio_features], dim=2)
        compressed_features = self.compression_layer(combining_features)

        # Generate predictions
        sdf = self.sdf_fcn(compressed_features).view(batch_size, num_locations)
        confidence = self.confidence_fcn(compressed_features).view(batch_size, num_locations)
        semantics = self.semantic_fcn(compressed_features)
        color_logits = self.color_fcn(compressed_features)
        traversability = self.traversability_fc(compressed_features).view(batch_size, num_locations)

        return sdf, confidence, semantics, color_logits, traversability

    def compute_distance_weight(self, locations, sigma=1):
        origin = torch.zeros_like(locations)
        distances = torch.norm(locations - origin, dim=-1, keepdim=True)
        weights = torch.exp(- distances / sigma)
        weights = weights / torch.max(weights)

        return weights
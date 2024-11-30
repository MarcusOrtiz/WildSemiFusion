import torch
import torch.nn as nn
from src.models.common_models import FourierFeatureLayer, ResidualBlock, SemanticNet, ColorNet, LABCNN, GrayscaleCNN, \
    CompressionLayer

class MultiModalNetwork(nn.Module):
    # TODO: Change masking to be a value outside of the color space
    # TOOD: Make sure not to count the void class against the model
    def __init__(self, num_bins, num_classes):
        super(MultiModalNetwork, self).__init__()
        self.fourier_layer = FourierFeatureLayer(in_dim=2, out_dim=256)
        self.lab_cnn = LABCNN(image_size=(224, 224), out_dim=256)
        self.gray_cnn = GrayscaleCNN(image_size=(224, 224), out_dim=256)

        self.compression_layer = CompressionLayer(in_dim=768, out_dim=256)

        self.semantic_fcn = SemanticNet(input_dim=256, hidden_dim=128, num_classes=num_classes)
        self.color_fcn = ColorNet(in_features=256, hidden_dim=128, num_bins=num_bins)

    def forward(self, locations, gray_images, lab_images):
        '''

        :param locations: (batch_size, num_locations, location_dim) or (batch_size, image_size[0] * image_size[1], 2)
        :param gray_images: (batch_size, 1, image_size[0], image_size[1])
        :param lab_images: (batch_size, 3, image_size[0], image_size[1])
        :return:
        '''
        batch_size, num_locations, _ = locations.shape

        location_features = self.fourier_layer(locations.view(-1, 2)) # (batch_size, num_locations, locations_dim) -> (batch_size * num_locations, 2) -> (batch_size * num_locations, 256)
        gray_features = self.gray_cnn(gray_images)  # (batch_size, 1, image_size[0], image_size[1]) -> (batch_size, 256)
        lab_features = self.lab_cnn(lab_images)  # (batch_size, 3, image_size[0], image_size[1]) -> (batch_size, 256)

        # Expand lab_features and gray_features to match the number of query locations
        gray_features = gray_features.repeat_interleave(num_locations, dim=0) # (batch_size, 256) -> (batch_size * num_locations, 256)
        lab_features = lab_features.repeat_interleave(num_locations, dim=0) # (batch_size, 256) -> (batch_size * num_locations, 256)

        # Concatenation and compression of encoding features
        combining_features = torch.cat([location_features, gray_features, lab_features], dim=-1) # (batch_size * num_locations, 768)
        compressed_features = self.compression_layer(combining_features)  # (batch_size * num_locations, 256)

        # Generate predictions
        semantics = self.semantic_fcn(compressed_features)  # (batch_size * num_locations, num_classes)
        color_logits = self.color_fcn(compressed_features)  # (batch_size * num_locations, 3, num_bins)

        return semantics, color_logits
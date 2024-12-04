import torch
import torch.nn as nn
from src.models.common_models import FourierFeatureLayer, ResidualBlock, SemanticNet, ColorNet, LABCNN, GrayscaleCNN, \
    CompressionLayer


class ColorExpert(nn.Module):
    # TODO: Change masking to be a value outside of the color space
    # TOOD: Make sure not to count the void class against the model
    def __init__(self, num_bins):
        super(ColorExpert, self).__init__()
        self.fourier_layer = FourierFeatureLayer(in_dim=2, out_dim=128)
        self.lab_cnn = LABCNN(image_size=(224, 224), out_dim=128)

        self.compression_layer = CompressionLayer(in_dim=256, out_dim=128)

        self.color_fcn = ColorNet(in_features=128, hidden_dim=64, num_bins=num_bins)

    def forward(self, locations, gray_images, lab_images):
        '''

        :param locations: (batch_size, num_locations, location_dim) or (batch_size, image_size[0] * image_size[1], 2)
        :param gray_images: (batch_size, 1, image_size[0], image_size[1])
        :param lab_images: (batch_size, 3, image_size[0], image_size[1])
        :return:
        '''
        batch_size, num_locations, _ = locations.shape

        locations = locations.reshape(-1, 2)
        location_features = self.fourier_layer(locations)  # (batch_size, num_locations, locations_dim) -> (batch_size * num_locations, 2) -> (batch_size * num_locations, 256)
        del locations
        gray_features = self.gray_cnn(gray_images)  # (batch_size, 1, image_size[0], image_size[1]) -> (batch_size, 256)
        del gray_images
        lab_features = self.lab_cnn(lab_images)  # (batch_size, 3, image_size[0], image_size[1]) -> (batch_size, 256)
        del lab_images

        lab_features = lab_features[:, None, :].expand(-1, num_locations, -1).reshape(-1, lab_features.size(-1))

        # Concatenation and compression of encoding features
        combining_features = torch.cat([location_features, gray_features, lab_features], dim=-1)  # (batch_size * num_locations, 768)
        del location_features, gray_features, lab_features
        compressed_features = self.compression_layer(combining_features)  # (batch_size * num_locations, 256)
        del combining_features

        # Generate predictions
        raw_color_logits = self.color_fcn(compressed_features)  # (batch_size * num_locations, 3, num_bins) of type

        return raw_color_logits

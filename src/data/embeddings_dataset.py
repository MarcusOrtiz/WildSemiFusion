import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data.utils.data_processing import (rgb_to_gray, rgb_to_lab_continuous, lab_continuous_to_lab_discretized,
                                            gray_continuous_to_gray_discretized, image_gaussian_noise, create_mask)
from src.data.utils.data_processing import image_to_array
from src.local_config import IMAGE_SIZE
import src.local_config as cfg
from src.models.common_models import FourierFeatureLayer, LABCNN, GrayscaleCNN
from src.utils import model_to_device, compile_model, generate_normalized_locations

ONTOLOGY_VOID = 0


class EmbeddingsDataset(Dataset):
    def __init__(self, preloaded_data, num_bins, image_size, image_noise, image_mask_rate, batch_size):
        self.num_bins = num_bins
        self.image_size = image_size
        self.image_noise = image_noise
        self.image_mask_rate = image_mask_rate
        self.batch_size = batch_size

        # Create models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fourier_feature_layer = FourierFeatureLayer(in_dim=2, out_dim=128)
        lab_cnn = LABCNN(image_size=cfg.IMAGE_SIZE, out_dim=128)
        self.fourier_feature_layer = model_to_device(fourier_feature_layer, device)
        self.lab_cnn = model_to_device(lab_cnn, device)
        self.fourier_feature_layer.load_state_dict(
            torch.load(os.path.join(cfg.AWS_SAVE_DIR, 'base/fourier_layer_model.pth'), map_location=device)
        )
        self.lab_cnn.load_state_dict(
            torch.load(os.path.join(cfg.AWS_SAVE_DIR, 'base/lab_cnn_model.pth'), map_location=device)
        )

        # self.fourier_feature_layer = compile_model(fourier_feature_layer)
        # self.lab_cnn = compile_model(lab_cnn)

        self.locations_features, self.lab_features, self.gt_lab_images = self._compute_embeddings(self.fourier_feature_layer, self.lab_cnn, preloaded_data['rgb_images'], device)
        print(f"Locations features device: {self.locations_features.device}, Locations features attached: {self.locations_features.is_attached()}")
        print(f"Lab features device: {self.lab_features[0].device}, Lab features attached: {self.lab_features[0].is_attached()}")
        print(f"GT Lab images device: {self.gt_lab_images[0].device}, GT Lab images attached: {self.gt_lab_images[0].is_attached()}")


    def _compute_embeddings(self, fourier_feature_layer, lab_cnn, rgb_images, device):
        normalized_locations = generate_normalized_locations()
        normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)
        locations_tensor = normalized_locations_tensor.reshape(-1, 2)
        with torch.inference_mode():
            locations_features = fourier_feature_layer(locations_tensor).detach()
        print(f"Locations Features Shape: {locations_features.shape}")

        gt_lab_images = []
        lab_features = []


        for i in range(0, len(rgb_images), self.batch_size):
            batch_images = rgb_images[i:i + self.batch_size]
            lab_tensors_batch = []
            for rgb_image in batch_images:
                lab_image = rgb_to_lab_continuous(rgb_image)
                gt_lab_image = lab_continuous_to_lab_discretized(lab_image, self.num_bins, void_bin=True)
                mask = create_mask(self.image_size[0], self.image_size[1], self.image_mask_rate)
                gt_lab_image[mask] = [self.num_bins - 1, self.num_bins - 1, self.num_bins - 1]
                gt_lab_image = torch.tensor(gt_lab_image, dtype=torch.long).to('cpu')
                gt_lab_images.append(gt_lab_image)

                noisy_lab_image = image_gaussian_noise(lab_image, self.image_noise, space='lab')
                noisy_lab_image_discretized = lab_continuous_to_lab_discretized(noisy_lab_image, self.num_bins, void_bin=True)
                noisy_lab_image_discretized[mask] = [self.num_bins - 1, self.num_bins - 1, self.num_bins - 1]
                masked_noisy_lab_image_discretized = noisy_lab_image_discretized
                lab_image_tensor = torch.tensor(masked_noisy_lab_image_discretized.transpose(2, 0, 1), dtype=torch.float32).to(device)
                lab_tensors_batch.append(lab_image_tensor)

            lab_tensors_batch = torch.stack(lab_tensors_batch)
            with torch.inference_mode():
                lab_features_batch = lab_cnn(lab_tensors_batch).detach()

            lab_features.extend(lab_features_batch)
        
        return locations_features, lab_features, gt_lab_images

    def __len__(self):
        return len(self.lab_features)

    def __getitem__(self, idx):
        return {
            'locations_features': self.locations_features,
            'lab_features': self.lab_features[idx],
            'gt_color': self.gt_lab_images[idx]
        }




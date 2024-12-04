import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data.utils.data_processing import (rgb_to_gray, rgb_to_lab_continuous, lab_continuous_to_lab_discretized,
                                            gray_continuous_to_gray_discretized, image_gaussian_noise, image_mask)
from src.data.utils.data_processing import image_to_array
from src.config import IMAGE_SIZE
import src.config as cfg

ONTOLOGY_VOID = 0


class Rellis2DDataset(Dataset):
    """
    Custom dataset for data loader to create lab and gray images with labels based off RGB images and image labels from Rellis-3D dataset that are preprocessed
    according to README. Newly created images have both a ground truth and a noisy + masked version.
    @:param preloaded_data: Dictionary of 'rgb_images' and 'labels' where both are image formatted and already preprocessed to correct shape and split
    @:param num_bins: Number of bins to discretize the Lab color space
    """

    def __init__(self, preloaded_data, num_bins, image_size, image_noise, image_mask_rate):
        self.num_bins = num_bins
        self.image_size = image_size
        self.image_noise = image_noise
        self.image_mask_rate = image_mask_rate

        data = preloaded_data

        self.rgb_images = data['rgb_images']
        self.gt_semantics_lst = data['gt_semantics']

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        lab_image = rgb_to_lab_continuous(rgb_image)
        gt_semantics = self.gt_semantics_lst[idx]
        # print(f"GT Semantics Shape In Dataset: {gt_semantics.shape}")

        # Convert RGB to discretized Lab and continuous Gray ground truth images
        gt_gray_image = rgb_to_gray(rgb_image)
        gt_lab_image = lab_continuous_to_lab_discretized(lab_image, self.num_bins)

        # Apply noise and masks to images
        noisy_lab_image = image_gaussian_noise(lab_image, self.image_noise, space='lab')
        masked_noisy_lab_image = image_mask(noisy_lab_image)
        masked_noisy_lab_image_discretized = lab_continuous_to_lab_discretized(masked_noisy_lab_image, self.num_bins)

        noisy_gray_image = image_gaussian_noise(gt_gray_image, self.image_noise, space='gray')
        masked_noisy_gray_image = image_mask(noisy_gray_image)

        # Convert images to tensors and adjust dimensions
        gt_semantics_tensor = torch.tensor(gt_semantics, dtype=torch.long)
        gt_color_tensor = torch.tensor(gt_lab_image, dtype=torch.long)
        lab_image_tensor = torch.tensor(masked_noisy_lab_image_discretized.transpose(2, 0, 1), dtype=torch.float32)  # Shape: (3, H, W)
        gray_image_tensor = torch.tensor(masked_noisy_gray_image[np.newaxis, ...], dtype=torch.float32)  # Shape: (1, H, W)

        # free memory


        return {
            'gt_semantics': gt_semantics_tensor,
            'gt_color': gt_color_tensor,
            'lab_image': lab_image_tensor,
            'gray_image': gray_image_tensor,
        }



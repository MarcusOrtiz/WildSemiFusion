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
        return len(self.gt_semantics_lst)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        lab_image = rgb_to_lab_continuous(rgb_image)
        gt_semantics = self.gt_semantics_lst[idx]

        # Convert RGB to discretized Lab and continuous Gray ground truth images
        gt_gray_image = rgb_to_gray(rgb_image)
        gt_lab_image = lab_continuous_to_lab_discretized(lab_image, self.num_bins)

        # Apply noise and masks to images
        noisy_lab_image = image_gaussian_noise(lab_image, self.image_noise, space='lab')
        masked_noisy_lab_image = image_mask(noisy_lab_image)
        masked_noisy_lab_image_discretized = lab_continuous_to_lab_discretized(masked_noisy_lab_image, self.num_bins)

        noisy_gray_image = image_gaussian_noise(gt_gray_image, self.image_noise, space='gray')
        masked_noisy_gray_image = image_mask(noisy_gray_image)

        """
        Generate Locations (currently all pixels TODO: always consistent on number of locations between indexes)
        Locations is somewhat misleading, it's actually the indices of the non-void pixels whereas WildFusion uses actual
        Locations are normalized to [0, 1]
        """
        y_coords, x_coords = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]), indexing='ij')
        locations = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)  # Shape: (H*W, 2)
        normalized_locations = locations.astype(np.float32)
        normalized_locations[:, 0] /=  self.image_size[0]
        normalized_locations[:, 1] /= self.image_size[1]


        # Convert images to tensors and adjust dimensions
        lab_image_tensor = torch.from_numpy(
            masked_noisy_lab_image_discretized.transpose(2, 0, 1)).float()  # Shape: (3, H, W)
        gray_image_tensor = torch.from_numpy(masked_noisy_gray_image[np.newaxis, ...]).float()  # Shape: (1, H, W)

        return {
            'locations': locations,
            'gt_semantics': gt_labels,
            'gt_lab_image': gt_lab_image,
            'lab_image': masked_noisy_lab_image_discretized,
            'gray_image': masked_noisy_gray_image_discretized,
            'gt_color': ...
        }


def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {key: [] for key in keys}

    for item in batch:
        for key in keys:
            collated[key].append(item[key])


# TODO: remove later, this is only to confirm that new ways are working
from skimage import color
def rgb_to_lab_discretized(rgb_colors: np.array, num_bins):
    """
    Taken from WildFusion/src/data_loader
    """
    colorless_mask = (rgb_colors == [-1, -1, -1]).all(axis=-1)
    lab_colors = color.rgb2lab(rgb_colors)  # Convert normalized RGB to LAB

    lab_colors_normalized = (lab_colors + [0, 128, 128]) / [100, 255, 255]
    lab_colors_normalized = np.clip(lab_colors_normalized, 0, 1)

    lab_colors_discretized = (lab_colors_normalized * (num_bins - 1)).astype(int)
    lab_colors_discretized[colorless_mask] = num_bins - 1

    assert np.all(lab_colors_discretized >= 0) and np.all(
        lab_colors_discretized < num_bins), "Discretized LAB values out of range"
    return lab_colors_discretized


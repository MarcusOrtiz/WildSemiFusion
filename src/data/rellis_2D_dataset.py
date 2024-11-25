import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.data.utils.data_processing import rgb_to_gray, rgb_to_lab_discretized, image_gaussian_noise, image_mask

ONTOLOGY_VOID = 0


class Rellis2DDataset(Dataset):
    """
    Custom dataset for data loader to create lab and gray images with labels based off RGB images and image labels from Rellis-3D dataset that are preprocessed
    according to README. Newly created images have both a ground truth and a noisy + masked version.
    @:param preloaded_data: Dictionary of 'rgb_images' and 'labels' where both are image formatted and already preprocessed to correct shape and split
    @:param num_bins: Number of bins to discretize the Lab color space
    """

    def __init__(self, preloaded_data, num_bins=313, image_size=(224, 224), image_noise=.1, image_mask_rate=(.1, .3)):
        self.num_bins = num_bins
        self.image_size = image_size
        self.rgb_images = preloaded_data['rgb_images']
        self.gt_labels = preloaded_data['labels']
        self.image_noise = image_noise
        self.image_mask_rate = image_mask_rate

    def __len__(self):
        return len(self.gt_labels)

    def __getitem__(self, idx):
        rgb_image = self.rgb_images[idx]
        gt_labels = self.gt_labels[idx]

        gt_gray_image = rgb_to_gray(rgb_image)
        gt_lab_image = rgb_to_lab_discretized(rgb_image, num_bins=self.num_bins)

        # Apply noise and masks to images (Question: Should I edit the labels as well?)
        noisy_lab_image = image_gaussian_noise(gt_lab_image, self.image_noise)
        masked_noisy_lab_image = image_mask(noisy_lab_image, mask_rate=self.image_mask_rate)
        noisy_gray_image = image_gaussian_noise(gt_gray_image, self.image_noise)
        masked_noisy_gray_image = image_mask(noisy_gray_image, mask_rate=self.image_mask_rate)

        # Locations is somewhat misleading, it's actually the indices of the non-void pixels whereas WildFusion uses actual
        # Avoids unlabeled semantics, similar to WildFusion
        locations = torch.tensor(np.argwhere(gt_labels != ONTOLOGY_VOID), dtype=torch.float32)

        return {
            'locations': locations,
            'gt_labels': gt_labels,
            'gt_gray_image': gt_gray_image,
            'gt_lab_image': gt_lab_image,
            'masked_noisy_lab_image': masked_noisy_lab_image,
            'maksed_noisy_gray_image': masked_noisy_gray_image,
        }

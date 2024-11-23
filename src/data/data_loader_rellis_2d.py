import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from cv2 import cvtColor, COLOR_RGB2GRAY
from src.data.utils.data_processing import rgb_to_gray, rgb_to_lab_discretized, image_gaussian_noise, image_mask


class CustomRellis2DDataset(Dataset):
    """
    Pixels per batch probably doesn't matter considering bottleneck will be CPU memory
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
        gt_label = self.gt_labels[idx]

        gt_gray_image = rgb_to_gray(rgb_image)
        gt_lab_image = rgb_to_lab_discretized(rgb_image, num_bins=self.num_bins)

        # Apply noise and masks to images (Question: Should I edit the labels as well?)
        noisy_image = image_gaussian_noise(gt_lab_image, self.image_noise)
        masked_noisy_lab_image = image_mask(noisy_image, mask_rate=self.image_mask_rate)

        # How does locations and selected_points work?
        # locations = self.locations[selected_points]


        return {
            'locations': locations,
            'gt_label': gt_label,
            'gt_gray_image': gt_gray_image,
            'gt_lab_image': gt_lab_image,
            'masked_noisy_lab_image': masked_noisy_lab_image,
        }


def custom_collate_fn(batch):
    keys = batch[0].keys()
    collated = {key: [] for key in keys}

    for item in batch:
        for key in keys:
            collated[key].append(item[key])

    # Stack tensors
    for key in ['point_clouds', 'mel_spectrograms', 'gt_travs']:
        collated[key] = torch.stack(collated[key])

    # Concatenate tensors
    for key in ['locations', 'gt_sdf', 'gt_confidence', 'gt_semantics', 'gt_color']:
        collated[key] = torch.cat(collated[key], dim=0)

    return collated

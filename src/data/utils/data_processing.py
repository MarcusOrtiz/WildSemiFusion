import os
import numpy as np
import random
from imageio.v3 import imread, imwrite
from typing import Tuple
from cv2 import imread, resize, cvtColor, COLOR_RGB2GRAY, INTER_NEAREST
from skimage import color


def annotated_image_to_array(label_path: str):
    return np.array(imread(label_path))


def resize_rgb_and_annotation(rgb_path: str, label_path: str, size: Tuple[int, int] = (224, 224)) -> Tuple[np.array, np.array]:
    rgb = imread(rgb_path)
    label = imread(label_path)

    resized_rgb = np.array(resize(rgb, size))
    resized_label = np.array(resize(label, size, interpolation=INTER_NEAREST))

    return resized_rgb, resized_label


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


def rgb_to_gray(rgb_colors: np.array):
    return cvtColor(rgb_colors, COLOR_RGB2GRAY)


def image_mask(image: np.array, mask_rate: Tuple[float, float] = (.1, .3), mask=None):
    """
    Apply a masking to an image
    :param image: (np.array) preloaded image
    :param mask_rate: (float, float) min and max rate of masking windows
    :param mask: (int, ...) value for each channel
    """
    MIN_MASK_SHAPE_DIVIDER = 10
    MAX_MASK_SHAPE_DIVIDER = 20
    image = image.copy()
    h, w, c = image.shape

    # Defining number of masking windows and their values and shapes
    window_shapes = [(np.random.randint(h // MIN_MASK_SHAPE_DIVIDER, h // MAX_MASK_SHAPE_DIVIDER),
                      np.random.randint(w // MIN_MASK_SHAPE_DIVIDER, w // MAX_MASK_SHAPE_DIVIDER))
                     for _ in range(3)]
    mask_rate = random.uniform(mask_rate[0], mask_rate[1])
    num_windows = [round((h * w) / (shape[0] * shape[1]) * mask_rate/3) for shape in window_shapes]
    if mask is None:
        mask = [0 for _ in range(c)]

    # Applying mask to random points
    for shape, num in zip(window_shapes, num_windows):
        for _ in range(num):
            y = np.random.randint(0, h - shape[0])
            x = np.random.randint(0, w - shape[1])
            if c == 1:
                image[y:y + shape[0], x:x + shape[1]] = mask[0]
            else:
                image[y:y + shape[0], x:x + shape[1], :] = mask
    return image


def image_gaussian_noise(image: np.array, noise: float, space: str = "lab"):
    noise = np.random.normal(0, noise, image.shape)
    noisy_image = image + noise
    if space == "lab":
        noisy_image = np.clip(noisy_image, 0, 1)
    return noisy_image

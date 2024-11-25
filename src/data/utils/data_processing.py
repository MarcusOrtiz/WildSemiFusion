import os

import cv2
import math as m
import numpy as np
import random
from imageio.v3 import imread, imwrite
from typing import Tuple
from cv2 import imread, resize, cvtColor, COLOR_RGB2GRAY, INTER_NEAREST
from skimage import color

GRAY_CHANNEL = (0, 1)
L_CHANNEL = (0, 100)
A_CHANNEL = (-128, 127)
B_CHANNEL = (-128, 127)


def image_to_array(label_path: str):
    return np.array(imread(label_path))


def resize_rgb_and_annotation(rgb_path: str, label_path: str, size: Tuple[int, int] = (224, 224)) \
        -> Tuple[np.array, np.array]:
    rgb = imread(rgb_path)
    label = imread(label_path)

    resized_rgb = np.array(resize(rgb, size))
    resized_label = np.array(resize(label, size, interpolation=INTER_NEAREST))

    return resized_rgb, resized_label


def rgb_to_lab_continuous(rgb_colors: np.array):
    lab_colors = rgb_colors.copy()
    return color.rgb2lab(lab_colors)


def lab_continuous_to_lab_discretized(lab_colors: np.array, num_bins):
    """
    Adapted from WildFusion/src/data_loader
    """
    lab_colors = lab_colors.copy()
    # print(f"max lab_colors original: {lab_colors.max()}")

    lab_colors_normalized = (lab_colors - [L_CHANNEL[0], A_CHANNEL[0], B_CHANNEL[0]]) / \
                            [channel_size(L_CHANNEL), channel_size(A_CHANNEL), channel_size(B_CHANNEL)]
    lab_colors_normalized = np.clip(lab_colors_normalized, 0, 1)
    # print(f"max lab_colors normalized: {lab_colors_normalized.max()}")


    lab_colors_discretized = (lab_colors_normalized * (num_bins-1)).astype(int)

    assert np.all(lab_colors_discretized >= 0) and np.all(
        lab_colors_discretized < num_bins), "Discretized LAB values out of range"
    return lab_colors_discretized

# TODO: Abstract with LAB
def gray_continuous_to_gray_discretized(gray_colors: np.array, num_bins):
    gray_colors = gray_colors.copy()

    gray_colors_normalized = (gray_colors - GRAY_CHANNEL[0]) / channel_size(GRAY_CHANNEL)
    gray_colors_normalized = np.clip(gray_colors_normalized, 0, 1)

    gray_colors_discretized = (gray_colors_normalized * (num_bins-1)).astype(int)

    assert np.all(gray_colors_discretized >= 0) and np.all(
        gray_colors_discretized < num_bins), "Discretized Gray values out of range"
    return gray_colors_discretized


def lab_discretized_to_rgb(lab_colors: np.array, num_bins):
    lab_colors = lab_colors.copy()

    lab_colors_normalized = (lab_colors.astype(float) / (num_bins-1))
    # print(f"lab_colors_normalized clipped: {lab_colors_normalized}")


    lab_colors_continuous = lab_colors_normalized * [channel_size(L_CHANNEL),
                                                     channel_size(A_CHANNEL),
                                                     channel_size(B_CHANNEL)] + \
                            [L_CHANNEL[0], A_CHANNEL[0], B_CHANNEL[0]]

    rgb_image = color.lab2rgb(lab_colors_continuous)

    return rgb_image



def rgb_to_gray(rgb_colors: np.array):
    gray_colors = rgb_colors.copy()
    return color.rgb2gray(gray_colors)


def image_mask(image: np.array, mask_rate_limits: Tuple[float, float] = (.0, .15), mask=None, space: str = "lab"):
    """
    Apply a masking to an image
    :param image: (np.array) preloaded image
    :param mask_rate: (float, float) min and max rate of masking windows
    :param mask: (int, ...) value for each channel
    """
    MIN_MASK_SHAPE_DIVIDER = 5
    MAX_MASK_SHAPE_DIVIDER = 12
    image = image.copy()

    if len(image.shape) == 2:  # Grayscale image
        h, w = image.shape
        c = 1
    else:
        h, w, c = image.shape

    # Defining number of masking windows and their values and shapes
    window_shapes = [(np.random.randint(h // MAX_MASK_SHAPE_DIVIDER, h // MIN_MASK_SHAPE_DIVIDER),
                      np.random.randint(w // MAX_MASK_SHAPE_DIVIDER, w // MIN_MASK_SHAPE_DIVIDER))
                     for _ in range(3)]
    # print(f"Window shapes: {window_shapes}")
    mask_rate = random.uniform(mask_rate_limits[0], mask_rate_limits[1])
    # print(f"Mask rate: {mask_rate}")
    mask_coverage_per_shape = [h * w * mask_rate / len(window_shapes)] * 3
    num_windows = [0 for _ in range(len(window_shapes))]
    for i, shape in enumerate(window_shapes):
        num_windows[i] = m.trunc(mask_coverage_per_shape[i] // (shape[0] * shape[1]))


    # print(f"Num windows: {num_windows}")
    if mask is None and space == "lab":
        mask = [0, 0, 0]
    elif mask is None and space == "gray":
        mask = [0]


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


# TODO: Decide if noise should be of total channel or of range
# TODO: Decide if noise and mask should be returning copies or modifying the original image, right now it is copies
def image_gaussian_noise(image: np.array, noise: float, space: str = "lab"):
    """
    Apply Gaussian noise to either grayscale or Lab image
    :param image:
    :param noise: from 0 to 1, it will be scaled by standard channel size constants
    :param space:
    :return:
    """

    if space == "lab":
        l_scale, a_scale, b_scale = channel_size(L_CHANNEL), channel_size(A_CHANNEL), channel_size(B_CHANNEL)

        noise_L = np.random.normal(0, noise * l_scale, image[..., 0].shape)
        noise_A = np.random.normal(0, noise * a_scale, image[..., 1].shape)
        noise_B = np.random.normal(0, noise * b_scale, image[..., 2].shape)

        L = np.clip(image[..., 0].copy() + noise_L, L_CHANNEL[0], L_CHANNEL[1])
        A = np.clip(image[..., 1].copy() + noise_A, A_CHANNEL[0], A_CHANNEL[1])
        B = np.clip(image[..., 2].copy() + noise_B, B_CHANNEL[0], B_CHANNEL[1])

        L += np.random.normal(0, noise, L.shape)
        A += np.random.normal(0, noise, A.shape)
        B += np.random.normal(0, noise, B.shape)

        noisy_image = np.stack([L, A, B], axis=-1)
    elif space == "gray":
        image = image.copy()
        gray_scale = channel_size(GRAY_CHANNEL)
        noise = np.random.normal(0, noise, image.shape) * gray_scale
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, GRAY_CHANNEL[0], GRAY_CHANNEL[1]).astype(image.dtype)
    else:
        raise ValueError(f"Space {space} not recognized")

    return noisy_image.astype(image.dtype)


def channel_size(channel_limits: Tuple[int, int]):
    return channel_limits[1] - channel_limits[0]

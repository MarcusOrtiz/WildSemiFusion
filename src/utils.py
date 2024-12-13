import numpy as np
import torch
import random
import torch.nn as nn
import src.local_config as cfg

"""
Generate Locations (currently all pixels TODO: always consistent on number of locations between indexes)
Locations in original use as a metric of space, here it is pixel correlation
Locations are normalized to [0, 1]
"""


def generate_normalized_locations(image_size) -> np.array:
    y_coords, x_coords = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]),
                                     indexing='ij')
    locations_grid = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)  # (IMAGE_SIZE[0]*IMAGE_SIZE[1], 2)
    normalized_locations = locations_grid.astype(np.float32)
    normalized_locations[:, 0] /= image_size[0]
    normalized_locations[:, 1] /= image_size[1]
    return normalized_locations


def model_to_device(model, device):
    # Must return model in case of DataParallel
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs!")
    return model


def compile_model(model):
    # backend = "cudagraphs" if torch.cuda.is_available() else "inductor"
    backend = "inductor"
    model = torch.compile(model=model, backend=backend)
    return model


def populate_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def generate_loss_trackers():
    training_losses = {'total': [], 'semantics': [], 'color': []}
    validation_losses = {'total': [], 'semantics': [], 'color': []}
    times = []
    return training_losses, validation_losses, times


def update_loss_trackers(losses, total_loss, semantics_loss, color_loss):
    losses['total'].append(total_loss)
    losses['semantics'].append(semantics_loss)
    losses['color'].append(color_loss)


def print_color_prediction_shapes(preds, gt):
    print(f"Preds Color Shape: {preds.view(-1, cfg.NUM_BINS).shape}")
    print(f"GT Color Shape: {gt.view(-1).shape}")


def print_semantic_prediction_shapes(preds, gt):
    print(f"Preds Semantics Shape: {preds.shape}")
    print(f"GT Semantics Shape: {gt.long().view(-1).shape}")

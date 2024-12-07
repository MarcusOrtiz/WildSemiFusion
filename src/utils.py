import numpy as np
import src.config as cfg

"""
Generate Locations (currently all pixels TODO: always consistent on number of locations between indexes)
Locations in original use as a metric of space, here it is pixel correlation
Locations are normalized to [0, 1]
"""
def generate_normalized_locations() -> np.array:
    y_coords, x_coords = np.meshgrid(np.arange(cfg.IMAGE_SIZE[0]), np.arange(cfg.IMAGE_SIZE[1]),
                                     indexing='ij')
    locations_grid = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)  # (IMAGE_SIZE[0]*IMAGE_SIZE[1], 2)
    normalized_locations = locations_grid.astype(np.float32)
    normalized_locations[:, 0] /= cfg.IMAGE_SIZE[0]
    normalized_locations[:, 1] /= cfg.IMAGE_SIZE[1]
    return normalized_locations

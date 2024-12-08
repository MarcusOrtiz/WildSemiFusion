import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.utils.data_processing import image_to_array, lab_discretized_to_rgb, lab_continuous_to_lab_discretized, \
    rgb_to_gray, rgb_to_lab_continuous, load_sequential_data
from src.data.rellis_2D_dataset import Rellis2DDataset
from torch.utils.data import DataLoader
from src.models.experts import ColorExpertModel
import src.local_config as cfg


test_preloaded_data = load_sequential_data(cfg.TEST_DIR)
print("Successfully loaded preprocessed test data")


# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ColorExpertModel(num_bins=cfg.NUM_BINS)
print("Model initialized successfully")

model = model.to(device)
print("Model moved to device")

state_dict = torch.load(cfg.SAVE_DIR_COLOR + 'best_model.pth', map_location=device)
model.load_state_dict(state_dict)

model.eval()
with torch.no_grad():
    print("Model set to evaluation mode")
    print("Model weights successfully loaded")
    y_coords, x_coords = np.meshgrid(np.arange(cfg.IMAGE_SIZE[0]), np.arange(cfg.IMAGE_SIZE[1]),
                                     indexing='ij')
    locations_grid = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)  # (IMAGE_SIZE[0]*IMAGE_SIZE[1], 2)
    normalized_locations = locations_grid.astype(np.float32)
    normalized_locations[:, 0] /= cfg.IMAGE_SIZE[0]
    normalized_locations[:, 1] /= cfg.IMAGE_SIZE[1]

    for rgb_image, gt_semantics in zip(test_preloaded_data['rgb_images'], test_preloaded_data['gt_semantics']):
        # Convert rgb_image to gray and lab
        lab_image = rgb_to_lab_continuous(rgb_image)
        lab_image_discretized = lab_continuous_to_lab_discretized(lab_image, cfg.NUM_BINS, void_bin=True)

        batch_size = 1

        locations = torch.from_numpy(normalized_locations).to(device).repeat(batch_size, 1, 1)
        gt_semantics_tensor = torch.tensor(gt_semantics, dtype=torch.long).unsqueeze(0)
        gt_color_tensor = torch.tensor(lab_image_discretized, dtype=torch.long).unsqueeze(0)
        lab_image_tensor = torch.tensor(lab_image_discretized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)  # Shape: (3, H, W)

        # Predictions from model
        preds_color_logits = model(locations, lab_image_tensor)
        # Extract the first sample from the batch
        assert preds_color_logits.shape[0] == batch_size * cfg.IMAGE_SIZE[0] * \
               cfg.IMAGE_SIZE[1], \
            "Should be num_locations * batch_size"
        assert preds_color_logits.shape[-1] == cfg.NUM_BINS, "Color logits should have bins as the last dimension"
        assert preds_color_logits.dtype == torch.float32, "Color logits are not class probabilities"

        # Reshape preds_color_logits
        preds_color_logits = preds_color_logits.view(batch_size, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3, cfg.NUM_BINS)
        print("Model forward pass successful")
        print(f"Predicted Color Logits Shape: {preds_color_logits.shape}")

        preds_colors = torch.argmax(preds_color_logits, dim=-1)  # Shape: (batchsize, H, W, 3)

        # Convert to NumPy array and translate to rgb
        preds_colors_np = preds_colors.cpu().numpy()
        preds_colors_rgb = lab_discretized_to_rgb(preds_colors_np[0], cfg.NUM_BINS, void_bin=True)
        print("After reshaping preds shape: ", preds_colors_np.shape)

        # Ground truth color
        gt_colors_np = gt_color_tensor.cpu().numpy()
        gt_colors_rgb = lab_discretized_to_rgb(gt_colors_np[0], cfg.NUM_BINS, void_bin=True)

        # Display gt vs pred
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(gt_colors_rgb)
        axs[0].set_title('Ground Truth Color')
        axs[1].imshow(preds_colors_rgb)
        axs[1].set_title('Predicted Color')
        plt.show()


        break


# Test model


# # Visualize results for the first sample in the batch
# idx = 0
# gt_color_rgb = lab_discretized_to_rgb(gt_colors[idx].cpu().numpy(), cfg.NUM_BINS)
# pred_color_rgb = preds_colors_rgb[idx]
# gt_semantics_image = gt_semantics[idx].cpu().numpy()
# pred_semantics_image = preds_semantics[idx].argmax(dim=0).cpu().numpy()  # Assuming logits for semantics
#
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
# # Gray image
# axs[0, 0].imshow(gray_images[idx].cpu().numpy().transpose(1, 2, 0), cmap='gray')
# axs[0, 0].set_title('Input Gray Image')
#
# # Ground truth LAB color image
# axs[0, 1].imshow(gt_color_rgb)
# axs[0, 1].set_title('Ground Truth Color (RGB)')
#
# # Predicted LAB color image
# axs[0, 2].imshow(pred_color_rgb)
# axs[0, 2].set_title('Predicted Color (RGB)')
#
# # Ground truth semantics
# axs[1, 0].imshow(gt_semantics_image, cmap='tab20')
# axs[1, 0].set_title('Ground Truth Semantics')
#
# # Predicted semantics
# axs[1, 1].imshow(pred_semantics_image, cmap='tab20')
# axs[1, 1].set_title('Predicted Semantics')
#
# # LAB image visualization (optional)
# axs[1, 2].imshow(lab_images[idx].cpu().numpy().transpose(1, 2, 0))
# axs[1, 2].set_title('Input LAB Image')
#
# plt.tight_layout()
# plt.show()

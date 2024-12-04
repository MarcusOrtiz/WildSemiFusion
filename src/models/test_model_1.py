import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data.utils.data_processing import image_to_array, lab_discretized_to_rgb
from src.data.rellis_2D_dataset import Rellis2DDataset
from torch.utils.data import DataLoader
from src.models.model_1 import MultiModalNetwork

import src.config as cfg

# Verify dataset
# for idx in range(len(train_dataset)):
#     print("Original RGB Shape: ", train_rgb_images[idx].shape)
#
#     sample = train_dataset[idx]
#     gt_semantics = sample['gt_semantics']
#     gt_color = sample['gt_color']
#     lab_image = sample['lab_image']
#     gray_image = sample['gray_image']
#
#     fig, axs = plt.subplots(3, 2, figsize=(15, 15))
#     print(f"GT Semantics Shaep: {gt_semantics.numpy().shape}")
#     print(f"GT Color Shape: {gt_color.numpy().shape}")
#     print(f"LAB Image Shape: {lab_image.numpy().shape}")
#     print(f"Gray Image Shape: {gray_image.numpy().shape}")
#     axs[0, 0].imshow(lab_discretized_to_rgb(gt_color.numpy(), cfg.NUM_BINS))
#     axs[0, 0].set_title('GT LAB Image to RGB')
#     axs[0, 1].imshow(lab_discretized_to_rgb(lab_image.numpy().transpose(1, 2, 0), cfg.NUM_BINS))
#     axs[0, 1].set_title('Masked Noisy LAB Image to RGB')
#
#     axs[1, 0].imshow(gray_image.numpy().transpose(1, 2, 0), cmap='gray')
#     axs[1, 0].set_title('Gray Image')
#     axs[1, 1].imshow(train_rgb_images[0])
#     axs[1, 1].set_title('Original RGB Image')
#
#     axs[2, 0].imshow(gt_semantics.to(torch.uint8).numpy())
#     axs[2, 0].set_title('GT Semantics')
#
#     plt.tight_layout()
#     plt.show()
#
#     break


# Verify dataloader
# for batch in train_dataloader:
#     # Access batch data
#     gt_semantics = batch['gt_semantics']
#     gt_color = batch['gt_color']
#     lab_image = batch['lab_image']
#     gray_image = batch['gray_image']
#
#     # Assuming `train_rgb_images` corresponds to a part of the dataset,
#     # visualize the first image in the batch for clarity
#     fig, axs = plt.subplots(2, 2, figsize=(15, 15))
#
#     print(f"GT Semantics Shape: {gt_semantics.shape}")
#     print(f"GT Color Shape: {gt_color.shape}")
#     print(f"LAB Image Shape: {lab_image.shape}")
#     print(f"Gray Image Shape: {gray_image.shape}")
#
#     axs[0, 0].imshow(lab_discretized_to_rgb(gt_color[0].numpy(), cfg.NUM_BINS))
#     axs[0, 0].set_title('GT LAB Image to RGB')
#     axs[0, 1].imshow(gt_semantics[0].to(torch.uint8).numpy())
#     axs[0, 1].set_title('GT Semantics')
#
#     axs[1, 0].imshow(lab_discretized_to_rgb(lab_image[0].numpy().transpose(1, 2, 0), cfg.NUM_BINS))
#     axs[1, 0].set_title('LAB Image to RGB')
#     axs[1, 1].imshow(gray_image[0].numpy().transpose(1, 2, 0), cmap='gray')
#     axs[1, 1].set_title('Gray Image')
#
#
#
#     plt.tight_layout()
#     plt.show()
#
#     # Break after verifying the first batch for simplicity
#     break
# Load preprocessed data
test_sequences = sorted([seq for seq in os.listdir(cfg.TEST_DIR) if not seq.startswith('.')])
test_rgb_images = []
test_semantics = []
for sequence in test_sequences:
    test_image_files = sorted(os.listdir(f"{cfg.TEST_DIR}/{sequence}/pylon_camera_node"))
    test_semantics_images = sorted(os.listdir(f"{cfg.TEST_DIR}/{sequence}/pylon_camera_node_label_id"))
    assert len(test_image_files) == len(test_semantics_images)
    for rgb_file, label_file in zip(test_image_files, test_semantics_images):
        assert rgb_file.split('.')[0] == label_file.split('.')[0]
    test_rgb_images.extend(
        [image_to_array(f"{cfg.TEST_DIR}/{sequence}/pylon_camera_node/{image_file}") for image_file in
         test_image_files])
    test_semantics.extend([image_to_array(f"{cfg.TEST_DIR}/{sequence}/pylon_camera_node_label_id/{label_file}", 1) for
                            label_file in test_semantics_images])
test_preloaded_data = {
    'rgb_images': test_rgb_images,
    'gt_semantics': test_semantics
}


# Create datasets
test_dataset = Rellis2DDataset(preloaded_data=test_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                               image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
print("Created testing dataset")

# Load the datasets
test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                             pin_memory=False, drop_last=True)
print("Created testing dataloader")

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultiModalNetwork(num_bins=cfg.NUM_BINS, num_classes=cfg.CLASSES)
print("Model initialized successfully")

model = model.to(device)
print("Model moved to device")

state_dict = torch.load(cfg.BEST_MODEL_PATH)
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

    for batch in test_dataloader:
        gt_semantics = batch['gt_semantics'].to(device)  # TODO: change dataset to follow format
        gt_colors = batch['gt_color'].to(device)  # TODO: change dataset to follow format
        gray_images = batch['gray_image'].to(device)
        lab_images = batch['lab_image'].to(device)
        batch_size = gt_semantics.shape[0]
        locations = torch.from_numpy(normalized_locations).to(device).repeat(batch_size, 1, 1)

        # Predictions from model
        preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)
        # Extract the first sample from the batch
        assert preds_semantics.shape[0] == preds_color_logits.shape[0] == cfg.BATCH_SIZE * cfg.IMAGE_SIZE[0] * \
               cfg.IMAGE_SIZE[1], \
            "Should be num_locations * batch_size"
        assert preds_semantics.shape[-1] == cfg.CLASSES, "Semantic logits should have classes as the last dimension"
        assert preds_color_logits.shape[-1] == cfg.NUM_BINS, "Color logits should have bins as the last dimension"
        assert preds_color_logits.dtype == torch.float32, "Color logits are not class probabilities"

        # Reshape preds_semantics
        preds_semantics = preds_semantics.view(batch_size, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], cfg.CLASSES)
        # Reshape preds_color_logits
        preds_color_logits = preds_color_logits.view(batch_size, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3, cfg.NUM_BINS)
        print("Model forward pass successful")
        print(f"Predicted Semantics Shape: {preds_semantics.shape}")
        print(f"Predicted Color Logits Shape: {preds_color_logits.shape}")

        preds_colors = torch.argmax(preds_color_logits, dim=-1)  # Shape: (batchsize, H, W, 3)

        # Convert to NumPy array and translate to rgb
        preds_colors_np = preds_colors.cpu().numpy()
        preds_colors_rgb = lab_discretized_to_rgb(preds_colors_np[0], cfg.NUM_BINS)
        print("After reshaping preds shape: ", preds_colors_np.shape)

        # Ground truth color
        gt_colors_np = gt_colors.cpu().numpy()
        gt_colors_rgb = lab_discretized_to_rgb(gt_colors_np[0], cfg.NUM_BINS)

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
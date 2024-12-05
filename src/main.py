import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.utils.data_processing import image_to_array, load_sequential_data
import numpy as np
import random
import src.config as cfg
from src.models.model_1 import MultiModalNetwork
from src.data.rellis_2D_dataset import Rellis2DDataset
from src.plotting import plot_base_losses
from matplotlib import pyplot as plt

"""
TODO: Set up argument parser for command-line flags for plotting and using previous weights
TODO: Consider adding confidence as output, this would also have an associated loss
TODO: Verify seed does not affect randomization used in the model and processing
"""

# Set the seed for random, NumPy, and PyTorch
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# Ensure the save directory exists
if not os.path.exists(cfg.SAVE_DIR_BASE):
    os.makedirs(cfg.SAVE_DIR_BASE)


# Initialize loss trackers
training_losses = []
validation_losses = []

training_losses_semantics = []
training_losses_color = []

validation_losses_semantics = []
validation_losses_color = []

t_losses = {
    'total': training_losses,
    'semantics': training_losses_semantics,
    'color': training_losses_color
}

v_losses = {
    'total': validation_losses,
    'semantics': validation_losses_semantics,
    'color': validation_losses_color
}

"""
Generate Locations (currently all pixels TODO: always consistent on number of locations between indexes)
Locations is somewhat misleading, it's actually the indices of the non-void pixels whereas WildFusion uses actual
Locations are normalized to [0, 1]
"""
y_coords, x_coords = np.meshgrid(np.arange(cfg.IMAGE_SIZE[0]), np.arange(cfg.IMAGE_SIZE[1]),
                                 indexing='ij')
locations_grid = np.stack([y_coords, x_coords], axis=-1).reshape(-1, 2)  # (IMAGE_SIZE[0]*IMAGE_SIZE[1], 2)
normalized_locations = locations_grid.astype(np.float32)
normalized_locations[:, 0] /= cfg.IMAGE_SIZE[0]
normalized_locations[:, 1] /= cfg.IMAGE_SIZE[1]


def train_val(model_local, dataloader, val_dataloader_local, epochs, lr, checkpoint_path, best_model_path):
    model_local.to(device)

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam([
            {'params': model_local.module.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
            {'params': model_local.module.color_fcn.parameters(), 'weight_decay': 1e-4},
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': model_local.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
            {'params': model_local.color_fcn.parameters(), 'weight_decay': 1e-4},
        ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')

    # Start timing
    start_time = time.time()

    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)  # TODO: Make sure to account for void class
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS-1)

    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0

    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)
    for epoch in range(start_epoch, epochs):
        model_local.train()
        epoch_loss = 0.0

        if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
            plot_base_losses(t_losses, v_losses)

        count = 0
        for batch in dataloader:
            count += 1

            # from src.data.utils.data_processing import lab_discretized_to_rgb
            # gt_semantics = batch['gt_semantics']
            # gt_color = batch['gt_color']
            # lab_image = batch['lab_image']
            # gray_image = batch['gray_image']
            #
            # # Assuming `train_rgb_images` corresponds to a part of the dataset,
            # # visualize the first image in the batch for clarity
            # fig, axs = plt.subplots(2, 2, figsize=(15, 15))
            #
            # print(f"GT Semantics Shape: {gt_semantics.shape}")
            # print(f"GT Color Shape: {gt_color.shape}")
            # print(f"LAB Image Shape: {lab_image.shape}")
            # print(f"Gray Image Shape: {gray_image.shape}")
            #
            # axs[0, 0].imshow(lab_discretized_to_rgb(gt_color[0].numpy(), cfg.NUM_BINS))
            # axs[0, 0].set_title('GT LAB Image to RGB')
            # axs[0, 1].imshow(gt_semantics[0].to(torch.uint8).numpy(), cmap='gray')
            # axs[0, 1].set_title('GT Semantics')
            #
            # axs[1, 0].imshow(lab_discretized_to_rgb(lab_image[0].numpy().transpose(1, 2, 0), cfg.NUM_BINS))
            # axs[1, 0].set_title('LAB Image to RGB')
            # axs[1, 1].imshow(gray_image[0].numpy().transpose(1, 2, 0), cmap='gray')
            # axs[1, 1].set_title('Gray Image')
            #
            #
            #
            # plt.tight_layout()
            # plt.show()


            # if (count < 10 or count % 10 == 0): print(f"Loading training batch {count}", flush=True)
            gt_semantics = batch['gt_semantics'].to(device)
            gt_color = batch['gt_color'].to(device)
            gray_images = batch['gray_image'].to(device)
            lab_images = batch['lab_image'].to(device)

            # Repeat locations along batch dimension
            batch_size = gt_semantics.shape[0]
            locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            # locations.requires_grad_(True)

            optimizer.zero_grad()

            # Predictions from model
            preds_semantics, preds_color_logits = model_local(locations, gray_images, lab_images)
            del locations, gray_images, lab_images

            # Semantic loss
            # print(f"Preds Semantics Shape: {preds_semantics.shape}")
            # print(f"GT Semantics Shape: {gt_semantics.long().view(-1).shape}")
            loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics, gt_semantics.long().view(-1))
            del preds_semantics, gt_semantics

            # Color loss
            # print(f"Preds Color Shape: {preds_color_logits.view(-1, cfg.NUM_BINS).shape}")
            # print(f"GT Color Shape: {gt_color.view(-1).shape}")
            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
            del preds_color_logits, gt_color

            # Total loss and optimization
            total_loss = loss_semantics + loss_color
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(dataloader)

        training_losses.append(average_epoch_loss)
        training_losses_semantics.append(loss_semantics.item())
        training_losses_color.append(loss_color.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss}")

        model_local.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_dataloader_local:
                count += 1
                # print(f"Loading validation batch {count}", flush=True)
                gt_semantics = batch['gt_semantics'].to(device)
                gt_color = batch['gt_color'].to(device)
                gray_images = batch['gray_image'].to(device)
                lab_images = batch['lab_image'].to(device)

                # Repeat locations along batch dimension
                batch_size = gt_semantics.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                # Predicting with model
                preds_semantics, preds_color_logits = model_local(locations, gray_images, lab_images)
                del locations, gray_images, lab_images

                # Semantic loss
                # print(f"Preds Semantics Shape: {preds_semantics.shape}")
                # print(f"GT Semantics Shape: {gt_semantics.long().view(-1).shape}")
                loss_semantics_val = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics,
                                                                               gt_semantics.long().view(-1))
                del preds_semantics, gt_semantics

                # Color loss
                # print(f"Preds Color Shape: {preds_color_logits.view(-1, cfg.NUM_BINS).shape}")
                # print(f"GT Color Shape: {gt_color.view(-1).shape}")
                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
                del preds_color_logits, gt_color

                val_loss += loss_semantics_val + loss_color_val

        average_val_loss = val_loss / len(val_dataloader_local)

        color_val_loss = loss_color_val.item()

        validation_losses.append(average_val_loss.item())
        validation_losses_semantics.append(loss_semantics_val.item())
        validation_losses_color.append(loss_color_val.item())

        print(f"Validation Loss: {average_val_loss.item()}")

        if color_val_loss < best_color_val_loss:
            best_color_val_loss = color_val_loss
            epochs_no_improve_color = 0
        else:
            epochs_no_improve_color += 1

        if (epochs_no_improve_color >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 200):
            print(
                f"Early stopping triggered at epoch {epoch + 1}. SDF validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            torch.save(model_local.state_dict(), best_model_path)
            print(f"Model saved at early stopping point with validation loss: {best_color_val_loss}")
            break

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model_local.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_loss}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_local.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': average_epoch_loss,
            'best_loss': best_loss
        }, checkpoint_path)

        scheduler.step(average_val_loss)

    # End timing and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds")

    return model_local


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")

else:
    device = torch.device("cpu")

print(f"Using device with cleared cache: {device}")

# Initialize model
model = MultiModalNetwork(cfg.NUM_BINS, cfg.CLASSES)
print("Model initialized successfully")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)
print("Model moved to device")

# Show memory reserved and allocated
# print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
# print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")

# Load preprocessed data
train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR)
val_preloaded_data = load_sequential_data(cfg.VAL_DIR)
print("Loaded training and validation preprocessed data")

# Create datasets
train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
print("Created training dataset")
val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                              image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
print("Created validation dataset")

# Load the datasets
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                              pin_memory=False, drop_last=True)
print("Created training dataloader")
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                            pin_memory=False, drop_last=True)
print("Created validation dataloader")

# Train and validate the model
trained_model = train_val(
    model,
    train_dataloader,
    val_dataloader,
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    checkpoint_path=cfg.CHECKPOINT_PATH_BASE,
    best_model_path=cfg.BEST_MODEL_PATH_BASE
)
print("Training complete")

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import src.config as cfg
from src.models.model_1 import MultiModalNetwork
from src.data.rellis_2D_dataset import Rellis2DDataset, custom_collate_fn

"""
TODO: Set up argument parser for command-line flags for plotting and using previous weights
TODO: Plot losses
TODO: Consider adding confidence as output, this would also have an associated loss
TODO: Verify seed does not affect randomization used in the model and processing
"""

# Set the seed for random, NumPy, and PyTorch
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# Ensure the save directory exists
if not os.path.exists(cfg.SAVE_DIR):
    os.makedirs(cfg.SAVE_DIR)

# Initialize loss trackers
training_losses = []
validation_losses = []

training_losses_semantics = []
training_losses_color = []

validation_losses_semantics = []
validation_losses_color = []

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


def train_val(model, dataloader, val_dataloader, epochs, lr, checkpoint_path, best_model_path):
    model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.module.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
        {'params': model.module.color_fcn.parameters(), 'weight_decay': 1e-4},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR,
                                                           patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')

    # Start timing
    start_time = time.time()

    criterion_mse = nn.MSELoss()
    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)  # TODO: Make sure to account for void class
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=312)  # TODO: Make sure to account for masking

    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            # TODO: Consider the dimensions later in the loss functions and see if they match the dataset loading
            locations = batch['locations'].to(device)
            gt_semantics = batch['gt_semantics'].to(device)  # TODO: change dataset to follow format
            gt_color = batch['gt_color'].to(device)  # TODO: change dataset to follow format
            gray_images = batch['gray_images'].to(device)
            lab_images = batch['lab_images'].to(device)

            locations.requires_grad_(True)

            optimizer.zero_grad()

            # Repeat locations along batch dimension
            batch_size = gt_semantics.shape[0]
            locations = torch.from_numpy(normalized_locations).to(device).repeat(batch_size, 1, 1)

            # Predictions from model
            preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)

            # Semantic loss
            loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics.view(-1, cfg.CLASSES),
                                                                           gt_semantics.long().view(-1))

            # Color loss
            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)

            # Total loss
            total_loss = loss_semantics + loss_color

            # Optimization
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(dataloader)

        training_losses.append(average_epoch_loss)
        training_losses_semantics.append(loss_semantics.item())
        training_losses_color.append(loss_color.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                gt_semantics = batch['gt_semantics'].to(device)  # TODO: change dataset to follow format
                gt_color = batch['gt_color'].to(device)  # TODO: change dataset to follow format
                gray_images = batch['gray_images'].to(device)
                lab_images = batch['lab_images'].to(device)

                # Repeat locations along batch dimension
                batch_size = gt_semantics.shape[0]
                locations = torch.from_numpy(normalized_locations).to(device).repeat(batch_size, 1, 1)

                # Predicting with model
                preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)

                # Semantic Loss | Assuming the semantic net is producing hot ones for classes
                loss_semantics_val = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(
                    preds_semantics.view(-1, cfg.CLASSES), gt_semantics.long().view(-1))

                # Color loss
                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)

                # Total loss
                val_loss += loss_semantics_val + loss_color_val

        average_val_loss = val_loss / len(val_dataloader)

        validation_losses.append(average_val_loss.item())
        validation_losses_semantics.append(loss_semantics_val.item())
        validation_losses_color.append(loss_color_val.item())

        print(f"Validation Loss: {average_val_loss.item()}")

        if loss_color_val < best_color_val_loss:
            best_color_val_loss = loss_color_val
            epochs_no_improve_color = 0
        else:
            epochs_no_improve_color += 1

        if (epochs_no_improve_color >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 100):
            print(
                f"Early stopping triggered at epoch {epoch + 1}. SDF validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at early stopping point with validation loss: {best_color_val_loss}")
            break

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_loss}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
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

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = MultiModalNetwork(cfg.NUM_BINS, cfg.CLASSES)
print("Model initialized successfully")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)
print("Model moved to device")

model = model.to(device)
print("Model moved to device")

# Show memory reserved and allocated
print(f"Memory reserved: {torch.cuda.memory_reserved()} bytes")
print(f"Memory allocated: {torch.cuda.memory_allocated()} bytes")

# TODO: Go through files correctly, maybe start preprocessing?
train_preprocessed_data = ...
val_preprocessed_data = ...

print("Data loaded successfully")

# TODO Change to new dataset
# Turns out that the bug isn't the batches in the later step its here where I am loading all the data
train_dataset = Rellis2DDataset(preloaded_data=train_preprocessed_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE, image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
val_dataset = Rellis2DDataset(preloaded_data=val_preprocessed_data, num_bins=cfg.NUM_BINS, points_per_scan=cfg.POINTS_PER_SCAN)
print("Datasets created successfully")

# Pass the datasets to the dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              collate_fn=custom_collate_fn, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True,
                            collate_fn=custom_collate_fn, drop_last=True)
print("DataLoaders created successfully")

# Train and validate the model
trained_model = train_val(
    model,
    train_dataloader,
    val_dataloader,
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    checkpoint_path=cfg.CHECKPOINT_PATH,
)
print("Training complete")

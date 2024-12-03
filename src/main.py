import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.utils.data_processing import image_to_array
import numpy as np
import random
import src.config as cfg
from src.models.model_1 import MultiModalNetwork
from src.data.rellis_2D_dataset import Rellis2DDataset

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

    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam([
            {'params': model.module.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
            {'params': model.module.color_fcn.parameters(), 'weight_decay': 1e-4},
        ], lr=lr)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
            {'params': model.color_fcn.parameters(), 'weight_decay': 1e-4},
        ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR,
                                                           patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')

    # Start timing
    start_time = time.time()

    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)  # TODO: Make sure to account for void class
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=312)  # TODO: Make sure to account for masking

    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0

    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        count = 0
        for batch in dataloader:
            count += 1
            print(f"Loading training batch {count}", flush=True)
            gt_semantics = batch['gt_semantics'].to(device)  # TODO: change dataset to follow format
            gt_color = batch['gt_color'].to(device)  # TODO: change dataset to follow format
            gray_images = batch['gray_image'].to(device)
            lab_images = batch['lab_image'].to(device)
            if torch.cuda.is_available():
                print(f"Allocated memory after batch load: {torch.cuda.memory_allocated()} bytes")
                print(f"Reserved memory after batch load: {torch.cuda.memory_reserved()} bytes")

            # Repeat locations along batch dimension
            batch_size = gt_semantics.shape[0]
            locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            if torch.cuda.is_available():
                print(f"Allocated memory after locations: {torch.cuda.memory_allocated()} bytes")
                print(f"Reserved memory after locations: {torch.cuda.memory_reserved()} bytes")
            # locations.requires_grad_(True)

            optimizer.zero_grad()

            # Predictions from model
            preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)
            if torch.cuda.is_available():
                print(f"Allocated memory after model: {torch.cuda.memory_allocated()} bytes")
                print(f"Reserved memory after model: {torch.cuda.memory_reserved()} bytes")
            del locations, gray_images, lab_images


            # Semantic loss
            # print(f"Preds Semantics Shape: {preds_semantics.shape}")
            # print(f"GT Semantics Shape: {gt_semantics.long().view(-1).shape}")
            loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics,
                                                                           gt_semantics.long().view(-1))
            del preds_semantics, gt_semantics

            # Color loss
            # print(f"Preds Color Shape: {preds_color_logits.view(-1, cfg.NUM_BINS).shape}")
            # print(f"GT Color Shape: {gt_color.view(-1).shape}")
            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
            del preds_color_logits, gt_color

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
        count = 0
        with torch.no_grad():
            for batch in val_dataloader:
                count += 1
                print(f"Loading validation batch {count}", flush=True)
                gt_semantics = batch['gt_semantics'].to(device)  # TODO: change dataset to follow format
                gt_color = batch['gt_color'].to(device)  # TODO: change dataset to follow format
                gray_images = batch['gray_image'].to(device)
                lab_images = batch['lab_image'].to(device)

                if torch.cuda.is_available():
                    print(f"Allocated memory after batch load: {torch.cuda.memory_allocated()} bytes")
                    print(f"Reserved memory after batch load: {torch.cuda.memory_reserved()} bytes")

                # Repeat locations along batch dimension
                batch_size = gt_semantics.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)
                if torch.cuda.is_available():
                    print(f"Allocated memory after locations: {torch.cuda.memory_allocated()} bytes")
                    print(f"Reserved memory after locations: {torch.cuda.memory_reserved()} bytes")


                # Predicting with model
                preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)
                if torch.cuda.is_available():
                    print(f"Allocated memory after model: {torch.cuda.memory_allocated()} bytes")
                    print(f"Reserved memory after model: {torch.cuda.memory_reserved()} bytes")
                del locations, gray_images, lab_images

                # Semantic loss
                # print(f"Preds Semantics Shape: {preds_semantics.shape}")
                # print(f"GT Semantics Shape: {gt_semantics.long().view(-1).shape}")
                loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics,
                                                                               gt_semantics.long().view(-1))
                del preds_semantics, gt_semantics

                # Color loss
                # print(f"Preds Color Shape: {preds_color_logits.view(-1, cfg.NUM_BINS).shape}")
                # print(f"GT Color Shape: {gt_color.view(-1).shape}")
                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
                del preds_color_logits, gt_color


                val_loss += loss_semantics + loss_color

        average_val_loss = val_loss / len(val_dataloader)

        validation_losses.append(average_val_loss.item())
        validation_losses_semantics.append(loss_semantics.item())
        validation_losses_color.append(loss_color.item())

        print(f"Validation Loss: {average_val_loss.item()}")

        if loss_color < best_color_val_loss:
            best_color_val_loss = loss_color
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

# TODO: Go through files correctly, maybe start preprocessing?
# # Load preprocessed data
# train_sequences = sorted(os.listdir(cfg.TRAIN_DIR))
# train_image_files = []
# train_semantics = []
# for sequence in train_sequences:
#     train_image_files.extend(sorted(os.listdir(f"{cfg.TRAIN_DIR}/{sequence}/pylon_camera_node")))
#     train_semantics.extend(sorted(os.listdir(f"{cfg.TRAIN_DIR}/{sequence}/pylon_camera_node_label_id")))
# print("Loaded training preprocessed data")
# val_sequences = sorted(os.listdir(cfg.VAL_DIR))
# val_image_files = []
# val_semantics = []
# for sequence in val_sequences:
#     val_image_files.extend(sorted(os.listdir(f"{cfg.VAL_DIR}/{sequence}/pylon_camera_node")))
#     val_semantics.extend(sorted(os.listdir(f"{cfg.VAL_DIR}/{sequence}/pylon_camera_node_label_id")))
# print("Loaded validation preprocessed data")




# train_image_files = sorted(os.listdir(f"{cfg.TRAIN_DIR}/pylon_camera_node"))
train_semantic_images = sorted(os.listdir(f"{cfg.TRAIN_DIR}/pylon_camera_node_label_id"))
train_rgb_images = [image_to_array(f"{cfg.TRAIN_DIR}/pylon_camera_node/{image_file}") for image_file in
                    train_image_files]
train_semantics = [image_to_array(f"{cfg.TRAIN_DIR}/pylon_camera_node_label_id/{label_file}", 1) for label_file in
                   train_semantic_images]
train_preloaded_data = {'rgb_images': train_rgb_images, 'gt_semantics': train_semantics}
val_image_files = sorted(os.listdir(f"{cfg.VAL_DIR}/pylon_camera_node"))
val_semantic_images = sorted(os.listdir(f"{cfg.VAL_DIR}/pylon_camera_node_label_id"))
val_rgb_images = [image_to_array(f"{cfg.VAL_DIR}/pylon_camera_node/{image_file}") for image_file in val_image_files]
val_semantics = [image_to_array(f"{cfg.VAL_DIR}/pylon_camera_node_label_id/{label_file}", 1) for label_file in
                 val_semantic_images]
val_preloaded_data = {'rgb_images': val_rgb_images, 'gt_semantics': val_semantics}
print("Loaded validation preprocessed data")

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
    checkpoint_path=cfg.CHECKPOINT_PATH,
    best_model_path=cfg.BEST_MODEL_PATH
)
print("Training complete")

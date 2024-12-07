import time
import os
import random
import torch
import torch.nn as nn
from src.data.utils.data_processing import load_sequential_data
from src.models.combined_1 import WeightedColorModel, DimensionalWeightedColorModel
from src.data.rellis_2D_dataset import Rellis2DDataset
import src.local_config as cfg
from torch.utils.data import DataLoader
import numpy as np

from src.plotting import plot_color_losses, plot_times
from src.utils import generate_normalized_locations, populate_random_seeds

populate_random_seeds()

if not os.path.exists(cfg.SAVE_DIR_COMBINED_WEIGHTED_COLOR):
    os.makedirs(cfg.SAVE_DIR_COMBINED_WEIGHTED_COLOR)

times = []
training_losses = []
validation_losses = []


def generate_plots(epoch):
    if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
        plot_color_losses(training_losses, validation_losses, cfg.SAVE_DIR_COMBINED_WEIGHTED_COLOR)
        plot_times(times, cfg.SAVE_DIR_COMBINED_WEIGHTED_COLOR)


def train_val(model, device, train_dataloader, val_dataloader, epochs, lr, save_dir: str):
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    model_module = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.Adam([
        {'params': [model_module.weights], 'weight_decay': 1e-4},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS - 1)
    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0

    normalized_locations = generate_normalized_locations()
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch)

        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            # if (idx < 2 or idx % 100 == 0): print(f"Loading training batch {idx}", flush=True)

            gt_semantics = batch['gt_semantics'].to(device)
            gt_color = batch['gt_color'].to(device)
            gray_images = batch['gray_image'].to(device)
            lab_images = batch['lab_image'].to(device)

            # Repeat locations along batch dimension
            batch_size = gt_semantics.shape[0]
            locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

            optimizer.zero_grad()

            preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)

            # Color loss
            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)

            total_loss = loss_color
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(train_dataloader)
        training_losses.append(average_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # print(f"Loading validation batch {batch_idx}", flush=True)
                gt_semantics = batch['gt_semantics'].to(device)
                gt_color = batch['gt_color'].to(device)
                gray_images = batch['gray_image'].to(device)
                lab_images = batch['lab_image'].to(device)

                # Repeat locations along batch dimension
                batch_size = gt_semantics.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                preds_semantics_logits, preds_color_logits = model(locations, gray_images, lab_images)

                # Color loss
                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)

                val_loss += loss_color_val

        average_val_loss = val_loss / len(val_dataloader)
        color_val_loss = average_val_loss.item()
        validation_losses.append(average_val_loss.item())
        times.append(time.time() - epoch_start_time)
        print(f"Validation Loss: {average_val_loss.item()}, total train time: {sum(times)}")

        if color_val_loss < best_color_val_loss:
            best_color_val_loss = color_val_loss
            epochs_no_improve_color = 0
        else:
            epochs_no_improve_color += 1

        if (epochs_no_improve_color >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 10):
            print(f"Early stop at epoch {epoch + 1}. Color validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
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

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        scheduler.step(average_val_loss)

    total_time = sum(times)
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = DimensionalWeightedColorModel(cfg.NUM_BINS, cfg.CLASSES, device)
    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs!")
    print(f"Model successfully initialized and moved to {device}")

    train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR)
    val_preloaded_data = load_sequential_data(cfg.VAL_DIR)
    print("Successfully loaded preprocessed training and validation data")

    train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                    image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                  image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                  pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                                pin_memory=cfg.PIN_MEMORY, drop_last=True)
    print(f"Created training dataloader with {len(train_dataset)} files and validation dataloader with {len(val_dataset)} files")

    # Train and validate the model
    trained_model = train_val(
        model,
        device,
        train_dataloader,
        val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COMBINED_WEIGHTED_COLOR
    )
    print("Training complete")


if __name__ == "__main__":
    main()

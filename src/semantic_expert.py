import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.utils.data_processing import image_to_array, load_sequential_data
import numpy as np
import random
import src.local_config as cfg
from src.models.experts import SemanticExpertModel
from src.data.rellis_2D_dataset import Rellis2DDataset
from src.plotting import plot_color_losses, plot_times
from src.utils import generate_normalized_locations, populate_random_seeds, model_to_device
from matplotlib import pyplot as plt

# Initialize loss trackers
training_losses = []
validation_losses = []
times = []


def generate_plots(epoch, save_dir):
    if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
        plot_color_losses(training_losses, validation_losses, save_dir)
        plot_times(times, save_dir)


def train_val(model, device, dataloader, val_dataloader, epochs, lr, save_dir: str):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    model_module = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.Adam([
        {'params': model_module.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR,
                                                           patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')
    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)
    best_semantics_val_loss = float('inf')
    epochs_no_improve_semantics = 0

    normalized_locations = generate_normalized_locations()
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, save_dir)

        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for idx, batch in enumerate(dataloader):
            # if (idx < 2 or idx % 100 == 0): print(f"Loading training batch {idx}", flush=True)

            gt_semantics = batch['gt_semantics'].to(device)
            gray_images = batch['gray_image'].to(device)

            # Repeat locations along batch dimension
            batch_size = gt_semantics.shape[0]
            locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

            optimizer.zero_grad()

            preds_semantics = model(locations, gray_images)
            del locations, gray_images

            gt_semantics = gt_semantics.long().view(-1)
            loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics, gt_semantics)
            del preds_semantics, gt_semantics

            total_loss = loss_semantics
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(dataloader)

        training_losses.append(average_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs})")
        print(f"Training Loss: {average_epoch_loss}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                # if (idx < 2 or idx % 100 == 0): print(f"Loading validation batch {idx}", flush=True)

                gt_semantics = batch['gt_semantics'].to(device)
                gray_images = batch['gray_image'].to(device)

                # Repeat locations along batch dimension
                batch_size = gt_semantics.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                preds_semantics = model(locations, gray_images)
                del locations, gray_images

                preds_semantics = preds_semantics
                gt_semantics = gt_semantics.long().view(-1)
                loss_semantics_val = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics, gt_semantics)
                del preds_semantics, gt_semantics

                val_loss += loss_semantics_val

        average_val_loss = val_loss / len(val_dataloader)
        semantics_val_loss = average_val_loss.item()
        validation_losses.append(average_val_loss.item())
        times.append(time.time() - epoch_start_time)

        print(f"Validation Loss: {average_val_loss.item()}, total train time: {sum(times)}")

        if semantics_val_loss < best_semantics_val_loss:
            best_semantics_val_loss = semantics_val_loss
            epochs_no_improve_semantics = 0
        else:
            epochs_no_improve_semantics += 1

        if (epochs_no_improve_semantics >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 200):
            print(f"Early stop at epoch {epoch + 1}. Semantics val loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at early stopping point with validation loss: {best_semantics_val_loss}")
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
    populate_random_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = SemanticExpertModel(cfg.NUM_BINS)
    model = model_to_device(model, device)

    # Create and load datasets
    train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR, cfg.TRAIN_FILES_LIMIT)
    val_preloaded_data = load_sequential_data(cfg.VAL_DIR, cfg.VAL_FILES_LIMIT)
    print("Successfully loaded preprocessed training and validation data")

    train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                    image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                  image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                  pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                                pin_memory=cfg.PIN_MEMORY, drop_last=True)
    print("Created training and validation dataloaders")

    # Train and validate the model
    trained_model = train_val(
        model,
        device,
        train_dataloader,
        val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_SEMANTICS_EXPERT
    )
    print("Training complete")


if __name__ == "__main__":
    main()

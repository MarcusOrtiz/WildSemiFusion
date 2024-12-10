import time
import os
import torch
import torch.nn as nn
import argparse
import importlib
from src.data.utils.data_processing import load_sequential_data
from src.models.color import WeightedColorModel, ChannelWeightedColorModel, ChannelBinWeightedColorModel, LinearColorModel
from src.data.rellis_2D_dataset import Rellis2DDataset
from torch.utils.data import DataLoader

from src.plotting import plot_losses, plot_times
from src.utils import generate_normalized_locations, populate_random_seeds, model_to_device


def generate_plots(epoch, training_losses, validation_losses, times, save_dir):
    if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
        plot_losses(training_losses, validation_losses, save_dir)
        plot_times(times, save_dir)


def train_val(model, device, optimizer, train_dataloader, val_dataloader, epochs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    training_losses = {
        'total': [],
        'semantics': [],
        'color': []
    }
    validation_losses = {
        'total': [],
        'semantics': [],
        'color': []
    }
    times = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS - 1)
    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)
    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0

    normalized_locations = generate_normalized_locations()
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, training_losses, validation_losses, times, save_dir)

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

            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)

            loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics.view(-1, cfg.CLASSES), gt_semantics.long().view(-1))

            total_loss = loss_color + loss_semantics
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(train_dataloader)

        training_losses['total'].append(average_epoch_loss)
        training_losses['semantics'].append(loss_semantics.item())
        training_losses['color'].append(loss_color.item())
        print(f"Epoch {epoch + 1}/{epochs})")
        print(f"Training Loss: {average_epoch_loss}")

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

                preds_semantics, preds_color_logits = model(locations, gray_images, lab_images)

                # Color loss
                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)

                loss_semantics_val = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics.view(-1, cfg.CLASSES), gt_semantics.long().view(-1))

                val_loss += loss_semantics_val + loss_color_val

        average_val_loss = val_loss / len(val_dataloader)
        color_val_loss = loss_color_val.item()

        validation_losses['total'].append(average_val_loss.item())
        validation_losses['semantics'].append(loss_semantics_val.item())
        validation_losses['color'].append(color_val_loss)
        times.append(time.time() - epoch_start_time)
        print(f"Validation Loss: {average_val_loss.item()}")
        print(f"Total time: {sum(times)}")

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
    populate_random_seeds()

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
    print(f"Created training dataloader with {len(train_dataset)} files and validation dataloader with {len(val_dataset)} files")

    # Train and validate each color model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weighted_model = WeightedColorModel(cfg.NUM_BINS, cfg.CLASSES, device)
    weighted_model = model_to_device(weighted_model, device)
    model_module = weighted_model.module if isinstance(weighted_model, nn.DataParallel) else weighted_model
    optimizer = torch.optim.Adam([
        {'params': model_module.color_fusion_weight, 'weight_decay': 1e-4},
    ], lr=cfg.LR)
    trained_weighted_model = train_val(
        weighted_model,
        device,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=cfg.EPOCHS,
        save_dir=cfg.SAVE_DIR_SEMANTICS_COLOR + "_weighted"
    )
    print("Training finished for weighted color model \n ---------------------")

    channel_weighted_model = ChannelWeightedColorModel(cfg.NUM_BINS, cfg.CLASSES, device)
    channel_weighted_model = model_to_device(channel_weighted_model, device)
    model_module = channel_weighted_model.module if isinstance(channel_weighted_model, nn.DataParallel) else channel_weighted_model
    optimizer = torch.optim.Adam([
        {'params': model_module.color_fusion_channel_weights.parameters(), 'weight_decay': 1e-4},
    ], lr=cfg.LR)
    trained_channel_weighted_model = train_val(
        channel_weighted_model,
        device,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=cfg.EPOCHS,
        save_dir=cfg.SAVE_DIR_SEMANTICS_COLOR + "_channel_weighted"
    )
    print("Training finished for channel weighted color model \n ---------------------")


    channel_bins_weighted_model = ChannelBinWeightedColorModel(cfg.NUM_BINS, cfg.CLASSES, device)
    channel_bins_weighted_model = model_to_device(channel_bins_weighted_model, device)
    model_module = channel_bins_weighted_model.module if isinstance(channel_bins_weighted_model, nn.DataParallel) else channel_bins_weighted_model
    optimizer = torch.optim.Adam([
        {'params': model_module.color_fusion_channel_bin_weights.parameters(), 'weight_decay': 1e-4},
    ], lr=cfg.LR)
    trained_channel_weighted_model = train_val(
        channel_bins_weighted_model,
        device,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=cfg.EPOCHS,
        save_dir=cfg.SAVE_DIR_SEMANTICS_COLOR + "channel_bins_weighted"
    )
    print("Training finished for channel and bins weighted color model \n ---------------------")

    linear_model = LinearColorModel(cfg.NUM_BINS, cfg.CLASSES, device)
    linear_model = model_to_device(linear_model, device)
    model_module = linear_model.module if isinstance(linear_model, nn.DataParallel) else linear_model
    optimizer = torch.optim.Adam([
        {'params': model_module.color_fusion_fc1.parameters(), 'weight_decay': 1e-4},
    ], lr=cfg.LR)
    trained_linear_model = train_val(
        linear_model,
        device,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs=cfg.EPOCHS,
        save_dir=cfg.SAVE_DIR_SEMANTICS_COLOR + "_linear"
    )
    print("Training finished for linear color model \n ---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a expert model foucsed on color prediction")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main()

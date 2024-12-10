import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.utils.data_processing import image_to_array, load_sequential_data
from src.models.experts import ColorExpertModel
from src.data.rellis_2D_dataset import Rellis2DDataset
from src.plotting import plot_color_loss, plot_times
from src.utils import generate_normalized_locations, populate_random_seeds, model_to_device
import argparse
import importlib


def generate_plots(epoch, training_losses, validation_losses, times, save_dir):
    if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
        plot_color_loss(training_losses, validation_losses, save_dir)
        plot_times(times, save_dir)


def train_val(model, device, train_dataloader, val_dataloader, epochs, lr, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    model_module = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.Adam([
        {'params': model_module.color_fcn.parameters(), 'weight_decay': 1e-4},
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR,
                                                           patience=cfg.PATIENCE)

    start_epoch = 0
    best_loss = float('inf')
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS - 1)
    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0
    training_losses, validation_losses, times = [], [], []

    normalized_locations = generate_normalized_locations()
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, training_losses, validation_losses, times, save_dir)

        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            # if (idx < 2 or idx % 100 == 0): print(f"Loading training batch {idx}", flush=True)

            gt_color = batch['gt_color'].to(device)
            lab_images = batch['lab_image'].to(device)

            # Repeat locations along batch dimension
            batch_size = gt_color.shape[0]
            locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

            optimizer.zero_grad()

            preds_color_logits = model(locations, lab_images)
            del locations, lab_images

            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
            del preds_color_logits, gt_color

            total_loss = loss_color
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(train_dataloader)

        training_losses.append(average_epoch_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {average_epoch_loss}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                # if (idx < 2 or idx % 100 == 0): print(f"Loading validation batch {idx}", flush=True)

                gt_color = batch['gt_color'].to(device)
                lab_images = batch['lab_image'].to(device)

                # Repeat locations along batch dimension
                batch_size = gt_color.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                preds_color_logits = model(locations, lab_images)
                del locations, lab_images

                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
                del preds_color_logits, gt_color

                val_loss += loss_color_val

        average_val_loss = val_loss / len(val_dataloader)
        color_val_loss = loss_color_val.item()

        validation_losses.append(average_val_loss.item())
        times.append(time.time() - epoch_start_time)
        print(f"Validation Loss: {average_val_loss.item()}")
        print(f"Total Time: {sum(times)}")

        if color_val_loss < best_color_val_loss:
            best_color_val_loss = color_val_loss
            epochs_no_improve_color = 0
        else:
            epochs_no_improve_color += 1

        if (epochs_no_improve_color >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 200):
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

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        scheduler.step(average_val_loss)

    total_time = sum(times)
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds")

    return model


def main():
    populate_random_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = ColorExpertModel(cfg.NUM_BINS)
    model = model_to_device(model, device)

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
        save_dir=cfg.SAVE_DIR_COLOR_EXPERT
    )
    print("Training complete for color expert model \n ---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a expert model foucsed on color prediction")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main()

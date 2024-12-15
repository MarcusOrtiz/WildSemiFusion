import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from src.data.utils.data_processing import load_sequential_data
from src.models.experts import ColorExpertModel
from src.data.rellis_2D_dataset import Rellis2DDataset
from src.plotting import generate_plots
from src.utils import generate_normalized_locations, populate_random_seeds, model_to_device, compile_model
import argparse
import importlib


def save_best_model(model, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    torch.save(model.lab_cnn.state_dict(), os.path.join(save_dir, "lab_cnn_model.pth"))
    torch.save(model.fourier_layer.state_dict(), os.path.join(save_dir, "fourier_layer_model.pth"))
    torch.save(model.compression_layer.state_dict(), os.path.join(save_dir, "compression_layer_model.pth"))
    torch.save(model.color_fcn.state_dict(), os.path.join(save_dir, "color_fcn_model.pth"))


def load_embeddings(model, device, embeddings_dir: str):
    fourier_layer_path = os.path.join(embeddings_dir, "fourier_layer_model.pth")
    lab_cnn_path = os.path.join(embeddings_dir, "lab_cnn_model.pth")

    model.fourier_layer.load_state_dict(torch.load(fourier_layer_path, map_location=device))
    model.lab_cnn.load_state_dict(torch.load(lab_cnn_path, map_location=device))


def freeze_embeddings(model):
    model.fourier_layer.eval()
    model.lab_cnn.eval()
    for param in model.fourier_layer.parameters():
        param.requires_grad = False
    for param in model.lab_cnn.parameters():
        param.requires_grad = False


def script_embeddings_inplace(model):
    model.fourier_layer = torch.jit.script(model.fourier_layer)
    model.lab_cnn = torch.jit.script(model.lab_cnn)


def train_val(model, device, train_dataloader, val_dataloader, epochs, lr, save_dir: str, use_checkpoint: bool):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    load_embeddings(model, device, cfg.SAVE_DIR_BASE)
    freeze_embeddings(model)
    script_embeddings_inplace(model)

    model = compile_model(model)

    optimizer = torch.optim.Adam([
        {'params': model.color_fcn.parameters(), 'weight_decay': 1e-4},
        {'params': model.compression_layer.parameters(), 'weight_decay': 1e-5},
    ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)
    scaler = GradScaler()

    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS - 1)
    training_losses = {'total': [], 'semantics': [], 'color': []}
    validation_losses = {'total': [], 'semantics': [], 'color': []}
    times = []
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if use_checkpoint and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_loss', best_val_loss)
        training_losses = checkpoint.get('training_losses', training_losses)
        validation_losses = checkpoint.get('validation_losses', validation_losses)
        times = checkpoint.get('times', times)

    normalized_locations = generate_normalized_locations(cfg.IMAGE_SIZE)
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, training_losses, validation_losses, times, save_dir, cfg.PLOT_INTERVAL)

        model.color_fcn.train()
        model.compression_layer.train()
        epoch_start_time = time.time()
        epoch_train_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            # if (idx < 2 or idx % 100 == 0): print(f"Loading training batch {idx}", flush=True)
            optimizer.zero_grad()
            with autocast():
                gt_color = batch['gt_color'].to(device, non_blocking=True)
                lab_images = batch['lab_image'].to(device, non_blocking=True)

                # Repeat locations along batch dimension
                batch_size = gt_color.shape[0]

                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                preds_color_logits = model(locations, lab_images)
                del locations, lab_images

                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.permute(0, 3, 1, 2)
                gt_color = gt_color.view(-1)
                loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
                del preds_color_logits, gt_color

                total_loss = loss_color

            epoch_train_loss += total_loss.item()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        average_epoch_train_loss = epoch_train_loss / len(train_dataloader)

        training_losses['total'].append(average_epoch_train_loss)
        training_losses['color'].append(average_epoch_train_loss)

        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        print(f"Training Loss: {average_epoch_train_loss}", flush=True)

        if torch.cuda.is_available() and not hasattr(model, "_torchdynamo_orig_callable"):
            torch.cuda.empty_cache()

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
                # if (idx < 2 or idx % 100 == 0): print(f"Loading validation batch {idx}", flush=True)
                with autocast():
                    gt_color = batch['gt_color'].to(device, non_blocking=True)
                    lab_images = batch['lab_image'].to(device, non_blocking=True)

                    # Repeat locations along batch dimension
                    batch_size = gt_color.shape[0]
                    locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                    preds_color_logits = model(locations, lab_images)
                    del locations, lab_images

                    preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                    gt_color = gt_color.view(-1)
                    loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
                    del preds_color_logits, gt_color

                    epoch_val_loss += loss_color_val.item()

        average_epoch_val_loss = epoch_val_loss / len(val_dataloader)

        validation_losses['total'].append(average_epoch_val_loss)
        validation_losses['color'].append(average_epoch_val_loss)
        times.append(time.time() - epoch_start_time)
        print(f"Validation Loss: {average_epoch_val_loss}", flush=True)
        print(f"Total Time: {sum(times)}", flush=True)


        epochs_no_improve += 1

        if average_epoch_val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = average_epoch_val_loss
            save_best_model(model, save_dir)
            print(f"New best model saved with validation loss: {best_val_loss}", flush=True)

        if (epochs_no_improve >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 10):
            print(f"Early stopping triggered at epoch {epoch + 1}. Validatio loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.", flush=True)
            print(f"Model saved at early stopping point with validation loss: {best_val_loss}", flush=True)
            break

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': average_epoch_val_loss,
                'best_loss': best_val_loss,
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'times': times
            }, checkpoint_path)


        if torch.cuda.is_available() and not hasattr(model, "_torchdynamo_orig_callable"):
            torch.cuda.empty_cache()

        scheduler.step(average_epoch_val_loss)

    total_time = sum(times)
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds", flush=True)

    return model


def main():
    populate_random_seeds(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColorExpertModel(cfg.NUM_BINS)
    model = model_to_device(model, device)
    print(f"Color expert model successfully initialized and moved to {device}")

    train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR, cfg.TRAIN_FILES_LIMIT)
    val_preloaded_data = load_sequential_data(cfg.VAL_DIR, cfg.VAL_FILES_LIMIT)
    print("Successfully loaded preprocessed training and validation data")

    train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                    image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                  image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE_COLOR_EXPERT, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                  pin_memory=cfg.PIN_MEMORY, drop_last=True, prefetch_factor=4)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE_COLOR_EXPERT, shuffle=False, num_workers=cfg.NUM_WORKERS,
                                pin_memory=cfg.PIN_MEMORY, drop_last=True, prefetch_factor=4)
    print(f"Created training dataloader with {len(train_dataset)} files and validation dataloader with {len(val_dataset)} files")

    # Train and validate the model
    trained_model = train_val(
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR_EXPERT,
        use_checkpoint=not args.scratch
    )
    print("Training complete for color expert model \n ---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a expert model foucsed on color prediction")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    parser.add_argument('--scratch', action='store_true', help='If not specified and checkpoint is stored, it will be used')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    torch.set_float32_matmul_precision('high')

    main()

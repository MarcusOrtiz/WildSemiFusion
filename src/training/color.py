import time
import sys
import os
import torch
import torch.nn as nn
import argparse
import importlib
from src.data.utils.data_processing import load_sequential_data
from src.models.color import ColorModelSimple, ColorModelLinear, ColorModelMLP
from src.data.rellis_2D_dataset import Rellis2DDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from src.models.base import BaseModel
from src.models.experts import ColorExpertModel

from src.plotting import generate_plots
from src.utils import generate_normalized_locations, populate_random_seeds, model_to_device, compile_model, generate_loss_trackers, update_loss_trackers



def save_best_model(model, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))


def load_sub_models(device, base_dir: str, color_expert_dir: str):
    base_path = os.path.join(base_dir, "best_model.pth")
    color_expert_path = os.path.join(color_expert_dir, "best_model.pth")

    base_model = BaseModel(cfg.NUM_BINS, cfg.CLASSES)
    color_expert_model = ColorExpertModel(cfg.NUM_BINS)

    base_model_state_dict = torch.load(base_path, map_location=device)
    base_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in base_model_state_dict.items()}
    base_model.load_state_dict(base_model_state_dict)
    color_expert_model_state_dict = torch.load(color_expert_path, map_location=device)
    color_expert_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in color_expert_model_state_dict.items()}
    color_expert_model.load_state_dict(color_expert_model_state_dict)
    print(f"Loaded base model from {base_path}, color expert from {color_expert_path}", flush=True)

    base_model = model_to_device(base_model, device)
    color_expert_model = model_to_device(color_expert_model, device)
    print(f"Moved base model and color expert to {device}", flush=True)

    return base_model, color_expert_model


def freeze_sub_models(base_model, color_expert_model):
    base_model.eval()
    color_expert_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    for param in color_expert_model.parameters():
        param.requires_grad = False


def script_sub_models(base_model, color_expert_model):
    base_model = torch.jit.script(base_model)
    color_expert_model = torch.jit.script(color_expert_model)
    return base_model, color_expert_model


def train_val(color_model, base_model, color_expert_model, device, train_dataloader, val_dataloader, epochs, lr, save_dir: str, use_checkpoint: bool):
    model_type = color_model.__class__.__name__.split("Model")[-1]
    os.makedirs(save_dir+model_type, exist_ok=True)
    checkpoint_path = os.path.join(save_dir+model_type, "checkpoint.pth")

    color_model = compile_model(color_model)

    optimizer = torch.optim.Adam([
        {'params': color_model.parameters(), 'lr': lr, 'weight_decay': 1e-4}, ], lr=lr)
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        color_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        training_losses = checkpoint.get('training_losses', [])
        validation_losses = checkpoint.get('validation_losses', [])
        times = checkpoint.get('times', [])

    normalized_locations = generate_normalized_locations(cfg.IMAGE_SIZE)
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, training_losses, validation_losses, times, save_dir, cfg.PLOT_INTERVAL)

        color_model.train()
        epoch_start_time = time.time()
        sub_model_time = 0
        epoch_train_loss = 0.0
        for idx, batch in enumerate(train_dataloader):
            # if (idx < 2 or idx % 100 == 0): print(f"Loading training batch {idx}", flush=True)
            optimizer.zero_grad()
            with autocast():
                gray_images = batch['gray_image'].to(device)
                lab_images = batch['lab_image'].to(device)

                # Repeat locations along batch dimension
                batch_size = gray_images.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                sub_model_start_time = time.time()
                preds_semantics_base, preds_color_base = base_model(locations, gray_images, lab_images)
                del gray_images
                preds_color_expert = color_expert_model(locations, lab_images)
                del lab_images, locations
                sub_model_time = time.time() - sub_model_start_time

                gt_color = batch['gt_color'].to(device)

                preds_semantics, preds_color = color_model(preds_semantics_base, preds_color_base, preds_color_expert)
                del preds_semantics_base, preds_color_base, preds_color_expert
                loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                del gt_color

                total_loss = loss_color

            epoch_train_loss += total_loss.item()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        average_epoch_train_loss = epoch_train_loss / len(train_dataloader)

        training_losses['total'].append(average_epoch_train_loss)
        training_losses['color'].append(average_epoch_train_loss)

        print(f"Epoch {epoch + 1}/{epochs} for {model_type} model)", flush=True)
        print(f"Training Loss Simple: {average_epoch_train_loss}", flush=True)

        if torch.cuda.is_available() and not hasattr(color_model, '_torchdynamo_orig_callable'):
            torch.cuda.empty_cache()


        color_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                with autocast():
                    gray_images = batch['gray_image'].to(device)
                    lab_images = batch['lab_image'].to(device)

                    # Repeat locations along batch dimension
                    batch_size = gray_images.shape[0]
                    locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                    sub_model_start_time = time.time()
                    preds_semantics_base, preds_color_base = base_model(locations, gray_images, lab_images)
                    del gray_images
                    preds_color_expert = color_expert_model(locations, lab_images)
                    del lab_images, locations
                    sub_model_time += time.time() - sub_model_start_time

                    gt_color = batch['gt_color'].to(device)

                    preds_semantics, preds_color = color_model(preds_semantics_base, preds_color_base, preds_color_expert)
                    del preds_semantics_base, preds_color_base, preds_color_expert
                    loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                    del gt_color

                    epoch_val_loss += loss_color_val.item()

        average_epoch_val_loss = epoch_val_loss / len(val_dataloader)

        validation_losses['total'].append(average_epoch_val_loss)
        validation_losses['color'].append(average_epoch_val_loss)
        times.append((time.time() - epoch_start_time) - sub_model_time)
        print(f"Validation Loss Simple: {average_epoch_val_loss}", flush=True)
        print(f"Time: {sum(times)}", flush=True)

        epochs_no_improve += 1

        if average_epoch_val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = average_epoch_val_loss
            save_best_model(color_model, save_dir+model_type)
            print(f"New best {model_type} model saved with validation loss: {best_val_loss}")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': color_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'times': times
            }, checkpoint_path)

        if (epochs_no_improve >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 50):
            total_time = sum(times)

            print(f"Early stop at epoch {epoch + 1} for {model_type} model. Val loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs)")
            print(f"Average time per epoch: {total_time / epochs}")
            print(f"Best validation loss: {best_val_loss}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scheduler.step(average_epoch_val_loss)


    total_time = sum(times)
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds", flush=True)
    print(f"Stopping since {model_type} since all epochs are done)", flush=True)
    print(f"Main model training time: {total_time}", flush=True)
    print(f"Average time per epoch: {total_time / epochs}", flush=True)
    print(f"Best validation loss: {best_val_loss}", flush=True)

    return color_model


def main():
    populate_random_seeds(cfg.SEED)

    train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR, cfg.TRAIN_FILES_LIMIT)
    val_preloaded_data = load_sequential_data(cfg.VAL_DIR, cfg.VAL_FILES_LIMIT)
    print("Successfully loaded preprocessed training and validation data")

    train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                    image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                  image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE_COLOR, shuffle=True, num_workers=cfg.NUM_WORKERS,
                                  pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE_COLOR, shuffle=False, num_workers=cfg.NUM_WORKERS,
                                pin_memory=cfg.PIN_MEMORY, drop_last=True)
    print(f"Created training dataloader with {len(train_dataset)} files and validation dataloader with {len(val_dataset)} files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, color_expert_model = load_sub_models(device, base_dir=cfg.SAVE_DIR_BASE, color_expert_dir=cfg.SAVE_DIR_COLOR_EXPERT)
    freeze_sub_models(base_model=base_model, color_expert_model=color_expert_model)
    base_model, color_expert_model = script_sub_models(base_model=base_model, color_expert_model=color_expert_model)

    color_model_simple = ColorModelSimple(num_bins=cfg.NUM_BINS)
    color_model_simple = model_to_device(color_model_simple, device)
    trained_color_model_simple = train_val(
        color_model=color_model_simple,
        base_model=base_model,
        color_expert_model=color_expert_model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR,
        use_checkpoint=not args.scratch
    )
    print("Training finished for Simple Color Model \n ---------------------", flush=True)
    del color_model_simple, trained_color_model_simple

    color_model_linear = ColorModelLinear(num_bins=cfg.NUM_BINS)
    color_model_linear = model_to_device(color_model_linear, device)
    trained_color_model_linear = train_val(
        color_model=color_model_linear,
        base_model=base_model,
        color_expert_model=color_expert_model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR,
        use_checkpoint=not args.scratch
    )
    print("Training finished for Linear Color Model \n ---------------------", flush=True)
    del color_model_linear, trained_color_model_linear

    color_model_mlp = ColorModelMLP(num_bins=cfg.NUM_BINS)
    color_model_mlp = model_to_device(color_model_mlp, device)
    trained_color_model_mlp = train_val(
        color_model=color_model_mlp,
        base_model=base_model,
        color_expert_model=color_expert_model,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR,
        use_checkpoint=not args.scratch
    )
    print("Training finished for MLP Color Model \n ---------------------", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a expert model foucsed on color prediction")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    parser.add_argument('--scratch', action='store_true', help='If not specified and checkpoint is stored, it will be used')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main()

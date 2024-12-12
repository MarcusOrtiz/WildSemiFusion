import time
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


def load_checkpoint(model, device, optimizer, scheduler, save_dir: str):
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    training_losses = checkpoint.get('training_losses', [])
    validation_losses = checkpoint.get('validation_losses', [])
    times = checkpoint.get('times', [])
    return start_epoch, best_loss, training_losses, validation_losses, times


def save_best_model(model, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    # Add the sub models


def load_sub_models(device, base_dir: str, color_expert_dir: str):
    base_path = os.path.join(base_dir, "best_model.pth")
    print(f"Loading base model from {base_path}")
    color_expert_path = os.path.join(color_expert_dir, "best_model.pth")
    print(f"Loading color expert model from {color_expert_path}")

    base_model = BaseModel(cfg.NUM_BINS, cfg.CLASSES)
    color_expert_model = ColorExpertModel(cfg.NUM_BINS)

    base_model_state_dict = torch.load(base_path, map_location=device)
    base_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in base_model_state_dict.items()}
    base_model.load_state_dict(base_model_state_dict)
    color_expert_model_state_dict = torch.load(color_expert_path, map_location=device)
    color_expert_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in color_expert_model_state_dict.items()}
    color_expert_model.load_state_dict(color_expert_model_state_dict)

    base_model = model_to_device(base_model, device)
    color_expert_model = model_to_device(color_expert_model, device)
    return base_model, color_expert_model


def freeze_script_compile_sub_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # model = torch.jit.script(model)
    return compile_model(model)


def create_optimization(model, lr):
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': 1e-5}, ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)
    scaler = GradScaler()
    return optimizer, scheduler, scaler


def create_directories(save_dir: str):
    save_dir_simple = save_dir + "_simple"
    save_dir_linear = save_dir + "_linear"
    save_dir_mlp = save_dir + "_mlp"
    os.makedirs(save_dir_simple, exist_ok=True)
    os.makedirs(save_dir_linear, exist_ok=True)
    os.makedirs(save_dir_mlp, exist_ok=True)
    return save_dir_simple, save_dir_linear, save_dir_mlp


def train_val(model_simple, model_linear, model_mlp, device, train_dataloader, val_dataloader, epochs, lr, save_dir: str, use_checkpoint: bool):
    save_dir_simple, save_dir_linear, save_dir_mlp = create_directories(save_dir)

    base_model, color_expert_model = load_sub_models(device, cfg.SAVE_DIR_BASE, cfg.SAVE_DIR_COLOR_EXPERT)
    base_model = freeze_script_compile_sub_model(base_model)
    color_expert_model = freeze_script_compile_sub_model(color_expert_model)

    # model_simple = compile_model(model_simple)
    # model_linear = compile_model(model_linear)

    optimizer_simple, scheduler_simple, scaler_simple = create_optimization(model_simple, lr)
    optimizer_linear, scheduler_linear, scaler_linear = create_optimization(model_linear, lr)
    optimizer_mlp, scheduler_mlp, scaler_mlp = create_optimization(model_mlp, lr)

    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS - 1)
    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)
    best_color_val_loss_simple, best_color_val_loss_linear, best_color_val_loss_mlp = float('inf'), float('inf'), float('inf')
    epochs_no_improve_color_simple, epochs_no_improve_color_linear, epochs_no_improve_color_mlp = 0, 0, 0
    training_losses_simple, validation_losses_simple, times_simple = generate_loss_trackers()
    training_losses_linear, validation_losses_linear, times_linear = generate_loss_trackers()
    training_losses_mlp, validation_losses_mlp, times_mlp = generate_loss_trackers()

    best_loss_simple, best_loss_linear, best_loss_mlp = float('inf'), float('inf'), float('inf')
    start_epoch = 0

    if use_checkpoint and os.path.exists(os.path.join(save_dir_simple, "checkpoint.pth")):
        _, best_loss_simple, training_losses_simple, validation_losses_simple, times_simple = load_checkpoint(model_simple, device, optimizer_simple,
                                                                                                              scheduler_simple, save_dir_simple)
    if use_checkpoint and os.path.exists(os.path.join(save_dir_linear, "checkpoint.pth")):
        _, best_loss_linear, training_losses_linear, validation_losses_linear, times_linear = load_checkpoint(model_linear, device, optimizer_linear,
                                                                                                              scheduler_linear, save_dir_linear)
    if use_checkpoint and os.path.exists(os.path.join(save_dir_mlp, "checkpoint.pth")):
        _, best_loss_mlp, training_losses_mlp, validation_losses_mlp, times_mlp = load_checkpoint(model_mlp, device, optimizer_mlp,
                                                                                                  scheduler_mlp, save_dir_mlp)

    normalized_locations = generate_normalized_locations()
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, training_losses_simple, validation_losses_simple, times_simple, save_dir_simple)
        generate_plots(epoch, training_losses_linear, validation_losses_linear, times_linear, save_dir_linear)
        generate_plots(epoch, training_losses_mlp, validation_losses_mlp, times_mlp, save_dir_mlp)

        model_simple.train()
        model_linear.train()
        model_mlp.train()
        epoch_loss_simple, epoch_loss_linear, epoch_loss_mlp = 0.0, 0.0, 0.0
        for idx, batch in enumerate(train_dataloader):
            # if (idx < 2 or idx % 100 == 0): print(f"Loading training batch {idx}", flush=True)
            optimizer_simple.zero_grad()
            optimizer_linear.zero_grad()
            optimizer_mlp.zero_grad()
            with autocast():
                gray_images = batch['gray_image'].to(device)
                lab_images = batch['lab_image'].to(device)

                # Repeat locations along batch dimension
                batch_size = gray_images.shape[0]
                locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                with torch.no_grad():
                    preds_semantics_base, preds_color_base = base_model(locations, gray_images, lab_images)
                    del gray_images
                    preds_color_expert = color_expert_model(locations, lab_images)
                    del lab_images, locations

                gt_semantics = batch['gt_semantics'].to(device)
                gt_color = batch['gt_color'].to(device)

                simple_start_time = time.time()
                preds_semantics_simple, preds_color_simple = model_simple(preds_semantics_base, preds_color_base, preds_color_expert)
                loss_semantics_simple = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics_simple, gt_semantics.long().view(-1))
                loss_color_simple = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_simple.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                del preds_semantics_simple, preds_color_simple
                times_simple.append(time.time() - simple_start_time)

                linear_start_time = time.time()
                preds_semantics_linear, preds_color_linear = model_linear(preds_semantics_base, preds_color_base, preds_color_expert)
                loss_semantics_linear = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics_linear, gt_semantics.long().view(-1))
                loss_color_linear = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_linear.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                del preds_semantics_linear, preds_color_linear
                times_linear.append(time.time() - linear_start_time)

                mlp_start_time = time.time()
                preds_semantics_mlp, preds_color_mlp = model_mlp(preds_semantics_base, preds_color_base, preds_color_expert)
                loss_semantics_mlp = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics_mlp, gt_semantics.long().view(-1))
                loss_color_mlp = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_mlp.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                del preds_semantics_mlp, preds_color_mlp
                times_mlp.append(time.time() - mlp_start_time)

                del preds_semantics_base, preds_color_base, preds_color_expert, gt_semantics, gt_color

                total_loss_simple = loss_semantics_simple + loss_color_simple
                total_loss_linear = loss_semantics_linear + loss_color_linear
                total_loss_mlp = loss_semantics_mlp + loss_color_mlp

            simple_start_time = time.time()
            epoch_loss_simple += total_loss_simple.item()
            scaler_simple.scale(total_loss_simple).backward()
            scaler_simple.step(optimizer_simple)
            scaler_simple.update()
            times_simple[-1] += time.time() - simple_start_time

            linear_start_time = time.time()
            times_simple.append(time.time() - simple_start_time)
            epoch_loss_linear += total_loss_linear.item()
            scaler_linear.scale(total_loss_linear).backward()
            scaler_linear.step(optimizer_linear)
            scaler_linear.update()
            times_linear[-1] += time.time() - linear_start_time

            mlp_start_time = time.time()
            epoch_loss_mlp += total_loss_mlp.item()
            scaler_mlp.scale(total_loss_mlp).backward()
            scaler_mlp.step(optimizer_mlp)
            scaler_mlp.update()
            times_mlp[-1] += time.time() - mlp_start_time

        average_epoch_loss_simple = epoch_loss_simple / len(train_dataloader)
        average_epoch_loss_linear = epoch_loss_linear / len(train_dataloader)
        average_epoch_loss_mlp = epoch_loss_mlp / len(train_dataloader)

        update_loss_trackers(training_losses_simple, average_epoch_loss_simple, loss_semantics_simple.item(), loss_color_simple.item())
        update_loss_trackers(training_losses_linear, average_epoch_loss_linear, loss_semantics_linear.item(), loss_color_linear.item())
        update_loss_trackers(training_losses_mlp, average_epoch_loss_mlp, loss_semantics_mlp.item(), loss_color_mlp.item())

        print(f"Epoch {epoch + 1}/{epochs})")
        print(f"Training Loss Simple: {average_epoch_loss_simple}")
        print(f"Training Loss Linear: {average_epoch_loss_linear}")
        print(f"Training Loss MLP: {average_epoch_loss_mlp}")

        if torch.cuda.is_available() and not hasattr(model_simple, '_torchdynamo_orig_callable') and not hasattr(model_linear, '_torchdynamo_orig_callable') and not hasattr(model_mlp, '_torchdynamo_orig_callable'):
            torch.cuda.empty_cache()

        for model in [model_simple, model_linear, model_mlp]:
            model.eval()

        val_loss_simple, val_loss_linear, val_loss_mlp = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                with autocast():
                    gray_images = batch['gray_image'].to(device)
                    lab_images = batch['lab_image'].to(device)

                    # Repeat locations along batch dimension
                    batch_size = gray_images.shape[0]
                    locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                    with torch.inference_mode():
                        preds_semantics_base, preds_color_base = base_model(locations, gray_images, lab_images)
                        del gray_images
                        preds_color_expert = color_expert_model(locations, lab_images)
                        del lab_images, locations

                    gt_semantics = batch['gt_semantics'].to(device)
                    gt_color = batch['gt_color'].to(device)

                    simple_start_time = time.time()
                    preds_semantics_simple, preds_color_simple = model_simple(preds_semantics_base, preds_color_base, preds_color_expert)
                    loss_semantics_val_simple = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics_simple, gt_semantics.long().view(-1))
                    loss_color_val_simple = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_simple.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                    del preds_semantics_simple, preds_color_simple
                    times_simple[-1] += time.time() - simple_start_time

                    linear_start_time = time.time()
                    preds_semantics_linear, preds_color_linear = model_linear(preds_semantics_base, preds_color_base, preds_color_expert)
                    loss_semantics_val_linear = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics_linear, gt_semantics.long().view(-1))
                    loss_color_val_linear = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_linear.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                    del preds_semantics_linear, preds_color_linear
                    times_linear[-1] += time.time() - linear_start_time

                    mlp_start_time = time.time()
                    preds_semantics_mlp, preds_color_mlp = model_mlp(preds_semantics_base, preds_color_base, preds_color_expert)
                    loss_semantics_val_mlp = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics_mlp, gt_semantics.long().view(-1))
                    loss_color_val_mlp = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_mlp.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                    del preds_semantics_mlp, preds_color_mlp
                    times_mlp[-1] += time.time() - mlp_start_time

                    del preds_semantics_base, preds_color_base, preds_color_expert, gt_semantics, gt_color

                    val_loss_simple += loss_semantics_simple + loss_color_simple
                    val_loss_linear += loss_semantics_linear + loss_color_linear
                    val_loss_mlp += loss_semantics_mlp + loss_color_mlp

            start_time = time.time()
            average_val_loss_simple = val_loss_simple / len(val_dataloader)
            average_val_loss_linear = val_loss_linear / len(val_dataloader)
            average_val_loss_mlp = val_loss_mlp / len(val_dataloader)
            color_val_loss_simple = loss_color_val_simple.item()
            color_val_loss_linear = loss_color_val_linear.item()
            color_val_loss_mlp = loss_color_val_mlp.item()
            semantics_val_loss_simple = loss_semantics_val_simple.item()
            semantics_val_loss_linear = loss_semantics_val_linear.item()
            semantics_val_loss_mlp = loss_semantics_val_mlp.item()

            update_loss_trackers(validation_losses_simple, average_val_loss_simple, semantics_val_loss_simple, color_val_loss_simple)
            update_loss_trackers(validation_losses_linear, average_val_loss_linear, semantics_val_loss_linear, color_val_loss_linear)
            update_loss_trackers(validation_losses_mlp, average_val_loss_mlp, semantics_val_loss_mlp, color_val_loss_mlp)
            times_simple[-1] += time.time() - start_time
            times_linear[-1] += time.time() - start_time
            times_mlp[-1] += time.time() - start_time

            print(f"Validation Loss Simple: {average_val_loss_simple}")
            print(f"Validation Loss Linear: {average_val_loss_linear}")
            print(f"Validation Loss MLP: {average_val_loss_mlp}")
            print(f"Training Time Simple: {sum(times_simple)}")
            print(f"Training Time Linear: {sum(times_linear)}")
            print(f"Training Time MLP: {sum(times_mlp)}")

        if color_val_loss_simple < best_color_val_loss_simple:
            best_color_val_loss_simple = color_val_loss_simple
            epochs_no_improve_color_simple = 0
        else:
            epochs_no_improve_color_simple += 1

        if color_val_loss_linear < best_color_val_loss_linear:
            best_color_val_loss_linear = color_val_loss_linear
            epochs_no_improve_color_linear = 0
        else:
            epochs_no_improve_color_linear += 1

        if color_val_loss_mlp < best_color_val_loss_mlp:
            best_color_val_loss_mlp = color_val_loss_mlp
            epochs_no_improve_color_mlp = 0
        else:
            epochs_no_improve_color_mlp += 1

        if average_val_loss_simple.item() < best_loss_simple:
            best_loss_simple = average_val_loss_simple
            save_best_model(model_simple, save_dir_simple)
            print(f"New best model saved with validation loss: {best_loss_simple}")

        if average_val_loss_linear.item() < best_loss_linear:
            best_loss_linear = average_val_loss_linear
            save_best_model(model_linear, save_dir_linear)
            print(f"New best model saved with validation loss: {best_loss_linear}")

        if average_val_loss_mlp.item() < best_loss_mlp:
            best_loss_mlp = average_val_loss_mlp
            save_best_model(model_mlp, save_dir_mlp)
            print(f"New best model saved with validation loss: {best_loss_mlp}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_simple.state_dict(),
            'optimizer_state_dict': optimizer_simple.state_dict(),
            'scheduler_state_dict': scheduler_simple.state_dict(),
            'loss': average_epoch_loss_simple,
            'best_loss': best_loss_simple
        }, os.path.join(save_dir_simple, "checkpoint.pth"))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_linear.state_dict(),
            'optimizer_state_dict': optimizer_linear.state_dict(),
            'scheduler_state_dict': scheduler_linear.state_dict(),
            'loss': average_epoch_loss_linear,
            'best_loss': best_loss_linear
        }, os.path.join(save_dir_linear, "checkpoint.pth"))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model_mlp.state_dict(),
            'optimizer_state_dict': optimizer_mlp.state_dict(),
            'scheduler_state_dict': scheduler_mlp.state_dict(),
            'loss': average_epoch_loss_mlp,
            'best_loss': best_loss_mlp
        }, os.path.join(save_dir_mlp, "checkpoint.pth"))

        if (epochs_no_improve_color_simple >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 75):
            print(f"Early stop at epoch {epoch + 1}. Color validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            print(f"Model saved at early stopping point with validation loss: {best_color_val_loss_simple}")

        if (epochs_no_improve_color_linear >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 75):
            print(f"Early stop at epoch {epoch + 1}. Color validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            print(f"Model saved at early stopping point with validation loss: {best_color_val_loss_linear}")

        if (epochs_no_improve_color_mlp >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 75):
            print(f"Early stop at epoch {epoch + 1}. Color validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            print(f"Model saved at early stopping point with validation loss: {best_color_val_loss_mlp}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        scheduler_simple.step(average_val_loss_simple)
        scheduler_linear.step(average_val_loss_linear)
        scheduler_mlp.step(average_val_loss_mlp)

    # total_time = sum(times)
    total_time = 0
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds")

    return model_simple, model_linear, model_mlp


def main():
    populate_random_seeds()

    train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR, cfg.TRAIN_FILES_LIMIT)
    val_preloaded_data = load_sequential_data(cfg.VAL_DIR, cfg.VAL_FILES_LIMIT)
    print("Successfully loaded preprocessed training and validation data")

    train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                    image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                  image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE_COLOR, shuffle=True, num_workers=0,
                                  pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE_COLOR, shuffle=False, num_workers=0,
                                pin_memory=cfg.PIN_MEMORY, drop_last=True)
    print(f"Created training dataloader with {len(train_dataset)} files and validation dataloader with {len(val_dataset)} files")

    # Train and validate each color model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_simple = ColorModelSimple(cfg.NUM_BINS)
    model_linear = ColorModelLinear(cfg.NUM_BINS)
    model_mlp = ColorModelMLP(cfg.NUM_BINS)
    model_simple = model_to_device(model_simple, device)
    model_linear = model_to_device(model_linear, device)
    model_mlp = model_to_device(model_mlp, device)

    trained_simple_model, trained_linear_model, trained_mlp_model = train_val(
        model_simple=model_simple,
        model_linear=model_linear,
        model_mlp=model_mlp,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR,
        use_checkpoint=not args.scratch
    )

    print("Training finished for color models \n ---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a expert model foucsed on color prediction")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    parser.add_argument('--scratch', action='store_true', help='If not specified and checkpoint is stored, it will be used')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main()

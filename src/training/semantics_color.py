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
from src.models.experts import ColorExpertModel, SemanticExpertModel

from src.plotting import generate_plots
from src.utils import generate_normalized_locations, populate_random_seeds, model_to_device, compile_model, generate_loss_trackers, \
    load_checkpoint


def save_best_model(model, save_dir):
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
    # TODO: Add the sub models


def load_sub_models(device, base_dir: str, color_expert_dir: str, semantics_expert_dir: str):
    base_path = os.path.join(base_dir, "best_model.pth")
    color_expert_path = os.path.join(color_expert_dir, "best_model.pth")
    semantics_expert_path = os.path.join(semantics_expert_dir, "best_model.pth")
    print(f"Loading color expert model from {color_expert_path}")

    base_model = BaseModel(cfg.NUM_BINS, cfg.CLASSES)
    color_expert_model = ColorExpertModel(cfg.NUM_BINS)
    semantics_expert_model = SemanticExpertModel(cfg.CLASSES)

    base_model_state_dict = torch.load(base_path, map_location=device)
    base_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in base_model_state_dict.items()}
    base_model.load_state_dict(base_model_state_dict)
    color_expert_model_state_dict = torch.load(color_expert_path, map_location=device)
    color_expert_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in color_expert_model_state_dict.items()}
    color_expert_model.load_state_dict(color_expert_model_state_dict)
    semantics_expert_model_state_dict = torch.load(semantics_expert_path, map_location=device)
    semantics_expert_model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in semantics_expert_model_state_dict.items()}
    semantics_expert_model.load_state_dict(semantics_expert_model_state_dict)

    base_model = model_to_device(base_model, device)
    print(f"Loaded base model from {base_path}")
    color_expert_model = model_to_device(color_expert_model, device)
    print(f"Loaded color expert model from {color_expert_path}")
    semantics_expert_model = model_to_device(semantics_expert_model, device)
    print(f"Loaded semantics expert model from {semantics_expert_path}")

    return base_model, color_expert_model, semantics_expert_model


def freeze_script_compile_sub_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # model = torch.jit.script(model)
    return compile_model(model)


def train_val(model, device, train_dataloader, val_dataloader, epochs, lr, save_dir: str, use_checkpoint: bool):
    model_type = model.__class__.__name__.split("_")[-1]

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    base_model, color_expert_model, semantics_expert_model = load_sub_models(device, cfg.SAVE_DIR_BASE, cfg.SAVE_DIR_COLOR_EXPERT, cfg.SAVE_DIR_SEMANTICS_EXPERT)
    base_model = freeze_script_compile_sub_model(base_model)
    color_expert_model = freeze_script_compile_sub_model(color_expert_model)
    semantics_expert_model = freeze_script_compile_sub_model(semantics_expert_model)

    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': lr, 'weight_decay': 1e-5}, ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)
    scaler = GradScaler()

    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=cfg.NUM_BINS - 1)
    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)

    best_color_val_loss = float('inf')
    epochs_no_improve_color = 0
    training_losses = {'total': [], 'semantics': [], 'color': []}
    validation_losses = {'total': [], 'semantics': [], 'color': []}
    times = []
    start_epoch = 0
    best_loss = float('inf')

    if use_checkpoint and os.path.exists(checkpoint_path):
        load_checkpoint(model, device, optimizer, scheduler, save_dir)

    normalized_locations = generate_normalized_locations(cfg.IMAGE_SIZE)
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    for epoch in range(start_epoch, epochs):
        generate_plots(epoch, training_losses, validation_losses, times, save_dir, cfg.PLOT_INTERVAL)

        model.train()
        epoch_start_time = time.time()
        sub_model_time = 0
        epoch_loss = 0.0
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
                with torch.no_grad():
                    preds_semantics_base, preds_color_base = base_model(locations, gray_images, lab_images)
                    preds_semantics_expert = semantics_expert_model(locations, gray_images)
                    del gray_images
                    preds_color_expert = color_expert_model(locations, lab_images)
                    del lab_images, locations
                sub_model_time = time.time() - sub_model_start_time

                gt_semantics = batch['gt_semantics'].to(device)
                gt_color = batch['gt_color'].to(device)

                preds_semantics, preds_color = model(preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert)
                loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics, gt_semantics.long().view(-1))
                loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                del preds_semantics, preds_color

                del preds_semantics_base, preds_color_base, preds_color_expert, gt_semantics, gt_color

                total_loss = loss_semantics + loss_color

            epoch_loss += total_loss.item()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()


        average_epoch_loss = epoch_loss / len(train_dataloader)

        training_losses['total'].append(average_epoch_loss)
        training_losses['semantics'].append(loss_semantics.item())
        training_losses['color'].append(loss_color.item())


        print(f"Epoch {epoch + 1}/{epochs})")
        print(f"Training Loss: {average_epoch_loss}")

        if torch.cuda.is_available() and not hasattr(model, '_torchdynamo_orig_callable'):
            torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                with autocast():
                    gray_images = batch['gray_image'].to(device)
                    lab_images = batch['lab_image'].to(device)

                    # Repeat locations along batch dimension
                    batch_size = gray_images.shape[0]
                    locations = normalized_locations_tensor.unsqueeze(0).expand(batch_size, -1, -1)

                    sub_model_start_time = time.time()
                    with torch.inference_mode():
                        preds_semantics_base, preds_color_base = base_model(locations, gray_images, lab_images)
                        preds_semantics_expert = semantics_expert_model(locations, gray_images)
                        del gray_images
                        preds_color_expert = color_expert_model(locations, lab_images)
                        del lab_images, locations
                    sub_model_time += time.time() - sub_model_start_time

                    gt_semantics = batch['gt_semantics'].to(device)
                    gt_color = batch['gt_color'].to(device)

                    preds_semantics, preds_color = model(preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert)
                    loss_semantics_val = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics, gt_semantics.long().view(-1))
                    loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color.view(-1, cfg.NUM_BINS), gt_color.view(-1))
                    del preds_semantics, preds_color

                    del preds_semantics_base, preds_color_base, preds_color_expert, gt_semantics, gt_color

                    val_loss += loss_semantics_val + loss_color_val

            average_val_loss = val_loss / len(val_dataloader)
            color_val_loss = loss_color_val.item()
            semantics_val_loss = loss_semantics.item()

            validation_losses['total'].append(average_val_loss.item())
            validation_losses['semantics'].append(semantics_val_loss)
            validation_losses['color'].append(color_val_loss)
            times.append((time.time() - epoch_start_time) - sub_model_time)
            print(f"Validation Loss: {average_val_loss.item()}")
            print(f"Training Time: {times[-1]}")

        if color_val_loss < best_color_val_loss:
            best_color_val_loss = color_val_loss
            epochs_no_improve_color = 0
        else:
            epochs_no_improve_color += 1

        if average_val_loss.item() < best_loss:
            best_loss = average_val_loss
            save_best_model(model, save_dir)
            print(f"New best {model_type} model saved with validation loss: {best_loss}")

        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': average_epoch_loss,
                'best_loss': best_loss,
                'training_losses': training_losses,
                'validation_losses': validation_losses,
                'times': times
            }, os.path.join(save_dir, "checkpoint.pth"))

        if (epochs_no_improve_color >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 75):
            print(f"Early stop at epoch {epoch + 1}. Color validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            print(f"{model_type} Model saved at early stopping point with validation loss: {best_color_val_loss}")
            break

        if torch.cuda.is_available() and not hasattr(model, '_torchdynamo_orig_callable'):
            torch.cuda.empty_cache()
        scheduler.step(average_val_loss)

    total_time = sum(times)
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds")

    return model


def main():
    populate_random_seeds(cfg.SEED)

    train_preloaded_data = load_sequential_data(cfg.TRAIN_DIR, cfg.TRAIN_FILES_LIMIT)
    val_preloaded_data = load_sequential_data(cfg.VAL_DIR, cfg.VAL_FILES_LIMIT)
    print("Successfully loaded preprocessed training and validation data")

    train_dataset = Rellis2DDataset(preloaded_data=train_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                    image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    val_dataset = Rellis2DDataset(preloaded_data=val_preloaded_data, num_bins=cfg.NUM_BINS, image_size=cfg.IMAGE_SIZE,
                                  image_noise=cfg.IMAGE_NOISE, image_mask_rate=cfg.IMAGE_MASK_RATE)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE_COLOR_SEMANTICS, shuffle=True, num_workers=0,
                                  pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE_COLOR_SEMANTICS, shuffle=False, num_workers=0,
                                pin_memory=cfg.PIN_MEMORY, drop_last=True)
    print(f"Created training dataloader with {len(train_dataset)} files and validation dataloader with {len(val_dataset)} files")

    # Train and validate each color model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_simple = ColorModelSimple(cfg.NUM_BINS)
    model_simple = model_to_device(model_simple, device)
    trained_simple_model = train_val(
        model=model_simple,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR_SEMANTICS + "_simple",
        use_checkpoint=not args.scratch
    )
    del trained_simple_model, model_simple
    print("Training finished for SemanticsColorSimpleModel \n ---------------------")

    quit()

    model_linear = ColorModelLinear(cfg.NUM_BINS)
    model_linear = model_to_device(model_linear, device)
    trained_linear_model = train_val(
        model=model_linear,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR_SEMANTICS + "_linear",
        use_checkpoint=not args.scratch
    )
    del trained_linear_model, model_linear
    print("Training finished for SemanticsColorLinearModel \n ---------------------")

    model_mlp = ColorModelMLP(cfg.NUM_BINS)
    model_mlp = model_to_device(model_mlp, device)
    trained_mlp_model = train_val(
        model=model_mlp,
        device=device,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=cfg.EPOCHS,
        lr=cfg.LR,
        save_dir=cfg.SAVE_DIR_COLOR_SEMANTICS + "_mlp",
        use_checkpoint=not args.scratch
    )
    print("Training finished for SemanticsColorMLPModel \n ---------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a expert model foucsed on color prediction")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    parser.add_argument('--scratch', action='store_true', help='If not specified and checkpoint is stored, it will be used')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main()

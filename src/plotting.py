import matplotlib.pyplot as plt
import src.local_config as cfg
from typing import Dict, List
import os


def generate_plots(epoch, training_losses, validation_losses, times, save_dir):
    if (epoch + 1) % cfg.PLOT_INTERVAL == 0:
        print(f"Made it in plotting loop for {save_dir}")
        plot_losses(training_losses, validation_losses, save_dir)
        print(f"Made it past plot losses")
        plot_times(times, save_dir)
        print(f"Made it past plot times")


def plot_times(times: List[float], save_dir: str):
    save_path = os.path.join(save_dir, "time_per_epoch.png")
    epochs = range(1, len(times) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, times, label="Time per Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Time (s)")
    plt.legend(loc="upper right")
    plt.title("Time per Epoch")

    plt.savefig(save_path)
    plt.close()
    print(f"Time plot saved to: {save_path}")

def plot_losses(training_losses: Dict[str, any], validation_losses: Dict[str, any], save_dir: str):
    LOSS_PLOT_PATH = os.path.join(save_dir, "loss_plot.png")
    INDIVIDUAL_LOSS_PLOT = os.path.join(save_dir, "losses_plot.png")

    plt.figure(figsize=(12, 6))  # Set figure size
    plt.subplot(1, 2, 1)
    plt.plot(training_losses['semantics'], label="Training Loss Semantics")
    plt.plot(validation_losses['semantics'], label="Validation Loss Semantics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Loss for Semantics")

    plt.subplot(1, 2, 2)
    plt.plot(training_losses['color'], label="Training Loss Color")
    plt.plot(validation_losses['color'], label="Validation Loss Color")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Loss for Color")

    plt.tight_layout()  # Ensure no overlap
    plt.savefig(INDIVIDUAL_LOSS_PLOT)
    plt.close()
    print(f"Individual loss plot saved to: {INDIVIDUAL_LOSS_PLOT}")

    print(f"Loss Plot path: {LOSS_PLOT_PATH}")
    print(f"Training Losses: {training_losses['total']}")
    print(f"Validation Losses: {validation_losses['total']}")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses['total'], label="Training Loss")
    plt.plot(validation_losses['total'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Losses")

    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    print(f"Plot saved to: {LOSS_PLOT_PATH}")


# from src.data.utils.data_processing import lab_discretized_to_rgb
# gt_semantics = batch['gt_semantics']
# gt_color = batch['gt_color']
# lab_image = batch['lab_image']
# gray_image = batch['gray_image']
#
# # Assuming `train_rgb_images` corresponds to a part of the dataset,
# # visualize the first image in the batch for clarity
# fig, axs = plt.subplots(2, 2, figsize=(15, 15))
#
# print(f"GT Semantics Shape: {gt_semantics.shape}")
# print(f"GT Color Shape: {gt_color.shape}")
# print(f"LAB Image Shape: {lab_image.shape}")
# print(f"Gray Image Shape: {gray_image.shape}")
#
# axs[0, 0].imshow(lab_discretized_to_rgb(gt_color[0].numpy(), cfg.NUM_BINS))
# axs[0, 0].set_title('GT LAB Image to RGB')
# axs[0, 1].imshow(gt_semantics[0].to(torch.uint8).numpy(), cmap='gray')
# axs[0, 1].set_title('GT Semantics')
#
# axs[1, 0].imshow(lab_discretized_to_rgb(lab_image[0].numpy().transpose(1, 2, 0), cfg.NUM_BINS))
# axs[1, 0].set_title('LAB Image to RGB')
# axs[1, 1].imshow(gray_image[0].numpy().transpose(1, 2, 0), cmap='gray')
# axs[1, 1].set_title('Gray Image')
#
#
#
# plt.tight_layout()
# plt.show()
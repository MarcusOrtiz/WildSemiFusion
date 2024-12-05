import matplotlib.pyplot as plt
import src.config as cfg
from typing import Dict, List

def plot_base_losses(training_losses: Dict[str, any], validation_losses: Dict[str, any]):
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
    plt.savefig(cfg.INDIVIDUAL_LOSS_PLOT_PATH_BASE)
    plt.close()
    print(f"Individual loss plot saved to: {cfg.INDIVIDUAL_LOSS_PLOT_PATH_BASE}")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses['total'], label="Training Loss")
    plt.plot(validation_losses['total'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Losses")

    plt.savefig(cfg.LOSS_PLOT_PATH_BASE)
    plt.close()
    print(f"Plot saved to: {cfg.LOSS_PLOT_PATH_BASE}")


def plot_color_losses(training_losses: List[any], validation_losses: List[any]):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Losses")

    plt.savefig(cfg.LOSS_PLOT_PATH_COLOR)
    plt.close()
    print(f"Plot saved to: {cfg.LOSS_PLOT_PATH_COLOR}")

def plot_semantics_losses():
    pass
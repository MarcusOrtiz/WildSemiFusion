"""
For compute_color_metrics and compute_semantic_metrics:
    Author: Duke General Robotics Lab, Yanbaihui Liu
    File: https://github.com/generalroboticslab/WildFusion/blob/main/evaluation/evaluation.py
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import torch

from src.models.base import BaseModel
from src.utils import model_to_device, compile_model


# Color Metrics (MSE, MAE, PSNR)
def compute_color_metrics(rgb1, rgb2):
    mse = np.mean((rgb1 - rgb2) ** 2)
    mae = np.mean(np.abs(rgb1 - rgb2))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float('inf')
    return mse, mae, psnr


# Semantic Metrics (Accuracy, Precision, Recall, F1, IoU)
def compute_semantic_metrics(semantics1, semantics2):
    accuracy = accuracy_score(semantics1, semantics2)
    precision = precision_score(semantics1, semantics2, average='weighted', zero_division=0)
    recall = recall_score(semantics1, semantics2, average='weighted', zero_division=0)
    f1 = f1_score(semantics1, semantics2, average='weighted', zero_division=0)
    iou = jaccard_score(semantics1, semantics2, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1, iou


def compute_average_metrics(gt_rgb, preds_rgb, gt_semantics, preds_semantics):
    mse_color, mae_color, pnsr_color = compute_color_metrics(gt_rgb, preds_rgb)
    accuracy, precision, recall, f1, iou = compute_semantic_metrics(gt_semantics, preds_semantics)

    return {
        "color_mse": mse_color,
        "color_mae": mae_color,
        "color_psnr": pnsr_color,
        "semantic_accuracy": accuracy,
        "semantic_precision": precision,
        "semantic_recall": recall,
        "semantic_f1": f1,
        "semantic_iou": iou
    }

def load_model(model, model_path, device):
    model_state_dict = torch.load(model_path, map_location=device)
    model_state_dict = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    base_model = model_to_device(model, device)
    print(f"Loaded model from {model_path} and moved to {device}")
    return base_model


def freeze_compile_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return compile_model(model)


import importlib
import os
import torch
import numpy as np
import time
import argparse

from evaluation.utils import compute_average_metrics, load_model, freeze_compile_model
from src.data.utils.data_processing import load_sequential_data, rgb_to_gray, rgb_to_lab_continuous, lab_continuous_to_lab_discretized, lab_discretized_to_rgb
from src.models.base import BaseModel
from src.models.color import ColorModelSimple, ColorModelLinear, ColorModelMLP
from src.models.experts import ColorExpertModel, SemanticExpertModel
from src.models.semantics_color import SemanticsColorModelSimple, SemanticsColorModelLinear, SemanticsColorModelMLP
from src.utils import model_to_device, compile_model, generate_normalized_locations, populate_random_seeds



def test_semantics_color_model(base_model, color_expert_model, semantics_expert_model, semantics_color_model, device, test_preloaded_data):
    """Test the model on the test dataset and save the results."""
    color_mse_sum, color_mae_sum, color_psnr_sum = 0, 0, 0
    accuracy_sum, precision_sum, recall_sum, f1_sum, iou_sum = 0, 0, 0, 0, 0

    normalized_locations = generate_normalized_locations(cfg.IMAGE_SIZE)
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    times = []
    for rgb_image, gt_semantics in zip(test_preloaded_data['rgb_images'], test_preloaded_data['gt_semantics']):
        with torch.no_grad():
            # Convert rgb_image to gray and lab
            gray_image = rgb_to_gray(rgb_image)
            lab_image = rgb_to_lab_continuous(rgb_image)
            lab_image_discretized = lab_continuous_to_lab_discretized(lab_image, cfg.NUM_BINS, void_bin=True)

            batch_size = 1

            locations = normalized_locations_tensor.repeat(batch_size, 1, 1)
            lab_image_tensor = torch.tensor(lab_image_discretized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (3, H, W)
            gray_image_tensor = torch.tensor(gray_image[np.newaxis, ...], dtype=torch.float32).unsqueeze(0).to(device)  # Shape: (1, H, W)

            # if torch.cuda.is_available(): torch.cuda.synchronize() #TODO Uncomment when trying out times after verifying results
            start_time = time.time()
            preds_semantics_base, preds_color_base = base_model(locations, gray_image_tensor, lab_image_tensor)
            preds_color_expert = color_expert_model(locations, lab_image_tensor)
            preds_semantics_expert = semantics_expert_model(locations, gray_image_tensor)
            preds_semantics, preds_color = semantics_color_model(preds_semantics_base, preds_semantics_expert, preds_color_base, preds_color_expert)
            # if torch.cuda.is_available(): torch.cuda.synchronize()
            times.append(time.time() - start_time)

            assert preds_semantics.shape[0] == preds_color.shape[0] == batch_size * cfg.IMAGE_SIZE[0] * \
                   cfg.IMAGE_SIZE[1], \
                "Should be num_locations * batch_size"
            assert preds_semantics.shape[-1] == cfg.CLASSES, "Semantic logits should have classes as the last dimension"
            assert preds_color.shape[-1] == cfg.NUM_BINS, "Color logits should have bins as the last dimension"
            assert preds_color.dtype == torch.float32, "Color logits are not class probabilities"

            # Reshape preds color and create rgb numpy to compare against original rgb
            preds_color = preds_color.view(batch_size, cfg.IMAGE_SIZE[0], cfg.IMAGE_SIZE[1], 3, cfg.NUM_BINS)
            preds_color = torch.argmax(preds_color, dim=-1)  # Shape: (batchsize, H, W, 3)
            preds_color_np = preds_color.cpu().numpy()
            preds_color_rgb_np = lab_discretized_to_rgb(preds_color_np[0], cfg.NUM_BINS, void_bin=True)

            # Convert gt rgb_image to float32, not using ints, it is relative
            gt_rgb_image = rgb_image.astype(np.float32)
            gt_rgb_image = gt_rgb_image / 255.0

            # flatten the semantics to compare metrics
            gt_semantics = gt_semantics.flatten()
            preds_semantics = torch.argmax(preds_semantics, dim=-1)
            preds_semantics = preds_semantics.cpu().numpy()
            preds_semantics = preds_semantics.flatten()

            metrics = compute_average_metrics(gt_rgb_image, preds_color_rgb_np, gt_semantics, preds_semantics)

            color_mse_sum += metrics['color_mse']
            color_mae_sum += metrics['color_mae']
            color_psnr_sum += metrics['color_psnr']

            accuracy_sum += metrics['semantic_accuracy']
            precision_sum += metrics['semantic_precision']
            recall_sum += metrics['semantic_recall']
            f1_sum += metrics['semantic_f1']
            iou_sum += metrics['semantic_iou']

    print(f"Average Color MSE: {color_mse_sum / len(test_preloaded_data['rgb_images'])}")
    print(f"Average Color MAE: {color_mae_sum / len(test_preloaded_data['rgb_images'])}")
    print(f"Average Color PSNR: {color_psnr_sum / len(test_preloaded_data['rgb_images'])}")

    print(f"Average Semantic Accuracy: {accuracy_sum / len(test_preloaded_data['rgb_images'])}")
    print(f"Average Semantic Precision: {precision_sum / len(test_preloaded_data['rgb_images'])}")
    print(f"Average Semantic Recall: {recall_sum / len(test_preloaded_data['rgb_images'])}")
    print(f"Average Semantic F1: {f1_sum / len(test_preloaded_data['rgb_images'])}")
    print(f"Average Semantic IoU: {iou_sum / len(test_preloaded_data['rgb_images'])}")

    print(f"Average Inference Time: {time.time() - start_time}")


def main():
    populate_random_seeds(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_path = os.path.join(cfg.TESTING_MODELS_DIR, "base", "best_model.pth")
    base_model_path = os.path.expanduser(base_model_path)
    base_model = BaseModel(cfg.NUM_BINS, cfg.CLASSES)
    base_model = load_model(base_model, base_model_path, device)
    base_model = freeze_compile_model(base_model)
    print("Base model frozen and compiled successfully")

    color_expert_model_path = os.path.join(cfg.TESTING_MODELS_DIR, "color_expert", "best_model.pth")
    color_expert_model_path = os.path.expanduser(color_expert_model_path)
    color_expert_model = ColorExpertModel(cfg.NUM_BINS)
    color_expert_model = load_model(color_expert_model, color_expert_model_path, device)
    color_expert_model = freeze_compile_model(color_expert_model)
    print("Color expert model frozen and compiled successfully")

    semantics_expert_model_path = os.path.join(cfg.TESTING_MODELS_DIR, "semantics_expert", "best_model.pth")
    semantics_expert_model_path = os.path.expanduser(semantics_expert_model_path)
    semantics_expert_model = SemanticExpertModel(cfg.CLASSES)
    semantics_expert_model = load_model(semantics_expert_model, semantics_expert_model_path, device)
    semantics_expert_model = freeze_compile_model(semantics_expert_model)
    print("Semantics expert model frozen and compiled successfully")

    if model_type == 'simple':
        semantics_color_model_path = os.path.join(cfg.TESTING_MODELS_DIR, "semantics_color_simple", "best_model.pth")
        semantics_color_model = SemanticsColorModelSimple(cfg.NUM_BINS, cfg.CLASSES)
    elif model_type == 'linear':
        semantics_color_model_path = os.path.join(cfg.TESTING_MODELS_DIR, "semantics_color_linear", "best_model.pth")
        semantics_color_model = SemanticsColorModelLinear(cfg.NUM_BINS, cfg.CLASSES)
    elif model_type == 'mlp':
        semantics_color_model_path = os.path.join(cfg.TESTING_MODELS_DIR, "semantics_color_mlp", "best_model.pth")
        semantics_color_model = SemanticsColorModelMLP(cfg.NUM_BINS, cfg.CLASSES)

    semantics_color_model = load_model(semantics_color_model, semantics_color_model_path, device)
    semantics_color_model = freeze_compile_model(semantics_color_model)
    print("SemanticsColor model frozen and compiled successfully")

    test_preloaded_data = load_sequential_data(cfg.TEST_DIR, cfg.TEST_FILES_LIMIT)
    print(f"Testing preloaded data gathered successfully for {len(test_preloaded_data['rgb_images'])} images")

    test_semantics_color_model(
        base_model=base_model,
        color_expert_model=color_expert_model,
        semantics_expert_model=semantics_expert_model,
        semantics_color_model=semantics_color_model,
        device=device,
        test_preloaded_data=test_preloaded_data
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test base model on testing dataset")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    parser.add_argument('--model', type=str, default='mlp',
                        help='Combination network type (simple | linear | mlp')
    args = parser.parse_args()
    model_type = args.model
    cfg = importlib.import_module(args.config)

    main()
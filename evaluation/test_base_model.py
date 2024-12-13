import importlib
import os
import torch
import numpy as np
import time
import argparse

from evaluation.utils import compute_average_metrics, load_model, freeze_compile_model
from src.data.utils.data_processing import load_sequential_data, rgb_to_gray, rgb_to_lab_continuous, lab_continuous_to_lab_discretized, lab_discretized_to_rgb
from src.models.base import BaseModel
from src.utils import model_to_device, compile_model, generate_normalized_locations, populate_random_seeds





def test_base_model(model, device, test_preloaded_data):
    """Test the model on the test dataset and save the results."""
    color_mse_sum, color_mae_sum, color_psnr_sum = 0, 0, 0
    accuracy_sum, precision_sum, recall_sum, f1_sum, iou_sum = 0, 0, 0, 0, 0

    normalized_locations = generate_normalized_locations(cfg.IMAGE_SIZE)
    normalized_locations_tensor = torch.from_numpy(normalized_locations).to(device)

    start_time = time.time()
    for rgb_image, gt_semantics in zip(test_preloaded_data['rgb_images'], test_preloaded_data['gt_semantics']):
        model.eval()
        with torch.no_grad():
            # Convert rgb_image to gray and lab
            gray_image = rgb_to_gray(rgb_image)
            lab_image = rgb_to_lab_continuous(rgb_image)
            lab_image_discretized = lab_continuous_to_lab_discretized(lab_image, cfg.NUM_BINS, void_bin=True)

            batch_size = 1

            locations = normalized_locations_tensor.repeat(batch_size, 1, 1)
            lab_image_tensor = torch.tensor(lab_image_discretized.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)  # Shape: (3, H, W)
            gray_image_tensor = torch.tensor(gray_image[np.newaxis, ...], dtype=torch.float32).unsqueeze(0)  # Shape: (1, H, W)

            preds_semantics, preds_color = model(locations, gray_image_tensor, lab_image_tensor)

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

            # flatten the semantics
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

    print(f"Total Time: {time.time() - start_time}")


def main():
    populate_random_seeds(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "~/Projects/WildSemiFusion/testing_models/base/best_model.pth"
    model_path = os.path.expanduser(model_path)
    model = BaseModel(cfg.NUM_BINS, cfg.CLASSES)
    model = load_model(model, model_path, device)
    model = freeze_compile_model(model)
    print("Model loaded successfully and compiled")

    test_preloaded_data = load_sequential_data(cfg.TEST_DIR, cfg.TEST_FILES_LIMIT)
    print(f"Testing preloaded data gathered successfully for {len(test_preloaded_data['rgb_images'])} images")

    test_base_model(model, device, test_preloaded_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test base model on testing dataset")
    parser.add_argument('--config', type=str, default='src.local_config',
                        help='Path to the configuration module (src.local_config | src.aws_config)')
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main()
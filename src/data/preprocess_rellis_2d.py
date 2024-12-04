import os
import argparse
import logging as log
from shutil import rmtree
from cv2 import imwrite
from src.data.utils.data_processing import resize_rgb_and_annotation

BASE_DIR = os.getcwd()
DEFAULT_ONTOLOGY_PATH = os.path.join(BASE_DIR, "../../input/ontology.yaml")
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "../../input/rellis_2d")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "../../input/rellis_2d_preprocessed")

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess rellis images")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_PATH,
                        help="Input directory for original images and labels")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Output directory for preprocessed images and labels")
    args, _ = parser.parse_known_args()
    parser.add_argument("--split_dir", type=str, default=os.path.join(args.input_dir, "./split"),
                        help="Directory containing split files")
    args = parser.parse_args()

    log.info(f"Creating Preprocessed Data Directories ...")
    if os.path.exists(args.output_dir):
        rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    sequence_dirs = [sequence_dir for sequence_dir in os.listdir(args.input_dir) if
                     not any(char.isalpha() for char in sequence_dir)]

    for env_lst in os.listdir(args.split_dir):
        for sequence_dir in sequence_dirs:
            os.makedirs(os.path.join(args.output_dir, env_lst[:-4], sequence_dir, "pylon_camera_node"))
            os.makedirs(os.path.join(args.output_dir, env_lst[:-4], sequence_dir, "pylon_camera_node_label_id"))
    log.info(f"Directories Created")

    log.info(f"Preprocessing rellis data ...")
    for env_lst in os.listdir(args.split_dir):
        log.info(f"Preprocessing rellis {env_lst[:-4]}ing data")
        with open(os.path.join(args.split_dir, env_lst), "r") as file:
            images_labels_mappings = file.readlines()

        for image_label_map in images_labels_mappings:
            image_label_map = image_label_map.split()
            image_file, label_file = image_label_map[0], image_label_map[1]
            image_resized, label_resized = resize_rgb_and_annotation(os.path.join(args.input_dir, image_file),
                                                                     os.path.join(args.input_dir, label_file))
            imwrite(os.path.join(args.output_dir, env_lst[:-4], image_file), image_resized)
            imwrite(os.path.join(args.output_dir, env_lst[:-4], label_file), label_resized)
        log.info(f"Finished preprocessing rellis {env_lst}ing data")
    log.info(f"Finished preprocessing rellis data")

import yaml
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONTOLOGY_PATH = os.path.join(BASE_DIR, "../../input/ontology.yaml")
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "../../input/relis_2d")
DEFAULT_OUTPUT_PATH = os.path.join(BASE_DIR, "../../input/relis_2d_preprocessed")


# Some of these are adopted from relis 3d

def load_label_mapping(ontology_path: str) -> dict:
    with open(ontology_path, "r") as file:
        ontology = yaml.safe_load(file)
    label_mapping = {k: k for k in ontology[0]}
    return label_mapping


def convert_label(label, label_mapping):
    temp = label.copy()
    for raw_id, mapped_id in label_mapping.items():
        temp[label == raw_id] = mapped_id
    return temp







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process relis images")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_PATH,
                        help="Input directory for labels")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Output directory for processed labels")
    args = parser.parse_args()

    # Check if default input is used
    if args.input_dir == DEFAULT_INPUT_PATH:
        for sequence_dir in os.listdir(args.input_dir):
            annotated_image_to_array(os.path.join(args.input_dir, sequence_dir, "pylon_camera_node_label_id"),
                                     os.path.join(args.output_dir, sequence_dir, "pylon_camera_node_label"))
    else:
        annotated_image_to_array(args.input_dir, args.output_dir)


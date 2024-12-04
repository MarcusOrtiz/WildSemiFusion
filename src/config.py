import os

# Random seed for reproducibility
SEED = 7

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
SAVE_DIR_BASE_1 = os.path.join(BASE_DIR, "saved_model_base_1")
SAVE_DIR_BASE_2 = os.path.join(BASE_DIR, "saved_model_base_2")
SAVE_DIR_COLOR_1 = os.path.join(BASE_DIR, "saved_expert_color_1")
SAVE_DIR_BASE_COLOR_1 = os.path.join(BASE_DIR, "saved_model_base_expert_color_1")
TRAIN_DIR = os.path.join(BASE_DIR, "../input/rellis_2d_preprocessed/train")
VAL_DIR = os.path.join(BASE_DIR, "../input/rellis_2d_preprocessed/val")
TEST_DIR = os.path.join(BASE_DIR, "../input/rellis_2d_preprocessed/test")
CHECKPOINT_PATH_BASE_1 = os.path.join(SAVE_DIR_BASE_1, "checkpoint.pth")
BEST_MODEL_PATH_BASE_1 = os.path.join(SAVE_DIR_BASE_1, "best_model.pth")

# Training parameters
BATCH_SIZE = 4
EPOCHS = 300
LR = 0.00002
EARLY_STOP_EPOCHS = 20

# Loss weights
WEIGHT_SEMANTICS = 1
WEIGHT_COLOR = 20

# Data processing parameters (for now to avoid error)
NUM_WORKERS = 0

# Model and Dataset parameters
NUM_BINS = 256 # including void
CLASSES = 35  # including void
IMAGE_SIZE = (224, 224)
IMAGE_NOISE = 0.01
IMAGE_MASK_RATE = (.1, .3)

# Other parameters
PATIENCE = 5
LR_DECAY_FACTOR = 0.6

# Plotting
PLOT_INTERVAL = 5
INDIVIDUAL_LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "loss_plot.png")
LOSS_PLOT_PATH = os.path.join(SAVE_DIR, "metric_plot.png")

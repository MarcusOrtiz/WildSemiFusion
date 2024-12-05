import os

# Random seed for reproducibility
SEED = 7

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
SAVE_DIR_BASE = os.path.join(BASE_DIR, "saved_model_base_1")
# SAVE_DIR_BASE = os.path.join(BASE_DIR, "saved_model_test_full")
SAVE_DIR_COLOR = os.path.join(BASE_DIR, "saved_expert_color_1")
SAVE_DIR_BASE_COLOR = os.path.join(BASE_DIR, "saved_model_base_expert_color_1")
TRAIN_DIR = os.path.join(BASE_DIR, "../input/rellis_2d_preprocessed/train")
VAL_DIR = os.path.join(BASE_DIR, "../input/rellis_2d_preprocessed/val")
TEST_DIR = os.path.join(BASE_DIR, "../input/rellis_2d_preprocessed/test")
CHECKPOINT_PATH_BASE = os.path.join(SAVE_DIR_BASE, "checkpoint.pth")
BEST_MODEL_PATH_BASE = os.path.join(SAVE_DIR_BASE, "best_model.pth")
CHECKPOINT_PATH_COLOR = os.path.join(SAVE_DIR_COLOR, "checkpoint.pth")
BEST_MODEL_PATH_COLOR = os.path.join(SAVE_DIR_COLOR, "best_model.pth")

# Training parameters
BATCH_SIZE = 8
EPOCHS = 400
LR = 0.00002
EARLY_STOP_EPOCHS = 30

# Loss weights
WEIGHT_SEMANTICS = 1
WEIGHT_COLOR = 20

# Data processing parameters (for now to avoid error)
NUM_WORKERS = 0

# Model and Dataset parameters
NUM_BINS = 257 # including void which must be last bin, consider switching to 193
CLASSES = 35  # including void
IMAGE_SIZE = (224, 224)
IMAGE_NOISE = 0.01
IMAGE_MASK_RATE = (.1, .3)

# Other parameters
PATIENCE = 5
LR_DECAY_FACTOR = 0.6

# Plotting
PLOT_INTERVAL = 5
INDIVIDUAL_LOSS_PLOT_PATH_BASE = os.path.join(SAVE_DIR_BASE, "losses_plot.png")
LOSS_PLOT_PATH_BASE = os.path.join(SAVE_DIR_BASE, "loss_plot.png")
INDIVIDUAL_LOSS_PLOT_PATH_COLOR = os.path.join(SAVE_DIR_COLOR, "losses_plot.png")
LOSS_PLOT_PATH_COLOR = os.path.join(SAVE_DIR_COLOR, "loss_plot.png")

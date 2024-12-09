import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Saved Model Directories
SAVE_DIR = os.path.join(SRC_DIR, "../saved_models")
SAVE_DIR_BASE = os.path.join(SAVE_DIR, "base_1")
SAVE_DIR_COLOR_EXPERT = os.path.join(SAVE_DIR, "color_expert_1")
SAVE_DIR_SEMANTICS_EXPERT = os.path.join(SAVE_DIR, "semantics_expert_1")
SAVE_DIR_COLOR = os.path.join(SAVE_DIR, "color_1")
SAVE_DIR_COLOR_SEMANTICS = os.path.join(SAVE_DIR, "color_semantics_1")

# Data Directories
INPUT_DIR = os.path.join(SRC_DIR, "../input")
TRAIN_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/train")
VAL_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/val")
TEST_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/test")

# Data processing parameters
NUM_WORKERS = 0
PIN_MEMORY = False
TRAIN_FILES_LIMIT = 100
VAL_FILES_LIMIT = 25

# Training parameters
BATCH_SIZE = 6
EPOCHS = 20
LR = 0.00002
EARLY_STOP_EPOCHS = 10
WEIGHT_SEMANTICS = 1
WEIGHT_COLOR = 20
PATIENCE = 5
LR_DECAY_FACTOR = 0.6

# Dataset and preprocessing parameters
NUM_BINS = 257  # including void which must be last bin, consider switching to 193
CLASSES = 35  # including void which must be first class
IMAGE_SIZE = (224, 224)
IMAGE_NOISE = 0.01
IMAGE_MASK_RATE = (.1, .25)

# Plotting
PLOT_INTERVAL = 5

# Random seed for reproducibility
SEED = 7

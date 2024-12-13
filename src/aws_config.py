import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Saved Model Directories
SAVE_DIR = os.path.join(SRC_DIR, "../saved_models_test")
SAVE_DIR_BASE = os.path.join(SAVE_DIR, "base")
SAVE_DIR_COLOR_EXPERT = os.path.join(SAVE_DIR, "color_expert")
SAVE_DIR_SEMANTICS_EXPERT = os.path.join(SAVE_DIR, "semantics_expert")
SAVE_DIR_COLOR = os.path.join(SAVE_DIR, "color")
SAVE_DIR_COLOR_SEMANTICS = os.path.join(SAVE_DIR, "color_semantics")

AWS_SAVE_DIR = os.path.join(SAVE_DIR, "../aws_saved_models_test")
AWS_SAVE_DIR_BASE = os.path.join(AWS_SAVE_DIR, "base")
AWS_SAVE_DIR_COLOR_EXPERT = os.path.join(AWS_SAVE_DIR, "color_expert")
AWS_SAVE_DIR_SEMANTICS_EXPERT = os.path.join(AWS_SAVE_DIR, "semantics_expert")
AWS_SAVE_DIR_COLOR = os.path.join(AWS_SAVE_DIR, "color")
AWS_SAVE_DIR_COLOR_SEMANTICS = os.path.join(AWS_SAVE_DIR, "color_semantics")

# Data Directories
INPUT_DIR = os.path.join(SRC_DIR, "../input")
TRAIN_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/train")
VAL_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/val")
TEST_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/test")

# Pytorch data processing parameters (limits should be None for full dataset)
NUM_WORKERS = 3
PIN_MEMORY = True
TRAIN_FILES_LIMIT = 825
VAL_FILES_LIMIT = 400
TEST_FILES_LIMIT = None

# Training parameters
BATCH_SIZE = 32
BATCH_SIZE_COLOR = 12
BATCH_SIZE_COLOR_EXPERT = 16
EPOCHS = 150
LR = 0.00002
EARLY_STOP_EPOCHS = 10
WEIGHT_SEMANTICS = 1
WEIGHT_COLOR = 1
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
SAVE_INTERVAL = 5

# Random seed for reproducibility
SEED = 7

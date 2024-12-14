import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Saved Model Directories
SAVE_DIR = os.path.join(SRC_DIR, "../saved_models")
SAVE_DIR_BASE = os.path.join(SAVE_DIR, "base")
SAVE_DIR_COLOR_EXPERT = os.path.join(SAVE_DIR, "color_expert")
SAVE_DIR_SEMANTICS_EXPERT = os.path.join(SAVE_DIR, "semantics_expert")
SAVE_DIR_COLOR = os.path.join(SAVE_DIR, "color")
SAVE_DIR_COLOR_SEMANTICS = os.path.join(SAVE_DIR, "color_semantics")

AWS_SAVE_DIR = os.path.join(SRC_DIR, "../aws_saved_models") # Just for visually testing aws locally
AWS_SAVE_DIR_BASE = os.path.join(AWS_SAVE_DIR, "base")
AWS_SAVE_DIR_COLOR_EXPERT = os.path.join(SAVE_DIR, "color_expert")
AWS_SAVE_DIR_SEMANTICS_EXPERT = os.path.join(AWS_SAVE_DIR, "semantics_expert")
AWS_SAVE_DIR_COLOR = os.path.join(AWS_SAVE_DIR, "color")
AWS_SAVE_DIR_COLOR_SEMANTICS = os.path.join(AWS_SAVE_DIR, "color_semantics")

# Data Directorie
INPUT_DIR = os.path.join(SRC_DIR, "../input")
TRAIN_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/train")
VAL_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/val")
TEST_DIR = os.path.join(INPUT_DIR, "rellis_2d_preprocessed/test")

# Data processing parameters (limits should be None for full dataset)
NUM_WORKERS = 0
PIN_MEMORY = False
TRAIN_FILES_LIMIT = 50
VAL_FILES_LIMIT = 10
TEST_FILES_LIMIT = None

# Training parameters
BATCH_SIZE = 6
EPOCHS = 25
LR = 0.00002
EARLY_STOP_EPOCHS = 5
WEIGHT_SEMANTICS = 1
WEIGHT_COLOR = 1
PATIENCE = 2
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

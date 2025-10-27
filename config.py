import os

# Paths (change accordingly)
DATA_DIR = "/content/drive/MyDrive/Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "Train")
VALID_DIR = os.path.join(DATA_DIR, "Validate")
TEST_DIR = os.path.join(DATA_DIR, "Test")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 10
SEED = 42

# Training defaults
LEARNING_RATE = 1e-3

# Runs and logging
RUNS_DIR = "runs"


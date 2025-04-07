# Hyperparameters for the model and training
WALK_LENGTH = 100
INPUT_DIM_DEFAULT = None  # Will be set dynamically based on data
HIDDEN_DIM = 128
OUTPUT_DIM_DEFAULT = None  # Will be set dynamically based on data
ENCODING_DIM = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 50
BATCH_SIZES = [1000]
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
DATA_PATH = "/content/amazon_electronics_photo.npz"
OUTPUT_CSV = "accuracy_table.csv"
"""
Project settings and hyperparameters for the LSTM experiments.
"""
import os
import random
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class Config:
    # Paths
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "LSTM", "results_hubert_embeddings")

    # Loader: 'handcrafted' or 'hubert'
    DATA_TYPE = 'hubert'

    # HuBERT inputs
    HUBERT_NPY = os.path.join(PROJECT_ROOT, "data", "features", "hubert", "frame_level")
    HUBERT_LABELS_CSV = os.path.join(PROJECT_ROOT, "data", "features", "hubert", "hubert_labels.csv")
    NPZ_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "prepared_hubert_data.npz")

    # Training
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    LEARNING_RATE = 3e-4
    PATIENCE = 15          # early-stopping patience (epochs)
    MIN_DELTA = 0.001      # minimum improvement for early stopping
    MAX_GRAD_NORM = 0.5    # gradient clipping

    # Regularisation
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.05
    USE_CLASS_WEIGHTS = True  # balanced class weights in CrossEntropyLoss

    # ReduceLROnPlateau
    USE_REDUCE_LR_ON_PLATEAU = True
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 7
    MIN_LR = 1e-6

    VAL_BATCH_SIZE = 8
    EARLY_STOP_METRIC = 'F1-Score'

    FEATURE_SELECTION = True
    FEATURE_SELECTION_METHOD = 'f_test'
    TOP_FEATURES_RATIO = 0.85

    # Model
    BIDIRECTIONAL = True
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # Data loading (0 = main process)
    NUM_WORKERS = 0

    # Speaker split
    TEST_SPEAKERS_PERCENTAGE = 0.20
    MAX_SPLIT_ITERATIONS = 100  # only used if USE_FIXED_SPEAKER_SPLIT=False
    FOLDS = 8

    # Use the shared fixed split (same as classical ML) for fair comparison
    USE_FIXED_SPEAKER_SPLIT = True
    SPEAKER_SPLIT_JSON = os.path.join(PROJECT_ROOT, "data", "splits", "speaker_split.json")

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    def pin_memory(self):
        return self.device.type == 'cuda'


def set_seeds(seed=42):
    """Seed numpy, random and torch for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_directories():
    """Create output folders."""
    config = Config()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

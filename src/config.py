# src/config.py

CMAPSS_DATASETS = {
    "FD001": {"desc": "1 cond, 1 fault (HPC)",        "n_train": 100, "n_test": 100},
    "FD002": {"desc": "6 cond, 1 fault (HPC)",        "n_train": 260, "n_test": 259},
    "FD003": {"desc": "1 cond, 2 faults (HPC+Fan)",   "n_train": 100, "n_test": 100},
    "FD004": {"desc": "6 cond, 2 faults (HPC+Fan)",   "n_train": 248, "n_test": 249},
}

MAX_RUL = 125
SEQUENCE_LENGTH = 30

HIDDEN_SIZE = 50
NUM_LAYERS = 2
OUTPUT_SIZE = 1

LEARNING_RATE = 1e-3
NUM_EPOCHS = 25  # oder 30, wenn du magst

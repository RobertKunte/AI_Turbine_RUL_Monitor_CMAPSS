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

# Sensors we keep (common across FD001–FD004)
GLOBAL_SENSOR_FEATURES = [
    "Sensor2", "Sensor3", "Sensor4",
    "Sensor6", "Sensor7", "Sensor8", "Sensor9",
    "Sensor11", "Sensor12", "Sensor13", "Sensor14",
    "Sensor15", "Sensor16", "Sensor17", "Sensor18",
    "Sensor19", "Sensor20", "Sensor21",
]

GLOBAL_SETTING_FEATURES = ["Setting1", "Setting2", "Setting3"]

GLOBAL_PHYS_FEATURES = [
    "Effizienz_HPC_Proxy",
    "EGT_Drift",
    "Fan_HPC_Ratio",
]

# FD-ID as simple numeric feature (0..3, will be scaled)
GLOBAL_META_FEATURES = ["FD_ID"]

GLOBAL_FEATURE_COLS = (
    GLOBAL_SETTING_FEATURES
    + GLOBAL_SENSOR_FEATURES
    + GLOBAL_PHYS_FEATURES
    + GLOBAL_META_FEATURES
)

GLOBAL_DROP_COLS = [
    "Sensor1", "Sensor5", "Sensor10",  # quasi-konstant
    "MaxTime",                         # nur Hilfsspalte für RUL
]
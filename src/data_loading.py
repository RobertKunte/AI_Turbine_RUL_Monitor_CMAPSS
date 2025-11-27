import os
import numpy as np
import pandas as pd

# Projekt-Root: eine Ebene über src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_cmapps_subset(fd_id, data_dir=DEFAULT_DATA_DIR, max_rul=125):
    """
    Load train, test, and RUL files for a given C-MAPSS subset (FD001–FD004)
    and compute RUL for the training trajectories.
    """
    assert fd_id in ["FD001", "FD002", "FD003", "FD004"], f"Unknown subset: {fd_id}"

    # <--- WICHTIG: hier data_dir verwenden --->
    train_path = os.path.join(data_dir, f"train_{fd_id}.txt")
    test_path  = os.path.join(data_dir, f"test_{fd_id}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{fd_id}.txt")

    col_names = ["UnitNumber", "TimeInCycles", "Setting1", "Setting2", "Setting3"] + \
                [f"Sensor{i}" for i in range(1, 22)]

    df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    df_test  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=col_names)
    rul_df   = pd.read_csv(rul_path,   sep=r"\s+", header=None)

    # RUL für Training berechnen
    df_max_time = df_train.groupby("UnitNumber")["TimeInCycles"].max().reset_index()
    df_max_time.rename(columns={"TimeInCycles": "MaxTime"}, inplace=True)
    df_train = df_train.merge(df_max_time, on="UnitNumber", how="left")

    df_train["RUL"] = df_train["MaxTime"] - df_train["TimeInCycles"]
    df_train["RUL"] = np.minimum(df_train["RUL"], max_rul)

    # True RUL für Test
    y_test_true = rul_df[0].values.astype(float)
    y_test_true = np.minimum(y_test_true, max_rul)

    return df_train, df_test, y_test_true

def get_feature_drop_cols(fd_id):
    # Sensoren, die fast konstant sind
    base_drop_sensors = ["Sensor1", "Sensor5", "Sensor10"]  # wie in deinem FD001-Setup

    drop_cols = base_drop_sensors.copy()

    # FD001/FD003: 1 Operating Condition -> Setting3 evtl. redundant
    if fd_id in ["FD001", "FD003"]:
        drop_cols += ["Setting3"]
    # FD002/FD004: 6 Conditions -> alle Settings behalten

    # Diese sind keine Eingangsfeatures
    drop_cols += ["MaxTime"]

    return drop_cols

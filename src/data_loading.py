import os
import numpy as np
import pandas as pd

# Projekt-Root: eine Ebene über src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FD_IDS_GLOBAL = ["FD001", "FD002", "FD003", "FD004"]

def _build_condition_mapping(df_train: pd.DataFrame, decimals: int = 3):
    """
    Erzeugt ein deterministisches Mapping von (Setting1, Setting2, Setting3) -> ConditionID.

    - Rundet die Settings, um Floating-Point-Rauschen zu entfernen.
    - Bestimmt eindeutige Kombinationen und weist ihnen IDs 0..N-1 zu.
    """
    rounded = df_train[["Setting1", "Setting2", "Setting3"]].round(decimals=decimals)

    unique_conds = (
        rounded.drop_duplicates()
        .sort_values(["Setting1", "Setting2", "Setting3"])
        .reset_index(drop=True)
    )

    cond_map = {}
    for cond_id, row in unique_conds.iterrows():
        key = (row["Setting1"], row["Setting2"], row["Setting3"])
        cond_map[key] = cond_id

    return cond_map


def _assign_condition_id(df: pd.DataFrame, cond_map, decimals: int = 3) -> pd.DataFrame:
    """
    Weist einer DataFrame die ConditionID anhand eines vorher berechneten Mappings zu.

    Wenn eine Setting-Kombination nicht im Mapping ist, wird ein Fehler geworfen
    (sollte nicht passieren, wenn Train/Test konsistent sind).
    """
    rounded = df[["Setting1", "Setting2", "Setting3"]].round(decimals=decimals)

    cond_ids = []
    for _, row in rounded.iterrows():
        key = (row["Setting1"], row["Setting2"], row["Setting3"])
        if key not in cond_map:
            raise ValueError(f"Unknown condition key {key} encountered in assign_condition_id.")
        cond_ids.append(cond_map[key])

    df = df.copy()
    df["ConditionID"] = cond_ids
    return df


def load_cmapps_subset(fd_id, data_dir=DEFAULT_DATA_DIR, max_rul=125):
    """
    Load train, test, and RUL files for a given C-MAPSS subset (FD001–FD004)
    and compute RUL for the training trajectories.
    """
    assert fd_id in ["FD001", "FD002", "FD003", "FD004"], f"Unknown subset: {fd_id}"

    train_path = os.path.join(data_dir, f"train_{fd_id}.txt")
    test_path  = os.path.join(data_dir, f"test_{fd_id}.txt")
    rul_path   = os.path.join(data_dir, f"RUL_{fd_id}.txt")

    col_names = ["UnitNumber", "TimeInCycles", "Setting1", "Setting2", "Setting3"] + \
                [f"Sensor{i}" for i in range(1, 22)]

    df_train = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    df_test  = pd.read_csv(test_path,  sep=r"\s+", header=None, names=col_names)
    rul_df   = pd.read_csv(rul_path,   sep=r"\s+", header=None)

    # --- RUL für Training ---
    df_max_time = df_train.groupby("UnitNumber")["TimeInCycles"].max().reset_index()
    df_max_time.rename(columns={"TimeInCycles": "MaxTime"}, inplace=True)
    df_train = df_train.merge(df_max_time, on="UnitNumber", how="left")

    df_train["RUL"] = df_train["MaxTime"] - df_train["TimeInCycles"]
    df_train["RUL"] = np.minimum(df_train["RUL"], max_rul)

    # --- True RUL für Test ---
    y_test_true = rul_df[0].values.astype(float)
    y_test_true = np.minimum(y_test_true, max_rul)

    # --- ConditionID hinzufügen ---
 # --- ConditionID initialisieren ---
    df_train["ConditionID"] = 0
    df_test["ConditionID"]  = 0

    # Nur FD002/FD004 haben mehrere Betriebsbedingungen
    if fd_id in ["FD002", "FD004"]:
        df_train, df_test = add_condition_id_from_rounded_settings(df_train, df_test)
    # FD001/FD003 bleiben bei ConditionID = 0

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


def add_condition_id_from_rounded_settings(df_train, df_test,
                                           s1_decimals=0,
                                           s2_decimals=1,
                                           s3_decimals=0):
    """
    Erzeugt eine diskrete ConditionID aus gerundeten Setting1/2/3.
    Die IDs basieren NUR auf den im Training vorkommenden (S1_r, S2_r, S3_r)-Tripeln.

    - df_train, df_test: DataFrames mit Spalten 'Setting1', 'Setting2', 'Setting3'
    - Rückgabe: df_train, df_test mit zusätzlicher Spalte 'ConditionID' (int)
    """

    # Kopien, um Seiteneffekte zu vermeiden
    df_tr = df_train.copy()
    df_te = df_test.copy()

    # 1) Rounded Settings erzeugen
    for df in (df_tr, df_te):
        df["S1_r"] = df["Setting1"].round(s1_decimals)
        df["S2_r"] = df["Setting2"].round(s2_decimals)
        df["S3_r"] = df["Setting3"].round(s3_decimals)

    # 2) Eindeutige Tripel im TRAIN finden
    unique_train = (
        df_tr[["S1_r", "S2_r", "S3_r"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Mapping: (S1_r, S2_r, S3_r) -> ConditionID (0..N-1)
    cond_map = {
        (row.S1_r, row.S2_r, row.S3_r): idx
        for idx, row in unique_train.iterrows()
    }

    print(f"[ConditionID] Found {len(cond_map)} unique (S1_r, S2_r, S3_r) combos in TRAIN.")

    # 3) Helper zum Zuordnen
    def _map_condition(row):
        key = (row["S1_r"], row["S2_r"], row["S3_r"])
        if key not in cond_map:
            raise ValueError(f"Unknown condition triplet in TEST data: {key}")
        return cond_map[key]

    df_tr["ConditionID"] = df_tr.apply(_map_condition, axis=1)
    df_te["ConditionID"] = df_te.apply(_map_condition, axis=1)

    # 4) Debug-Ausgabe
    print("[ConditionID] Train ConditionIDs:", np.sort(df_tr["ConditionID"].unique()))
    print("[ConditionID] Test  ConditionIDs:", np.sort(df_te["ConditionID"].unique()))

    # Optional: die gerundeten Hilfsspalten wieder entfernen
    df_tr.drop(columns=["S1_r", "S2_r", "S3_r"], inplace=True)
    df_te.drop(columns=["S1_r", "S2_r", "S3_r"], inplace=True)

    return df_tr, df_te

def load_cmapps_global(fd_ids=FD_IDS_GLOBAL,
                       data_dir=DEFAULT_DATA_DIR,
                       max_rul=125):
    """
    Load all requested C-MAPSS subsets and build ONE global training DataFrame
    plus per-subset test DataFrames + true RUL arrays.

    Returns
    -------
    df_train_all : pd.DataFrame
        Global training dataframe with column 'FD_ID'.
    test_dfs : dict[str, pd.DataFrame]
        Mapping fd_id -> test dataframe with 'FD_ID' column.
    test_ruls : dict[str, np.ndarray]
        Mapping fd_id -> y_test_true (clamped).
    """
    train_frames = []
    test_dfs = {}
    test_ruls = {}

    for fd_idx, fd_id in enumerate(fd_ids):
        df_train, df_test, y_test_true = load_cmapps_subset(
            fd_id, data_dir=data_dir, max_rul=max_rul
        )

        # Add FD_ID as integer (0..N-1)
        df_train["FD_ID"] = fd_idx
        df_test["FD_ID"] = fd_idx

        train_frames.append(df_train)
        test_dfs[fd_id] = df_test
        test_ruls[fd_id] = y_test_true

    df_train_all = pd.concat(train_frames, ignore_index=True)
    return df_train_all, test_dfs, test_ruls

def load_cmapps_global_test():
    """
    Lädt alle Test-Sets FD001–FD004, fügt FD_ID hinzu
    und liefert ein einheitliches DataFrame zurück.
    """
    dfs = []
    for fd_id in ["FD001", "FD002", "FD003", "FD004"]:
        df_train, df_test, rul_test = load_cmapps_subset(fd_id)

        df_test = df_test.copy()
        df_test["FD_ID"] = fd_id
        dfs.append(df_test)

    return pd.concat(dfs, ignore_index=True)
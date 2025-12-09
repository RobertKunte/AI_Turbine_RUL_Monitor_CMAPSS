"""
Driver-Script für Full-Trajectory LSTM Training auf FD001-FD004.

Dieses Script:
- Lädt die CMAPSS-Daten FD001-FD004
- Fügt physikalische Features und ConditionID hinzu
- Baut Full-Trajectory Sequenzen (Sliding Window über alle Zyklen)
- Trainiert ein EOLFullLSTM-Modell
- Evaluiert mit NASA PHM08 Metriken
"""

import os
import sys
from pathlib import Path

# Projekt-Root zum Python-Pfad hinzufügen
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np

from src.data_loading import load_cmapps_subset
from src.additional_features import create_physical_features
from src.config import GLOBAL_FEATURE_COLS
from src.eol_full_lstm import (
    build_full_eol_sequences_from_df,
    create_full_dataloaders,
    EOLFullLSTM,
    train_eol_full_lstm,
    evaluate_eol_full_lstm,
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===================================================================
# 1. Daten laden: FD001-FD004
# ===================================================================
print("=" * 60)
print("[1] Loading CMAPSS Data (FD001-FD004)")
print("=" * 60)

dfs = []
for fd_id in ["FD001", "FD002", "FD003", "FD004"]:
    print(f"Loading {fd_id}...")
    df_train, _, _ = load_cmapps_subset(
        fd_id,
        max_rul=None,  # Kein Clipping im Train (wird später in build_full_eol_sequences gecappt)
        clip_train=False,
        clip_test=True,  # Test weiter clampen für NASA-Score
    )
    df_train = create_physical_features(df_train)
    # FD_ID als numerisches Feature (0=FD001, 1=FD002, 2=FD003, 3=FD004)
    # Wichtig: FD_ID muss numerisch sein, da es als Feature verwendet wird
    fd_id_map = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}
    df_train["FD_ID"] = fd_id_map[fd_id]
    dfs.append(df_train)
    print(f"  {fd_id}: {len(df_train)} rows, {df_train['UnitNumber'].nunique()} engines")

df_train_global = pd.concat(dfs, ignore_index=True)
print(f"\nTotal: {len(df_train_global)} rows, {df_train_global['UnitNumber'].nunique()} engines")

# ===================================================================
# 2. Feature-Liste definieren
# ===================================================================
print("\n" + "=" * 60)
print("[2] Defining Feature Columns")
print("=" * 60)

# Numerische Features aus GLOBAL_FEATURE_COLS
numeric_cols = df_train_global[GLOBAL_FEATURE_COLS].select_dtypes(
    include=["number"]
).columns.tolist()

feature_cols = numeric_cols
print(f"Using {len(feature_cols)} features:")
print(f"  Features: {', '.join(feature_cols[:10])}..." if len(feature_cols) > 10 else f"  Features: {', '.join(feature_cols)}")

# ===================================================================
# 3. Full-Trajectory Sequenzen bauen
# ===================================================================
print("\n" + "=" * 60)
print("[3] Building Full-Trajectory Sequences")
print("=" * 60)

X_full, y_full, unit_ids_full, cond_ids_full = build_full_eol_sequences_from_df(
    df=df_train_global,
    feature_cols=feature_cols,
    past_len=30,
    max_rul=125,  # NASA-Style: RUL wird auf 125 gecappt
    unit_col="UnitNumber",
    cycle_col="TimeInCycles",
    rul_col="RUL",
)

# ===================================================================
# 4. Dataloaders erstellen (Engine-basierter Split)
# ===================================================================
print("\n" + "=" * 60)
print("[4] Creating DataLoaders (Engine-based Split)")
print("=" * 60)

train_loader, val_loader, scaler, train_unit_ids, val_unit_ids = create_full_dataloaders(
    X=X_full,
    y=y_full,
    unit_ids=unit_ids_full,
    cond_ids=cond_ids_full,
    batch_size=256,
    engine_train_ratio=0.8,
    shuffle_engines=True,
    random_seed=42,
)

# ===================================================================
# 5. Modell initialisieren
# ===================================================================
print("\n" + "=" * 60)
print("[5] Initializing EOLFullLSTM Model")
print("=" * 60)

model = EOLFullLSTM(
    input_dim=len(feature_cols),
    hidden_dim=64,
    num_layers=2,
    dropout=0.1,
    bidirectional=False,
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ===================================================================
# 6. Training
# ===================================================================
print("\n" + "=" * 60)
print("[6] Training EOLFullLSTM")
print("=" * 60)

from src.config import (
    HI_CONDITION_CALIB_WEIGHT,
    HI_CONDITION_CALIB_PLATEAU_THRESH,
)

model, history = train_eol_full_lstm(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=80,
    lr=1e-4,
    weight_decay=1e-4,
    patience=8,
    device=device,
    results_dir="../results/eol_full_lstm",
    run_name="fd001_fd004",
    use_health_head=True,  # Enable multi-task mode for condition calibration
    hi_condition_calib_weight=HI_CONDITION_CALIB_WEIGHT,
    hi_plateau_threshold=HI_CONDITION_CALIB_PLATEAU_THRESH,
)

# ===================================================================
# 7. Evaluation
# ===================================================================
print("\n" + "=" * 60)
print("[7] Evaluating EOLFullLSTM")
print("=" * 60)

metrics = evaluate_eol_full_lstm(
    model=model,
    val_loader=val_loader,
    device=device,
)

# ===================================================================
# 8. Zusammenfassung
# ===================================================================
print("\n" + "=" * 60)
print("[8] Final Summary")
print("=" * 60)
print("Pointwise Metrics (all validation samples):")
print(f"  RMSE: {metrics['pointwise']['rmse']:.4f} cycles")
print(f"  MAE: {metrics['pointwise']['mae']:.4f} cycles")
print(f"  Bias: {metrics['pointwise']['bias']:.4f} cycles")
print(f"  R²: {metrics['pointwise']['r2']:.4f}")
print(f"  NASA Score (pointwise, sum): {metrics['nasa_pointwise']['score_sum']:.2f}")
print(f"  NASA Score (pointwise, mean): {metrics['nasa_pointwise']['score_mean']:.4f}")

if "eol" in metrics:
    print("\nEOL Metrics (per engine, last cycle):")
    print(f"  RMSE_eol: {metrics['eol']['rmse']:.4f} cycles")
    print(f"  MAE_eol: {metrics['eol']['mae']:.4f} cycles")
    print(f"  Bias_eol: {metrics['eol']['bias']:.4f} cycles")
    print(f"  NASA Score (EOL, sum): {metrics['eol']['nasa_score_sum']:.2f}")
    print(f"  NASA Score (EOL, mean): {metrics['eol']['nasa_score_mean']:.4f}")
    print(f"  Num engines: {metrics['eol']['num_engines']}")

print("=" * 60)
print("Training complete!")
print("=" * 60)


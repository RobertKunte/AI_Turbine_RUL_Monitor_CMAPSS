import numpy as np
import pandas as pd

def create_physical_features(df):
    """
    Creates three new, physically motivated features based on the sensor data 
    for improved RUL prediction.
    """
    df_new = df.copy()

    # --- 1. HPC Efficiency Proxy (Total Pressure / Total Temperature) ---
    # Sensor 12: Total pressure at High Pressure Turbine (T4)
    # Sensor 7: Total temperature at High Pressure Compressor (T2)
    df_new['Effizienz_HPC_Proxy'] = df_new['Sensor12'] / df_new['Sensor7']

    # --- 2. Exhaust Gas Temperature Drift (EGT Drift) ---
    # Sensor 17: Exhaust Gas Temperature (EGT)
    
    # Calculate the mean of Sensor 17 for the first 10 cycles of each unit 
    # as the "healthy" baseline reference.
    df_ref = df_new[df_new['TimeInCycles'] <= 10].groupby('UnitNumber')['Sensor17'].mean().reset_index()
    df_ref.rename(columns={'Sensor17': 'Sensor17_Base'}, inplace=True)
    
    # Merge the baseline back into the main DataFrame
    df_new = pd.merge(df_new, df_ref, on='UnitNumber', how='left')
    
    # Calculate the Drift: (Current Value - Healthy Baseline)
    df_new['EGT_Drift'] = df_new['Sensor17'] - df_new['Sensor17_Base']
    
    # Cleanup: remove the temporary base column
    df_new.drop(columns=['Sensor17_Base'], inplace=True)


    # --- 3. Fan-HPC Degradation Ratio ---
    # Sensor 2: Fan Speed
    # Sensor 3: HPC Speed
    df_new['Fan_HPC_Ratio'] = df_new['Sensor2'] / df_new['Sensor3']

    print(f"New columns successfully added. Current number of columns: {len(df_new.columns)}")
    
    # RUL ist nicht immer vorhanden (z.B. im Testset) → optional ausgeben
    debug_cols = ['Effizienz_HPC_Proxy', 'EGT_Drift', 'Fan_HPC_Ratio']
    if 'RUL' in df_new.columns:
        debug_cols.append('RUL')
    print(df_new[debug_cols].head())
    
    return df_new
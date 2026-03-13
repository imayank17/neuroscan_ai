import pandas as pd
import numpy as np
import os

data_path = "backend/data.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("Columns:", df.columns.tolist()[:10], "...", df.columns.tolist()[-5:])
    print("Class distribution (y):")
    print(df['y'].value_counts().sort_index())
    
    # Try to inspect model if possible
    try:
        from tensorflow.keras.models import load_model
        model_path = "backend/Epilepsy.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("\nModel Input Shape:", model.input_shape)
        else:
            print(f"\nModel {model_path} not found")
    except Exception as e:
        print("\nCould not inspect model:", e)
else:
    print("data.csv not found")

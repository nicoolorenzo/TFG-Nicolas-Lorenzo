import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.data_loading import get_my_data

print("Getting data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()
print("Data was extracted")

# Tomar los RTs originales
SMRT_RTs_raw = y[17388:97425].reshape(-1, 1)

# Calcular el scaler
scaler = StandardScaler()
scaler.fit(SMRT_RTs_raw)

# Mostrar solo scaler.scale_[0]
print(f"Valor de scaler.scale_[0]: {scaler.scale_[0]}")

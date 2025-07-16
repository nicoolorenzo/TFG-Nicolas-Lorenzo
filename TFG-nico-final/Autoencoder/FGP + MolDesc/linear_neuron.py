print("Starting script")

# ================================
# IMPORTACIONES
# ================================
from tensorflow import keras
import pandas as pd
from utils.data_loading import get_my_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from keras import layers
from keras.layers import Input, Concatenate, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping

from BlackBox.Preprocessors import DescriptorsPreprocessor

# Estilo de gráficos
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# ================================
# 1. CARGA DE DATOS
# ================================
print("Getting data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()
print("Data was extracted")

# Índices SMRT
start_idx = 17388
end_idx = 97425

# ================================
# 2. PROCESAMIENTO DE DESCRIPTORES
# ================================
X_raw_desc = X.iloc[start_idx:end_idx, descriptors_columns]
RTs_raw = y[start_idx:end_idx]

# Normalizar descriptores
desc_cols = list(range(X_raw_desc.shape[1]))
preproc = DescriptorsPreprocessor(desc_cols=desc_cols, cor_th=0.9, k=1977)
preproc.fit(X_raw_desc, RTs_raw)
X_proc_desc = preproc.transform(X_raw_desc)

# Normalizar RTs
RTs_array = RTs_raw.reshape(-1, 1)
scaler = StandardScaler()
RTs_scaled = scaler.fit_transform(RTs_array)


# ================================
# 3. PROCESAMIENTO DE FINGERPRINTS
# ================================
X_raw_fgp = X.iloc[start_idx:end_idx, fingerprints_columns[2:]].astype("float32")

X_ALL = pd.concat([X_proc_desc, X_raw_fgp], axis=0)

X_ALL_train, X_ALL_val, RT_train, RT_val = train_test_split(
    X_ALL, RTs_scaled, test_size=0.3, random_state=42
)
##################################################### Neurona lineal: ########################################################

#Neurona lineal
single_neuron = keras.Sequential([
    layers.Dense(1, activation='linear', name="pred")
], name="single_neuron")

# Compilación
single_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mae", "mse"]
)

# ================================
# 6. ENTRENAMIENTO
# ================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = single_neuron.fit(
    X_ALL_train,
    RT_train,
    validation_data=(X_ALL_val, RT_val),
    epochs=500,
    batch_size=64,
    callbacks=callbacks
)

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("linear_neuron_ALL.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("linear_neuron_ALL.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("linear_neuron_ALL.png", dpi=300, bbox_inches='tight')
plt.close()



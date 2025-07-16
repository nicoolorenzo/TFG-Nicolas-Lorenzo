print("Starting script")
from tensorflow import keras
import pandas as pd
from utils.data_loading import get_my_data
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
from BlackBox.Preprocessors import DescriptorsPreprocessor

print("Getting data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()

# Extraer datos
X_raw = X.iloc[:, descriptors_columns]

# 1. Obtener índices de columnas (todas las del nuevo X_train)
desc_cols = list(range(X_raw.shape[1]))

# 2. Crear el preprocesador
preproc = DescriptorsPreprocessor(desc_cols=desc_cols, cor_th=0.9, k='all')

# 3. Ajustar el preprocesador con los datos y (RTs) correspondientes
preproc.fit(X_raw, y)

# 4. Transformar los datos con el preprocesador ya ajustado
X_proc = preproc.transform(X_raw)

# 5. División en entrenamiento y validación
X_train, X_val = train_test_split(X_proc, test_size=0.2, random_state=42)

########################################## Predictor descriptores solos (MAE): ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")

# 10. Compilar el modelo
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mae", "mse"]
)

print(out_neuron.summary())

#Callbacks de entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])

# 13. Guardar modelo
out_neuron.save("linear_neuron_desc.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("linear_neuron_desc.png", dpi=300, bbox_inches='tight')
df.to_csv('linear_neuron_desc.csv', index=False)


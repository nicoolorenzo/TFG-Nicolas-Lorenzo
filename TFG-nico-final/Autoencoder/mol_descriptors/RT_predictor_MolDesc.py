print("Starting script")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.data_loading import get_my_data
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from BlackBox.Preprocessors import DescriptorsPreprocessor

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

##
print("Gettig data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()
print("Data was extracted")


### Extraemos Descriptores y RTs de SMRT:

# 1. Extraer descriptores y RTs
X_raw = X.iloc[17388:97425, descriptors_columns]
SMRT_RTs_raw0 = y[17388:97425]  # y como vector 1D (Series)
desc_cols = list(range(X_raw.shape[1]))
print(SMRT_RTs_raw0.shape)

# 2. Crear y ajustar el preprocesador (usa X e y para selección supervisada)
preproc = DescriptorsPreprocessor(desc_cols=desc_cols, cor_th=0.9, k=1977)
preproc.fit(X_raw, SMRT_RTs_raw0)

# 3. Transformar X
X_proc = preproc.transform(X_raw)  # ahora sí, sin error

# 4. Normalizar y (RTs)
SMRT_RTs_raw = SMRT_RTs_raw0.reshape(-1, 1)  # convertir a array 2D
scaler = StandardScaler()
SMRT_RTs = scaler.fit_transform(SMRT_RTs_raw)
#SMRT_RTs_raw=scaler.inverse_transform(SMRT_RTs)
print(SMRT_RTs.shape)

# 5. División de datos en entrenamiento y validación
X_train, X_val, RT_train, RT_val = train_test_split(X_proc, SMRT_RTs, test_size=0.3, random_state=42)

# 6. Conversión a float (por compatibilidad con Keras)
X_train = X_train.astype(float)
X_val = X_val.astype(float)
RT_train = RT_train.astype(float)
RT_val = RT_val.astype(float)




########################################## Asim 3 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])


# 13. Guardar modelo
out_neuron.save("ASIM3_DESC.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("ASIM3_DESC.png", dpi=300, bbox_inches='tight')
df.to_csv('ASIM3_DESC.csv', index=False)

######################################### Asim 4 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])


# 13. Guardar modelo
out_neuron.save("ASIM4_DESC.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("ASIM4_DESC.png", dpi=300, bbox_inches='tight')
df.to_csv('ASIM4_DESC.csv', index=False)

######################################## Asim 5 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(61, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(30, activation='selu', kernel_initializer='lecun_normal'),
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])


# 13. Guardar modelo
out_neuron.save("ASIM5_DESC.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("ASIM5_DESC.png", dpi=300, bbox_inches='tight')
df.to_csv('ASIM5_DESC.csv', index=False)

######################################## Asim 6 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(61, activation='selu', kernel_initializer='lecun_normal'),
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])


# 13. Guardar modelo
out_neuron.save("ASIM6_DESC.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("ASIM6_DESC.png", dpi=300, bbox_inches='tight')
df.to_csv('ASIM6_DESC.csv', index=False)

######################################## Asim 5 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(61, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(30, activation='selu', kernel_initializer='lecun_normal'),
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])


# 13. Guardar modelo
out_neuron.save("ASIM7_DESC.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("ASIM7_DESC.png", dpi=300, bbox_inches='tight')
df.to_csv('ASIM7_DESC.csv', index=False)

######################################## Asim 8 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(246, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(61, activation='selu', kernel_initializer='lecun_normal'),
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])


# 13. Guardar modelo
out_neuron.save("ASIM8_DESC.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("ASIM8_DESC.png", dpi=300, bbox_inches='tight')
df.to_csv('ASIM8_DESC.csv', index=False)
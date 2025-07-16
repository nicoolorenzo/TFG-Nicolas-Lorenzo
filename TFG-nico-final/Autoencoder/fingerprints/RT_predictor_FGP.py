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



mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
##
print("Gettig data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()
print("Data was extracted")

#Extraemos fgps y RTs de SMRT:

#Fingerprints:
SMRT_fgps = X.iloc[17388:97425, fingerprints_columns[2:]]
print(SMRT_fgps.shape)

#RTs:
SMRT_RTs_raw0 = y[17388:97425]
SMRT_RTs_raw = SMRT_RTs_raw0.reshape(-1, 1)
scaler = StandardScaler()
SMRT_RTs = scaler.fit_transform(SMRT_RTs_raw)
print(f"Desviación típica (scaler.scale_[0]) = {scaler.scale_[0]:.4f} segundos")

#Definimos los conjuntos de entrenamiento y validación, para fgps y para RTs:
X_train, X_val, RT_train, RT_val = train_test_split(SMRT_fgps, SMRT_RTs, test_size=0.3, random_state=42)

X_train = X_train.astype(float)
X_val = X_val.astype(float)
RT_train = RT_train.astype(float)
RT_val = RT_val.astype(float)

########################################## Predictor 9 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")



# 10. Compilar el modelo
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mae", "mse"]
)

#print(predictor.summary())

#Callbacks de entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])



# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("LINEAL_FGP.png", dpi=300, bbox_inches='tight')
df.to_csv('LINEAL_FGP.csv', index=False)


exit()
########################################################## Asim 10: ##################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_123"),
    layers.Dense(70, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_1232"),
    layers.Dense(35, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_61"),
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")


#Nota: Puede que Adam sea mejor que RMSprop
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mse"]
)

print(out_neuron.summary())

#Entrenamiento: Encoder+Neurona lineal vs Neurona lineal
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)

#Encoder+Neurona lineal
history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])
out_neuron.save("asim10_fgp.keras")

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("asim10_fgp.png", dpi=300, bbox_inches='tight')

df.to_csv('asim10_fgp.csv', index=False)


exit()


########################################################## Asim 6: ##################################################

out_neuron = keras.Sequential([
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(70, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(35, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")


#Nota: Puede que Adam sea mejor que RMSprop
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mse"]
)

print(out_neuron.summary())

#Entrenamiento: Encoder+Neurona lineal vs Neurona lineal
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)

#Encoder+Neurona lineal
history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])
out_neuron.save("asim6_fgp.keras")

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("asim6_fgp.png", dpi=300, bbox_inches='tight')

df.to_csv('asim6_fgp.csv', index=False)

########################################################## Asim 7: ##################################################
out_neuron = keras.Sequential([
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(70, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")

#Nota: Puede que Adam sea mejor que RMSprop
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mse"]
)

print(out_neuron.summary())

#Entrenamiento: Encoder+Neurona lineal vs Neurona lineal
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)

#Encoder+Neurona lineal
history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])
out_neuron.save("asim7_fgp.keras")

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("asim7_fgp.png", dpi=300, bbox_inches='tight')

df.to_csv('asim7_fgp.csv', index=False)

########################################################## Asim 8: ##################################################


out_neuron = keras.Sequential([
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(70, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(70, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(35, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")


#Nota: Puede que Adam sea mejor que RMSprop
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mse"]
)

print(out_neuron.summary())

#Entrenamiento: Encoder+Neurona lineal vs Neurona lineal
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)

#Encoder+Neurona lineal
history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])
out_neuron.save("asim8_fgp.keras")

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("asim8_fgp.png", dpi=300, bbox_inches='tight')

df.to_csv('asim8_fgp.csv', index=False)

########################################################## Asim 9: ##################################################


out_neuron = keras.Sequential([
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(140, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(70, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")


#Nota: Puede que Adam sea mejor que RMSprop
out_neuron.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mse"]
)

print(out_neuron.summary())

#Entrenamiento: Encoder+Neurona lineal vs Neurona lineal
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)

#Encoder+Neurona lineal
history = out_neuron.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])
out_neuron.save("asim9_fgp.keras")

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("asim9_fgp.png", dpi=300, bbox_inches='tight')

df.to_csv('asim9_fgp.csv', index=False)
print("Starting script")

# Librerías estándar y de terceros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from utils.data_loading import get_my_data


print("Gettig data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()

def metric_common_zeros_ones(y_true, y_pred):
    # Redondear las predicciones a 0 o 1
    y_pred_rounded = tf.round(y_pred)
    # Convertimos los valores booleanos a enteros para hacer la comparación
    y_true = tf.cast(y_true, tf.int32)
    y_pred_rounded = tf.cast(y_pred_rounded, tf.int32)
    # Contamos cuántos 1 y 0 coinciden entre las predicciones y los valores originales
    common_ones_zeros = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_rounded), tf.float32))
    return common_ones_zeros

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Sacamos X_train
X_train = X.iloc[:, fingerprints_columns]
X_train = X_train.astype("float32")
X_train = X_train.iloc[:,  2:]
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)

#AUTOENCODER 21: Cuello de botella de 550

stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[2212]),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu", input_shape=[280]),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(2212, activation="sigmoid"),
])

# Autoencoder
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

# Compilación
stacked_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mae", "mse", metric_common_zeros_ones]
)

print(stacked_encoder.summary())
print(stacked_decoder.summary())

# Entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)
history = stacked_ae.fit(X_train, X_train, epochs=500, validation_data=(X_val, X_val), callbacks=[callback, reduce_lr])
stacked_encoder.save("autoencoder_run22.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics22.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics22.csv', index=False)
#########################################################################################################################

SMRT_fgps = X.iloc[17388:97425, fingerprints_columns[2:]]

#RTs:
SMRT_RTs_raw0 = y[17388:97425]
SMRT_RTs_raw = SMRT_RTs_raw0.reshape(-1, 1)
scaler = StandardScaler()
SMRT_RTs = scaler.fit_transform(SMRT_RTs_raw)

#Definimos los conjuntos de entrenamiento y validación, para fgps y para RTs:
X_train, X_val, RT_train, RT_val = train_test_split(SMRT_fgps, SMRT_RTs, test_size=0.3, random_state=42)

X_train = X_train.astype(float)
X_val = X_val.astype(float)
RT_train = RT_train.astype(float)
RT_val = RT_val.astype(float)
#################################################################################################################################################################

#Autoencoder 4 + (280/280/280/280/1) -------> Predictor 7_2

# Definir el submodelo de salida con una capa oculta antes de la salida final
out_neuron = keras.Sequential([
    keras.Input(shape=(280,)),
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(280, activation='selu', kernel_initializer='lecun_normal'),
    layers.Dense(1, activation='linear')
], name="output_neuron")

# Cargar y clonar el autoencoder (solo el encoder se usará)
model_A = keras.models.load_model("autoencoder_run4.keras")
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
model_A_clone._name = "encoder_model"

# Combinar encoder + output_neuron
predictor = keras.models.Sequential([
    model_A_clone,
    out_neuron
])

# Compilar el modelo
predictor.compile(
    loss="mae",
    optimizer=keras.optimizers.Adam(),
    metrics=["mse"]
)

#Entrenamiento: Encoder+Neurona lineal vs Neurona lineal
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)

#Encoder+Neurona lineal
history = predictor.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])
predictor.save("predictor_run7_2.keras")

df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("predictor_run7_2.png", dpi=300, bbox_inches='tight')

df.to_csv('predictor_run7_2.csv', index=False)



#########################################################################################################################3

print("Starting script")

# Librerías estándar y de terceros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import ReduceLROnPlateau
from utils.data_loading import get_my_data

print("Gettig data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()

def metric_common_zeros_ones(y_true, y_pred):
    # Redondear las predicciones a 0 o 1
    y_pred_rounded = tf.round(y_pred)
    # Convertimos los valores booleanos a enteros para hacer la comparación
    y_true = tf.cast(y_true, tf.int32)
    y_pred_rounded = tf.cast(y_pred_rounded, tf.int32)
    # Contamos cuántos 1 y 0 coinciden entre las predicciones y los valores originales
    common_ones_zeros = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_rounded), tf.float32))
    return common_ones_zeros

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Extraemos los fgp de get_my_data()
print("Gettig data")
X, y, usp_columns, chromatography_columns, descriptors_columns, fingerprints_columns = get_my_data()

#Sacamos X_train
X_train = X.iloc[:, fingerprints_columns]
X_train = X_train.astype("float32")
X_train = X_train.iloc[:,  2:]
X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=42)


stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='glorot_normal', activation="selu", input_shape=[2212]),
    keras.layers.Dense(1100, kernel_initializer='glorot_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='glorot_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='glorot_normal', activation="selu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(550, kernel_initializer='glorot_normal', activation="selu", input_shape=[280]),
    keras.layers.Dense(1100, kernel_initializer='glorot_normal', activation="selu"),
    keras.layers.Dense(2212, activation="sigmoid"),
])

# Autoencoder
stacked_ae = keras.models.Sequential([stacked_encoder, stacked_decoder])

# Compilación
stacked_ae.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mae", "mse", metric_common_zeros_ones]
)

print(stacked_encoder.summary())
print(stacked_decoder.summary())

# Entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)
history = stacked_ae.fit(X_train, X_train, epochs=500, validation_data=(X_val, X_val), callbacks=[callback, reduce_lr])
stacked_encoder.save("autoencoder_run23.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics23.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics23.csv', index=False)

########################################## Predictor 1 descriptores: ###########################################################

 7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")

# 8. Cargar y clonar el autoencoder
model_A = keras.models.load_model("auto_desc_1.keras")
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
model_A_clone._name = "encoder_model"

# 9. Crear el modelo predictor (autoencoder + neurona lineal)
predictor = keras.models.Sequential([
    model_A_clone,
    out_neuron
])

# 10. Compilar el modelo
predictor.compile(
    loss="mse",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mae", "mse"]
)

print(predictor.summary())

history = predictor.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])

# 13. Guardar modelo
predictor.save("pred_desc_1.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("pred_desc_1.png", dpi=300, bbox_inches='tight')
df.to_csv('pred_desc_1.csv', index=False)

########################################## Predictor 9 descriptores: ###########################################################

#7. Crear la neurona de salida
out_neuron = keras.Sequential([
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_123"),
    layers.Dense(123, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_1232"),
    layers.Dense(61, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_61"),
    layers.Dense(30, activation='selu', kernel_initializer='lecun_normal', name="dense_selu_30"),
    layers.Dense(1, activation='linear', name="pred")
], name="output_neuron")

# 8. Cargar y clonar el autoencoder
model_A = keras.models.load_model("auto_desc_1_NEW.keras")
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
model_A_clone._name = "encoder_model9"
model_A_clone.trainable= False

# 9. Crear el modelo predictor (autoencoder + neurona lineal)
predictor = keras.models.Sequential([
    model_A_clone,
    out_neuron
])

# 10. Compilar el modelo
predictor.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["mae", "mse"]
)

print(predictor.summary())

#Callbacks de entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = predictor.fit(X_train, RT_train, epochs=500, validation_data=(X_val, RT_val), callbacks=[callback, reduce_lr])

#Y_val_pred_scal = predictor.predict(X_val)
#Y_val_pred scaler.inverse_transform(Y_val_pred_scal)
#RT_val_Seg = scaler.inverse_transform(RT_val)
#Calcular MAE /MSE /lo quese aentre Y_val_pred y RT_val_Seg


# 13. Guardar modelo
predictor.save("pred_desc9_frozen.keras")

# 14. Guardar e imprimir resultados
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.75)
plt.savefig("pred_desc9_frozen.png", dpi=300, bbox_inches='tight')
df.to_csv('pred_desc9_frozen.csv', index=False)

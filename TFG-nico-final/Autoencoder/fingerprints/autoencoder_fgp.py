print("Starting script")
from tensorflow import keras
import pandas as pd
from utils.data_loading import get_my_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau



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

#AUTOENCODER 22: Autoencoder 4 + RMSprop + Lecun + selu

# Encoder
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
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)
history = stacked_ae.fit(X_train, X_train, epochs=500, validation_data=(X_val, X_val), callbacks=[callback, reduce_lr])


#Gráficas
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="cross_entropy")
plt.plot(history.history['val_loss'], label="val_cross_entropy")
plt.plot(history.history['mse'], label="MSE")
plt.plot(history.history['val_mse'], label="val_MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss (cross_entropy)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("autoencoder_4_RERUN.png", dpi=300, bbox_inches='tight')
plt.close()




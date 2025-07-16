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
from keras.initializers import lecun_normal

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

# División en train/val
X_train_desc, X_val_desc, RT_train, RT_val = train_test_split(
    X_proc_desc, RTs_scaled, test_size=0.3, random_state=42
)

# ================================
# 3. PROCESAMIENTO DE FINGERPRINTS
# ================================
X_raw_fgp = X.iloc[start_idx:end_idx, fingerprints_columns[2:]].astype("float32")

X_train_fgp, X_val_fgp, _, _ = train_test_split(
    X_raw_fgp, RTs_scaled, test_size=0.3, random_state=42
)


#######################################################  ASIM 5: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(526, activation='selu', kernel_initializer=lecun_normal())(merged)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim5.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim5.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim5.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 10: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim10.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim10.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim10.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 11: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim11.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim11.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim11.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 12: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim12.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim12.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim12.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 9: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(526, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(131, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim13.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim13.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim13.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 10: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(193, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(96, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim14.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim14.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim14.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 11: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(201, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(100, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim15.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim15.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim15.png", dpi=300, bbox_inches='tight')
plt.close()


####################################################### Predictor 12: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(131, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(65, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim16.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim16.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim16.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 9: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(526, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(526, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(131, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim17.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim17.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim17.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 10: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(193, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(96, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim18.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim18.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim18.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 11: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(201, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(100, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim19.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim19.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim19.png", dpi=300, bbox_inches='tight')
plt.close()


####################################################### Predictor 12: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(merged)
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(131, activation='selu', kernel_initializer=lecun_normal())(x)
x = Dense(65, activation='selu', kernel_initializer=lecun_normal())(x)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)


full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="RT_Predictor_NoEncoders")


full_model.compile(
    loss="mae",
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.7, momentum=0.7),
    metrics=["mae", "mse"]
)


callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = full_model.fit(
    [X_train_desc, X_train_fgp],
    RT_train,
    validation_data=([X_val_desc, X_val_fgp], RT_val),
    epochs=500,
    batch_size=16,
    callbacks=callbacks
)


full_model.save("ALL_Asim20.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim20.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim20.png", dpi=300, bbox_inches='tight')
plt.close()



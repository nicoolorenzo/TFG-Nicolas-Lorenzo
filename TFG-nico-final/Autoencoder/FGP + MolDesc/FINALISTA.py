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


####################################################### Predictor 9: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_6_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run4.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])

# Red de prediccion de RT
out_neuron = Dense(1, activation='linear', name="predicted_RT")(merged)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_1.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_1.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_1.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 10: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_6_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run11.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])


out_neuron = Dense(1, activation='linear', name="predicted_RT")(merged)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_2.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_2.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_2.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 11: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_1_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run4.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])


out_neuron = Dense(1, activation='linear', name="predicted_RT")(merged)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_3.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_3.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_3.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 12: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_1_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run11.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])


out_neuron = Dense(1, activation='linear', name="predicted_RT")(merged)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_4.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_4.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_4.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 9: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_6_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run4.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])

# Red de prediccion de RT
x = Dense(526, activation='selu', kernel_initializer=lecun_normal())(merged)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_5.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_5.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_5.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 10: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_6_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run11.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])

# Red de prediccion de RT
x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(merged)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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
# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_6.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_6.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_6.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 11: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_1_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run4.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])

# Red de prediccion de RT
x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(merged)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_7.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_7.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_7.png", dpi=300, bbox_inches='tight')
plt.close()

####################################################### Predictor 12: #######################################################

# Cargar modelo DESCRIPTORES
autoencoder_desc = keras.models.load_model("auto_desc_1_NEW.keras")
# Crear Input explícito
input_desc_dummy = Input(shape=(X_train_desc.shape[1],), name="forced_input_desc")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_desc(input_desc_dummy)
# Crear modelo funcional completo
autoencoder_desc_func = Model(inputs=input_desc_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_desc_func.layers[-2].output
encoder_desc = Model(inputs=autoencoder_desc_func.input, outputs=new_output, name="encoder_model_desc")
encoder_desc.trainable = False


# Cargar modelo FINGERPRINTS
autoencoder_fgp = keras.models.load_model("autoencoder_run11.keras")
# Crear Input explícito
input_fgp_dummy = Input(shape=(X_train_fgp.shape[1],), name="forced_input_fgp")
# Aplicar el modelo secuencial como si fuera una capa
output_dummy = autoencoder_fgp(input_fgp_dummy)
# Crear modelo funcional completo
autoencoder_fgp_func = Model(inputs=input_fgp_dummy, outputs=output_dummy)
# Cortar la última capa
new_output = autoencoder_fgp_func.layers[-2].output
encoder_fgp = Model(inputs=autoencoder_fgp_func.input, outputs=new_output, name="encoder_model_fgp")
encoder_fgp.trainable = False


for layer in encoder_fgp.layers:
    layer._name = layer.name + str("_2")

#Modelo conjunto
input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")

# Embeddings
encoded_desc = encoder_desc(input_desc)
encoded_fgp  = encoder_fgp(input_fgp)

# Salida concatenada de ambos encoders
merged = Concatenate(name="concat_latents")([encoded_desc, encoded_fgp])

# Red de prediccion de RT
x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(merged)
out_neuron = Dense(1, activation='linear', name="predicted_RT")(x)

# Modelo completo
full_model = Model(inputs=[input_desc, input_fgp], outputs=out_neuron, name="LinearNeuronFromEncoders")

# Compilación
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

# ================================
# 7. GUARDADO Y VISUALIZACIÓN
# ================================
full_model.save("predALL_8.keras")

# Guardar historial
hist_df = pd.DataFrame(history.history)
hist_df.to_csv("predALL_8.csv", index=False)

# Graficar evolución
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("predALL_8.png", dpi=300, bbox_inches='tight')
plt.close()

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
#######################################################  ASIM 5: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(386, activation='selu', kernel_initializer=lecun_normal())(merged)
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


full_model.save("ALL_Asim6.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim6.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim6.png", dpi=300, bbox_inches='tight')
plt.close()
#######################################################  ASIM 5: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(403, activation='selu', kernel_initializer=lecun_normal())(merged)
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


full_model.save("ALL_Asim7.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim7.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim7.png", dpi=300, bbox_inches='tight')
plt.close()
#######################################################  ASIM 8: #######################################################

input_desc = Input(shape=(X_train_desc.shape[1],), name="input_descriptors")
input_fgp  = Input(shape=(X_train_fgp.shape[1],), name="input_fingerprints")


merged = Concatenate(name="concat_raw_inputs")([input_desc, input_fgp])


x = Dense(263, activation='selu', kernel_initializer=lecun_normal())(merged)
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


full_model.save("ALL_Asim8.keras")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("ALL_Asim8.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.title("Training History")
plt.legend()
plt.grid(True)
plt.savefig("ALL_Asim8.png", dpi=300, bbox_inches='tight')
plt.close()

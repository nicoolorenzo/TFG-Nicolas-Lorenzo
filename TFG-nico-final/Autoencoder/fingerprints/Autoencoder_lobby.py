#AUTOENCODER 13: Todas duplicadas + Lecun + selu

# Encoder
stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[2212]),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(140, kernel_initializer='lecun_normal', activation="selu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu", input_shape=[140]),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
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
stacked_encoder.save("autoencoder_run13.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics13.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics13.csv', index=False)

#AUTOENCODER 14: Todas triplicadas + Lecun + selu

# Encoder
stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[2212]),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(140, kernel_initializer='lecun_normal', activation="selu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu", input_shape=[140]),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
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
stacked_encoder.save("autoencoder_run14.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics14.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics14.csv', index=False)

#AUTOENCODER 15: Truncado + Lecun + Selu (Todas duplicadas)

stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[2212]),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(140, kernel_initializer='lecun_normal', activation="selu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[140]),
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
#stacked_ae.build(input_shape=(None, 2212))
print(stacked_encoder.summary())
print(stacked_decoder.summary())

# Entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)
history = stacked_ae.fit(X_train, X_train, epochs=500, validation_data=(X_val, X_val), callbacks=[callback, reduce_lr])
stacked_encoder.save("autoencoder_run15.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics15.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics15.csv', index=False)

#SAUTOENCODER 16: Truncado + Lecun + Selu (Todas triplicadas)

stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[2212]),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(1100, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(550, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(280, kernel_initializer='lecun_normal', activation="selu"),
    keras.layers.Dense(140, kernel_initializer='lecun_normal', activation="selu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu", input_shape=[140]),
    keras.layers.Dense(2212, kernel_initializer='lecun_normal', activation="selu"),
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
#stacked_ae.build(input_shape=(None, 2212))
print(stacked_encoder.summary())
print(stacked_decoder.summary())

# Entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)
history = stacked_ae.fit(X_train, X_train, epochs=500, validation_data=(X_val, X_val), callbacks=[callback, reduce_lr])
stacked_encoder.save("autoencoder_run16.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics16.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics16.csv', index=False)

#AUTOENCODER 17: Estándar con "he_normal" y relu

# Encoder
stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='he_normal', activation="relu", input_shape=[2212]),
    keras.layers.Dense(1100, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(550, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(280, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(140, kernel_initializer='he_normal', activation="relu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(280, kernel_initializer='he_normal', activation="relu", input_shape=[140]),
    keras.layers.Dense(550, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(1100, kernel_initializer='he_normal', activation="relu"),
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
stacked_encoder.save("autoencoder_run17.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics17.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics17.csv', index=False)

#AUTOENCODER 18: Truncado + "he_normal" + relu
stacked_encoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='he_normal', activation="relu", input_shape=[2212]),
    keras.layers.Dense(1100, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(550, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(280, kernel_initializer='he_normal', activation="relu"),
    keras.layers.Dense(140, kernel_initializer='he_normal', activation="relu")
])

# Decoder
stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(2212, kernel_initializer='he_normal', activation="relu", input_shape=[140]),
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
#stacked_ae.build(input_shape=(None, 2212))
print(stacked_encoder.summary())
print(stacked_decoder.summary())

# Entrenamiento
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',  factor=0.5,  patience=3, verbose=1)
history = stacked_ae.fit(X_train, X_train, epochs=500, validation_data=(X_val, X_val), callbacks=[callback, reduce_lr])
stacked_encoder.save("autoencoder_run18.keras")

#Gráficas
df = pd.DataFrame(history.history)
df.plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.125)
plt.savefig("training_metrics18.png", dpi=300, bbox_inches='tight')


#Guardar las métricas como .csv por seguridad
df.to_csv('training_metrics18.csv', index=False)
from tensorflow import keras

# Cargar el modelo
model = keras.models.load_model("pred_fgp9_frozen.keras")

# Mostrar resumen general
print("Resumen del modelo:")
model.summary()

# Detalle capa por capa
print("\nDetalles capa por capa:")
for i, layer in enumerate(model.layers):
    print(f"Capa {i} - Nombre: {layer.name}")
    print(f"  Tipo: {type(layer).__name__}")

    # Ver número de neuronas (si aplica)
    if hasattr(layer, "units"):
        print(f"  Neuronas (units): {layer.units}")

    # Función de activación
    if hasattr(layer, "activation"):
        print(f"  Activación: {layer.activation.__name__}")

    # Inicializador de pesos (kernel)
    if hasattr(layer, "kernel_initializer"):
        print(f"  Kernel initializer: {layer.kernel_initializer.__class__.__name__}")

    print("-" * 50)




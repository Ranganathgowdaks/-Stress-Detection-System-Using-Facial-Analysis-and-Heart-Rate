from tensorflow.keras.models import load_model

def reset_weights(model):
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

# Load the model you want to reset
model = load_model("src/models/saved_model/mobilenet_model.h5")

# Reset the weights
reset_weights(model)

# Save the reset model if you want to reuse it later
model.save("src/models/saved_model/mobilenet_model_untrained.h5")
print("Model has been reset and saved as 'mobilenet_model_untrained.h5'")

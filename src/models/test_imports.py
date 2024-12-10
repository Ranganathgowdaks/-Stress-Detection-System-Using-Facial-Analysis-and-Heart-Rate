import tensorflow as tf

print(tf.__version__)  # Check if TensorFlow is correctly imported

# Test Keras imports
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

print("Keras imports successful!")

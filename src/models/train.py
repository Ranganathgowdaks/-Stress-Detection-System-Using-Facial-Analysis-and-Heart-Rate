from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Directory paths for the dataset
base_dir = 'src/data'  # Base directory containing 'stress' and 'no_stress' folders

# Image data generator with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use flow_from_directory to read images from the base_dir which contains subfolders 'stress' and 'no_stress'
train_generator = train_datagen.flow_from_directory(
    base_dir,  # Base directory containing 'stress' and 'no_stress' folders
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    shuffle=True
)

# Load MobileNetV2 model pre-trained on ImageNet and fine-tune
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers for initial training

# Add custom layers for stress classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)  # Two classes: stress and no stress

model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=15)

# Save the fine-tuned model
model_save_path = "src/models/saved_model/mobilenet_finetuned.h5"
model.save(model_save_path)
print(f"Model has been successfully trained and saved to '{model_save_path}'")

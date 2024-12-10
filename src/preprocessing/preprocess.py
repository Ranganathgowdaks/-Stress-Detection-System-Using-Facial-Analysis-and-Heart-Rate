# /src/preprocessing/preprocess.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, batch_size=32, target_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_gen, val_gen

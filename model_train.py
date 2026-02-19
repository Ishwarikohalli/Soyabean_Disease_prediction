import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- Configuration ---
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
MODEL_SAVE_PATH = 'model.h5'
CLASS_NAMES_PATH = 'class_names.json'

# --- Data Preprocessing (Grayscale) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'   # ✅ IMPORTANT
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale'   # ✅ IMPORTANT
)

num_classes = len(train_generator.class_indices)
print("Class indices:", train_generator.class_indices)

# Save class names
class_names = list(train_generator.class_indices.keys())
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# --- CNN Model (Input shape changed to 1 channel) ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),  # ✅ changed
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator
)

model.save(MODEL_SAVE_PATH)
print("Grayscale Model saved successfully!")

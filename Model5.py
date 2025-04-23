import kagglehub
# Download latest version
path = kagglehub.dataset_download("riyaelizashaju/isic-skin-disease-image-dataset-labelled")
print("Path to dataset files:", path)

import os
import json
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define the dataset path
dataset_path = r"C:\Users\dubey\.cache\kagglehub\datasets\riyaelizashaju\isic-skin-disease-image-dataset-labelled\versions\1"

if os.path.exists(dataset_path):
    print(os.listdir(dataset_path))
else:
    print("Dataset path does not exist.")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Define path to images
image_folder_path = os.path.join(dataset_path, 'ISIC_Labelled')

# List all subdirectories (i.e., classes) in the dataset
classes = os.listdir(image_folder_path)
print("Classes found in dataset:", classes)

# Visualize a sample image from one of the classes
sample_class = classes[0]  # Select the first class for example
sample_image_path = os.path.join(image_folder_path, sample_class, os.listdir(os.path.join(image_folder_path, sample_class))[0])  # Get the first image in that class

# Load and display the image
img = mpimg.imread(sample_image_path)
plt.imshow(img)
plt.axis('off')  # Turn off axis numbers
plt.show()

import splitfolders

splitfolders.ratio(
    r"C:\Users\dubey\.cache\kagglehub\datasets\riyaelizashaju\isic-skin-disease-image-dataset-labelled\versions\1/ISIC_Labelled",
    output=r"C:\Users\dubey\ISIC_dataset",
    seed=42,
    ratio=(0.7, 0.15, 0.15),  # 70% Train, 15% Validation, 15% Test
    group_prefix=None
)

base_dir=r'C:\Users\dubey\.cache\kagglehub\datasets\riyaelizashaju\isic-skin-disease-image-dataset-labelled\versions\1/ISIC_Labelled'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator for Train & Validation (With Augmentation for Train)
train_val_gen = ImageDataGenerator(rescale=1./255)

test_gen = ImageDataGenerator(rescale=1./255)  # No data augmentation for test

# Define dataset directories
train_dir = r"C:\Users\dubey\ISIC_dataset/train"
val_dir = r"C:\Users\dubey\ISIC_dataset/val"
test_dir = r"C:\Users\dubey\ISIC_dataset/test"

# Train Generator
train_generator = train_val_gen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,

    class_mode='categorical'
)

# Validation Generator
validation_generator = train_val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,

    class_mode='categorical'
)

# Test Generator (No Shuffling)
test_generator = test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,

    class_mode='categorical',
    shuffle=False
)

from keras.layers import Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.applications.mobilenet import MobileNet

# Load Pretrained MobileNet Model
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Custom Classification Head
X = Flatten()(base_model.output)
X = Dense(256, activation='relu')(X)
X = BatchNormalization()(X)  # Normalize activations
X = Dropout(0.5)(X)  # Prevent overfitting
X = Dense(128, activation='relu')(X)
X = BatchNormalization()(X)
X = Dropout(0.5)(X)
X = Dense(8, activation='softmax')(X)  # Output Layer (73 Classes)

# Define New Model
model = Model(inputs=base_model.input, outputs=X)

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print Model Summary
model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Callbacks for Training
mc = ModelCheckpoint(
    monitor='val_accuracy',
    filepath='ISIC_deseases.keras',
    verbose=1,
    save_best_only=True,
    mode='max'  # Ensure highest accuracy is saved
)

es = EarlyStopping(
    monitor='val_loss',  # Can also monitor 'val_loss' to prevent overfitting
    min_delta=0.01,
    patience=3,
    verbose=1,
    mode='auto'
)

cb = [es, mc]

# Calculate steps per epoch dynamically
steps_per_epoch = train_generator.samples // 32
validation_steps = validation_generator.samples // 32

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=100,  # Can fine-tune based on model's performance
    verbose=1,
    callbacks=cb,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Model Evaluation
print("Evaluating model...")
val_loss, val_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // 32)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on the Test Data
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // 32)
print(f"Test Accuracy: {test_accuracy}")

model.save('ISIC_deseases1.keras', save_format="keras")

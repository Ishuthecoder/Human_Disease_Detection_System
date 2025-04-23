import os
import random
import shutil
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Optional: Mixed precision for memory-efficient training
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

# Set base dataset path (as you mentioned)
base_path = "C:/Users/dubey/.cache/kagglehub/datasets/shuvokumarbasak2030/medical-imaging-bone-fracture-colorized-img-data/versions/1"

# Constants
IMG_SIZE = 224
BATCH_SIZE = 16  # Lowered batch size for GPU safety
EPOCHS = 30

# Sub-directory paths from the base path
train_dir = os.path.join("fracture_dataset_split", "train")
val_dir = os.path.join("fracture_dataset_split", "val")
test_dir = os.path.join("fracture_dataset_split", "test")

# Data generators
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.1, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')

# Load MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_data.num_classes, activation='softmax', dtype='float32')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile initial model
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Unfreeze last 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile for fine-tuning
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

# Save training history
history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
df = pd.DataFrame(history_dict)
df.to_csv('training_history.csv', index=False)

# Evaluate model
loss, accuracy = model.evaluate(test_data)
print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

# Save final model in .keras format
model.save("Completed_project2.keras", save_format="keras")

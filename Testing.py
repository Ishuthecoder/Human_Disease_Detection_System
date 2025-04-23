from keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load your unified model
model = load_model("Unified_Five_Models.keras")
print("✅ Model loaded successfully!")

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Define the class labels according to your final model output
class_labels = ["Cancer", "COVID", "Skin Disease","Bone Fracture","Normal"]  # 🔁 Update these if needed

# Path to test image
test_image_path = r"C:\Users\dubey\Downloads\test.webp"  # 🔁 Replace with your actual image path

# Check if file exists
if not os.path.exists(test_image_path):
    print(f"❌ File not found: {test_image_path}")
else:
    # Preprocess and predict
    img = preprocess_image(test_image_path)
    prediction = model.predict(img)
    
    # Show results
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]

    print("✅ Prediction successful!")
    print("Raw output:", prediction)
    print(f"Predicted class index: {predicted_index}")
    print(f"🧠 Predicted Disease Label: {predicted_label}")

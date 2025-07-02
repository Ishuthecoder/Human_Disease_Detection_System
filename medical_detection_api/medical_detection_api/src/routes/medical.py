from flask import Blueprint, request, jsonify
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import base64
import json
import os
from scipy.ndimage import zoom
from matplotlib.colors import LinearSegmentedColormap
import ollama # Import the ollama library

medical_bp = Blueprint("medical", __name__)

# User-friendly model names
MODEL_DISPLAY_NAMES = {
    "Multi_Cancer_Model1": "Cancer Detection Model",
    "Covid_Model_Infection": "COVID Infection Detection",
    "Covid_Model_Segmentation": "COVID Segmentation Model",
    "ISIC_deseases": "Skin Disease Detection (ISIC)",
    "Completed_project": "Bone Fracture Detection"
}

# Prediction labels for each model
PREDICTION_LABELS = {
    "Multi_Cancer_Model1": [
        "Benign", "all_early", "all_pre", "all_pro", "Brain Glioma", "Brain_Menin", "Brain Tumor",
        "Breast Benign", "Breast Maligant", "cervical dysplasia", "Squamous cell carcinoma", "Cervical MEP",
        "Cervix Pain", "Cervix Cancer", "Cervix Insufficiency", "Kidney Tumor", "Normal Kidney", "Colon ACA",
        "Colon BNT", "Lung ACA", "Lung SSC", "Lung BNT", "LYMPH CLL", "Lymph fluid", "LYMPH MCL",
        "Normal Oral", "Oral SSC"
    ],
    "Covid_Model_Infection": ["COVID Positive", "COVID Negative", "Normal"],
    "Covid_Model_Segmentation": ["COVID Positive", "COVID Negative", "Normal"],
    "ISIC_deseases": [
        "Actinic keratosis", "Basal cell carcinoma", "Benign keratosis", "Dermatofibroma",
        "Melanocytic Nevus", "Melanoma", "Vascular Lesion", "Normal", "Skin Disease"
    ],
    "Completed_project": [
        "Avulsion fracture", "Comminuted fracture", "Fracture Dislocation", "Greenstick fracture",
        "Hairline Fracture", "Impacted fracture", "Longitudinal fractur", "Oblique fracture",
        "Pathological fracture", "Spiral Fracture", "Pelvis Fracture", "Non_Fractured"
    ]
}

def fake_predict(image, model_name):
    """Enhanced prediction function with visualization data"""
    labels = PREDICTION_LABELS[model_name]
    
    # Main prediction
    pred_class = random.choice(labels)
    confidence = round(random.uniform(80, 99), 2)
    
    # Generate additional related predictions for visualization
    all_classes = labels
    confidences = {}
    confidences[pred_class] = confidence
    
    # Generate realistic confidence scores for other classes
    remaining_classes = [c for c in all_classes if c != pred_class]
    
    # Select 4 more classes for display (or fewer if not enough remaining)
    display_classes = random.sample(remaining_classes, min(4, len(remaining_classes)))
    
    # Assign decreasing confidences to other classes, ensuring they sum to approximately 100
    remaining_confidence = 100 - confidence
    for i, cls in enumerate(display_classes):
        if i < len(display_classes) - 1:
            cls_conf = round(random.uniform(1, remaining_confidence/(len(display_classes)-i)), 2)
            confidences[cls] = cls_conf
            remaining_confidence -= cls_conf
        else:
            # Last class gets the remainder
            confidences[cls] = round(remaining_confidence, 2)
    
    # Generate a fake heatmap data for visualization
    heatmap_data = np.random.rand(10, 10)
    # Make it more concentrated in certain areas to simulate detection regions
    x, y = np.random.randint(0, 10, 2)
    heatmap_data[max(0, x-2):min(10, x+3), max(0, y-2):min(10, y+3)] += 1.5
    heatmap_data = heatmap_data / np.max(heatmap_data)  # Normalize
    
    # Generate random ROI boxes for visualization
    num_rois = 1 if confidence > 90 else 2
    rois = []
    for _ in range(num_rois):
        x = random.uniform(0.1, 0.7)
        y = random.uniform(0.1, 0.7)
        width = random.uniform(0.1, 0.3)
        height = random.uniform(0.1, 0.3)
        rois.append((x, y, width, height))
    
    return pred_class, confidence, confidences, heatmap_data, rois

def create_overlay_image(image, heatmap_data):
    """Create heatmap overlay on the image and return as base64"""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Resize heatmap to match image dimensions
    zoom_factors = (img_array.shape[0]/heatmap_data.shape[0], img_array.shape[1]/heatmap_data.shape[1])
    heatmap_resized = zoom(heatmap_data, zoom_factors, order=1)
    
    # Create colormap (red for hotspots)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [(0, 0, 0, 0), (1, 0, 0, 0.7)])
    
    # Create figure with subplots
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    heatmap = ax.imshow(heatmap_resized, cmap=cmap, alpha=0.6)
    ax.axis("off")
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64

def create_roi_image(image, rois):
    """Create ROI visualization on the image and return as base64"""
    img_array = np.array(image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)
    
    height, width = img_array.shape[0], img_array.shape[1]
    
    for x, y, w, h in rois:
        # Convert normalized coordinates to pixel coordinates
        x_px, y_px = x * width, y * height
        w_px, h_px = w * width, h * height
        
        # Create a rectangle patch
        rect = patches.Rectangle((x_px, y_px), w_px, h_px, 
                                 linewidth=3, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
    
    ax.axis("off")
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    
    # Convert to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64

def load_history():
    """Load history from JSON file"""
    history_file = "history.json"
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    """Save history to JSON file"""
    history_file = "history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

@medical_bp.route("/models", methods=["GET"])
def get_models():
    """Get available models"""
    return jsonify({
        "models": [
            {"id": model_id, "name": display_name}
            for model_id, display_name in MODEL_DISPLAY_NAMES.items()
        ]
    })

@medical_bp.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        if "model" not in request.form:
            return jsonify({"error": "No model specified"}), 400
        
        image_file = request.files["image"]
        model_name = request.form["model"]
        
        if model_name not in PREDICTION_LABELS:
            return jsonify({"error": "Invalid model name"}), 400
        
        # Open and process the image
        image = Image.open(image_file.stream)
        
        # Run prediction
        pred_class, confidence, all_confidences, heatmap_data, rois = fake_predict(image, model_name)
        
        # Generate visualizations
        heatmap_image = create_overlay_image(image, heatmap_data)
        roi_image = create_roi_image(image, rois)
        
        # Convert original image to base64 for frontend display
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        original_image_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        
        result = {
            "prediction": pred_class,
            "confidence": confidence,
            "all_confidences": all_confidences,
            "original_image": original_image_base64,
            "heatmap_image": heatmap_image,
            "roi_image": roi_image,
            "model_name": model_name,
            "model_display_name": MODEL_DISPLAY_NAMES[model_name]
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@medical_bp.route("/chat", methods=["POST"])
def chat():
    prompt = "" # Initialize prompt to an empty string
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        latest_prediction = data.get("latest_prediction", None) # Assuming frontend sends this

        # Construct the prompt based on whether there's a recent prediction
        if latest_prediction:
            # Use .get() with a default value to prevent KeyError if a key is missing
            selected_model_internal = latest_prediction.get("model_name", "Unknown Model")
            pred_class = latest_prediction.get("prediction", "Unknown Prediction")
            confidence = latest_prediction.get("confidence", 0)

            prompt = f"""
You are a trusted and intelligent virtual medical assistant designed to help patients understand their health conditions—both predicted by the system and those described by the user. You provide well-structured, informative, and empathetic responses to guide patients about their health concerns.

**IMPORTANT: Format your entire response using Markdown. Use clear headings (e.g., ###), bullet points (*), and bold text (**) where appropriate to enhance readability. Ensure each section is clearly delineated.**

Your task includes two parts:

---

**Part 1: Prediction-Based Assistance**  
Use the following model output to explain the possible condition, symptoms, and treatment options.

Model: {selected_model_internal}  
Prediction: {pred_class}  
Confidence: {confidence}%

For this prediction, provide a structured and patient-friendly explanation with these sections:
1. **Understanding the Condition** – Briefly describe the predicted disease/condition.
2. **Common Symptoms** – List symptoms typically associated with it.
3. **Preventive Measures** – Suggest general preventive steps and lifestyle tips.
4. **Treatment and Management** – Outline common medical approaches to manage or treat the condition.
5. **What to Do Next** – Recommend what type of doctor or test the patient might consider and encourage consulting a healthcare provider.

---

**Part 2: General Health Concern Assistance**  
If the user describes a medical concern or asks a question **not related to the prediction**, such as stomach gas, headache, weakness, skin issues, etc., provide suggestions in the same structure:

1. **Possible Causes** – List likely general causes of the user’s concern.
2. **Common Symptoms (if applicable)** – Describe any signs to watch for.
3. **Home Remedies or Lifestyle Advice** – Share safe, evidence-based suggestions the user may try at home.
4. **Medical Treatments** – Briefly explain standard treatments or medications used (without prescribing).
5. **When to Seek Medical Help** – Tell the patient when it’s important to consult a doctor and what kind of specialist they should consider.

Always conclude with a polite reminder that while you're providing medical guidance, in-person consultation with a licensed healthcare provider is essential for accurate diagnosis and treatment.
"""
        else:
            # If no prediction, just use the user's message as the prompt for general assistance
            prompt = f"""
You are a trusted and intelligent virtual medical assistant designed to help patients understand their health concerns. You provide well-structured, informative, and empathetic responses to guide patients about their health concerns.

**IMPORTANT: Format your entire response using Markdown. Use clear headings (e.g., ###), bullet points (*), and bold text (**) where appropriate to enhance readability. Ensure each section is clearly delineated.**

Based on the user's question: "{user_message}", provide suggestions with these sections:

1. **Possible Causes** – List likely general causes of the user’s concern.
2. **Common Symptoms** – List symptoms typically associated with it.
3. **Home Remedies or Lifestyle Advice** – Share safe, evidence-based suggestions the user may try at home.
4. **Medical Treatments** – Briefly explain standard treatments or medications used (without prescribing).
5. **When to Seek Medical Help** – Tell the patient when it’s important to consult a doctor and what kind of specialist they should consider.

Always conclude with a polite reminder that while you're providing medical guidance, in-person consultation with a licensed healthcare provider is essential for accurate diagnosis and treatment.
"""

        # Call Ollama
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return jsonify({
            "response": response["message"]["content"],
            "status": "success"
        })

    except Exception as e:
        print(f"Error calling Ollama: {e}") # This will now print the actual error
        # Fallback to placeholder if Ollama fails (e.g., not running, model not found)
        placeholder_response = f"""Thank you for your question: "{user_message}"

I'm currently experiencing technical difficulties and cannot connect to the AI model. In a full implementation, I would provide detailed medical guidance based on your question and any recent diagnosis results.

For now, I recommend:
1. Consult with a healthcare professional for personalized advice
2. Follow up on any concerning symptoms
3. Maintain a healthy lifestyle with proper diet and exercise

Please note: This is not a substitute for professional medical advice."""
        
        return jsonify({
            "response": placeholder_response,
            "status": "fallback",
            "error": str(e) # Include the error in the response for debugging
        })

@medical_bp.route("/history", methods=["GET"])
def get_history():
    """Get prediction history"""
    try:
        history = load_history()
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@medical_bp.route("/history", methods=["POST"])
def add_to_history():
    """Add prediction to history"""
    try:
        data = request.get_json()
        
        # Load existing history
        history = load_history()
        
        # Add new entry
        history.append({
            "timestamp": data.get("timestamp"),
            "filename": data.get("filename"),
            "model": data.get("model"),
            "prediction": data.get("prediction"),
            "confidence": data.get("confidence")
        })
        
        # Save updated history
        save_history(history)
        
        return jsonify({"status": "success"})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
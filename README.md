
##  Project Description

The **Human Disease Detection System** is an AI-powered diagnostic platform that enables users to upload medical images for disease prediction and receive intelligent health recommendations. This system integrates deep learning models and natural language processing to provide a seamless and interactive healthcare experience.


##  Key Functionalities

1. **Medical Image-Based Disease Detection**
   - Users can upload various medical images (e.g., chest X-rays, skin scans, fracture images).
   - The system identifies potential diseases by processing the image through deep learning models.
   - Outputs include:
     - **Predicted disease name**
     - **Confidence score (%)**
     - **Region of Interest (ROI)** highlighting the affected area (if applicable)

2. **AI-Powered Health Query Interface**
   - After a prediction, users can interact with an integrated AI assistant (BioGPT or LLaMA-based).
   - The assistant can answer:
     - Follow-up questions about the diagnosed disease
     - General health-related queries
     - Advice on symptoms, prevention, and treatments

3. **General Health Assistance**
   - Users can ask open-ended medical questions, even without uploading an image.
   - Ideal for learning about diseases, medications, or lifestyle recommendations.

4. **Personalized History Dashboard**
   - Each user session is tracked to maintain:
     - A record of uploaded images
     - Disease prediction results
     - Past health-related conversations with the AI
   - Enables users to monitor their health over time

###  Behind the Scenes

- **Model Inference** is handled via `.h5` files loaded using TensorFlow/Keras.
- **Web Interface** is built with Flask and HTML templates.
- **Prediction Logic** is managed by `Mergerd_Model.py` and `ModelX.py` files.
- **Chat Interface** is powered by a local or cloud LLM that handles medical Q&A.
- **Data Persistence** is managed through flat files or local databases for history tracking.



##  Technologies Used

The Human Disease Detection System is built using a combination of modern AI frameworks, web development tools, and language models to deliver real-time disease prediction and intelligent health assistance.

| Technology / Tool         | Description                                                                                  |
|--------------------------|----------------------------------------------------------------------------------------------|
| **Python 3.10+**          | Core programming language used to develop backend logic, AI pipelines, and utility scripts. |
| **TensorFlow / Keras**    | Deep learning frameworks used for training and loading disease classification models.        |
| **Flask**                 | Lightweight web framework for building the backend API, routing, and serving the web app.   |
| **HTML / CSS / Jinja2**   | Frontend templating for rendering the web interface and integrating dynamic data.            |
| **OpenCV**                | Used for image preprocessing and Region of Interest (ROI) extraction.                        |
| **NumPy & Pandas**        | Essential libraries for data manipulation and numerical operations.                          |
| **Matplotlib / Seaborn**  | Visualization libraries (used optionally for debugging or rendering ROIs/statistics).        |
| **Pre-trained LLMs (BioGPT / LLaMA)** | Used to provide interactive, text-based medical recommendations and health advice. |
| **SQLite / JSON Storage** | Lightweight database or flat file format for storing user history and prediction logs.       |
| **Git & GitHub**          | Version control and collaborative development platform.                                      |
| **VS Code**               | Preferred IDE for development, debugging, and testing.                                       |
| **Ollama**                | Local LLM inference backend for privacy-preserving medical chats without needing the cloud.  |


###  Project Folder Structure

```

Human_Disease_Detection_System/
├── .gitignore # Files and folders to ignore by Git
├── LICENSE # Project license (MIT)
├── README.md # Project documentation
├── Model1.py - Model5.py # Separate disease model loaders/predictors
├── Mergerd_Model.py # Unified prediction script for all models
├── Model_labels.json # Label mapping for disease models
├── Testing.py # Script for running predictions manually
├── mediscan_complete.html # Web template for results display
└── medical_detection_api/
└── medical_detection_api/
├── app.py # Flask web server and routing
├── static/ # CSS, JS, images
├── templates/ # HTML templates for frontend
├── model/ # Pre-trained models (.h5 files)
├── utils/ # Helper functions (e.g., preprocessing, prediction)
└── requirements.txt # Python dependencies

```


##  Setup Instructions

Follow the steps below to set up and run the Human Disease Detection System locally on your machine.

###  Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

###  Step 1: Clone the Repository

```bash
git clone https://github.com/Ishuthecoder/Human_Disease_Detection_System.git

```

###  Step 2: Navigate to the Project Directory

```bash
cd Human_Disease_Detection_System/medical_detection_api/medical_detection_api

```

###  Step 3: Install Required Python Libraries

```bash
pip install -r requirements.txt/pip install -r requirements2.txt

```

###  Step 4: Run the Flask Application

```bash
python app.py

```

###  Step 5: Open the Web App

```bash
http://localhost:5000

```


##  Sample Predictions

| Input Image | Predicted Disease | Confidence |
|-------------|-------------------|------------|
| ![xray]("C:\Pictures\bone_fracture.jpg") | Greenstick | 98.7% |
| ![skin]("C:\Pictures\skin_disease.jpg") | Melanoma | 92.4% |


##  Demo –  Video

A quick walkthrough of the Human Disease Detection System in action.

> 📽️ Watch the system in action – from uploading a disease image to receiving AI-generated health recommendations.

[![Watch Demo]("C:\Users\dubey\OneDrive\Pictures\Screenshots\Screenshot 2025-06-14 225504.png")]("C:\Documents\ezyZip.mp4")

> _Click the thumbnail above or [click here to view the video]("C:\Documents\ezyZip.mp4")

##  Contributing

Contributions are welcome! Please open issues or submit pull requests for:
- Model improvements
- UI enhancements
- Bug fixes

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) before you begin.


##  License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


##  Author

**Ishika Dubey**  
GitHub: [@Ishuthecoder](https://github.com/Ishuthecoder)  


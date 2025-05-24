# Plant Disease Detection System

A comprehensive deep learning system for identifying plant diseases from leaf images, providing treatment recommendations, and helping farmers make informed decisions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project uses a custom Convolutional Neural Network (CNN) to identify plant diseases from images of plant leaves. The system provides both chemical and natural treatment recommendations based on the detected disease.

The application consists of a FastAPI backend for prediction serving and a Streamlit frontend for user interaction. The model is trained on a dataset of berry plant disease images.

## Features

- **Disease Detection**: Identify 7 different plant diseases from leaf images
- **Treatment Recommendations**: Get both chemical and organic treatment options
- **User-friendly Interface**: Easy-to-use web interface with image upload capability
- **Confidence Scoring**: View the confidence level of predictions
- **Preventive Measures**: Access tips to prevent future disease spread

## System Architecture

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │
│   Streamlit   │ --> │    FastAPI    │ --> │  Deep Learning│
│  Frontend UI  │     │   Backend     │     │     Model     │
│               │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
```

- **Frontend**: Streamlit web interface for user interaction
- **Backend**: FastAPI service for image processing and model inference
- **Model**: Custom CNN architecture trained on plant disease dataset

## Installation

### Prerequisites

- Python 3.11 or higher
- Git (optional, for cloning the repository)

### Setup

1. **Clone the repository (optional):**
   ```bash
   git clone <repository-url>
   cd plant-disease-detection
   ```

2. **Create a virtual environment:**
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Or using conda
   conda create -n plant-disease python=3.11
   conda activate plant-disease
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the model:**
   Ensure the `strawbery_disease_model.pth` file is in the project root directory. If not available, you can train the model using the training notebook.

## Usage

### Running the Application

1. **Start the backend server:**
   ```bash
   # Terminal 1
   uvicorn main:app --reload --port 8080
   ```

2. **Start the frontend application:**
   ```bash
   # Terminal 2
   streamlit run streamlit_app.py
   ```

3. **Access the application:**
   Open your web browser and go to [http://localhost:8501](http://localhost:8501)

### Using the Web Interface

1. Upload a clear image of a plant leaf
2. Click "Analyze Image"
3. View the prediction results, confidence score, and treatment recommendations

## Project Structure

```
plant-disease-detection/
├── augment.py              # Image augmentation script
├── berry/                  # Dataset directory
│   ├── Train/              # Training images
│   └── Test/               # Testing images
├── main.py                 # FastAPI backend service
├── streamlit_app.py        # Streamlit frontend interface
├── train.ipynb             # Model training notebook
├── strawbery_disease_model.pth  # Trained model weights
└── requirements.txt        # Project dependencies
```

## Model Information

### Architecture

The model uses a custom CNN architecture with:
- 6 convolutional layers with batch normalization
- Max pooling after each convolutional layer
- 4 fully connected layers with dropout for regularization

### Disease Classes

The model can identify the following diseases:
1. Angular Leaf Spot (Bacterial)
2. Anthracnose Fruit Rot (Fungal)
3. Blossom Blight (Fungal)
4. Gray Mold (Fungal)
5. Leaf Spot (Fungal)
6. Powdery Mildew Fruit (Fungal)
7. Powdery Mildew Leaf (Fungal)

## API Documentation

### FastAPI Endpoints

#### `/predict/`
- **Method**: POST
- **Description**: Analyzes an uploaded leaf image and returns disease prediction
- **Parameters**: Image file (multipart/form-data)
- **Returns**: JSON object with prediction details
  ```json
  {
    "predicted_class": "powdery_mildew_leaf",
    "confidence": 0.95,
    "chemical_solution": "Apply triadimefon or wettable sulfur.",
    "natural_solution": "Spray with baking soda + vegetable oil + water mix."
  }
  ```

## Training a New Model

To train your own model:
1. Prepare your dataset in the `berry/` directory
2. Open and run `train.ipynb` in Jupyter Notebook or JupyterLab
3. Adjust hyperparameters as needed
4. The trained model will be saved as `strawbery_disease_model.pth`

## Data Augmentation

To augment your dataset for improved model performance:
1. Update input and output paths in `augment.py`
2. Run the script:
   ```bash
   python augment.py
   ```
3. The augmented dataset will be created in the specified output directory

## Troubleshooting

### Common Issues

#### FastAPI Server Won't Start
- Check if port 8080 is already in use
- Verify that all dependencies are installed correctly
- Ensure the model file exists in the correct location

#### Streamlit Connection Error
- Confirm that the FastAPI server is running on port 8080
- Check the API_URL in streamlit_app.py matches your FastAPI server address

#### Model Loading Error
- Verify that the model file path is correct
- Ensure that the model architecture in main.py matches the one used during training

#### Image Processing Issues
- Make sure uploaded images are in supported format (JPG, PNG)
- Check that the image preprocessing in main.py matches the preprocessing used during training


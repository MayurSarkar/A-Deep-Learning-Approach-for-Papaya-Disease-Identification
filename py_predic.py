import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Load the trained model for disease detection
disease_model = load_model('c1plant_disease_model.weights(1).h5')

# Define disease labels
disease_labels = {
    0: "Anthracnose",
    1: "Black Spot",
    2: "Phytophthora",
    3: "Powdery Mildew",
    4: "Ring Spot"
}

# Load the trained model for freshness detection
freshness_model = load_model('c3plant_disease_model.weights.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch
    img_array /= 255.0  # Normalize image
    return img_array

# Function to classify the freshness of papayas
def classify_freshness(image_path):
    try:
        img = image.load_img(image_path, target_size=(200, 200, 3))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        val = freshness_model.predict(images)
        if val == 1:
            return "Fresh Papaya"
        else:
            return "Disease Affected Papaya"
    except Exception as e:
        return str(e)

# Endpoint to classify images
@app.route('/classify', methods=['GET', 'POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        image_path = 'temp_image.jpg'  # Save the image temporarily
        file.save(image_path)

        # Check freshness of papaya
        freshness_result = classify_freshness(image_path)
        if freshness_result == "Fresh Papaya":
            return jsonify({'prediction': freshness_result})

        # If papaya is not fresh, classify disease
        processed_image = preprocess_image(image_path)
        predictions = disease_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_disease = disease_labels[predicted_class_index]
        return jsonify({'prediction': predicted_disease})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

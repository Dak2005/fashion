from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "fashion_recommender.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = preprocess_input(img)  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Route for home
@app.route('/')
def home():
    return render_template('index.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    # Preprocess and predict
    input_image = preprocess_image(file_path)
    prediction = model.predict(input_image)
    predicted_class = np.argmax(prediction)

    return jsonify({"predicted_class": int(predicted_class)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

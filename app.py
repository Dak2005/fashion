from flask import Flask, request, render_template
import os

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400  # Return error if no file is found

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400  # Return error if file is not selected

    # Save file for debugging
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)
    
    return f"File received: {file.filename}", 200  # Temporary response

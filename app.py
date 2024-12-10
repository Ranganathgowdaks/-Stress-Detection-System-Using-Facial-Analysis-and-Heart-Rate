from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # Only needed during development when using different ports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time
import logging

# Initialize Flask app and configure static folder for React frontend
app = Flask(__name__, static_folder='build', static_url_path='')


# Enable CORS (optional for development with React on a different port)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the fine-tuned model for stress detection
MODEL_PATH = "src/models/saved_model/mobilenet_finetuned.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
else:
    logging.error(f"Model not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Serve the React frontend (index.html)
@app.route('/')
def serve_react_app():
    return send_from_directory(app.static_folder, 'index.html')

# Serve static files (e.g., JS, CSS, images)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Prepare the image for model prediction
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize pixel values
    return img_array

# Classify heart rate into stress levels
def classify_heart_rate(heart_rate):
    if heart_rate < 80:
        return "no stress"
    elif 80 <= heart_rate < 100:
        return "happy/excited"
    elif 100 <= heart_rate < 120:
        return "mild stress"
    else:
        return "high stress"

# Predict endpoint for stress detection
@app.route('/predict', methods=['POST'])
def predict():
    # Check if both image and heart rate are provided
    if 'file' not in request.files or 'heart_rate' not in request.form:
        return jsonify({'error': 'Image and heart rate are required'}), 400

    # Process the uploaded file
    file = request.files['file']
    filename = f"{int(time.time())}_{file.filename}"
    img_path = os.path.join("uploads", filename)
    file.save(img_path)

    # Process the heart rate input
    try:
        heart_rate = int(request.form['heart_rate'])
    except ValueError:
        return jsonify({'error': 'Invalid heart rate value'}), 400

    # Prepare the image for prediction
    img = prepare_image(img_path)
    predictions = model.predict(img)

    # Cleanup - delete the uploaded file after prediction
    os.remove(img_path)

    # Interpret predictions from the model
    stress_prob = float(predictions[0][1])  # Assuming index 1 is the 'stress' class
    image_stress_level = "stress" if stress_prob > 0.5 else "no stress"
    image_confidence = round(stress_prob * 100, 2)  # Convert to percentage

    # Classify based on heart rate
    heart_rate_category = classify_heart_rate(heart_rate)

    # Assign numerical values to the heart rate categories
    heart_rate_values = {
        "no stress": 0,
        "happy/excited": 30,
        "mild stress": 60,
        "high stress": 90
    }

    # Get heart rate percentage
    heart_rate_percentage = heart_rate_values.get(heart_rate_category, 0)

    # Calculate the final stress percentage as the average of image confidence and heart rate percentage
    final_stress_percentage = round((image_confidence + heart_rate_percentage) / 2, 2)

    # Combine the results
    final_stress_level = "stress" if final_stress_percentage > 50 else "no stress"

    # Prepare the result with image category
    result = {
        'final_stress_level': final_stress_level,
        'stress_percentage': final_stress_percentage,
        'confidence': {
            'image_confidence': image_confidence,
            'heart_rate_category': heart_rate_category,
            'image_category': image_stress_level,  # Add Image Category to the response
            'heart_rate_percentage': heart_rate_percentage
        },
        'raw_predictions': predictions.tolist(),
        'heart_rate_input': heart_rate
    }

    return jsonify(result)


if __name__ == '__main__':
    # Start the Flask app
    app.run(debug=True)

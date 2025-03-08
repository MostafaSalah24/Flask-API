import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = "siamese_model_script.pt"

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = torch.jit.load(MODEL_PATH, map_location=device)
    model.eval()
    return model

model = load_model()

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

def preprocess_image(image_path):
    """Load and preprocess an image from file path."""
    image = Image.open(image_path).convert("L")
    return transform(image).unsqueeze(0).to(device)

def compare_images(image1_path, image2_path):
    """Compare two images using the Siamese model."""
    with torch.no_grad():
        # Preprocess images
        img1 = preprocess_image(image1_path)
        img2 = preprocess_image(image2_path)

        # Get model predictions
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2).item()

        threshold = 0.2152
        is_different = distance > threshold
        confidence = 1 - (distance / (2 * threshold))
        confidence = max(min(confidence, 1.0), 0.0)  # Clip to [0,1]

        # Return JSON response
        return {
            'status': 'fake' if is_different else 'valid',
            'distance': distance,
            'threshold': threshold,
            'is_different': bool(is_different),
            'confidence': confidence,
            'similarity_score': 1.0 - min(distance / (2 * threshold), 1.0) if distance <= threshold else 0.0
        }

@app.route("/verify", methods=["POST"])
def verify():
    if "original" not in request.files or "current" not in request.files:
        return jsonify({"error": "Both images are required"}), 400

    original = request.files["original"]
    current = request.files["current"]

    orig_path = "temp_original.png"
    curr_path = "temp_current.png"

    original.save(orig_path)
    current.save(curr_path)

    try:
        # Compare images
        result = compare_images(orig_path, curr_path)
    except Exception as e:
        result = {"error": str(e)}
    
    # Clean up temporary files
    os.remove(orig_path)
    os.remove(curr_path)

    return jsonify(result)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


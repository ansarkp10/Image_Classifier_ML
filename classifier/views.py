# classifier/views.py
from django.shortcuts import render
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Where your model file should be placed:
# <project_root>/classifier/models/cnn_model.h5
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "cnn_model.h5")
model = load_model(MODEL_PATH)  # load once at import time

# Map numeric class -> human label (set according to how you trained)
# If 0=Dog and 1=Cat during training, keep this exact mapping.
CLASS_NAMES = {0: "Dog", 1: "Cat"}

def preprocess_image(image_path):
    """Open image, convert to RGB, resize to 100x100, normalize to 0..1."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((100, 100))
    arr = np.array(img).astype("float32")  # shape (100,100,3)
    # NOTE: your training used 0..1 normalized images (you confirmed earlier)
    arr /= 255.0
    arr = arr.reshape(1, 100, 100, 3)
    return arr

def index(request):
    prediction = None
    img_url = None

    if request.method == "POST":
        uploaded_file = request.FILES.get("image")
        if uploaded_file:
            # Ensure media folder exists
            media_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media")
            os.makedirs(media_dir, exist_ok=True)

            # Save uploaded file into project/media/
            img_path = os.path.join(media_dir, uploaded_file.name)
            with open(img_path, "wb+") as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)

            # Preprocess image and predict
            img_array = preprocess_image(img_path)
            pred_prob = float(model.predict(img_array, verbose=0)[0][0])  # 0..1
            pred_class = 1 if pred_prob >= 0.5 else 0

            # Prepare user-friendly values
            probability_percent = round(pred_prob * 100, 2)  # e.g. 92.53
            prediction = {
                "class": int(pred_class),
                "name": CLASS_NAMES.get(pred_class, str(pred_class)),
                "probability": probability_percent  # percent 0..100
            }

            # For template, expose a URL path relative to project root: "media/filename"
            img_url = os.path.join("media", uploaded_file.name).replace("\\", "/")

    return render(request, "index.html", {"prediction": prediction, "img_url": img_url})

from django.db import models

# Create your models here.
# models.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import io
from PIL import Image

# Load the model once (singleton)
model = EfficientNetV2B3(weights='imagenet')

def predict_label_from_image(file):
    try:
        # Load image from InMemoryUploadedFile
        img = Image.open(file).convert('RGB')
        img = img.resize((300, 300))  # EfficientNetV2B3 input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]

        # Format result
        results = [{"label": label, "probability": float(prob)} for (_, label, prob) in decoded]
        return results

    except Exception as e:
        return [{"error": str(e)}]

from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

BUCKET_NAME = 'dr-tf-models'
class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

model = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)



def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/diabetic_retinopathy_model.h5",
            "/tmp/diabetic_retinopathy_model.h5"
        )
        model = tf.keras.models.load_model("/tmp/diabetic_retinopathy_model.h5")

    # Check if file is present in the request
    if 'file' not in request.files:
        return {'error': 'No file provided.'}, 400

    image = request.files["file"]

    preprocessed_image = np.array(Image.open(image).convert("RGB").resize((256, 256)))

    img_array = tf.expand_dims(preprocessed_image, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    response = jsonify({"class": predicted_class, "confidence": confidence})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')

    return response

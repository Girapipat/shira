from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# โหลดโมเดลเพียงครั้งเดียว
MODEL_PATH = "classifier_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

@app.route("/upload", methods=["POST"])
def upload():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        # แปลงรูปเป็น array
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # ทำนาย
        pred = model.predict(arr)[0][0]
        confidence = float(pred)

        # กำหนด intensity (ตัวอย่าง ใช้ confidence mapping)
        intensity = int(confidence * 255)

        return jsonify({
            "is_solution": confidence > 0.5,
            "confidence": confidence,
            "intensity": intensity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "✅ AI Solution Classifier API is running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

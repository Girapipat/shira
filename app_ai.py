from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder="static", template_folder="static")

# ✅ โหลดโมเดลตอนเริ่ม (Render จะโหลดทีเดียว)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ฟังก์ชันเตรียมภาพก่อน predict
def preprocess_image(filepath):
    img = Image.open(filepath).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ✅ หน้าเว็บหลัก
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ✅ API สำหรับอัปโหลดและวิเคราะห์
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "ไม่มีไฟล์"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ไฟล์ไม่ถูกต้อง"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("/tmp", filename)  # ใช้ /tmp สำหรับ Render
    file.save(filepath)

    try:
        img_tensor = preprocess_image(filepath)
        prediction = model.predict(img_tensor)[0]

        # สมมติ output = [probability_of_solution]
        confidence = float(prediction[0])
        is_solution = confidence > 0.5
        intensity = int(confidence * 255)  # ตัวอย่างคำนวณ intensity

        return jsonify({
            "is_solution": is_solution,
            "confidence": confidence,
            "intensity": intensity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

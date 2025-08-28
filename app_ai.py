import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# -------------------------------
# 🔹 กำหนด URL โมเดล (ใส่ลิงก์จริงของคุณแทน)
# -------------------------------
MODEL_URL = "https://huggingface.co/username/model/resolve/main/classifier_model.h5"
MODEL_PATH = "models/classifier_model.h5"

# -------------------------------
# 🔹 โหลดโมเดลอัตโนมัติถ้าไม่มีในเครื่อง
# -------------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("📥 กำลังดาวน์โหลดโมเดลจาก:", MODEL_URL)
        r = requests.get(MODEL_URL, stream=True)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            print("✅ ดาวน์โหลดเสร็จสิ้น")
        else:
            raise Exception(f"❌ โหลดโมเดลไม่สำเร็จ: {r.status_code}")

download_model()

# -------------------------------
# 🔹 โหลดโมเดล
# -------------------------------
model = load_model(MODEL_PATH)

# -------------------------------
# 🔹 หน้าเว็บหลัก
# -------------------------------
@app.route('/')
def index():
    return send_from_directory("static", "index.html")

# -------------------------------
# 🔹 API อัปโหลดภาพ
# -------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))  # ขึ้นกับโมเดลที่ train
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        confidence = float(prediction)

        is_solution = confidence > 0.5
        intensity = int(confidence * 255)

        return jsonify({
            "is_solution": is_solution,
            "confidence": confidence,
            "intensity": intensity
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------
# 🔹 Run local (ตอน dev ใช้ python app_ai.py)
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

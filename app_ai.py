import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# CONFIG
# -------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "classifier_model.h5"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# โหลดโมเดลล่วงหน้า
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    print("❌ โหลดโมเดลล้มเหลว:", str(e))
    model = None


# -------------------------
# HELPER
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path, target_size=(224, 224)):
    """แปลงภาพให้เป็น Tensor สำหรับโมเดล"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "ไม่พบไฟล์ในคำขอ"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "ไม่ได้เลือกไฟล์"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(save_path)

            # -------------------------
            # รันโมเดล
            # -------------------------
            if model is None:
                return jsonify({"error": "โมเดลยังไม่ถูกโหลด"}), 500

            img_tensor = preprocess_image(save_path)
            pred = model.predict(img_tensor)[0][0]

            # binary classification
            is_solution = bool(pred > 0.5)
            confidence = float(pred if is_solution else 1 - pred)

            # mock intensity (อาจเปลี่ยนตาม use case จริง)
            intensity = int(np.mean(img_tensor) * 255)

            result = {
                "is_solution": is_solution,
                "confidence": confidence,
                "intensity": intensity,
            }
            return jsonify(result), 200

        return jsonify({"error": "ชนิดไฟล์ไม่รองรับ"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

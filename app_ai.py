import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "classifier_model.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# โหลดโมเดล
try:
    model = load_model(MODEL_PATH)
    print("✅ โหลดโมเดลสำเร็จ")
except Exception as e:
    print("❌ โหลดโมเดลไม่สำเร็จ:", e)
    model = None


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    # เสิร์ฟหน้า index.html ที่อยู่ใน templates/
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ที่อัปโหลด"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "กรุณาเลือกไฟล์"}), 400

    # เซฟไฟล์
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # วิเคราะห์ภาพ
    try:
        if model is None:
            return jsonify({"error": "ไม่สามารถโหลดโมเดลได้"}), 500

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]
        is_solution = prediction > 0.5
        confidence = float(prediction if is_solution else 1 - prediction)

        # ตัวอย่าง intensity (mock จาก mean pixel)
        intensity = int(np.mean(img_array) * 255)

        return jsonify({
            "is_solution": bool(is_solution),
            "confidence": confidence,
            "intensity": intensity
        })

    except Exception as e:
        return jsonify({"error": f"วิเคราะห์ไม่สำเร็จ: {str(e)}"}), 500


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# โหลดโมเดล
MODEL_PATH = os.path.join("models", "classifier_model.h5")
model = load_model(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")  # เรียกใช้ index.html จาก templates/

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "ไม่มีไฟล์ที่อัปโหลด"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "กรุณาเลือกไฟล์"})

    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    try:
        # โหลดรูปและแปลงให้อยู่ในรูปแบบที่โมเดลต้องการ
        img = image.load_img(filepath, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # พยากรณ์
        prediction = model.predict(img_array)[0][0]  
        confidence = float(prediction)

        # ค่าความเข้มจำลอง (mock)
        intensity = int(confidence * 255)

        result = {
            "is_solution": confidence > 0.5,
            "confidence": confidence,
            "intensity": intensity
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

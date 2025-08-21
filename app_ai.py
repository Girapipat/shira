from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# โหลดโมเดล
model_class = tf.keras.models.load_model("classifier_model.h5")
model_reg = tf.keras.models.load_model("regression_model.h5", compile=False)

IMG_SIZE = (224, 224)
THRESHOLD = 0.6              # ความมั่นใจขั้นต่ำที่ถือว่า "ใช่สารละลาย"
INTENSITY_MIN = 10           # ความเข้มข้นขั้นต่ำที่ถือว่าเป็นสารละลายจริง

def prepare_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "ไม่พบไฟล์ภาพ"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ยังไม่ได้เลือกไฟล์"})

    try:
        img = prepare_image(file)

        # ขั้นที่ 1: ทำนายว่าเป็นสารละลายหรือไม่
        prob = float(model_class.predict(img)[0][0])  # 0.0–1.0 จาก sigmoid

        if prob >= THRESHOLD:
            # ขั้นที่ 2: ทำนายค่าความเข้มข้น
            pred_value = float(model_reg.predict(img)[0][0])
            intensity = int(pred_value * 255 / 0.9)
            intensity = max(0, min(intensity, 255))

            # ตรวจสอบว่าเข้มข้นต่ำผิดปกติหรือไม่
            if intensity < INTENSITY_MIN:
                return jsonify({
                    "is_solution": False,
                    "confidence": round(prob, 2),
                    "intensity": intensity,
                    "note": "ค่าความเข้มข้นต่ำเกินไป"
                })

            return jsonify({
                "is_solution": True,
                "confidence": round(prob, 2),
                "intensity": intensity
            })
        else:
            return jsonify({
                "is_solution": False,
                "confidence": round(prob, 2)
            })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

import tensorflow as tf

# ------------------------
# CONFIG
# ------------------------
models = [
    ("classifier_model.h5", "classifier_model.tflite"),
    ("regression_model.h5", "regression_model.tflite")
]

# ------------------------
# CONVERT
# ------------------------
for h5_path, tflite_path in models:
    try:
        print(f"🔄 กำลังโหลดโมเดล {h5_path} ...")
        model = tf.keras.models.load_model(h5_path)

        print(f"➡️ กำลังแปลง {h5_path} เป็น {tflite_path} ...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optional: เปิดการ optimize (ลดขนาดไฟล์และเพิ่มความเร็ว)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # แปลง
        tflite_model = converter.convert()

        # บันทึกไฟล์
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"✅ แปลงสำเร็จ: {tflite_path}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดกับ {h5_path}: {e}")

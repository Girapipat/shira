import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
BINARY_FOLDER = "_binary"

# -------------------------
# STEP 1: สร้างโฟลเดอร์ binary classifier
# -------------------------
def simplify_class_folder_structure(base_dir):
    new_dir = os.path.join(base_dir, BINARY_FOLDER)
    os.makedirs(os.path.join(new_dir, "not_solution"), exist_ok=True)
    os.makedirs(os.path.join(new_dir, "solution"), exist_ok=True)

    for folder in os.listdir(base_dir):
        src = os.path.join(base_dir, folder)
        if not os.path.isdir(src) or folder == BINARY_FOLDER:
            continue

        target_folder = "not_solution" if folder == "not_solution" else "solution"
        dst_base = os.path.join(new_dir, target_folder)

        for fname in os.listdir(src):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_file = os.path.join(src, fname)
                dst_file = os.path.join(dst_base, f"{folder}_{fname}")
                tf.io.gfile.copy(src_file, dst_file, overwrite=True)

    return new_dir

# -------------------------
# MAIN TRAIN FUNCTION
# -------------------------
def train_classifier(data_dir="dataset", model_output="classifier_model.h5"):
    print("🔧 เตรียมโฟลเดอร์ binary...")
    binary_data_dir = simplify_class_folder_structure(data_dir)

    # -------------------------
    # STEP 2: เตรียมข้อมูล + Augment
    # -------------------------
    print("📦 เตรียมข้อมูล...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        binary_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        binary_data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    if train_gen.samples == 0:
        raise ValueError("❌ ไม่มีภาพสำหรับฝึกในโฟลเดอร์")

    # -------------------------
    # STEP 3: คำนวณ class weights
    # -------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_gen.classes
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("⚖️ Class weights:", class_weight_dict)

    # -------------------------
    # STEP 4: สร้างโมเดล
    # -------------------------
    print("🧠 สร้างโมเดล...")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # -------------------------
    # STEP 5: เทรน
    # -------------------------
    print("🚀 เริ่มเทรน...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weight_dict
    )

    # -------------------------
    # STEP 6: บันทึกโมเดล
    # -------------------------
    model.save(model_output)
    print(f"✅ บันทึกโมเดลเป็น {model_output} สำเร็จแล้ว")

# -------------------------
# หากเรียกใช้โดยตรง
# -------------------------
if __name__ == "__main__":
    train_classifier()

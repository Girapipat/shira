import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import re
import numpy as np

# พารามิเตอร์
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# ===== ฟังก์ชันแมปชื่อโฟลเดอร์เป็นค่าความเข้มข้น =====
def folder_to_intensity(folder_name):
    match = re.search(r'\((.*?)\)', folder_name)
    level = match.group(1).lower().strip() if match else ""
    levels = [
        "lowest", "lower", "low",
        "midlow", "mid", "midhigh",
        "high", "higher", "highest"
    ]
    if level in levels:
        return levels.index(level) / (len(levels) - 1)
    else:
        return None

# ===== โหลด path ของภาพพร้อม label =====
def load_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(full_path))
                if folder_name.startswith("solution_") and folder_name != "not_solution":
                    intensity = folder_to_intensity(folder_name)
                    if intensity is not None:
                        image_paths.append(full_path)
                        labels.append(intensity)

    return image_paths, labels

# ===== แปลง path → dataset =====
def prepare_dataset(image_paths, labels):
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    def process_path(file_path, label):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.float32)

    ds = path_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

# ===== สร้างโมเดล Regression =====
def build_regression_model():
    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.3)(x)
    x = layers.RandomZoom(0.3)(x)
    x = layers.RandomContrast(0.3)(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # output: 0.0 - 1.0

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
        metrics=['mae']
    )
    return model

# ===== เทรนโมเดล Regression =====
def train_regression(data_dir):
    image_paths, labels = load_image_paths_and_labels(data_dir)
    if len(image_paths) == 0:
        raise ValueError("❌ ไม่มีข้อมูลที่ตรงกับรูปแบบระดับความเข้มข้องสารในโฟลเดอร")

    # สุ่มข้อมูลก่อนแบ่ง
    combined = list(zip(image_paths, labels))
    np.random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined)

    train_size = int(0.8 * len(image_paths))
    train_paths = image_paths[:train_size]
    train_labels = labels[:train_size]
    val_paths = image_paths[train_size:]
    val_labels = labels[train_size:]

    train_ds = prepare_dataset(train_paths, train_labels)
    val_ds = prepare_dataset(val_paths, val_labels)

    model = build_regression_model()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("regression_model.h5", save_best_only=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # Fine-tune รอบ 2
    base_model = model.get_layer('mobilenetv2_1.00_224')
    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='mean_squared_error',
        metrics=['mae']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    print("✅ เทรน regression_model.h5 เสร็จสิ้น")

# ===== ทดสอบแยกไฟล์ =====
if __name__ == "__main__":
    train_regression("dataset")

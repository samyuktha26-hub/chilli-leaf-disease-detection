import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

# --- CONFIG ---
UPLOADS_DIR = "uploads"
NEW_MODEL_PATH = "chilli_model_updated.tflite"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 3

# --- STEP 1: Load dataset ---
if not os.path.exists(UPLOADS_DIR):
    raise FileNotFoundError(f"Uploads folder not found: {UPLOADS_DIR}")

train_ds = image_dataset_from_directory(
    UPLOADS_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes detected: {class_names}")

# --- STEP 2: Build model ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- STEP 3: Retrain ---
model.fit(train_ds, epochs=EPOCHS)

# --- STEP 4: Convert to TFLite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(NEW_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Retrained TFLite model saved at {NEW_MODEL_PATH}")


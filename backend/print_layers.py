from tensorflow.keras.models import load_model
import tensorflow as tf
model = load_model("my_trained_model.h5")
for layer in model.layers[::-1]:
    if isinstance(layer, tf.keras.layers.Conv2D):
        print(layer.name)
        break


import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model

def generate_gradcam_overlay(pil_img, model, last_conv_layer_name):
    img = pil_img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)

        # FIX — robust unpacking
        if isinstance(outputs, (tuple, list)):
            flat = []
            for x in outputs:
                if isinstance(x, (tuple, list)):
                    flat.extend(x)
                else:
                    flat.append(x)
            conv_outputs = flat[0]
            predictions = flat[1]
        else:
            raise ValueError("Unexpected outputs from grad_model")

        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1).numpy()

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    heatmap = cv2.resize(cam, (pil_img.width, pil_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np.array(pil_img), 0.6, heatmap, 0.4, 0)
    success, encoded = cv2.imencode(".jpg", overlay)
    return encoded.tobytes()


import base64
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os, time, subprocess
from gradcam_utils import generate_gradcam_overlay
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# --- NEW: Configure OpenAI ---
# REPLACE 'sk-...' WITH YOUR ACTUAL API KEY
client = OpenAI(api_key="aaa")

script_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(script_dir, "chilli_disease_mobilenetv2.tflite")
keras_model_path = os.path.join(script_dir, "my_trained_model.h5")
labels_path = os.path.join(script_dir, "labels.txt")
LAST_CONV_LAYER_NAME = "Conv_1_bn"   # for MobileNetV2 — update if model summary says otherwise
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open(labels_path) as f:
    labels = [line.strip() for line in f]

# app.py - somewhere near the top after labels are loaded
treatment_dict = {
    "healthy": "No treatment required. Continue good farm practices.",
    "leaf curl": "Remove infected leaves. Apply Imidacloprid 0.3ml/litre.",
    "leaf spot": "Use Mancozeb 2g/litre. Improve field aeration.",
    "whitefly": "Spray Neem oil 3ml/litre. Maintain crop hygiene.",
    "yellowish": "Apply urea-based fertilizer carefully. Remove stressed leaves."
}

# Keras model only for Grad-CAM
MODEL_KERAS = tf.keras.models.load_model(keras_model_path)

# ---------------------------------------------------------
# --- NEW: Chatbot Route ---
# ---------------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # We feed your treatment_dict into the AI's instructions so it stays consistent
        system_instruction = f"""
        You are an expert agriculturalist specializing in Chilli Plant diseases.
        
        Here are the standard treatments we recommend:
        {str(treatment_dict)}
        
        1. If the user asks about a specific disease in this list, recommend the treatment above.
        2. If the user asks general questions, provide short, practical farming advice.
        3. Keep answers concise (under 3 sentences) as farmers are reading this on mobile phones.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Use "gpt-4o" if you have access for better results
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ]
        )

        bot_reply = response.choices[0].message.content
        return jsonify({"reply": bot_reply})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"error": "Something went wrong with the chatbot"}), 500


# ---------------------------------------------------------
# --- Existing Routes (Unchanged) ---
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # --- Read image ---
    img_bytes = file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = pil_img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    arr_tflite = np.expand_dims(arr, axis=0).astype(np.float32)

    # --- TFLite inference ---
    interpreter.set_tensor(input_details[0]['index'], arr_tflite)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])
    label = labels[idx]
    treatment = treatment_dict.get(label.lower(), "Treatment not available.")
    # --- Severity ---
    severity = "Severe" if confidence > 0.90 else "Moderate" if confidence > 0.60 else "Mild"

    # --- Grad-CAM with text overlay ---
    heatmap_b64 = None
    try:
        cam_jpg = generate_gradcam_overlay(pil_img, MODEL_KERAS, LAST_CONV_LAYER_NAME)

        # Convert to OpenCV image
        img_cv = cv2.imdecode(np.frombuffer(cam_jpg, np.uint8), cv2.IMREAD_COLOR)

        # Overlay text
        text = f"{label} ({confidence*100:.1f}%) "
        text1=f"{severity}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0) if severity=="Mild" else (0, 255, 255) if severity=="Moderate" else (0, 0, 255)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.putText(img_cv, text, (10, 25), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(img_cv, text1, (10, 55), font, font_scale, color, thickness, cv2.LINE_AA)
        # Encode back to JPEG
        _, cam_buf = cv2.imencode(".jpg", img_cv)
        heatmap_b64 = base64.b64encode(cam_buf.tobytes()).decode("utf-8")

    except Exception as e:
        print("Grad-CAM failed:", e)

    # --- Save original to dataset ---
    ts = int(time.time())
    save_dir = os.path.join(UPLOAD_FOLDER, label)
    os.makedirs(save_dir, exist_ok=True)
    pil_img.save(os.path.join(save_dir, f"{label}_{ts}.jpg"))

    # --- Response ---
    response = {
        "label": label,
        "confidence": confidence,
        "severity": severity,
        "treatment": treatment,  # you can still show in frontend card
        "all_probs": {labels[i]: float(pred[i]) for i in range(len(labels))}
    }
    if heatmap_b64:
        response["gradcam"] = heatmap_b64

    return jsonify(response)


@app.route("/upload-labeled", methods=["POST"])
def upload_labeled():
    file = request.files.get("file")
    label = request.form.get("label")
    if not file or not label:
        return jsonify({"error": "Need file + label"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, label)
    os.makedirs(save_path, exist_ok=True)
    file.save(os.path.join(save_path, f"{label}_{int(time.time())}.jpg"))
    return jsonify({"message": "Saved"})


# ----------------- Retrain route stays unchanged -----------------
@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        result = subprocess.run(["python", "retrain.py"], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({"status": "failed", "error": result.stderr})

        global interpreter, input_details, output_details
        interpreter = tf.lite.Interpreter(model_path="chilli_model_updated.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return jsonify({"status": "success", "message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
# app.py - combined server with improved chilli-leaf validation (OpenCV + ORB reference matching)
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

# ----------------------------
# Configuration / Globals
# ----------------------------
app = Flask(__name__)
CORS(app)

# Use environment variable for API key; fallback to None
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set this in your shell/IDE
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set. Chatbot will fail if used.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

script_dir = os.path.dirname(os.path.abspath(__file__))
tflite_model_path = os.path.join(script_dir, "chilli_disease_mobilenetv2.tflite")
keras_model_path = os.path.join(script_dir, "my_trained_model.h5")
labels_path = os.path.join(script_dir, "labels.txt")
LAST_CONV_LAYER_NAME = "Conv_1_bn"   # adjust if your model uses a different layer
UPLOAD_FOLDER = os.path.join(script_dir, "uploads")
REF_FOLDER = os.path.join(script_dir, "chilli_refs")   # put reference images here
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REF_FOLDER, exist_ok=True)

# ----------------------------
# Load models
# ----------------------------
# load tflite interpreter
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# load keras model for grad-cam (optional; if absent, grad-cam calls will be caught)
MODEL_KERAS = None
try:
    MODEL_KERAS = tf.keras.models.load_model(keras_model_path)
except Exception as e:
    print("Warning: Could not load Keras model for Grad-CAM:", e)
    MODEL_KERAS = None

# labels
with open(labels_path) as f:
    labels = [line.strip() for line in f]

# treatment dictionary
treatment_dict = {
    "healthy": "No treatment required. Continue good farm practices.",
    "leaf curl": "Remove infected leaves. Apply Imidacloprid 0.3ml/litre.",
    "leaf spot": "Use Mancozeb 2g/litre. Improve field aeration.",
    "whitefly": "Spray Neem oil 3ml/litre. Maintain crop hygiene.",
    "yellowish": "Apply urea-based fertilizer carefully. Remove stressed leaves."
}

# ----------------------------
# Reference descriptors (ORB) loader
# ----------------------------
ORB = cv2.ORB_create(nfeatures=1000)
BF = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# list of tuples: (ref_filename, kp_ref, des_ref)
REF_DESCS = []

def load_reference_descriptors():
    """Load ORB keypoints/descriptors for all images in chilli_refs/"""
    global REF_DESCS
    REF_DESCS = []
    if not os.path.exists(REF_FOLDER):
        print("Reference folder not found:", REF_FOLDER)
        return
    files = sorted([f for f in os.listdir(REF_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    for fn in files:
        path = os.path.join(REF_FOLDER, fn)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = ORB.detectAndCompute(img, None)
        if des is None:
            continue
        REF_DESCS.append((fn, kp, des))
    print(f"Loaded {len(REF_DESCS)} reference descriptor(s) from {REF_FOLDER}.")

# load on startup
load_reference_descriptors()

# ----------------------------
# Utility: ORB matching function
# ----------------------------
def orb_match_score(des1, des2, k=2, ratio_thresh=0.75):
    """
    Compute number of 'good matches' using Lowe's ratio test.
    Returns integer good_matches_count and ratio (good / min_kp)
    """
    if des1 is None or des2 is None:
        return 0, 0.0
    try:
        matches = BF.knnMatch(des1, des2, k=k)
    except Exception:
        return 0, 0.0
    good = []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    good_count = len(good)
    # return also proportion relative to average descriptors
    denom = max(1, min(len(des1), len(des2)))
    ratio = good_count / float(denom)
    return good_count, ratio

# ----------------------------
# Leaf validation (relaxed + ORB reference matching)
# ----------------------------
def is_chilli_leaf_opencv(pil_img, use_ref_matching=True):
    """
    Returns (is_leaf_bool, details_dict)
    - uses relaxed HSV green ratio + contour checks + texture
    - if use_ref_matching and REF_DESCS exists, runs ORB match scoring and uses thresholds
    details_dict includes debug numbers for tuning
    """
    details = {}
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img_small = cv2.resize(img_bgr, (350, 350))
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)

    # relaxed green range for variable lighting
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = float(np.sum(mask > 0)) / (mask.shape[0] * mask.shape[1])
    details['green_ratio'] = green_ratio

    # basic contour check
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    details['contour_count'] = len(contours)
    if len(contours) == 0:
        return False, details

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(h) / (w + 1e-9)
    details['aspect_ratio'] = aspect_ratio

    # relaxed thresholds: allow skewed, rotated leaves
    if green_ratio < 0.12:   # extremely low green => reject
        details['reason'] = 'too_little_green'
        return False, details

    # texture check (very low texture often indicates paper/plastic or extremely blurred)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    details['texture_var'] = float(texture)
    if texture < 6:
        details['reason'] = 'low_texture'
        return False, details

    # If reference descriptors available, run ORB matching and require at least a minimal good match
    details['ref_matches'] = None
    if use_ref_matching and len(REF_DESCS) > 0:
        # compute descriptors for uploaded image (grayscale)
        gray_full = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        kp1, des1 = ORB.detectAndCompute(gray_full, None)
        if des1 is None or len(des1) < 8:
            details['reason'] = 'insufficient_keypoints'
            return False, details
        best_good = 0
        best_ratio = 0.0
        best_ref = None
        for (fn, kp_ref, des_ref) in REF_DESCS:
            good_count, ratio = orb_match_score(des1, des_ref, k=2, ratio_thresh=0.75)
            if good_count > best_good:
                best_good = good_count
                best_ratio = ratio
                best_ref = fn
        details['ref_matches'] = {'best_good': int(best_good), 'best_ratio': float(best_ratio), 'best_ref': best_ref}
        # Decide acceptance: medium strictness
        # Accept if good matches >= 12 OR ratio >= 0.04
        if not (best_good >= 12 or best_ratio >= 0.04):
            details['reason'] = 'low_reference_match'
            return False, details

    # passed all checks
    details['reason'] = 'passed'
    return True, details

# ----------------------------
# Chatbot route (uses OpenAI client if available)
# ----------------------------
@app.route('/chat', methods=['POST'])
def chat():
    if client is None:
        return jsonify({"error": "Chatbot not configured (OPENAI_API_KEY missing)."}), 500
    try:
        data = request.json or {}
        user_message = data.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        system_instruction = f"""
        You are an expert agriculturalist specializing in Chilli Plant diseases.
        Here are the treatments: {str(treatment_dict)}
        Always reply under 3 sentences.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ]
        )
        bot_reply = response.choices[0].message.content
        return jsonify({"reply": bot_reply})
    except Exception as e:
        print("Chat Error:", e)
        return jsonify({"error": "Chatbot error"}), 500

# ----------------------------
# validate_leaf endpoint (exposes debug info)
# ----------------------------
@app.route("/validate_leaf", methods=["POST"])
def validate_leaf():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    img_bytes = file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    is_leaf, details = is_chilli_leaf_opencv(pil_img, use_ref_matching=True)
    return jsonify({
        "is_chilli_leaf": bool(is_leaf),
        "details": details,
        "message": "Valid chilli leaf" if is_leaf else "Not a chilli leaf"
    })

# ----------------------------
# predict endpoint (unchanged core behavior, returns gradcam)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    img_bytes = file.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Validate first
    is_leaf, details = is_chilli_leaf_opencv(pil_img, use_ref_matching=True)
    if not is_leaf:
        return jsonify({
            "is_chilli_leaf": False,
            "message": "Not a chilli leaf – please upload chilli leaf image",
            "details": details
        })

    # Continue prediction
    img_resized = pil_img.resize((224, 224))
    arr = np.array(img_resized) / 255.0
    arr_tflite = np.expand_dims(arr, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], arr_tflite)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]
    idx = int(np.argmax(pred))
    confidence = float(pred[idx])
    label = labels[idx]
    treatment = treatment_dict.get(label.lower(), "Treatment not available.")
    severity = "Severe" if confidence > 0.90 else "Moderate" if confidence > 0.60 else "Mild"

    heatmap_b64 = None
    if MODEL_KERAS is not None:
        try:
            cam_jpg = generate_gradcam_overlay(pil_img, MODEL_KERAS, LAST_CONV_LAYER_NAME)
            # cam_jpg is bytes of a jpg image; encode as base64
            heatmap_b64 = base64.b64encode(cam_jpg).decode("utf-8")
        except Exception as e:
            print("Grad-CAM failed:", e)

    # Save to dataset folder
    ts = int(time.time())
    save_dir = os.path.join(UPLOAD_FOLDER, label)
    os.makedirs(save_dir, exist_ok=True)
    pil_img.save(os.path.join(save_dir, f"{label}_{ts}.jpg"))

    response = {
        "is_chilli_leaf": True,
        "label": label,
        "confidence": confidence,
        "severity": severity,
        "treatment": treatment,
        "details": details,
        "all_probs": {labels[i]: float(pred[i]) for i in range(len(labels))}
    }
    if heatmap_b64:
        response["gradcam"] = heatmap_b64

    return jsonify(response)

# ----------------------------
# upload-labeled (unchanged)
# ----------------------------
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

# ----------------------------
# retrain (unchanged)
# ----------------------------
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

        # reload ref descriptors in case retrain added refs
        load_reference_descriptors()

        return jsonify({"status": "success", "message": "Model retrained successfully"})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)})

# ----------------------------
# run
# ----------------------------
if __name__ == "__main__":
    print("Reference folder:", REF_FOLDER)
    print("Ensure you placed some chilli reference images (jpg/png) in the folder above.")
    load_reference_descriptors()
    app.run(debug=True)

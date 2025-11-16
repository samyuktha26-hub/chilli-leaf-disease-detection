import requests
import os

url = "http://127.0.0.1:5000/predict"

# Use absolute path to be safe
file_path = r"chilli.jpg"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

with open(file_path, "rb") as f:
    files = {"file": ("chilli.jpg", f, "image/jpeg")}
    res = requests.post(url, files=files)

print("Status code:", res.status_code)
print("Response content:", res.text)

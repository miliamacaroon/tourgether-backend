import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ===============================
# HF SETUP
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "intxnk01/tourgether-models"

def download_yolo_model(filename="models/best.pt", local_dir="downloads") -> str:
    """Download YOLO model from Hugging Face if not present locally."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if not os.path.exists(local_path):
        print(f"📥 Downloading YOLO model from Hugging Face...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN,
            cache_dir=local_dir
        )
        print(f"✅ YOLO model downloaded to {local_path}")
    return local_path

# ===============================
# LOAD YOLO
# ===============================
YOLO_PATH = download_yolo_model("models/best.pt")
yolo_model = YOLO(YOLO_PATH)

# ===============================
# DETECTION FUNCTION
# ===============================
def detect_objects(image_path):
    results = yolo_model(image_path)
    return results

import os
import logging
from huggingface_hub import hf_hub_download, snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use the mount path from your Volume (defaults to /app/models if not set)
MODEL_DIR = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "/app/models")
HF_REPO_ID = "intxnk01/tourgether-models"
HF_TOKEN = os.getenv("HF_TOKEN")

def download_all_models():
    """Main entry point for Railway startup"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Download YOLO Model
    yolo_dest = os.path.join(MODEL_DIR, "best.pt")
    if not os.path.exists(yolo_dest):
        logger.info(f"📥 Downloading YOLO model to {yolo_dest}...")
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="models/best.pt",
            token=HF_TOKEN,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
    else:
        logger.info("✅ YOLO model already exists in persistent volume.")

    # 2. Download FAISS Index
    faiss_dest = os.path.join(MODEL_DIR, "faiss_embeddings_region")
    if not os.path.exists(faiss_dest):
        logger.info(f"📥 Downloading FAISS embeddings to {faiss_dest}...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns="faiss_embeddings_region/*",
            token=HF_TOKEN,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
    else:
        logger.info("✅ FAISS embeddings already exist in persistent volume.")

if __name__ == "__main__":
    download_all_models()
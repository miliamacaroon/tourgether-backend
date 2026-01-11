"""
Download models and embeddings from Hugging Face
Run this on Railway startup before the app starts
"""
import os
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

# Your Hugging Face repository details
HF_REPO_ID = "your-username/your-repo-name"  # e.g., "john/tourgether-models"
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional, only needed for private repos

def download_yolo_model():
    """Download YOLOv11 model"""
    print("📥 Downloading YOLO model from Hugging Face...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    try:
        # Download the best.pt file
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="models/best.pt",
            token=HF_TOKEN,
            cache_dir="./models",
            local_dir="./models",
            local_dir_use_symlinks=False
        )
        print(f"✅ YOLO model downloaded to: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLO model: {e}")
        return False

def download_faiss_embeddings():
    """Download FAISS embeddings folder"""
    print("📥 Downloading FAISS embeddings from Hugging Face...")
    
    try:
        # Download entire faiss_embeddings_region folder
        snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns="faiss_embeddings_region/*",
            token=HF_TOKEN,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print("✅ FAISS embeddings downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download FAISS embeddings: {e}")
        return False

def download_all_models():
    """Download all required models"""
    print("\n" + "="*50)
    print("🚀 TourGether Model Download Starting...")
    print("="*50 + "\n")
    
    yolo_ok = download_yolo_model()
    faiss_ok = download_faiss_embeddings()
    
    if yolo_ok and faiss_ok:
        print("\n✅ All models downloaded successfully!")
        return True
    else:
        print("\n⚠️ Some models failed to download")
        return False

if __name__ == "__main__":
    download_all_models()
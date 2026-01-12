import os
import logging
from ultralytics import YOLO

# Set up logging to see what's happening in Railway logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the loaded model (Singleton pattern)
_vision_model = None

def load_model(model_path: str):
    """
    Loads the YOLO model into memory. 
    Using a global variable ensures we don't load it multiple times.
    """
    global _vision_model
    
    if _vision_model is not None:
        return _vision_model
    
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file {model_path} is missing. Check download logs.")
    
    try:
        logger.info(f"🔄 Loading YOLO model from {model_path}...")
        # Use task='detect' to be explicit
        _vision_model = YOLO(model_path, task='detect')
        logger.info("✅ Vision model loaded successfully.")
        return _vision_model
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise

def detect_attraction(image_path: str, model=None):
    """
    Run inference on an image. 
    Accepts an optional model instance to avoid re-loading.
    """
    if model is None:
        # Fallback if no model is passed, though main.py should provide it
        model = load_model(os.getenv("MODEL_PATH", "models/best.pt"))
    
    try:
        results = model(image_path)
        
        # Extract the top detected class and confidence
        if len(results) > 0 and len(results[0].boxes) > 0:
            top_box = results[0].boxes[0]
            class_id = int(top_box.cls[0])
            label = model.names[class_id]
            confidence = float(top_box.conf[0])
            return label, confidence
        
        return "Unknown", 0.0
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return "Error", 0.0
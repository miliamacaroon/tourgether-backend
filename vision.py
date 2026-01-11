from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def detect_attraction(image_path, model):
    results = model(image_path)
    probs = results[0].probs

    if probs is None:
        return "no_detection", 0.0

    cls_id = int(probs.top1)
    label = model.names[cls_id]
    confidence = float(probs.top1conf)

    return label, confidence

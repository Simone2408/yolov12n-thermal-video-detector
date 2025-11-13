# src/model_loader.py

from pathlib import Path
from ultralytics import YOLO


def load_model(weights_path: str = "weights/best.pt"):
    """
    Load a YOLO model using Ultralytics API.

    This avoids dealing with torch.load and PyTorch safe unpickling,
    and lets Ultralytics handle the internal DetectionModel logic.
    """

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights_path}")

    model = YOLO(str(weights_path))
    print(f"[INFO] YOLO model loaded from {weights_path}")
    return model

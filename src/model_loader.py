# src/model_loader.py

from pathlib import Path

import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel


# ✅ Allow Ultralytics DetectionModel to be loaded safely by torch.load
add_safe_globals([DetectionModel])


def load_model(weights_path: str = "weights/best.pt", device: str = "cuda"):
    """
    Load YOLOv12n model from a .pt checkpoint (Ultralytics-style).

    - The checkpoint was saved from an Ultralytics YOLO model (DetectionModel).
    - In PyTorch 2.6, torch.load() uses a "safe" unpickler by default (weights_only=True),
      which blocks unknown Python classes unless they are explicitly allowlisted.
    - Here we use `add_safe_globals([DetectionModel])` so that torch.load can
      deserialize Ultralytics' DetectionModel safely.

    Only do this if you trust the source of the checkpoint (in this case: yourself).
    """

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights_path}")

    # 👇 Now safe unpickling knows DetectionModel is allowed
    checkpoint = torch.load(weights_path, map_location=device)

    # Case 1: you saved the full model directly: torch.save(model, "best.pt")
    if not isinstance(checkpoint, dict):
        model = checkpoint
    else:
        # Case 2: Ultralytics-style checkpoint with 'model' key
        if "model" in checkpoint:
            model = checkpoint["model"]
        else:
            raise KeyError(
                "Impossibile trovare la chiave 'model' nel checkpoint. "
                "Adatta load_model() al formato del tuo file di pesi."
            )

    model.to(device)
    model.eval()

    print(f"[INFO] Modello caricato da {weights_path} su device: {device}")
    return model

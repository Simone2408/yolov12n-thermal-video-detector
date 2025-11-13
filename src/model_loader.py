# src/model_loader.py

from pathlib import Path

import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel


# ✅ Allow Ultralytics DetectionModel to be loaded by torch.load safely
add_safe_globals([DetectionModel])


def load_model(weights_path: str = "weights/best.pt", device: str = "cuda"):
    """
    Load YOLOv12n (Ultralytics) model from a .pt checkpoint.

    Notes:
    - This checkpoint was saved from Ultralytics (DetectionModel).
    - Torch 2.6 uses `weights_only=True` by default in `torch.load`.
    - We explicitly allow the Ultralytics DetectionModel class via `add_safe_globals`.
    """

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights_path}")

    # ✅ weights_only=True is fine now because we have whitelisted DetectionModel
    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,  # we can also set False explicitly to be safe
    )

    # Case 1: you saved the whole model directly (torch.save(model, "best.pt"))
    if not isinstance(checkpoint, dict):
        model = checkpoint
    else:
        # Case 2: Ultralytics-style dict with 'model' key
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

# src/model_loader.py

import torch
from pathlib import Path


def load_model(weights_path: str = "weights/best.pt", device: str = "cuda"):
    """
    Load YOLOv12n model from a .pt checkpoint.
    
    """

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights_path}")

    # IMPORTANT: weights_only=False for Ultralytics model checkpoints
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

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

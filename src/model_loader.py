# src/model_loader.py

import torch
from pathlib import Path


def load_model(weights_path: str = "weights/best.pt", device: str = "cuda"):
    """
    Carica il modello YOLOv12n da un file di pesi.
    """

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # Caso 1: hai salvato direttamente il modello (torch.save(model, ...))
    if not isinstance(checkpoint, dict):
        model = checkpoint
    else:
        # Caso 2: hai salvato un dict con dentro il modello
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

# src/model_loader.py

from pathlib import Path
import torch


def load_model(weights_path: str = "weights/best.pt", device: str = "cuda"):
    """
    Load YOLOv12n model from a .pt checkpoint.

    IMPORTANT:
    - This checkpoint was saved from an Ultralytics YOLO model (DetectionModel).
    - In PyTorch 2.6, torch.load() uses `weights_only=True` by default.
    - Here we explicitly set `weights_only=False` because we want to load
      the full model object, not only raw weights.

    Only do this if you trust the source of the checkpoint (in this case: yourself).
    """

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pesi non trovati: {weights_path}")

    # 🔴 KEY POINT: explicitly disable weights_only safety restriction
    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False,  # <--- this is the fix
    )

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

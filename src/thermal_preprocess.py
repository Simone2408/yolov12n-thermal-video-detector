# src/thermal_preprocess.py

import cv2
import numpy as np
import torch


def normalize_thermal_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalizza un frame termico in range [0, 255] (uint8).

    Gestisce:
    - input 8 bit
    - input 16 bit
    - input BGR già "colorato" (lo converte in grayscale)
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    if gray.dtype == np.uint16:
        # schiaccia [min, max] -> [0, 255]
        gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-6) * 255.0).astype(
            np.uint8
        )
    elif gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


def apply_hist_equalization(gray: np.ndarray) -> np.ndarray:
    """
    Equalizzazione dell'istogramma per aumentare il contrasto.
    """
    return cv2.equalizeHist(gray)


def apply_colormap(gray: np.ndarray, mode: str = "inferno") -> np.ndarray:
    """
    Applica una colormap al frame termico.

    mode:
        - 'gray'     -> scala di grigi
        - 'inferno'  -> colormap inferno
        - 'jet'      -> colormap tipo arcobaleno
    """
    if mode == "gray":
        colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "inferno":
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    elif mode == "jet":
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    else:
        colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return colored


def preprocess_for_model(
    frame: np.ndarray,
    img_size: int = 640,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Preprocessing finale per il modello YOLOv12n.

    Step:
    1. normalizzazione termica
    2. equalizzazione
    3. colormap (inferno)
    4. resize a img_size x img_size
    5. BGR -> RGB
    6. [0,255] -> [0,1]
    7. HWC -> CHW + batch dim
    """
    gray = normalize_thermal_frame(frame)
    gray = apply_hist_equalization(gray)
    colored = apply_colormap(gray, mode="inferno")

    # Resize
    resized = cv2.resize(colored, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # [0,255] -> [0,1]
    img = rgb.astype(np.float32) / 255.0

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)

    # Aggiungi batch dim: (1, C, H, W)
    img = np.expand_dims(img, axis=0)

    tensor = torch.from_numpy(img).to(device)
    return tensor

def preprocess_frame_for_yolo(frame: np.ndarray, mode: str = "inferno") -> np.ndarray:
    """
    Prepare a thermal frame for YOLO:
    - normalize
    - histogram equalization
    - apply colormap (default: inferno)
    Returns a BGR uint8 image, same size as input.
    """
    gray = normalize_thermal_frame(frame)
    gray = apply_hist_equalization(gray)
    colored = apply_colormap(gray, mode=mode)
    return colored
# src/thermal_preprocess.py

import cv2
import numpy as np


def normalize_thermal_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize a thermal frame to [0, 255] uint8.
    Handles:
    - 8-bit images
    - 16-bit images
    - 3-channel BGR images (converted to grayscale)
    """
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    if gray.dtype == np.uint16:
        gray = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-6) * 255.0).astype(
            np.uint8
        )
    elif gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return gray


def apply_hist_equalization(gray: np.ndarray) -> np.ndarray:
    """Increase contrast via histogram equalization."""
    return cv2.equalizeHist(gray)


def apply_colormap(gray: np.ndarray, mode: str = "inferno") -> np.ndarray:
    """
    Apply a colormap to the thermal frame.

    mode:
        - 'gray'     -> grayscale
        - 'inferno'  -> inferno colormap
        - 'jet'      -> rainbow-style
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


def preprocess_frame_for_yolo(frame: np.ndarray, mode: str = "inferno") -> np.ndarray:
    """
    Full preprocessing pipeline for YOLO:

    1. normalize thermal intensities
    2. histogram equalization
    3. apply colormap

    Returns:
        BGR uint8 image (same size as input) ready for YOLO.
    """
    gray = normalize_thermal_frame(frame)
    gray = apply_hist_equalization(gray)
    colored = apply_colormap(gray, mode=mode)
    return colored

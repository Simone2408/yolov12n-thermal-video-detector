# src/utils.py

import time
from typing import List

import cv2
import numpy as np

# Classi del modello (ordine coerente con il training)
CLASS_NAMES = [
    "train",
    "signal",
    "railway switch",
    "pole",
    "balise",
    "overhead support arm",
]

# Colori BGR per ogni classe (diversi)
CLASS_COLORS = {
    0: (255, 0, 0),      # train        -> blu
    1: (0, 0, 255),      # signal       -> rosso
    2: (255, 255, 0),    # railway switch -> giallo
    3: (0, 255, 255),    # pole         -> ciano
    4: (255, 0, 255),    # balise       -> magenta
    5: (0, 255, 0),      # overhead support arm -> verde
}


def draw_detections(
    frame: np.ndarray,
    detections: np.ndarray,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """
    Disegna i bounding box sul frame.

    Si assume che `detections` sia un array Nx6:
        [x1, y1, x2, y2, score, class_id]

    TODO:
    - Se il tuo modello produce un formato diverso, adatta questa funzione.
    """
    h, w = frame.shape[:2]

    if detections is None or len(detections) == 0:
        return frame

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det

        if score < conf_threshold:
            continue

        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        cls_id = int(cls_id)
        if 0 <= cls_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[cls_id]
        else:
            class_name = str(cls_id)

        color = CLASS_COLORS.get(cls_id, (0, 255, 0))  # default verde
        label = f"{class_name} {score:.2f}"

        # Rettangolo e sfondo etichetta
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


class FPSCounter:
    """
    Semplice FPS counter per mostrare i frame per secondo.
    """

    def __init__(self, avg_over: int = 30):
        self.avg_over = avg_over
        self.times: List[float] = []

    def tick(self) -> float:
        now = time.time()
        self.times.append(now)
        if len(self.times) > self.avg_over:
            self.times.pop(0)

        if len(self.times) <= 1:
            return 0.0

        fps = (len(self.times) - 1) / (self.times[-1] - self.times[0] + 1e-6)
        return fps

    @staticmethod
    def put_fps_on_frame(frame: np.ndarray, fps: float) -> np.ndarray:
        text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

# src/video_inference.py

import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from .thermal_preprocess import preprocess_frame_for_yolo
from .utils import draw_detections, FPSCounter


def yolo_infer_frame(model, frame_bgr: np.ndarray, img_size: int = 640, device: str = "cuda"):
    """
    Performs YOLO inference on a BGR frame and returns detections in Nx6 format:
    [x1, y1, x2, y2, conf, cls]
    """
    results = model(frame_bgr, imgsz=img_size, device=device, verbose=False)

    # If results is a list/tuple, take first element
    if isinstance(results, (tuple, list)):
        results = results[0]

    detections = None
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        conf = boxes.conf.cpu().numpy()  # (N,)
        cls = boxes.cls.cpu().numpy()    # (N,)

        detections = np.concatenate(
            [xyxy, conf[:, None], cls[:, None]],
            axis=1
        )

    return detections


def run_inference_colab_stream(
    weights: str = "weights/best.pt",
    source: str = "",
    img_size: int = 640,
    conf_thres: float = 0.25,
    device: str = "cuda",
):
    """
    Runs YOLO inference on a thermal video and streams annotated frames
    in real-time (Colab cell).
    """
    # Colab-specific imports
    try:
        from google.colab.patches import cv2_imshow
        from IPython.display import clear_output
    except ImportError:
        raise RuntimeError("run_inference_colab_stream is intended for Google Colab.")

    # Load model
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    print("[INFO] Loading YOLO model...")
    model = YOLO(str(weights_path))
    print(f"[INFO] YOLO model loaded from: {weights_path}")

    # Open video
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found at: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    fps_counter = FPSCounter()
    frame_id = 0

    print("[INFO] Starting streaming inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thermal → BGR
        frame_bgr = preprocess_frame_for_yolo(frame)

        # YOLO inference
        detections = yolo_infer_frame(model, frame_bgr, img_size, device)

        # Draw detections
        drawn = draw_detections(frame_bgr.copy(), detections, conf_threshold=conf_thres)

        # FPS overlay
        fps = fps_counter.tick()
        drawn = fps_counter.put_fps_on_frame(drawn, fps)

        # Show frame in notebook (override previous)
        clear_output(wait=True)
        cv2_imshow(drawn)

        frame_id += 1

    cap.release()
    print("[INFO] Streaming finished.")

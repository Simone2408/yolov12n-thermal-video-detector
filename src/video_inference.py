# src/video_inference.py

import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from .utils import draw_detections, FPSCounter


def yolo_infer_frame(model, frame_bgr: np.ndarray, img_size: int = 640, device: str = "cuda"):
    """
    Performs YOLO inference on a BGR frame and returns detections in Nx6 format:
    [x1, y1, x2, y2, conf, cls]
    """
    # Ultralytics accetta direttamente un np.ndarray BGR
    results = model(frame_bgr, imgsz=img_size, device=device, verbose=False)

    # Se results è una lista/tupla, prendiamo il primo elemento
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
    Real-time style streaming in Google Colab:

    - legge il video termico frame-by-frame
    - usa direttamente i frame così come sono (nessun preprocessing termico)
    - esegue YOLO su ogni frame
    - disegna i box colorati
    - mostra il risultato nella cella Colab
    """
    # Import specifici di Colab
    try:
        from google.colab.patches import cv2_imshow
        from IPython.display import clear_output
    except ImportError:
        raise RuntimeError("run_inference_colab_stream è pensato per essere usato in Google Colab.")

    # Carica il modello YOLO con le API ufficiali Ultralytics
    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    print("[INFO] Loading YOLO model...")
    model = YOLO(str(weights_path))
    print(f"[INFO] YOLO model loaded from: {weights_path}")

    # Apri il video
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

        # 🔹 Nessun preprocessing: usiamo il frame così com'è
        frame_bgr = frame  # (BGR 8-bit da OpenCV)

        # YOLO inference
        detections = yolo_infer_frame(model, frame_bgr, img_size, device)

        # Disegna i box
        drawn = draw_detections(frame_bgr.copy(), detections, conf_threshold=conf_thres)

        # Overlay FPS
        fps = fps_counter.tick()
        drawn = fps_counter.put_fps_on_frame(drawn, fps)

        # Mostra il frame nella cella (sovrascrivendo il precedente)
        clear_output(wait=True)
        cv2_imshow(drawn)

        frame_id += 1

    cap.release()
    print("[INFO] Streaming finished.")

def run_inference_to_file(
    weights: str = "weights/best.pt",
    source: str = "",
    output_path: str = "outputs/output.mp4",
    img_size: int = 640,
    conf_thres: float = 0.25,
    device: str = "cuda",
):
    """
    Runs YOLO inference on a video and saves the annotated output to a file.
    """

    from ultralytics import YOLO

    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    fps_counter = FPSCounter()

    print("[INFO] Starting offline inference (writing to file)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_bgr = frame

        detections = yolo_infer_frame(model, frame_bgr, img_size, device)
        drawn = draw_detections(frame_bgr.copy(), detections, conf_threshold=conf_thres)

        fps = fps_counter.tick()
        drawn = fps_counter.put_fps_on_frame(drawn, fps)

        out.write(drawn)

    cap.release()
    out.release()
    print(f"[INFO] Saved output video to: {output_path}")

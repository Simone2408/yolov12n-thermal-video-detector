# src/video_inference.py

import argparse
from pathlib import Path

import cv2
import numpy as np

from .model_loader import load_model
from .thermal_preprocess import preprocess_frame_for_yolo
from .utils import draw_detections, FPSCounter


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv12n Thermal Video Inference (Ultralytics)")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input thermal video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.mp4",
        help="Path to save output video (offline mode)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="file",
        choices=["file", "colab_stream"],
        help="Inference mode: 'file' (save video) or 'colab_stream' (real-time in Colab)",
    )
    return parser.parse_args()


def yolo_infer_frame(model, frame_bgr: np.ndarray, img_size: int, device: str):
    """
    Run a single YOLO inference on a BGR frame using Ultralytics API.

    Returns:
        detections: Nx6 array [x1, y1, x2, y2, conf, cls]
        frame_bgr: the same processed frame (for drawing)
    """
    # Ultralytics YOLO accepts numpy BGR images directly
    results = model(
        frame_bgr,
        imgsz=img_size,
        device=device,
        verbose=False,
    )

    # results is usually a list-like, take first element
    if isinstance(results, (list, tuple)):
        results = results[0]

    detections = None
    if results and results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        conf = boxes.conf.cpu().numpy()  # (N,)
        cls = boxes.cls.cpu().numpy()    # (N,)

        detections = np.concatenate(
            [xyxy, conf[:, None], cls[:, None]],
            axis=1
        )

    return detections, frame_bgr


def run_inference_to_file(
    weights: str = "weights/best.pt",
    source: str = "",
    output: str = "outputs/result.mp4",
    img_size: int = 640,
    conf_thres: float = 0.25,
    device: str = "cuda",
):
    """
    Offline mode:
    - read video from file
    - run YOLO on each frame
    - save annotated video to disk
    """
    print("[INFO] Loading YOLO model...")
    model = load_model(weights)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output, fourcc, fps_src, (width, height))

    fps_counter = FPSCounter()

    print("[INFO] Starting offline inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thermal preprocessing -> BGR frame for YOLO
        proc_frame = preprocess_frame_for_yolo(frame)

        detections, proc_frame = yolo_infer_frame(model, proc_frame, img_size, device)

        drawn = draw_detections(proc_frame.copy(), detections, conf_threshold=conf_thres)
        fps = fps_counter.tick()
        drawn = fps_counter.put_fps_on_frame(drawn, fps)

        out.write(drawn)

    cap.release()
    out.release()
    print(f"[INFO] Offline inference complete. Saved to: {output}")


def run_inference_colab_stream(
    weights: str = "weights/best.pt",
    source: str = "",
    img_size: int = 640,
    conf_thres: float = 0.25,
    device: str = "cuda",
):
    """
    Colab streaming mode:
    - read thermal video
    - preprocess each frame
    - run YOLO
    - display annotated frame in the cell (quasi real-time)
    """
    # Imports specific to Colab
    try:
        from google.colab.patches import cv2_imshow
        from IPython.display import clear_output
    except ImportError:
        raise ImportError(
            "run_inference_colab_stream is intended to be used inside Google Colab."
        )

    print("[INFO] Loading YOLO model...")
    model = load_model(weights)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source_path}")

    fps_counter = FPSCounter()

    print("[INFO] Starting real-time streaming in Colab...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Thermal preprocessing -> BGR for YOLO and display
        proc_frame = preprocess_frame_for_yolo(frame)

        detections, proc_frame = yolo_infer_frame(model, proc_frame, img_size, device)

        drawn = draw_detections(proc_frame.copy(), detections, conf_threshold=conf_thres)
        fps = fps_counter.tick()
        drawn = fps_counter.put_fps_on_frame(drawn, fps)

        # Show in cell (overwrite previous frame)
        clear_output(wait=True)
        cv2_imshow(drawn)
        frame_idx += 1

    cap.release()
    print("[INFO] Streaming finished.")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "file":
        run_inference_to_file(
            weights=args.weights,
            source=args.source,
            output=args.output,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            device=args.device,
        )
    else:
        run_inference_colab_stream(
            weights=args.weights,
            source=args.source,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            device=args.device,
        )

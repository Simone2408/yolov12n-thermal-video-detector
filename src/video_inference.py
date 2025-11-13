# src/video_inference.py

import argparse
from pathlib import Path

import cv2
import torch

from .model_loader import load_model
from .thermal_preprocess import preprocess_for_model
from .utils import draw_detections, FPSCounter


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv12n Thermal Video Inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/best.pt",
        help="Percorso ai pesi del modello",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Percorso al video sorgente (termico)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.mp4",
        help="Percorso di salvataggio del video di output",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Dimensione di input del modello (lato)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="Soglia di confidenza per visualizzare le predizioni",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo: cuda o cpu",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="file",
        choices=["file", "colab_stream"],
        help="Modalità: 'file' per salvare video, 'colab_stream' per streaming in Colab",
    )
    return parser.parse_args()


def postprocess_predictions(preds, img_size, width, height):
    """
    Adatta le predizioni dal sistema di coordinate del modello (img_size x img_size)
    alla risoluzione originale del frame (width x height).

    Si assume che preds sia un array Nx6:
        [x1, y1, x2, y2, score, class_id]
    """
    if preds is None:
        return None

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()

    if len(preds) == 0:
        return preds

    preds = preds.copy()
    scale_x = width / float(img_size)
    scale_y = height / float(img_size)

    preds[:, 0] *= scale_x
    preds[:, 2] *= scale_x
    preds[:, 1] *= scale_y
    preds[:, 3] *= scale_y

    return preds


def _infer_frame(model, frame, img_size, device):
    """
    Un singolo passo di inferenza su un frame:
    - preprocess termico
    - forward del modello
    - ritorna le predizioni grezze
    """
    tensor = preprocess_for_model(frame, img_size=img_size, device=device)

    with torch.no_grad():
        raw_preds = model(tensor)

        # Se il tuo modello restituisce più output, prendi quello giusto
        if isinstance(raw_preds, (list, tuple)):
            raw_preds = raw_preds[0]

    return raw_preds


def run_inference_to_file(
    weights: str = "weights/best.pt",
    source: str = "",
    output: str = "outputs/result.mp4",
    img_size: int = 640,
    conf_thres: float = 0.25,
    device: str = "cuda",
):
    """
    Modalità "classica": legge un video, fa inferenza frame-by-frame,
    salva il risultato in un nuovo MP4 con bounding box e FPS.
    """
    # Device check
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA non disponibile, uso CPU.")
        device = "cpu"

    print(f"[INFO] Caricamento modello da {weights} su device: {device}")
    model = load_model(weights, device=device)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video sorgente non trovato: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {source_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output, fourcc, fps_src, (width, height))

    fps_counter = FPSCounter()

    print("[INFO] Inizio inferenza video (salvataggio su file)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_preds = _infer_frame(model, frame, img_size, device)
        preds_scaled = postprocess_predictions(raw_preds, img_size, width, height)

        frame_drawn = draw_detections(frame.copy(), preds_scaled, conf_threshold=conf_thres)

        fps = fps_counter.tick()
        frame_drawn = fps_counter.put_fps_on_frame(frame_drawn, fps)

        out.write(frame_drawn)

    cap.release()
    out.release()
    print(f"[INFO] Inferenza completata. Video salvato in: {output}")


def run_inference_colab_stream(
    weights: str = "weights/best.pt",
    source: str = "",
    img_size: int = 640,
    conf_thres: float = 0.25,
    device: str = "cuda",
):
    """
    Modalità pensata per Google Colab:
    - mostra il video direttamente nella cella
    - aggiorna il frame ad ogni iterazione
    - effetto “quasi real time” mentre il modello annota

    Nota: in Colab l'aggiornamento non sarà perfetto come una GUI locale,
    ma l'esperienza è molto più "live" rispetto a salvare e poi aprire il video.
    """
    # Import specifici per Colab
    try:
        from google.colab.patches import cv2_imshow
        from IPython.display import clear_output
    except ImportError:
        raise ImportError(
            "run_inference_colab_stream è pensato per Google Colab. "
            "Assicurati di eseguirlo in un notebook Colab."
        )

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA non disponibile, uso CPU.")
        device = "cpu"

    print(f"[INFO] Caricamento modello da {weights} su device: {device}")
    model = load_model(weights, device=device)

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video sorgente non trovato: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire il video: {source_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_counter = FPSCounter()

    print("[INFO] Inizio inferenza video in streaming su Colab...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_preds = _infer_frame(model, frame, img_size, device)
        preds_scaled = postprocess_predictions(raw_preds, img_size, width, height)

        frame_drawn = draw_detections(frame.copy(), preds_scaled, conf_threshold=conf_thres)

        fps = fps_counter.tick()
        frame_drawn = fps_counter.put_fps_on_frame(frame_drawn, fps)

        # Mostra il frame nella cella, aggiornando
        clear_output(wait=True)
        cv2_imshow(frame_drawn)
        frame_idx += 1

    cap.release()
    print("[INFO] Streaming terminato.")
    

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
        # Modalità streaming Colab da terminale ha poco senso,
        # ma lasciamo la possibilità.
        run_inference_colab_stream(
            weights=args.weights,
            source=args.source,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            device=args.device,
        )

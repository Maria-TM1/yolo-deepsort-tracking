#!/usr/bin/env python3
"""YOLO + DeepSORT con ciclo 4+1 y filtrado de clases reales.

Con --skip_frames 4:
- YOLO se ejecuta en 4 fotogramas consecutivos.
- En el quinto fotograma no se ejecuta YOLO; DeepSORT predice con Kalman.
- No se elimina ningún fotograma del vídeo de salida.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

REAL_CLASS_NAMES = {
    "bicycle",
    "bus",
    "car",
    "human",
    "motorcycle",
    "trafficcone",
    "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_model", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--conf_thres", type=float, default=0.35)
    parser.add_argument("--max_age", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=2)
    parser.add_argument("--max_cosine_distance", type=float, default=0.2)
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=4,
        help="Fotogramas consecutivos con YOLO antes de uno solo con predicción Kalman.",
    )
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def xyxy_to_xywh(box: np.ndarray) -> list[int]:
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def main() -> None:
    args = parse_args()

    if args.skip_frames < 1:
        raise ValueError("--skip_frames debe ser >= 1")
    if not args.yolo_model.exists():
        raise FileNotFoundError(f"No existe el modelo: {args.yolo_model}")
    if not args.video_path.exists():
        raise FileNotFoundError(f"No existe el vídeo: {args.video_path}")

    yolo = YOLO(str(args.yolo_model))
    names = yolo.names

    if isinstance(names, dict):
        allowed_class_ids = {
            int(class_id)
            for class_id, class_name in names.items()
            if str(class_name).lower() in REAL_CLASS_NAMES
        }
    else:
        allowed_class_ids = {
            class_id
            for class_id, class_name in enumerate(names)
            if str(class_name).lower() in REAL_CLASS_NAMES
        }

    if not allowed_class_ids:
        raise RuntimeError(f"No se encontraron las clases reales. Clases disponibles: {names}")

    print("Clases permitidas:", flush=True)
    for class_id in sorted(allowed_class_ids):
        print(f"  {class_id}: {names[class_id]}", flush=True)

    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.max_cosine_distance,
    )

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear: {args.output_path}")

    frame_idx = 0
    cycle_length = args.skip_frames + 1

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            run_detection = (frame_idx % cycle_length) < args.skip_frames

            if run_detection:
                results = yolo(frame, verbose=False)
                boxes = results[0].boxes

                boxes_xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                keep = (confs >= args.conf_thres) & np.isin(
                    class_ids, list(allowed_class_ids)
                )

                detections = [
                    (xyxy_to_xywh(box), float(conf), int(class_id))
                    for box, conf, class_id in zip(
                        boxes_xyxy[keep], confs[keep], class_ids[keep], strict=True
                    )
                ]

                tracks = tracker.update_tracks(detections, frame=frame)
            else:
                # Mantiene todos los fotogramas del vídeo y predice solo este fotograma.
                tracks = tracker.update_tracks([], frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                # Evita dibujar tracks antiguos no asociados, que producían cajas duplicadas.
                if run_detection and track.time_since_update != 0:
                    continue
                if not run_detection and track.time_since_update > 1:
                    continue

                left, top, right, bottom = map(int, track.to_ltrb())
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{track.track_id}",
                    (left, max(20, top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            writer.write(frame)

            if args.display:
                cv2.imshow("YOLO + DeepSORT", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0 or frame_idx == total_frames:
                print(f"Procesados {frame_idx}/{total_frames} fotogramas", flush=True)

    finally:
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Vídeo guardado en: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

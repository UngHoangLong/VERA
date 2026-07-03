"""
Annotate a video using the exact same face detection logic as Module 1 (AdvancedFaceCropper).

Usage:
    python scripts/debug_face_detection.py data/raw/genuine/04PmEJaYKd0_16_3.mp4
    python scripts/debug_face_detection.py data/raw/genuine/04PmEJaYKd0_16_3.mp4 --out /tmp/debug.mp4
"""

import argparse
import sys
import os
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.face_crop import AdvancedFaceCropper

GREEN  = (0, 200,   0)
RED    = (0,   0, 200)
YELLOW = (0, 200, 200)
BLACK  = (0,   0,   0)


def annotate_video(video_path: Path, out_path: Path):
    cropper = AdvancedFaceCropper()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    frame_idx   = 0
    fatal_count = 0
    last_bbox   = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_img, bbox, landmarks, is_real, is_fatal = cropper.process_frame(frame, fallback_bbox=last_bbox)

        if is_fatal:
            fatal_count += 1
            banner_color = RED
            status_text  = f"Frame {frame_idx:03d}  |  FATAL"
        elif face_img is not None:
            last_bbox = bbox
            if bbox is not None:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), GREEN, 2)
            banner_color = GREEN
            status_text  = f"Frame {frame_idx:03d}  |  OK  {'(real detect)' if is_real else '(fallback bbox)'}"
        else:
            banner_color = YELLOW
            status_text  = f"Frame {frame_idx:03d}  |  no face"

        cv2.rectangle(frame, (0, 0), (W, 28), banner_color, -1)
        cv2.putText(frame, status_text, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Total frames : {frame_idx}")
    print(f"FATAL frames : {fatal_count}  ({100*fatal_count/max(frame_idx,1):.1f}%)")
    print(f"Output       : {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path   = Path(args.out) if args.out else video_path.with_name(video_path.stem + "_debug.mp4")

    annotate_video(video_path, out_path)


if __name__ == "__main__":
    main()

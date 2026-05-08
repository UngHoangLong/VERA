import argparse
from pathlib import Path
import json
import math
import numpy as np
import cv2


class MP468:
    NOSE_TIP = 1
    UPPER_LIP = 13
    LOWER_LIP = 14
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]


COLORS = {
    "bbox": (255, 255, 0),
    "left_eye": (0, 255, 0),
    "right_eye": (0, 200, 255),
    "nose": (0, 0, 255),
    "mouth_left": (255, 0, 255),
    "mouth_right": (255, 0, 255),
    "mouth_center": (255, 255, 255),
    "upper_lip": (180, 0, 255),
    "lower_lip": (180, 0, 255),
}


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def finite_mask(lm: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(lm[:, :2]), axis=1)


def point(lm: np.ndarray, idx: int):
    pt = lm[idx, :2]
    return pt.astype(np.float32) if np.all(np.isfinite(pt)) else None


def center(lm: np.ndarray, indices):
    pts = lm[indices, :2]
    valid = np.all(np.isfinite(pts), axis=1)
    if not np.any(valid):
        return None
    return np.mean(pts[valid], axis=0).astype(np.float32)


def infer_bbox_from_landmarks(lm: np.ndarray):
    valid = finite_mask(lm)
    if not np.any(valid):
        return None
    pts = lm[valid, :2]
    x1, y1 = np.min(pts, axis=0)
    x2, y2 = np.max(pts, axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def expand_bbox_xyxy(bbox: np.ndarray, margin_ratio: float):
    x1, y1, x2, y2 = bbox.tolist()
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    mx = w * margin_ratio
    my = h * margin_ratio
    return np.array([x1 - mx, y1 - my, x2 + mx, y2 + my], dtype=np.float32)


def map_orig_to_crop(pt: np.ndarray, crop_bbox_xyxy: np.ndarray, out_w: int, out_h: int):
    x1, y1, x2, y2 = crop_bbox_xyxy.tolist()
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    x = (pt[0] - x1) * out_w / w
    y = (pt[1] - y1) * out_h / h
    return np.array([x, y], dtype=np.float32)


def draw_point(img, pt, label, color):
    if pt is None:
        return
    x, y = int(round(pt[0])), int(round(pt[1]))
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (x, y), 3, color, -1)
        cv2.putText(img, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_bbox(img, bbox_xyxy, color):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy.tolist()]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def build_debug_frame(face_img: np.ndarray, lm: np.ndarray, margin_ratio: float):
    debug = face_img.copy()
    out_h, out_w = debug.shape[:2]

    bbox = infer_bbox_from_landmarks(lm)
    if bbox is None:
        cv2.putText(debug, "No valid landmarks", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return debug

    crop_bbox = expand_bbox_xyxy(bbox, margin_ratio)

    # Convert key points from original landmark coordinates to crop coordinates.
    raw = {
        "left_eye": center(lm, MP468.LEFT_EYE),
        "right_eye": center(lm, MP468.RIGHT_EYE),
        "nose": point(lm, MP468.NOSE_TIP),
        "mouth_left": point(lm, MP468.MOUTH_LEFT),
        "mouth_right": point(lm, MP468.MOUTH_RIGHT),
        "upper_lip": point(lm, MP468.UPPER_LIP),
        "lower_lip": point(lm, MP468.LOWER_LIP),
    }
    raw["mouth_center"] = None if raw["mouth_left"] is None or raw["mouth_right"] is None else (raw["mouth_left"] + raw["mouth_right"]) / 2.0

    mapped = {k: (None if v is None else map_orig_to_crop(v, crop_bbox, out_w, out_h)) for k, v in raw.items()}

    # In crop coordinates, the crop itself spans the whole image.
    draw_bbox(debug, np.array([0, 0, out_w - 1, out_h - 1], dtype=np.float32), COLORS["bbox"])
    for key, pt2 in mapped.items():
        draw_point(debug, pt2, key, COLORS[key])

    # Extra guide text.
    cv2.putText(debug, "Check if eyes/nose/mouth lie on the correct face parts", (10, out_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return debug


def make_contact_sheet(images, cols=3, pad=8, bg=(30, 30, 30)):
    if not images:
        return None
    h, w = images[0].shape[:2]
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    sheet = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, dtype=np.uint8)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y+h, x:x+w] = img
    return sheet


def process_slide(face_file: Path, landmark_file: Path, out_dir: Path, margin_ratio: float, max_frames: int):
    faces = load_array(face_file)
    landmarks = load_array(landmark_file)

    num = min(len(faces), len(landmarks), max_frames if max_frames > 0 else len(faces))
    vis_frames = []
    for i in range(num):
        face = faces[i]
        lm = landmarks[i]
        vis = build_debug_frame(face, lm, margin_ratio)
        # Tag frame index.
        cv2.putText(vis, f"frame={i}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        vis_frames.append(vis)
        cv2.imwrite(str(out_dir / f"{face_file.stem}_frame_{i:02d}.png"), vis)

    sheet = make_contact_sheet(vis_frames)
    if sheet is not None:
        cv2.imwrite(str(out_dir / f"{face_file.stem}_contact_sheet.png"), sheet)


def discover_pairs(interim_root: Path):
    for video_dir in sorted([p for p in interim_root.iterdir() if p.is_dir()]):
        for chunk_dir in sorted([p for p in video_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")]):
            slides_dir = chunk_dir / "slides"
            if not slides_dir.exists():
                continue
            face_files = sorted(slides_dir.glob("slide_*_faces.npy"))
            for face_file in face_files:
                landmark_file = slides_dir / face_file.name.replace("_faces.npy", "_landmarks.npy")
                if landmark_file.exists():
                    yield video_dir.name, chunk_dir.name, face_file, landmark_file


def main():
    parser = argparse.ArgumentParser(description="Visualize eye, nose, and mouth coordinates on cropped faces to verify Module 2 inputs.")
    parser.add_argument("interim_root", type=str, help="Path to data/interim")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save debug images")
    parser.add_argument("--margin_ratio", type=float, default=0.2, help="Must match face_crop.py margin_ratio used in Module 1")
    parser.add_argument("--max_frames", type=int, default=4, help="How many frames per slide to visualize")
    parser.add_argument("--max_slides", type=int, default=0, help="How many slides total to visualize; 0 = all")
    args = parser.parse_args()

    interim_root = Path(args.interim_root)
    out_dir = Path(args.output_dir) if args.output_dir else interim_root / "_debug_landmark_projection"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "interim_root": str(interim_root),
        "output_dir": str(out_dir),
        "margin_ratio": args.margin_ratio,
        "max_frames": args.max_frames,
        "slides": []
    }

    count = 0
    for video_name, chunk_name, face_file, landmark_file in discover_pairs(interim_root):
        if args.max_slides > 0 and count >= args.max_slides:
            break
        slide_out = out_dir / video_name / chunk_name
        slide_out.mkdir(parents=True, exist_ok=True)
        process_slide(face_file, landmark_file, slide_out, args.margin_ratio, args.max_frames)
        manifest["slides"].append({
            "video": video_name,
            "chunk": chunk_name,
            "face_file": str(face_file),
            "landmark_file": str(landmark_file),
            "output_dir": str(slide_out),
        })
        count += 1

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved debug images for {count} slide(s) to: {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


Number = float


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def recursive_find_number(obj: Any, candidate_keys: Iterable[str]) -> Optional[float]:
    candidate_keys = set(candidate_keys)

    def _search(x: Any) -> Optional[float]:
        if isinstance(x, dict):
            for k in candidate_keys:
                if k in x and is_number(x[k]):
                    return float(x[k])
            for v in x.values():
                found = _search(v)
                if found is not None:
                    return found
        elif isinstance(x, list):
            for item in x:
                found = _search(item)
                if found is not None:
                    return found
        return None

    return _search(obj)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def minmax_normalize_map(raw_map: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    if not raw_map:
        return {}
    vals = list(raw_map.values())
    vmin = min(vals)
    vmax = max(vals)
    if math.isclose(vmin, vmax):
        return {k: 1.0 for k in raw_map}
    return {k: (v - vmin) / (vmax - vmin) for k, v in raw_map.items()}


def parse_chunk_meta(chunk_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    video_id = chunk_obj.get("video_id")
    chunk_id = chunk_obj.get("chunk_id")

    if video_id is None:
        video_id = recursive_find_number(chunk_obj, [])  # no-op to keep structure symmetric
    return video_id, chunk_id


def iter_chunk_jsons(root: Path) -> Iterable[Tuple[str, str, Path, Dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")

    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue
        video_id = video_dir.name
        for json_path in sorted(video_dir.glob("*.json")):
            if json_path.name in {"summary.json", "manifest.json"}:
                continue
            chunk_id = json_path.stem
            data = load_json(json_path)
            yield video_id, chunk_id, json_path, data


def load_scfd_from_root(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    score_keys = [
        "scfd_score",
        "semantic_consistency_score",
        "score",
        "video_score",
        "consistency_score",
        "semantic_score",
        "chunk_score",
    ]
    for video_id, chunk_id, path, data in iter_chunk_jsons(root):
        raw = recursive_find_number(data, score_keys)
        if raw is None:
            raise ValueError(f"Không tìm thấy score SCFD trong: {path}")
        out[(video_id, chunk_id)] = {
            "video_id": video_id,
            "chunk_id": chunk_id,
            "scfd_score_raw": float(raw),
            "source_path": str(path),
        }
    return out


def load_ccfd_from_root(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    wer_keys = ["wer", "word_error_rate", "ccfd_wer", "wer_raw"]
    score_keys = ["ccfd_score", "content_consistency_score", "score", "video_score", "chunk_score"]

    for video_id, chunk_id, path, data in iter_chunk_jsons(root):
        wer = recursive_find_number(data, wer_keys)
        score = recursive_find_number(data, score_keys)
        item: Dict[str, Any] = {
            "video_id": video_id,
            "chunk_id": chunk_id,
            "source_path": str(path),
        }
        if wer is not None:
            item["ccfd_wer_raw"] = float(wer)
            item["ccfd_score_raw"] = 1.0 - min(float(wer), 1.0)
            item["ccfd_score_source"] = "1 - min(WER, 1)"
        elif score is not None:
            item["ccfd_score_raw"] = clamp01(float(score))
            item["ccfd_score_source"] = "precomputed_score_clamped"
        else:
            raise ValueError(f"Không tìm thấy WER/score CCFD trong: {path}")
        out[(video_id, chunk_id)] = item
    return out


def load_tcfd_from_json(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    data = load_json(path)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    if isinstance(data, dict) and "videos" in data:
        for video_obj in data.get("videos", []):
            video_id = video_obj.get("video_id")
            if not video_id:
                continue
            for chunk in video_obj.get("chunks", []):
                chunk_id = chunk.get("chunk_id")
                if not chunk_id:
                    continue
                raw = None
                for k in ("tcfd_score", "temporal_consistency_score", "score", "video_score", "chunk_score"):
                    if is_number(chunk.get(k)):
                        raw = float(chunk[k])
                        break
                if raw is None:
                    continue
                out[(video_id, chunk_id)] = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "tcfd_score_raw": raw,
                    "num_windows": chunk.get("num_windows"),
                    "chunk_start_sec": chunk.get("chunk_start_sec"),
                    "chunk_end_sec": chunk.get("chunk_end_sec"),
                    "source_path": str(path),
                }
        return out

    if isinstance(data, dict) and "results" in data:
        for item in data.get("results", []):
            video_id = item.get("video_id")
            chunk_id = item.get("chunk_id")
            raw = recursive_find_number(item, ["tcfd_score", "temporal_consistency_score", "score", "video_score", "chunk_score"])
            if video_id and chunk_id and raw is not None:
                out[(video_id, chunk_id)] = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "tcfd_score_raw": float(raw),
                    "source_path": str(path),
                }
        return out

    raise ValueError(f"Không nhận diện được format TCFD JSON: {path}")


def load_generic_json_map(path: Path, score_keys: Iterable[str], field_name: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    data = load_json(path)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add_item(obj: Dict[str, Any], default_video_id: Optional[str] = None) -> None:
        video_id = obj.get("video_id", default_video_id)
        chunk_id = obj.get("chunk_id")
        raw = recursive_find_number(obj, score_keys)
        if video_id and chunk_id and raw is not None:
            out[(video_id, chunk_id)] = {
                "video_id": video_id,
                "chunk_id": chunk_id,
                field_name: float(raw),
                "source_path": str(path),
            }

    if isinstance(data, dict) and "videos" in data:
        for video_obj in data.get("videos", []):
            video_id = video_obj.get("video_id")
            for chunk in video_obj.get("chunks", []):
                add_item(chunk, default_video_id=video_id)
        return out

    if isinstance(data, dict) and "results" in data:
        for item in data.get("results", []):
            add_item(item)
        return out

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                add_item(item)
        return out

    raise ValueError(f"Không nhận diện được format JSON: {path}")


def group_by_video(flat_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in flat_items:
        grouped[item["video_id"]].append(item)

    videos: List[Dict[str, Any]] = []
    for video_id in sorted(grouped):
        chunks = sorted(grouped[video_id], key=lambda x: x["chunk_id"])
        scores = [c["fusion_score"] for c in chunks if is_number(c.get("fusion_score"))]
        anomalies = [c["fusion_anomaly"] for c in chunks if is_number(c.get("fusion_anomaly"))]
        videos.append({
            "video_id": video_id,
            "num_chunks": len(chunks),
            "fusion_score_mean": float(sum(scores) / len(scores)) if scores else None,
            "fusion_anomaly_mean": float(sum(anomalies) / len(anomalies)) if anomalies else None,
            "chunks": chunks,
        })
    return videos


def summarize(flat_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [x["fusion_score"] for x in flat_items if is_number(x.get("fusion_score"))]
    return {
        "num_chunks_total": len(flat_items),
        "num_chunks_ok": sum(1 for x in flat_items if x.get("status") == "ok"),
        "num_chunks_missing": sum(1 for x in flat_items if x.get("status") != "ok"),
        "fusion_score_mean": float(sum(scores) / len(scores)) if scores else None,
        "fusion_score_std": (
            float((sum((s - (sum(scores) / len(scores))) ** 2 for s in scores) / len(scores)) ** 0.5)
            if scores else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="System fusion for module 2.2 (SCFD + TCFD + CCFD)")
    parser.add_argument("--scfd-root", type=str, default=None, help="Thư mục output SCFD dạng per-video/per-chunk")
    parser.add_argument("--ccfd-root", type=str, default=None, help="Thư mục output CCFD dạng per-video/per-chunk")
    parser.add_argument("--tcfd-json", type=str, default=None, help="File output TCFD JSON")
    parser.add_argument("--scfd-json", type=str, default=None, help="File SCFD JSON dạng gộp nếu có")
    parser.add_argument("--ccfd-json", type=str, default=None, help="File CCFD JSON dạng gộp nếu có")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--require-all-three", action="store_true")
    args = parser.parse_args()

    if not any([args.scfd_root, args.scfd_json]):
        raise ValueError("Cần cung cấp --scfd-root hoặc --scfd-json")
    if not args.tcfd_json:
        raise ValueError("Cần cung cấp --tcfd-json")
    if not any([args.ccfd_root, args.ccfd_json]):
        raise ValueError("Cần cung cấp --ccfd-root hoặc --ccfd-json")

    if args.scfd_root:
        scfd_map = load_scfd_from_root(Path(args.scfd_root))
    else:
        scfd_map = load_generic_json_map(Path(args.scfd_json), ["scfd_score", "semantic_consistency_score", "score"], "scfd_score_raw")

    if args.ccfd_root:
        ccfd_map = load_ccfd_from_root(Path(args.ccfd_root))
    else:
        raw_ccfd = load_json(Path(args.ccfd_json))
        # Convert generic ccfd json to same shape
        ccfd_map = {}
        generic_map = load_generic_json_map(Path(args.ccfd_json), ["ccfd_score", "content_consistency_score", "score", "wer", "word_error_rate"], "tmp")
        for key, item in generic_map.items():
            # if generic hit WER, can't distinguish reliably; keep clamped score behavior
            ccfd_map[key] = {
                "video_id": item["video_id"],
                "chunk_id": item["chunk_id"],
                "ccfd_score_raw": clamp01(float(item["tmp"])),
                "ccfd_score_source": "generic_json_fallback",
                "source_path": item.get("source_path"),
            }

    tcfd_map = load_tcfd_from_json(Path(args.tcfd_json))

    scfd_norm = minmax_normalize_map({k: v["scfd_score_raw"] for k, v in scfd_map.items()})
    tcfd_norm = minmax_normalize_map({k: v["tcfd_score_raw"] for k, v in tcfd_map.items()})

    keys_union = set(scfd_map) | set(tcfd_map) | set(ccfd_map)
    keys_intersection = set(scfd_map) & set(tcfd_map) & set(ccfd_map)
    keys = sorted(keys_intersection if args.require_all_three else keys_union)

    flat_items: List[Dict[str, Any]] = []
    for key in keys:
        video_id, chunk_id = key
        item: Dict[str, Any] = {"video_id": video_id, "chunk_id": chunk_id}

        if key in scfd_map:
            item.update({k: v for k, v in scfd_map[key].items() if k not in {"video_id", "chunk_id"}})
            item["scfd_score_norm"] = scfd_norm[key]
        else:
            item["scfd_score_raw"] = None
            item["scfd_score_norm"] = None

        if key in tcfd_map:
            item.update({k: v for k, v in tcfd_map[key].items() if k not in {"video_id", "chunk_id", "source_path"}})
            item["tcfd_score_norm"] = tcfd_norm[key]
        else:
            item["tcfd_score_raw"] = None
            item["tcfd_score_norm"] = None

        if key in ccfd_map:
            for kk, vv in ccfd_map[key].items():
                if kk not in {"video_id", "chunk_id", "source_path"}:
                    item[kk] = vv
            item["ccfd_score_norm"] = clamp01(float(ccfd_map[key]["ccfd_score_raw"]))
        else:
            item["ccfd_score_raw"] = None
            item["ccfd_score_norm"] = None

        available = [
            item.get("scfd_score_norm"),
            item.get("tcfd_score_norm"),
            item.get("ccfd_score_norm"),
        ]
        available = [float(x) for x in available if is_number(x)]

        if args.require_all_three and len(available) < 3:
            item["status"] = "missing_branch"
            item["fusion_score"] = None
            item["fusion_anomaly"] = None
        elif not available:
            item["status"] = "missing_branch"
            item["fusion_score"] = None
            item["fusion_anomaly"] = None
        else:
            fusion = float(sum(available) / len(available))
            item["status"] = "ok"
            item["fusion_score"] = fusion
            item["fusion_anomaly"] = 1.0 - fusion

        flat_items.append(item)

    output = {
        "config": {
            "scfd_root": args.scfd_root,
            "scfd_json": args.scfd_json,
            "tcfd_json": args.tcfd_json,
            "ccfd_root": args.ccfd_root,
            "ccfd_json": args.ccfd_json,
            "output_json": args.output_json,
            "require_all_three": bool(args.require_all_three),
            "fusion_rule": "mean(normalized_scfd, normalized_tcfd, normalized_ccfd)",
            "normalization": {
                "scfd": "min-max",
                "tcfd": "min-max",
                "ccfd": "1 - min(WER, 1), or precomputed score clamped to [0,1] if WER missing",
            },
        },
        "summary": summarize(flat_items),
        "videos": group_by_video(flat_items),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "saved_to": str(out_path),
        "num_chunks_total": output["summary"]["num_chunks_total"],
        "num_chunks_ok": output["summary"]["num_chunks_ok"],
        "num_chunks_missing": output["summary"]["num_chunks_missing"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

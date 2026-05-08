import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

SLIDE_FACE_RE = re.compile(r'^(slide_\d+)_faces\.npy$')
SLIDE_LMK_RE = re.compile(r'^(slide_\d+)_landmarks\.npy$')

@dataclass
class PairInfo:
    slide_id: str
    faces_path: Path
    landmarks_path: Path

def collect_slide_pairs(slides_dir: Path) -> List[PairInfo]:
    face_map: Dict[str, Path] = {}
    lmk_map: Dict[str, Path] = {}
    for p in sorted(slides_dir.glob('*.npy')):
        m = SLIDE_FACE_RE.match(p.name)
        if m:
            face_map[m.group(1)] = p
            continue
        m = SLIDE_LMK_RE.match(p.name)
        if m:
            lmk_map[m.group(1)] = p

    pairs_with_idx: List[Tuple[int, PairInfo]] = []
    for slide_id, faces_path in sorted(face_map.items()):
        landmarks_path = lmk_map.get(slide_id)
        if landmarks_path is not None:
            idx_match = re.search(r'\d+', slide_id)
            idx = int(idx_match.group()) if idx_match else -1
            pairs_with_idx.append((idx, PairInfo(slide_id=slide_id, faces_path=faces_path, landmarks_path=landmarks_path)))

    if not pairs_with_idx:
        return []

    pairs_with_idx.sort(key=lambda x: x[0])
    longest_seq: List[PairInfo] = []
    current_seq: List[PairInfo] = [pairs_with_idx[0][1]]
    last_idx = pairs_with_idx[0][0]

    for i in range(1, len(pairs_with_idx)):
        curr_idx, pair_info = pairs_with_idx[i]
        if curr_idx == last_idx + 1:
            current_seq.append(pair_info)
        else:
            if len(current_seq) > len(longest_seq):
                longest_seq = current_seq
            current_seq = [pair_info]
        last_idx = curr_idx

    if len(current_seq) > len(longest_seq):
        longest_seq = current_seq

    return longest_seq
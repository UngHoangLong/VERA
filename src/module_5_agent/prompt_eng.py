"""
prompt_eng.py — Module 5 prompt construction.

Turns a Module 4 prompt_package.json into a multi-block prompt:

    System prompt : forensic-analyst role + visual checklist + metric glossary
    Block A       : video-level summary + temporal pattern
    Block B       : per top-K chunk evidence (compact NORMAL, full ELEVATED/CRITICAL)
    Block C       : 5-step reasoning (visual → metrics → cross-ref → verdict)
"""

import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.module_3_autoencoder.config import FEATURE_INTERPRETATIONS  # noqa: F401

METRIC_GLOSSARY = {
    "max_blur_flicker":      "frame-to-frame sharpness flicker; high -> synthesized/blended region",
    "blur_flicker_variance": "blur flicker instability; high -> intermittent blending artifacts",
    "max_texture_flicker":   "skin-texture flicker; high -> generated texture inconsistent across frames",
    "asymmetry_max":         "left-right facial texture asymmetry; high -> imperfect face blending",
    "max_blending_flicker":  "face-blend boundary flicker; high -> visible compositing seam",
    "blending_variance":     "blending artifact instability; high -> ongoing compositing seam",
    "mean_landmark_jitter":  "average landmark instability; high -> jittery non-physical facial motion",
    "max_kinematic_flicker": "facial-geometry flicker; high -> geometry snapping between frames",
    "max_rigid_violation":   "rigid motion violation; high -> face parts moving independently of head",
    "blinking_variance":     "blink activity variance; LOW -> absent blinking (synthesis giveaway)",
    "mouth_movement_variance": "mouth movement variance; LOW -> static mouth during speech (poor lip-sync)",
    "gaze_anomaly":          "gaze-pose mismatch; high -> eyes not tracking naturally with head",
    "iris_jitter_variance":  "iris position variance; LOW -> unnaturally still eyes (dead eye)",
    "wer_score":             "ASR vs VSR disagreement; high -> audio doesn't match lip motion",
    "semantic_anomaly":      "audio-visual semantic drop; high -> speech meaning misaligns with face",
    "min_cosine_anomaly":    "worst-point semantic mismatch; high -> clear mismatch moment",
    "temporal_anomaly":      "audio-visual temporal desync; high -> lips and sound out of phase",
    "min_temporal_anomaly":  "worst-point temporal desync; high -> clear out-of-sync moment",
    "temporal_sync_variance": "sync quality fluctuation; high -> unstable lip-sync",
    "vocal_jitter_relative": "vocal pitch micro-fluctuation; LOW -> too-perfect pitch (TTS/voice-cloning)",
    "vocal_shimmer_relative": "vocal amplitude micro-fluctuation; LOW -> too-perfect amplitude (synthetic voice)",
}

GROUP_1_VISUAL_ARTIFACTS = [
    "max_blur_flicker", "blur_flicker_variance",
    "max_texture_flicker", "asymmetry_max",
    "max_blending_flicker", "blending_variance",
]
GROUP_2_FACIAL_DYNAMICS = [
    "mean_landmark_jitter", "max_kinematic_flicker", "max_rigid_violation",
    "blinking_variance", "mouth_movement_variance",
    "gaze_anomaly", "iris_jitter_variance",
]
GROUP_3_AV_COHERENCE = [
    "wer_score", "semantic_anomaly", "min_cosine_anomaly",
    "temporal_anomaly", "min_temporal_anomaly", "temporal_sync_variance",
]
GROUP_4_AUDIO_ARTIFACTS = [
    "vocal_jitter_relative", "vocal_shimmer_relative",
]

GROUPS = [
    ("GROUP 1 — VISUAL ARTIFACTS",        GROUP_1_VISUAL_ARTIFACTS),
    ("GROUP 2 — FACIAL DYNAMICS",         GROUP_2_FACIAL_DYNAMICS),
    ("GROUP 3 — AUDIO-VISUAL COHERENCE",  GROUP_3_AV_COHERENCE),
    ("GROUP 4 — AUDIO ARTIFACTS",         GROUP_4_AUDIO_ARTIFACTS),
]

LOW_IS_SUSPICIOUS = {
    "vocal_jitter_relative", "vocal_shimmer_relative",
    "blinking_variance", "mouth_movement_variance", "iris_jitter_variance",
}


def _severity(name: str, signal: str) -> str:
    if name in LOW_IS_SUSPICIOUS:
        if signal == "FAR_BELOW_NORMAL":
            return "CRITICAL"
        if signal == "BELOW_NORMAL":
            return "ELEVATED"
        return "NORMAL"
    if signal == "FAR_ABOVE_NORMAL":
        return "CRITICAL"
    if signal == "ABOVE_NORMAL":
        return "ELEVATED"
    return "NORMAL"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_metric_glossary() -> str:
    lines = [
        "METRIC GLOSSARY (what each feature measures; direction stated explicitly).",
    ]
    for gname, gfeats in GROUPS:
        lines.append(f"\n{gname}")
        for fn in gfeats:
            desc = METRIC_GLOSSARY.get(fn)
            if desc:
                lines.append(f"  - {fn}: {desc}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a digital forensics analyst specializing in deepfake video detection.\n\n"

        "YOUR TASK: Analyze video frames AND automated metrics to determine if a video "
        "contains manipulated visual content (face-swap/reenactment), manipulated audio "
        "(voice cloning/TTS), or both.\n\n"

        "EVIDENCE SOURCES (use BOTH, do not rely on only one):\n"
        "1. VISUAL FRAMES — attached images from sampled chunks. Inspect them carefully for:\n"
        "   - Face-background boundary: blending seams, color/lighting mismatch\n"
        "   - Skin texture: unnatural smoothness, inconsistency across face regions\n"
        "   - Eyes: reflection consistency, symmetry, natural movement\n"
        "   - Temporal consistency: compare frames within and across chunks for flickering/morphing\n"
        "   - Overall naturalness: does the face look photorealistic or synthetic?\n"
        "2. AUTOMATED METRICS — per-chunk features compared against a genuine baseline.\n"
        "   Each metric has a severity tag:\n"
        "     CRITICAL = far into the suspicious direction\n"
        "     ELEVATED = moderately suspicious\n"
        "     NORMAL   = within genuine range\n"
        "   For most features HIGH is suspicious, but some are suspicious when LOW "
        "(blinking_variance, mouth_movement_variance, iris_jitter_variance, "
        "vocal_jitter_relative, vocal_shimmer_relative) — see glossary.\n\n"

        "CALIBRATION RULES:\n"
        "- A single CRITICAL metric can be noise. Multiple CRITICAL in the SAME group is a strong signal.\n"
        "- Genuine videos may show a few ELEVATED features — judge the whole picture.\n"
        "- Visual inspection can catch artifacts that metrics miss, and vice versa.\n"
        "- When evidence is weak or contradictory, lean toward UNCERTAIN.\n\n"

        + build_metric_glossary()
    )


# ---------------------------------------------------------------------------
# Block A — video summary
# ---------------------------------------------------------------------------

def build_block_a(video_id: str, summary: Dict[str, Any], top_chunks: List[Dict]) -> str:
    thr = summary.get("threshold")
    thr_str = f"{thr:.4f}" if isinstance(thr, (int, float)) else "n/a"
    lines = [
        "=" * 60,
        "VIDEO ANALYSIS REQUEST",
        "=" * 60,
        f"Video ID     : {video_id}",
        f"Duration     : {summary.get('video_duration_sec', 0)}s",
        f"Total chunks : {summary.get('total_chunks', 0)} analyzed",
        f"Threshold    : {thr_str} (95th percentile of genuine baseline)",
        "",
        "ANOMALY SCORE DISTRIBUTION:",
        f"  Mean score : {summary.get('mean_score', 0):.4f}",
        f"  Max score  : {summary.get('max_score', 0):.4f}",
        f"  Chunks above threshold: {summary.get('chunks_above_threshold', 0)} / {summary.get('total_chunks', 0)}",
        "",
        "TOP CHUNKS:",
    ]
    for c in top_chunks:
        tm = c.get("time_metadata", {})
        an = c.get("anomaly", {})
        start = tm.get("start_sec", 0.0)
        end = tm.get("end_sec", 0.0)
        score = an.get("joint_anomaly_score", 0.0)
        level = an.get("level", "?").upper()
        lines.append(f"  {c.get('chunk_id','?')}  [{start:.1f}s-{end:.1f}s]  score={score:.4f}  {level}")
    lines.append(f"\n  Temporal pattern: {summary.get('temporal_pattern', 'NONE')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block B — per-chunk evidence (compact NORMAL, full ELEVATED/CRITICAL)
# ---------------------------------------------------------------------------

def _fmt_val(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _feature_row(name: str, entry: Dict[str, Any]) -> str:
    val = entry.get("value")
    if val is None:
        return None
    signal = entry.get("signal", "NORMAL")
    severity = _severity(name, signal)
    if severity == "NORMAL":
        return None
    p50 = entry.get("genuine_p50")
    prank = entry.get("percentile_rank")
    p50s = _fmt_val(p50) if p50 is not None else "-"
    pranks = f"{prank}" if prank is not None else "-"
    return (f"  {name:<24} {_fmt_val(val):>10}  genP50={p50s:>8}  "
            f"pct={pranks:>3}  ** {severity} **")


def _normal_summary(names: List[str]) -> str:
    if not names:
        return ""
    return f"  [NORMAL] {', '.join(names)}"


def build_block_b(top_chunks: List[Dict]) -> str:
    blocks: List[str] = []
    for c in top_chunks:
        tm = c.get("time_metadata", {})
        an = c.get("anomaly", {})
        feats: Dict[str, Any] = {}
        feats.update(c.get("features", {}).get("visual", {}))
        feats.update(c.get("features", {}).get("audio_visual", {}))

        header = [
            "-" * 60,
            f"{c.get('chunk_id','?')}  |  Time: {tm.get('start_sec',0):.1f}s-{tm.get('end_sec',0):.1f}s"
            f"  |  Frames: {len(c.get('frame_files', []))} attached",
            f"Anomaly Score: {an.get('joint_anomaly_score',0):.4f}  "
            f"(level {an.get('level','?')})",
            f"Modalities analyzed: {', '.join(c.get('modalities_analyzed', [])) or 'none'}",
        ]
        if c.get("modalities_missing"):
            header.append(f"Modalities ABSENT: {', '.join(c['modalities_missing'])}")

        group_texts = []
        for gname, gfeats in GROUPS:
            elevated_rows = []
            normal_names = []
            for fn in gfeats:
                if fn not in feats:
                    continue
                entry = feats[fn]
                if not isinstance(entry, dict):
                    continue
                row = _feature_row(fn, entry)
                if row:
                    elevated_rows.append(row)
                elif entry.get("value") is not None:
                    normal_names.append(fn)

            if elevated_rows or normal_names:
                parts = [gname]
                parts.extend(elevated_rows)
                ns = _normal_summary(normal_names)
                if ns:
                    parts.append(ns)
                group_texts.append("\n".join(parts))

        blocks.append("\n".join(header) + "\n\n" + "\n\n".join(group_texts))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Block C — reasoning instructions + output format
# ---------------------------------------------------------------------------

def build_block_c() -> str:
    return (
        "=" * 60 + "\n"
        "TASK: Determine whether this video contains manipulated content.\n"
        "You must assess video manipulation and audio manipulation SEPARATELY.\n\n"

        "Follow these steps:\n\n"

        "STEP 1 — VISUAL INSPECTION:\n"
        "  Carefully examine the attached frames. For each chunk, describe:\n"
        "  - Face quality: skin texture, boundary with background, blending artifacts\n"
        "  - Eye region: reflections, symmetry, natural movement\n"
        "  - Temporal consistency: do frames within/across chunks look consistent?\n"
        "  - Any synthetic or unnatural appearance?\n\n"

        "STEP 2 — METRIC REVIEW:\n"
        "  Review ELEVATED and CRITICAL metrics across all chunks.\n"
        "  - GROUP 1+2 anomalies suggest VIDEO manipulation (face-swap, reenactment)\n"
        "  - GROUP 3 anomalies suggest lip-sync mismatch or dubbed audio\n"
        "  - GROUP 4 anomalies suggest AUDIO manipulation (voice cloning, TTS)\n\n"

        "STEP 3 — CROSS-REFERENCE:\n"
        "  Do visual findings agree with metrics? For example:\n"
        "  - Visual blending artifacts + CRITICAL blending metrics = strong face-swap signal\n"
        "  - Frames look natural but metrics are CRITICAL = possible subtle manipulation\n"
        "  - Frames look synthetic but metrics are NORMAL = metrics may have missed it\n\n"

        "STEP 4 — SEPARATE ASSESSMENT:\n"
        "  Based on all evidence, determine independently:\n"
        "  a) Is the VIDEO (visual) manipulated? (face-swap, reenactment, synthesis)\n"
        "  b) Is the AUDIO manipulated? (voice cloning, TTS, dubbed)\n\n"

        "STEP 5 — FINAL VERDICT:\n"
        "  - If either video or audio is manipulated → overall label is FAKE\n"
        "  - If neither is manipulated → GENUINE\n"
        "  - If unsure → UNCERTAIN\n\n"

        "Respond in EXACTLY this format:\n\n"

        "<verdict>\n"
        "{\n"
        '  "video_fake": true or false,\n'
        '  "audio_fake": true or false,\n'
        '  "label": "GENUINE | FAKE | UNCERTAIN",\n'
        '  "confidence": "LOW | MEDIUM | HIGH",\n'
        '  "video_manipulation_type": "FACE_SWAP | REENACTMENT | NONE | UNCERTAIN",\n'
        '  "audio_manipulation_type": "VOICE_CLONING | TTS | DUBBED | NONE | UNCERTAIN",\n'
        '  "primary_evidence": ["brief description of key evidence"],\n'
        '  "visual_inspection_summary": "what you observed in the frames",\n'
        '  "metric_summary": "key metric findings",\n'
        '  "key_reasoning": "1-2 sentence overall justification"\n'
        "}\n"
        "</verdict>"
    )


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_user_prompt(package: Dict[str, Any]) -> str:
    video_id = package.get("video_id", "unknown")
    summary = package.get("video_summary", {})
    top_chunks = package.get("top_chunks", [])

    if not top_chunks:
        return (
            build_block_a(video_id, summary, top_chunks)
            + "\n\nNOTE: No valid chunks were extracted for this video. "
            "Answer UNCERTAIN with LOW confidence.\n\n"
            + build_block_c()
        )

    return (
        build_block_a(video_id, summary, top_chunks)
        + "\n\n"
        + build_block_b(top_chunks)
        + "\n\n"
        + build_block_c()
    )

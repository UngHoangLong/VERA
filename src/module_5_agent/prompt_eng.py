"""
prompt_eng.py — Module 5 prompt construction.

Turns a Module 4 prompt_package.json into the 3-block prompt described in
PROMPT_DESIGN_PROPOSAL.md:

    System prompt : forensic-analyst role + calibration rules + metric glossary
    Block A       : video-level summary + temporal pattern
    Block B       : per top-K chunk evidence, grouped into 4 feature groups
    Block C       : 4-step reasoning instructions + required output format

The evidence values themselves come pre-calibrated from Module 3
(percentile_rank vs the genuine baseline + a NORMAL/ABOVE/FAR_ABOVE signal),
so the prompt never asks the MLLM to judge "high vs low" on its own.
"""

import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.module_3_autoencoder.config import FEATURE_INTERPRETATIONS  # noqa: F401 (kept for reference)

# ---------------------------------------------------------------------------
# Metric glossary for the MLLM.
#
# Unlike FEATURE_INTERPRETATIONS (terse, used inside the evidence JSON), these
# entries also state WHAT A HIGH VALUE IMPLIES for deepfake detection, so the
# MLLM can reason from a metric to a manipulation type instead of guessing.
# ---------------------------------------------------------------------------

METRIC_GLOSSARY = {
    # GROUP 1 — visual artifacts (blending boundary / texture inconsistency)
    "max_blur_flicker":      "frame-to-frame sharpness flicker on the face; high -> a synthesized/blended "
                             "region whose sharpness does not match the real frame (face-swap)",
    "blur_flicker_variance": "how unstable the blur flicker is over time; high -> intermittent blending artifacts",
    "max_texture_flicker":   "surface skin-texture flicker; high -> generated texture that fails to stay "
                             "consistent across frames (synthesis/swap)",
    "asymmetry_max":         "left-right facial texture asymmetry; high -> imperfect blending of a pasted face",
    "max_blending_flicker":  "flicker at the face-blend boundary; high -> visible seam where a fake face is "
                             "composited onto the head (strong face-swap signal)",
    "blending_variance":     "instability of blending artifacts over time; high -> ongoing compositing seam",

    # GROUP 2 — facial dynamics (unnatural motion / reenactment)
    "mean_landmark_jitter":  "average landmark instability; high -> jittery, non-physical facial motion",
    "max_kinematic_flicker": "facial-geometry flicker; high -> geometry that snaps between frames (reenactment)",
    "max_rigid_violation":   "violation of rigid head motion; high -> face parts moving independently of the "
                             "head, as in puppeteering/reenactment",
    "blinking_variance":     "irregularity of blink timing; high -> unnatural/absent blinking typical of synthesis",
    "mouth_movement_variance":"irregularity of mouth motion; high -> mouth driven by a model rather than real speech",
    "gaze_anomaly":          "mismatch between gaze direction and head pose; high -> eyes that do not track "
                             "naturally with the head (face-swap/reenactment)",
    "iris_jitter_variance":  "abnormal iris-region jitter; high -> eyes synthesized or poorly aligned",

    # GROUP 3 — audio-visual coherence (lip-sync / content mismatch)
    "wer_score":             "disagreement between what is HEARD (ASR) and what the LIPS say (VSR); high -> "
                             "audio track does not match lip motion (lip-sync deepfake or dubbed audio)",
    "semantic_anomaly":      "drop in audio-visual semantic agreement; high -> speech meaning and face/mouth "
                             "do not align",
    "min_cosine_anomaly":    "worst-point audio-visual semantic disagreement in the chunk; high -> a clear "
                             "moment of mismatch",
    "temporal_anomaly":      "drop in audio-visual temporal sync; high -> lips and sound are out of phase "
                             "(lip-sync manipulation)",
    "min_temporal_anomaly":  "worst-point temporal desync in the chunk; high -> a clear out-of-sync moment",
    "temporal_sync_variance":"how much sync quality fluctuates; high -> unstable lip-sync",

    # GROUP 4 — audio artifacts (synthetic voice)
    "vocal_jitter_relative": "micro instability of vocal pitch; high -> synthetic/cloned voice (TTS/vocoder)",
    "vocal_shimmer_relative":"micro instability of vocal amplitude; high -> synthetic/cloned voice (TTS/vocoder)",
}

# ---------------------------------------------------------------------------
# Feature grouping (4 groups for the MLLM to reason modality-by-modality)
# ---------------------------------------------------------------------------

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
    ("GROUP 1 — VISUAL ARTIFACTS",     GROUP_1_VISUAL_ARTIFACTS),
    ("GROUP 2 — FACIAL DYNAMICS",      GROUP_2_FACIAL_DYNAMICS),
    ("GROUP 3 — AUDIO-VISUAL COHERENCE", GROUP_3_AV_COHERENCE),
    ("GROUP 4 — AUDIO ARTIFACTS",      GROUP_4_AUDIO_ARTIFACTS),
]

_SIGNAL_TO_SEVERITY = {
    "FAR_ABOVE_NORMAL": "CRITICAL",
    "ABOVE_NORMAL": "ELEVATED",
    "NORMAL": "NORMAL",
    "BELOW_NORMAL": "NORMAL",
    "FAR_BELOW_NORMAL": "NORMAL",
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_metric_glossary() -> str:
    lines = ["METRIC GLOSSARY (what each feature measures and what a HIGH value implies):"]
    for gname, gfeats in GROUPS:
        lines.append(f"\n{gname}")
        for fn in gfeats:
            desc = METRIC_GLOSSARY.get(fn)
            if desc:
                lines.append(f"  - {fn}: {desc}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a digital forensics analyst specializing in deepfake video detection. "
        "Your job is to read evidence produced by an automated anomaly-detection system "
        "and reach a grounded conclusion.\n\n"
        "IMPORTANT CALIBRATION RULES:\n"
        "- The primary evidence is the physical metrics, NOT your visual impression of the frames. "
        "Frames are provided only as grounding context.\n"
        "- Each metric is compared against a GENUINE baseline. 'percentile_rank' is where the value "
        "falls within genuine videos (0-100). 'signal' summarizes it:\n"
        "    CRITICAL  = far above the genuine range (>= 95th percentile)\n"
        "    ELEVATED  = above the genuine range (>= 80th percentile)\n"
        "    NORMAL    = within the genuine range\n"
        "- A single CRITICAL feature can be noise. Multiple CRITICAL features in the SAME group "
        "is a strong signal.\n"
        "- Genuine videos may still show a few ELEVATED features — judge the whole picture.\n"
        "- When evidence is weak or contradictory, answer UNCERTAIN rather than forcing a verdict.\n\n"
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
        "TEMPORAL ANOMALY PATTERN:",
    ]
    for c in top_chunks:
        tm = c.get("time_metadata", {})
        an = c.get("anomaly", {})
        start = tm.get("start_sec", 0.0)
        end = tm.get("end_sec", 0.0)
        score = an.get("joint_anomaly_score", 0.0)
        level = an.get("level", "?").upper()
        lines.append(f"  {c.get('chunk_id','?')}  [{start:.1f}s-{end:.1f}s]  score={score:.4f}  {level}")
    lines.append(f"\n  Distribution: {summary.get('temporal_pattern', 'NONE')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Block B — per-chunk evidence
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
        return f"  {name:<24} null (not observed)"
    p50 = entry.get("genuine_p50")
    p95 = entry.get("genuine_p95")
    prank = entry.get("percentile_rank")
    signal = entry.get("signal", "NORMAL")
    severity = _SIGNAL_TO_SEVERITY.get(signal, "NORMAL")
    p50s = _fmt_val(p50) if p50 is not None else "  -"
    p95s = _fmt_val(p95) if p95 is not None else "  -"
    pranks = f"{prank}" if prank is not None else "-"
    return (f"  {name:<24} {_fmt_val(val):>10}  genP50={p50s:>8}  "
            f"genP95={p95s:>8}  pct={pranks:>3}  {severity}")


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
            f"(normalized {an.get('normalized_anomaly_score',0):.2f}, level {an.get('level','?')})",
            f"Modalities analyzed: {', '.join(c.get('modalities_analyzed', [])) or 'none'}",
        ]
        if c.get("modalities_missing"):
            header.append(f"Modalities ABSENT : {', '.join(c['modalities_missing'])}")

        group_texts = []
        for gname, gfeats in GROUPS:
            rows = [_feature_row(fn, feats[fn]) for fn in gfeats if fn in feats]
            if rows:
                group_texts.append(gname + "\n" + "\n".join(rows))

        recon = (f"VISUAL RECON SCORE : {an.get('visual_reconstruction_score')}\n"
                 f"AUDIO RECON SCORE  : {an.get('audio_reconstruction_score')}\n"
                 f"KL DIVERGENCE      : {an.get('kl_divergence')}")

        blocks.append("\n".join(header) + "\n\n" + "\n\n".join(group_texts) + "\n\n" + recon)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Block C — reasoning instructions + output format
# ---------------------------------------------------------------------------

def build_block_c() -> str:
    return (
        "=" * 60 + "\n"
        "TASK: Analyze the evidence above and determine whether this video is a deepfake.\n\n"
        "STEP 1 - FAST SCAN: Across all chunks, which feature group has the most "
        "CRITICAL/ELEVATED features? Is there a consistent pattern?\n\n"
        "STEP 2 - DEEP ANALYSIS (chunks with high score): For each suspicious chunk, which "
        "group is most abnormal, and what deepfake type does that combination suggest?\n"
        "  - Visual artifacts CRITICAL, audio NORMAL        -> FACE_SWAP\n"
        "  - Audio-visual mismatch + poor temporal sync     -> LIP_SYNC / reenactment\n"
        "  - Both visual and audio abnormal                 -> FULL_SYNTHESIS\n"
        "  - Audio abnormal, visual NORMAL                  -> AUDIO_ONLY\n"
        "  - No group abnormal                              -> likely GENUINE\n\n"
        "STEP 3 - TEMPORAL PATTERN: Are anomalies SCATTERED (whole-video deepfake) or "
        "CONCENTRATED (localized edit)?\n\n"
        "STEP 4 - SYNTHESIS: Combine all evidence into a verdict with confidence.\n\n"
        "Respond in EXACTLY this format:\n"
        "<think>\n[your step-by-step reasoning]\n</think>\n"
        "<verdict>\n"
        "{\n"
        '  "assessment": "GENUINE | UNCERTAIN | SUSPICIOUS | LIKELY_DEEPFAKE",\n'
        '  "confidence": "LOW | MEDIUM | HIGH",\n'
        '  "primary_evidence": ["feature (chunk_id)", ...],\n'
        '  "deepfake_type": "FACE_SWAP | LIP_SYNC | FULL_SYNTHESIS | AUDIO_ONLY | NONE",\n'
        '  "temporal_pattern": "SCATTERED | CONCENTRATED | ISOLATED | NONE",\n'
        '  "key_reasoning": "1-2 sentence justification",\n'
        '  "limitations": ["what the pipeline cannot determine"]\n'
        "}\n"
        "</verdict>"
    )


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_user_prompt(package: Dict[str, Any]) -> str:
    """Build the full user prompt (Block A + B + C) from a prompt_package dict."""
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

"""
Feature configuration for Module 3.

Each feature is extracted from the nested Module 2 JSON.
The `transform` field controls how the raw value is converted before normalization.

Supported transforms:
- identity: keep raw value
- one_minus: transform x into 1 - x
"""

FEATURE_SPECS = [
    # --- Audio-Visual Consistency (module 22) ---
    {"name": "wer_score",              "path": ["audio_visual_consistency", "transcripts", "wer_score"],                        "transform": "identity",  "group": "content_consistency"},
    {"name": "semantic_anomaly",       "path": ["audio_visual_consistency", "semantic_consistency", "percentile_3rd_cosine"],    "transform": "one_minus", "group": "semantic_consistency"},
    {"name": "min_cosine_anomaly",     "path": ["audio_visual_consistency", "semantic_consistency", "min_cosine_similarity"],    "transform": "one_minus", "group": "semantic_consistency"},
    {"name": "temporal_anomaly",       "path": ["audio_visual_consistency", "temporal_sync", "sync_score"],                     "transform": "one_minus", "group": "temporal_sync"},
    {"name": "min_temporal_anomaly",   "path": ["audio_visual_consistency", "temporal_sync", "min_sync_score"],                 "transform": "one_minus", "group": "temporal_sync"},
    {"name": "temporal_sync_variance", "path": ["audio_visual_consistency", "temporal_sync", "variance"],                       "transform": "identity",  "group": "temporal_sync"},
    # --- Visual Spatial (module 21) ---
    {"name": "max_blur_flicker",         "path": ["visual_spatial", "blur",      "max_blur_flicker"],         "transform": "identity",  "group": "visual_spatial"},
    {"name": "blur_flicker_variance",    "path": ["visual_spatial", "blur",      "blur_flicker_variance"],    "transform": "identity",  "group": "visual_spatial"},
    {"name": "max_texture_flicker",      "path": ["visual_spatial", "texture",   "max_texture_flicker"],      "transform": "identity",  "group": "visual_spatial"},
    {"name": "asymmetry_max",            "path": ["visual_spatial", "texture",   "asymmetry_max"],            "transform": "identity",  "group": "visual_spatial"},
    {"name": "mean_landmark_jitter",     "path": ["visual_spatial", "kinematics","mean_landmark_jitter"],     "transform": "identity",  "group": "visual_spatial"},
    {"name": "max_kinematic_flicker",    "path": ["visual_spatial", "kinematics","max_kinematic_flicker"],    "transform": "identity",  "group": "visual_spatial"},
    {"name": "max_rigid_violation",      "path": ["visual_spatial", "kinematics","max_rigid_violation"],      "transform": "identity",  "group": "visual_spatial"},
    {"name": "blinking_variance",        "path": ["visual_spatial", "kinematics","blinking_variance"],        "transform": "identity",  "group": "visual_spatial"},
    {"name": "mouth_movement_variance",  "path": ["visual_spatial", "kinematics","mouth_movement_variance"],  "transform": "identity",  "group": "visual_spatial"},
    {"name": "gaze_anomaly",             "path": ["visual_spatial", "eye_gaze",  "gaze_pose_sync_score"],     "transform": "one_minus", "group": "visual_spatial"},
    {"name": "iris_jitter_variance",     "path": ["visual_spatial", "iris_jitter","iris_jitter_variance"],    "transform": "identity",  "group": "visual_spatial"},
    {"name": "max_blending_flicker",     "path": ["visual_spatial", "blending",  "max_blending_flicker"],    "transform": "identity",  "group": "visual_spatial"},
    {"name": "blending_variance",        "path": ["visual_spatial", "blending",  "blending_variance"],       "transform": "identity",  "group": "visual_spatial"},
    # --- Audio Artifacts (module 23) ---
    {"name": "vocal_jitter_relative",  "path": ["audio_artifacts", "vocal_jitter_relative"],  "transform": "identity", "group": "audio_artifacts"},
    {"name": "vocal_shimmer_relative", "path": ["audio_artifacts", "vocal_shimmer_relative"], "transform": "identity", "group": "audio_artifacts"},
]

FEATURE_NAMES = [spec["name"] for spec in FEATURE_SPECS]
FEATURE_GROUPS = {spec["name"]: spec["group"] for spec in FEATURE_SPECS}

VISUAL_FEATURE_NAMES = [spec["name"] for spec in FEATURE_SPECS if spec["group"] == "visual_spatial"]
AUDIO_FEATURE_NAMES = [spec["name"] for spec in FEATURE_SPECS if spec["group"] != "visual_spatial"]
FEATURE_INTERPRETATIONS = {
    "wer_score":              "content mismatch between ASR and VSR transcripts",
    "semantic_anomaly":       "drop in audio-visual semantic consistency (3rd percentile)",
    "min_cosine_anomaly":     "drop in semantic consistency at the weakest point",
    "temporal_anomaly":       "drop in audio-visual temporal synchronization",
    "min_temporal_anomaly":   "weakest temporal sync score within the chunk",
    "temporal_sync_variance": "variance of temporal synchronization within the chunk",
    "max_blur_flicker":         "frame-to-frame blur flicker",
    "blur_flicker_variance":    "variance of blur flicker",
    "max_texture_flicker":      "surface texture flicker",
    "asymmetry_max":            "facial texture asymmetry",
    "mean_landmark_jitter":     "average facial landmark jitter",
    "max_kinematic_flicker":    "facial geometry flicker",
    "max_rigid_violation":      "rigid motion violation of facial landmarks",
    "blinking_variance":        "variance in blinking frequency",
    "mouth_movement_variance":  "variance in mouth movement",
    "gaze_anomaly":             "inconsistency between gaze direction and head pose",
    "iris_jitter_variance":     "abnormal iris region jitter",
    "max_blending_flicker":     "face-blending artifact flicker",
    "blending_variance":        "variance of face-blending artifacts",
    "vocal_jitter_relative":  "micro-fluctuation in vocal pitch (jitter)",
    "vocal_shimmer_relative": "micro-fluctuation in vocal amplitude (shimmer)",
}

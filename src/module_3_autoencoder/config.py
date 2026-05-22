"""
Feature configuration for Module 3.

Each feature is extracted from the nested Module 2 JSON.
The `transform` field controls how the raw value is converted before normalization.

Supported transforms:
- identity: keep raw value
- one_minus: transform x into 1 - x
"""

FEATURE_SPECS = [
    {"name": "wer_score", "path": ["audio_visual_consistency", "transcripts", "wer_score"], "transform": "identity", "group": "content_consistency"},
    {"name": "semantic_anomaly", "path": ["audio_visual_consistency", "semantic_consistency", "percentile_3rd_cosine"], "transform": "one_minus", "group": "semantic_consistency"},
    {"name": "temporal_anomaly", "path": ["audio_visual_consistency", "temporal_sync", "sync_score"], "transform": "one_minus", "group": "temporal_sync"},
    {"name": "min_temporal_anomaly", "path": ["audio_visual_consistency", "temporal_sync", "min_sync_score"], "transform": "one_minus", "group": "temporal_sync"},
    {"name": "max_blur_flicker", "path": ["visual_spatial", "blur", "max_blur_flicker"], "transform": "identity", "group": "visual_spatial"},
    {"name": "blur_flicker_variance", "path": ["visual_spatial", "blur", "blur_flicker_variance"], "transform": "identity", "group": "visual_spatial"},
    {"name": "max_texture_flicker", "path": ["visual_spatial", "texture", "max_texture_flicker"], "transform": "identity", "group": "visual_spatial"},
    {"name": "asymmetry_max", "path": ["visual_spatial", "texture", "asymmetry_max"], "transform": "identity", "group": "visual_spatial"},
    {"name": "mean_landmark_jitter", "path": ["visual_spatial", "kinematics", "mean_landmark_jitter"], "transform": "identity", "group": "visual_spatial"},
    {"name": "max_kinematic_flicker", "path": ["visual_spatial", "kinematics", "max_kinematic_flicker"], "transform": "identity", "group": "visual_spatial"},
    {"name": "max_rigid_violation", "path": ["visual_spatial", "kinematics", "max_rigid_violation"], "transform": "identity", "group": "visual_spatial"},
    {"name": "gaze_anomaly", "path": ["visual_spatial", "eye_gaze", "gaze_pose_sync_score"], "transform": "one_minus", "group": "visual_spatial"},
    {"name": "iris_jitter_variance", "path": ["visual_spatial", "iris_jitter", "iris_jitter_variance"], "transform": "identity", "group": "visual_spatial"},
    {"name": "vocal_jitter_relative", "path": ["audio_artifacts", "vocal_jitter_relative"], "transform": "identity", "group": "audio_artifacts"},
    {"name": "vocal_shimmer_relative", "path": ["audio_artifacts", "vocal_shimmer_relative"], "transform": "identity", "group": "audio_artifacts"},
]

FEATURE_NAMES = [spec["name"] for spec in FEATURE_SPECS]
FEATURE_GROUPS = {spec["name"]: spec["group"] for spec in FEATURE_SPECS}
FEATURE_INTERPRETATIONS = {
    "wer_score": "độ lệch nội dung giữa ASR và VSR",
    "semantic_anomaly": "độ suy giảm nhất quán ngữ nghĩa audio-visual",
    "temporal_anomaly": "độ suy giảm đồng bộ thời gian audio-visual",
    "min_temporal_anomaly": "điểm đồng bộ thời gian yếu nhất trong chunk",
    "max_blur_flicker": "dao động độ mờ giữa các frame",
    "blur_flicker_variance": "phương sai dao động độ mờ",
    "max_texture_flicker": "dao động kết cấu bề mặt",
    "asymmetry_max": "bất đối xứng kết cấu vùng mặt",
    "mean_landmark_jitter": "dao động landmark trung bình",
    "max_kinematic_flicker": "dao động hình học khuôn mặt",
    "max_rigid_violation": "vi phạm chuyển động cứng của landmark",
    "gaze_anomaly": "suy giảm nhất quán giữa hướng nhìn và tư thế đầu",
    "iris_jitter_variance": "dao động bất thường vùng mống mắt",
    "vocal_jitter_relative": "dao động vi mô tần số giọng nói",
    "vocal_shimmer_relative": "dao động vi mô biên độ giọng nói",
}

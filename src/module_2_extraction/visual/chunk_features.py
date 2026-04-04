import numpy as np

def aggregate_chunk_features(slide_features_list):
    """Tổng hợp đặc trưng cho chunk từ các slide."""
    chunk_summary = {}
    for key in slide_features_list[0]:
        vals = [slide[key] for slide in slide_features_list]
        chunk_summary[f"{key}_mean"] = float(np.mean(vals))
        chunk_summary[f"{key}_std"] = float(np.std(vals))
    return chunk_summary

import numpy as np
from pathlib import Path

def check_landmarks_integrity(base_dir):
    base_path = Path(base_dir)
    landmark_files = list(base_path.glob("**/slides/*_landmarks.npy"))
    
    if not landmark_files:
        print(f"❌ Không tìm thấy file landmarks nào tại: {base_dir}")
        return

    print(f"🔍 Đang kiểm tra {len(landmark_files)} file slide landmarks...")
    print("-" * 50)

    total_slides = 0
    slides_with_nan = 0
    total_frames = 0
    nan_frames_count = 0

    for lm_file in sorted(landmark_files):
        total_slides += 1
        data = np.load(lm_file) # Shape thường là (12, 468, 2)
        
        # Kiểm tra xem có bất kỳ giá trị NaN nào trong slide này không
        # Vì video_slicer của cậu điền NaN cho các frame không có landmark
        nan_mask = np.isnan(data)
        
        # Kiểm tra từng frame trong slide
        # data.shape[0] là số frame (thường là 12)
        for i in range(data.shape[0]):
            total_frames += 1
            if np.isnan(data[i]).any():
                nan_frames_count += 1
        
        if nan_mask.any():
            slides_with_nan += 1
            # print(f"⚠️ Slide lỗi: {lm_file.relative_to(base_path)}") # Bỏ comment nếu muốn xem chi tiết từng file

    print("-" * 50)
    print(f"📊 KẾT QUẢ KIỂM TRA:")
    print(f"✅ Tổng số Slide đã xử lý: {total_slides}")
    print(f"⚠️ Số Slide chứa ít nhất 1 frame lỗi (NaN): {slides_with_nan}")
    print(f"🖼️ Tổng số Frame đã quét: {total_frames}")
    print(f"❌ Số Frame bị mất Landmark (NaN): {nan_frames_count}")
    
    if nan_frames_count == 0:
        print("\n🎉 TUYỆT VỜI! 100% frame đều có Landmark đầy đủ.")
    else:
        percentage = (nan_frames_count / total_frames) * 100
        print(f"\n💡 Tỷ lệ mất landmark: {percentage:.2f}%")
        print("Điều này là bình thường do cơ chế Fallback bọc lót của cậu.")

if __name__ == "__main__":
    # Thay đổi đường dẫn này trỏ đến thư mục interim của cậu
    check_landmarks_integrity("data/interim/mavos-sample")
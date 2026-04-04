## Cách chạy

Ví dụ kiểm tra toàn bộ `data/interim`:

```bash
python ./src/module_2_extraction/visual1/verify_landmark_projection.py data/interim
```

Nếu chỉ muốn xem ít mẫu trước:

```bash
python /src/module_2_extraction/visual1/verify_landmark_projection.py data/interim --max_slides 5 --max_frames 4
```

## Đầu ra

Mặc định sẽ lưu vào:

```bash
data/interim/_debug_landmark_projection/
```

Trong đó có dạng:

* `sample/chunk_0000/slide_00_faces_frame_00.png`
* `sample/chunk_0000/slide_00_faces_contact_sheet.png`
* `manifest.json`
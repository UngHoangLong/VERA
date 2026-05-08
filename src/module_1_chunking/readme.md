## Mục đích
`export_face_full_preview.py` dùng để gộp toàn bộ các file `slide_*_faces.npy` trong các thư mục `chunk_*` thành một video preview tổng (`full_face_preview.mp4`).

Script sẽ đọc `metadata.json`, sắp xếp các slide theo thời gian, loại bớt phần bị trùng do overlap giữa các chunk, rồi ghi tất cả frame khuôn mặt đã crop vào một file video duy nhất để dễ kiểm tra kết quả face crop của toàn bộ video. :contentReference[oaicite:0]{index=0}

## Lệnh chạy
```bash
python export_face_full_preview.py --input-root data/interim/mavos-sample --overwrite
````

## Tham số

* `--input-root`: thư mục chứa các `chunk_*`
* `--output`: đường dẫn file video đầu ra
* `--fps`: đặt FPS thủ công
* `--overwrite`: ghi đè file đầu ra nếu đã tồn tại
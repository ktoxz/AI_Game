# Mine Cart Camera Collector

Trò chơi Pygame sử dụng webcam để nhận diện mũi người chơi và điều khiển xe mỏ thu thập tiền rơi. Nhặt tiền để tăng điểm, tránh bom (trúng 3 lần sẽ thua) và xem bảng xếp hạng bên cạnh.

## Cách chạy
1. Cài đặt phụ thuộc (Python 3.11):
   ```bash
   pip install -r requirements.txt
   ```
2. Kết nối webcam và chạy trò chơi:
   ```bash
   python main.py
   ```

## Điều khiển
- Di chuyển mũi của bạn trái/phải trước webcam để điều khiển xe mỏ.
- Nhấn **R** để chơi lại sau khi thua.
- Nhấn **Esc** hoặc đóng cửa sổ để thoát.

## Cơ chế trò chơi
- Hai loại vật phẩm: **tiền** (vàng) và **bom** (đỏ) rơi từ trên xuống.
- Nhặt tiền +1 điểm, trúng bom kích hoạt hiệu ứng nổ và cộng vào số lần dính bom.
- Sau mỗi **15 giây**, tốc độ rơi và tần suất xuất hiện vật phẩm sẽ tăng.
- Bảng xếp hạng bên phải lưu lại 5 điểm cao nhất trong file `scores.json`.
- Góc dưới phải hiển thị khung webcam thu nhỏ kèm chấm xanh tại vị trí mũi đang được nhận diện.

import cv2
import torch
import numpy as np
import argparse # Thêm thư viện đọc tham số dòng lệnh
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

def main():
    # --- 1. Khởi tạo bộ đọc tham số (Argparse) ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='weights/yolov8n.pt', help='Đường dẫn file model (.pt, .xml, folder ncnn)')
    parser.add_argument('--source', type=str, default='0', help='ID camera (0) hoặc "picamera0" cho RPi 5 CSI')
    parser.add_argument('--resolution', type=str, default='640x480', help='Độ phân giải (ví dụ: 1280x720)')
    parser.add_argument('--conf', type=float, default=0.5, help='Ngưỡng tin cậy (Confidence threshold)')
    args = parser.parse_args()

    # Xử lý độ phân giải
    width, height = map(int, args.resolution.split('x'))

    # --- 2. Cấu hình thiết bị và Model ---
    # Ép buộc dùng CPU vì RPi 5 không có CUDA
    device = torch.device("cpu")
    print(f"--- Đang nạp model từ: {args.model} ---")
    
    # Load model linh hoạt theo tham số truyền vào
    model = DetectMultiBackend(weights=args.model, device=device, fuse=True)
    model = AutoShape(model)

    # Load classname
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # --- 3. Khởi tạo Camera cho Raspberry Pi 5 ---
    if args.source == "picamera0":
        # Pipeline tối ưu cho Camera Module trên Pi 5
        pipeline = (
            f"libcamerasrc ! video/x-raw, width={width}, height={height}, framerate=30/1 ! "
            "videoconvert ! appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        # Dùng cho USB Webcam
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Lỗi: Không thể mở camera!")
        return

    tracker = DeepSort(max_age=30)
    print("--- Hệ thống đã sẵn sàng! Bấm 'Q' để thoát ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        detect = []

        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            class_id = int(label)

            # Lọc theo ngưỡng tin cậy (dùng args.conf)
            if confidence < args.conf:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật Tracking
        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                
                color = [int(c) for c in colors[class_id]]
                label_str = f"{class_names[class_id]}-{track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label_str, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Object-tracking RPi5", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

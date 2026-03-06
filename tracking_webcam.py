import cv2 # thu vien opencv
import torch # thu vien AI
import numpy as np # xu ly mang so
from deep_sort_realtime.deepsort_tracker import DeepSort # thuat toan tracking
from models.common import DetectMultiBackend, AutoShape #load model YOLO// tu resize anh va xu li input

# Config value
    #video_path = "data_ext/traffic.mp4" #dan duong file video
conf_threshold = 0.5 #do tin cay toi thieu de chap nhan detection
tracking_class = 0 # None: track all theo doi het trong file classes names

# Khởi tạo DeepSort
tracker = DeepSort(max_age=30) #khoi tao deepsort dung de gan ID cho object khi di chuyen

# Khởi tạo YOLOv9
#device = "cpu" # "cuda": GPU, "cpu": CPU, "mps:0"
device = torch.device("cuda:0") #su dung gpu 0
#load model yolov9
model  = DetectMultiBackend(weights="weights/yolov9c.pt", device=device, fuse=True ) # tối ưu model bằng cách gộp convolution + batchnorm
model  = AutoShape(model) #resize anh, convert tensor, NMS

# Load classname từ file classes.names
with open("data_ext/classes.names") as f:# Mở file chứa danh sách tên vật thể và tự động đóng sau khi dùng.
    class_names = f.read().strip().split('\n')# Đọc nội dung file và xóa bỏ các khoảng trắng thừa ở đầu/cuối
    # Cắt văn bản thành một danh sách (List) các tên riêng biệt theo từng dòng.
colors = np.random.randint(0,255, size=(len(class_names),3 ))  #tao mau random
tracks = []# luu vat the check duoc //#luu object dang tracking.

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(1)

# Tiến hành đọc từng frame từ video
while True:
    # Đọc
    ret, frame = cap.read()
    if not ret:#neu video loi bo qua frame
        continue
    # Đưa qua model để detect
    results = model(frame)# frame dua vao yolov9 de phat hien vat the
    detect = []#danh sach cac object da phat hien trong frame hien tai
    #deepsort khong nhan truc tiep tu output cua  yolo, can format rieng
    for detect_object in results.pred[0]:#pred chứa các object mà mô hình phát hiện được trong ảnh [x1, y1, x2, y2, confidence, class]
        # lap qua tung object ma yolo detect duoc
        # bbox toa do, confidence do tin cay, Id class loai object = label
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox) #lay toa do bounding box
        class_id = int(label)

        if tracking_class is None: #kiem tra co tracking tat ca object khong, linh hoat, neu muon doi chi can thay trachking_class = None
            if confidence < conf_threshold:
                continue
        else:
            if class_id != tracking_class or confidence < conf_threshold:#neu class_id tuc la da qua detect ma khac object ma
                #chung ta muon detect hoac neu dung la no nhung do tin cay thap hon nguong thi cung loai
                continue
        #dua object tu yolo sang deepsort de tracking
        detect.append([ [x1, y1, x2-x1, y2 - y1], confidence, class_id ])#

    # Cập nhật,gán ID bằng DeepSort
    tracks = tracker.update_tracks(detect, frame = frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb() #lay toa do bounding box cua object dang duoc track
            class_id = track.get_det_class()#lay class cua object ma yolo dang detect
            x1, y1, x2, y2 = map(int, ltrb)#map(function, iterable) ap dung 1 ham len tung phan tu cua mot danh sach
            color = colors[class_id]#chon mau cho object
            B, G, R = map(int,color)#tach mau thanh 3 phan
            #tao text hien thi tren video
            label = "{}-{}".format(class_names[class_id], track_id)
            #ve khung chu nhat ( bounding box ) quanh object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)# B, G, R lấy ở tren kia
            #ve nen mau phia sau chu label
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            #cv2.putText(image, text, org, font, fontScale, color, thickness)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show hình ảnh lên màn hình
    cv2.namedWindow("Object-tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object-tracking", 800, 600)
    cv2.imshow("Object-tracking", frame)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
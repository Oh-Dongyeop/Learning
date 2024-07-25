from ultralytics import YOLO
import cv2


# model = YOLO("yolov8x.pt")
model = YOLO("./runs/detect/train70/weights/best.pt")
cap = cv2.VideoCapture("./smd_plus/VIS_Onshore/Videos/MVI_1582_VIS.avi")

#
# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        results = model(frame)
        cv2.imshow("Results", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

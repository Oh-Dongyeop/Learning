import time, cv2
from ultralytics import YOLO
from glob import glob

# Load a model
model = YOLO("yolov8n.pt") # load an pretrained model for COCO dataset
classnames = model.names
img_files = glob('*.jpg')
for img_file in img_files:
    img_bgr = cv2.imread(img_file)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Predict with the model
    t0 = time.time()
    results = model(img_rgb, conf=0.25)[0]
    xyxys = results.boxes.xyxy.cpu().numpy().astype(int)
    confids = results.boxes.conf.cpu().numpy()
    clsids = results.boxes.cls.cpu().numpy().astype(int)
    print('추론시간 : {:.2f}ms'.format((time.time() - t0) * 1000))
    for xyxy, confid, clsid in zip(xyxys, confids, clsids):
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        text = '{}({:.2f})'.format(classnames[clsid], confid)
        cv2.putText(img_bgr, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    resized_img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.5, fy=0.5) # dsize=(0, 0) or None이면fx, fy 적용
    cv2.imshow('Result: {} object detected '.format(len(clsids)), resized_img_bgr)
    key = cv2.waitKey()
cv2.destroyAllWindows()
import cv2
import time
from ultralytics import YOLO
from ultralytics import RTDETR
from pathlib import Path
import numpy as np 

ROOT = Path(__file__).parent.parent

#model_path = ROOT / "models" / "fire_yolo26n.engine"
#predict_dir = ROOT / "predict" / "fire_yolo26n"
#model = YOLO(str(model_path))

model_path = ROOT / "models" / "fire_rtdetr.engine"
predict_dir = ROOT / "predict" / "fire_rtdetr"
model = RTDETR(str(model_path))

target_folder = ROOT / "dataset" / "test"
predict_dir.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["fire", "smoke"]

FIRE_ID = 0
SMOKE_ID = 1

COLORS = {
    0: (0, 0, 255),
    1: (255, 0, 0)
}

def filter_by_priority(boxes, scores, class_ids, fire_id=0, smoke_id=1):
    """
    Keep only fire detections if any fire is found.
    Otherwise, keep only smoke detections.
    """
    if len(class_ids) == 0:
        return boxes, scores, class_ids
    fire_mask = (class_ids == fire_id)
    if np.any(fire_mask):
        return boxes[fire_mask], scores[fire_mask], class_ids[fire_mask]
    smoke_mask = (class_ids == smoke_id)
    return boxes[smoke_mask], scores[smoke_mask], class_ids[smoke_mask]

def draw_detections(frame, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(cls_id)
        
        color = COLORS.get(cls_id, (0, 255, 0))
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        label = f"{name}: {score:.2f}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

for file_path in target_folder.iterdir():
    ext = file_path.suffix.lower()

    # Imagens
    if ext in ['.jpg', '.jpeg', '.png']:
        output_path = predict_dir / f"{file_path.stem}_output.jpg"
        
        frame = cv2.imread(str(file_path))
        if frame is None:
            continue
            
        results = model.predict(frame, conf=0.5, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        # Filtro nas imagens
        boxes, scores, class_ids = filter_by_priority(boxes, scores, class_ids, FIRE_ID, SMOKE_ID)
        
        frame = draw_detections(frame, boxes, scores, class_ids)
        cv2.imwrite(str(output_path), frame)

    # Videos
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        output_path = predict_dir / f"{file_path.stem}_output.mp4"
        
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            continue
            
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps_video, (width, height))

        # Loop videos
        while True:
            t_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(frame, conf=0.5, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()

           # Filtro nos videos 
            boxes, scores, class_ids = filter_by_priority(boxes, scores, class_ids, FIRE_ID, SMOKE_ID)
            
            frame = draw_detections(frame, boxes, scores, class_ids)
            
            t_end = time.time()
            fps_atual = 1.0 / (t_end - t_start)
            
            cv2.putText(frame, f"FPS: {fps_atual:.1f}", (width - 300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
            
            writer.write(frame)
            
        cap.release()
        writer.release()

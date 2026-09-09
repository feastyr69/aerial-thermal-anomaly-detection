from ultralytics import YOLO

class AnomalyDetector:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        # Class 0: Person, 1: Car, 2: Bicycle
        
    def analyze_frame(self, frame, conf_threshold=0.5):
        # Run YOLO inference
        results = self.model(frame, verbose=False)
        
        detections = []
        is_anomaly = False
        highest_score = 0.0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Confidence thresholding
                if conf > conf_threshold:
                    detections.append({
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
                    is_anomaly = True
                    if conf > highest_score:
                        highest_score = conf
                        
        return {
            "detections": detections,
            "anomaly_score": highest_score,
            "is_anomaly": is_anomaly
        }

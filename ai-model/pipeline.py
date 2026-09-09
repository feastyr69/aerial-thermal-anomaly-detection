import cv2
import time
from inference import AnomalyDetector
from pathlib import Path

# Configuration
STREAM_URL = "http://localhost:8001/stream"
WEIGHTS_PATH = Path("weights/best.pt")

def run_pipeline():
    print(f"Loading model from {WEIGHTS_PATH}...")
    if not WEIGHTS_PATH.exists():
        print(f"Warning: {WEIGHTS_PATH} not found. Please ensure the weights are downloaded from Colab and placed here.")
        
    detector = AnomalyDetector(WEIGHTS_PATH)
    
    print(f"Connecting to stream at {STREAM_URL}...")
    # OpenCV can read HTTP MJPEG streams directly
    cap = cv2.VideoCapture(STREAM_URL)
    
    if not cap.isOpened():
        print("Error: Could not connect to the stream. Make sure thermal-streaming/mock_stream.py is running.")
        return
        
    print("Pipeline started. Processing frames... Press Ctrl+C to stop.")
    
    frame_count = 0
    start_time_total = time.time()
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or disconnected. Reconnecting in 2s...")
                time.sleep(2)
                cap = cv2.VideoCapture(STREAM_URL)
                continue
                
            # Run inference
            result = detector.analyze_frame(frame)
            
            # Simulated geo-tagging (e.g. border coordinates)
            lat, lon = 28.6139, 77.2090 
            
            latency = (time.time() - start_time) * 1000
            
            if result['is_anomaly']:
                print(f"[ALERT] Anomaly Detected! Score: {result['anomaly_score']:.2f} | Detections: {len(result['detections'])} | Inference Latency: {latency:.1f}ms | Geo: {lat}, {lon}")
                # In Phase 5, we will POST this to the backend API instead of just printing
                
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time_total
                fps = frame_count / elapsed
                # print(f"--- Pipeline Status: Running at {fps:.1f} FPS ---")
                
    except KeyboardInterrupt:
        print("Pipeline stopped.")

if __name__ == "__main__":
    run_pipeline()

import cv2
import time
import asyncio
import argparse
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from pathlib import Path

app = FastAPI()

class MockStreamer:
    def __init__(self, video_path, latency_ms=0, fps=30):
        self.video_path = video_path
        self.latency_ms = latency_ms
        self.fps = fps
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            print(f"Warning: Could not open {video_path}. Will generate dummy frames.")
            self.cap = None

    def generate_dummy_frame(self):
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "No Video Source", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    async def get_frames(self):
        target_frame_time = 1.0 / self.fps
        
        while True:
            start_time = time.time()
            
            if self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
            else:
                frame = self.generate_dummy_frame()

            if self.latency_ms > 0:
                await asyncio.sleep(self.latency_ms / 1000.0)

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Maintain FPS
            elapsed = time.time() - start_time
            if elapsed < target_frame_time:
                await asyncio.sleep(target_frame_time - elapsed)

streamer = None

@app.get("/stream")
async def video_feed():
    return StreamingResponse(streamer.get_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Thermal Video Feed Generator")
    parser.add_argument("--video", type=str, default="sample_videos/mock_thermal.mp4", help="Path to mp4 file")
    parser.add_argument("--latency", type=int, default=0, help="Simulate latency in ms")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()

    # Ensure sample dir exists
    Path("sample_videos").mkdir(exist_ok=True)
    
    streamer = MockStreamer(args.video, args.latency, args.fps)
    print(f"Starting mock stream on http://localhost:8001/stream")
    print(f"Using video: {args.video} | Latency: {args.latency}ms")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

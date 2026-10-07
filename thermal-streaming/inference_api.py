"""Local API for browser-based video inference."""

import shutil
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT_DIR / "ai-model" / "weights" / "uavbest.pt"
ALLOWED_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = None
    application.state.model_error = None
    try:
        application.state.model = await run_in_threadpool(_load_model)
    except Exception as error:
        application.state.model_error = str(error)
    yield


app = FastAPI(title="Thermal Video Inference API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_inference_lock = threading.Lock()


def _load_model():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


def _process_video(model, input_path: Path, intermediate_path: Path, output_path: Path, confidence: float):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to create a browser-playable MP4")

    with _inference_lock:
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open this video. Try an MP4, MOV, AVI, MKV, or WebM file.")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError("The uploaded video has invalid dimensions.")
        if fps <= 0:
            fps = 30.0

        writer = cv2.VideoWriter(
            str(intermediate_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Could not initialize the video encoder.")

        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
                annotated_frame = result.plot()
                if annotated_frame.shape[:2] != (height, width):
                    raise RuntimeError("The inference output dimensions do not match the source frame.")
                writer.write(annotated_frame)
                frame_count += 1
        finally:
            capture.release()
            writer.release()

        if frame_count == 0:
            raise ValueError("No readable frames were found in the uploaded video.")

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(intermediate_path),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        encoded = subprocess.run(command, capture_output=True, text=True)
        if encoded.returncode != 0 or not output_path.is_file():
            detail = encoded.stderr.strip().splitlines()
            message = detail[-1] if detail else "Unknown FFmpeg error"
            raise RuntimeError(f"Could not encode the annotated video: {message}")


@app.get("/api/health")
async def health(request: Request):
    model_ready = request.app.state.model is not None
    return {
        "status": "ready" if model_ready else "model_error",
        "weights": WEIGHTS_PATH.name,
        "detail": request.app.state.model_error,
    }


@app.post("/api/infer")
async def infer_video(
    request: Request,
    video: UploadFile = File(...),
    confidence: float = Form(0.25, ge=0.0, le=1.0),
):
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Choose a video file (MP4, MOV, AVI, MKV, or WebM).")

    work_dir = Path(tempfile.mkdtemp(prefix="thermal-inference-"))
    input_path = work_dir / f"input{suffix}"
    intermediate_path = work_dir / "annotated.avi"
    output_path = work_dir / "annotated.mp4"

    try:
        model = request.app.state.model
        if model is None:
            detail = request.app.state.model_error or "The model has not loaded. Restart the inference API."
            raise HTTPException(status_code=503, detail=detail)

        with input_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                destination.write(chunk)
        if input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="The selected video is empty.")

        await run_in_threadpool(_process_video, model, input_path, intermediate_path, output_path, confidence)
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="annotated_video.mp4",
            background=BackgroundTask(shutil.rmtree, work_dir, ignore_errors=True),
        )
    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except FileNotFoundError as error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        await video.close()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

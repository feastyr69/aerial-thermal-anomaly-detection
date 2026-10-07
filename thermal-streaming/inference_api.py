"""Local API for browser-based video inference."""

import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT_DIR / "ai-model" / "weights" / "uavbest.pt"
ALLOWED_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.model = None
    application.state.model_error = None
    try:
        application.state.model = _load_model()
    except Exception as error:
        application.state.model_error = str(error)
    yield


app = FastAPI(title="Thermal Video Inference API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Inference-ID"],
)

_inference_lock = threading.Lock()
_inference_jobs = {}
_JOB_TTL_SECONDS = 15 * 60


def _load_model():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


async def _stream_video_frames(
    model, input_path: Path, intermediate_path: Path, output_path: Path, confidence: float
):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to create a browser-playable MP4")

    if not _inference_lock.acquire(blocking=False):
        raise RuntimeError("Another video is already being processed. Try again when it finishes.")

    capture = None
    writer = None
    try:
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
            raise RuntimeError("Could not initialize the video encoder.")

        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=confidence, imgsz=640, verbose=False)[0]
            annotated_frame = result.plot()
            if annotated_frame.shape[:2] != (height, width):
                raise RuntimeError("The inference output dimensions do not match the source frame.")
            writer.write(annotated_frame)
            encoded_frame, jpeg = cv2.imencode(".jpg", annotated_frame)
            if not encoded_frame:
                raise RuntimeError("Could not encode an inference frame for live display.")
            frame_bytes = jpeg.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame_bytes)).encode("ascii")
                + b"\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            frame_count += 1

        if frame_count == 0:
            raise ValueError("No readable frames were found in the uploaded video.")

        writer.release()
        writer = None

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
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        _inference_lock.release()


def _cleanup_inference_job(inference_id: str):
    job = _inference_jobs.pop(inference_id, None)
    if job:
        shutil.rmtree(job["work_dir"], ignore_errors=True)


def _prune_inference_jobs():
    now = time.monotonic()
    expired = [
        inference_id
        for inference_id, job in _inference_jobs.items()
        if job["complete"] and now - job["created_at"] > _JOB_TTL_SECONDS
    ]
    for inference_id in expired:
        _cleanup_inference_job(inference_id)


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

    model = request.app.state.model
    if model is None:
        detail = request.app.state.model_error or "The model has not loaded. Restart the inference API."
        video.file.close()
        raise HTTPException(status_code=503, detail=detail)

    _prune_inference_jobs()
    inference_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix="thermal-inference-"))
    input_path = work_dir / f"input{suffix}"
    intermediate_path = work_dir / "annotated.avi"
    output_path = work_dir / "annotated.mp4"

    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(video.file, destination, length=1024 * 1024)
        if input_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="The selected video is empty.")
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        video.file.close()

    job = {
        "work_dir": work_dir,
        "output_path": output_path,
        "complete": False,
        "error": None,
        "error_status": 500,
        "created_at": time.monotonic(),
    }
    _inference_jobs[inference_id] = job

    async def live_frames():
        stream_completed = False
        try:
            async for part in _stream_video_frames(
                model, input_path, intermediate_path, output_path, confidence
            ):
                yield part
            stream_completed = True
        except Exception as error:
            job["error"] = str(error)
            if isinstance(error, ValueError):
                job["error_status"] = 400
            elif isinstance(error, FileNotFoundError):
                job["error_status"] = 503
        finally:
            if stream_completed or job["error"]:
                job["complete"] = True
            else:
                _cleanup_inference_job(inference_id)

    return StreamingResponse(
        live_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "X-Inference-ID": inference_id,
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/infer/{inference_id}/result")
async def inference_result(inference_id: str):
    _prune_inference_jobs()
    job = _inference_jobs.get(inference_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Inference result expired or does not exist.")
    if not job["complete"]:
        raise HTTPException(status_code=409, detail="Inference is still processing.")
    if job["error"]:
        detail = job["error"]
        status = job["error_status"]
        _cleanup_inference_job(inference_id)
        raise HTTPException(status_code=status, detail=detail)
    if not job["output_path"].is_file():
        _cleanup_inference_job(inference_id)
        raise HTTPException(status_code=500, detail="Inference finished without producing a video.")

    return FileResponse(
        job["output_path"],
        media_type="video/mp4",
        filename="annotated_video.mp4",
        background=BackgroundTask(_cleanup_inference_job, inference_id),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)

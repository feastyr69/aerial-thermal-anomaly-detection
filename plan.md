# Implementation Plan — Software-Only Phase
## Drone-Based Thermal Anomaly Detection System (5G-Enabled)

This plan covers building and validating the **software stack only** — no physical drone, thermal camera, or 5G hardware required yet. Real thermal footage (FLIR ADAS dataset) and locally stored/looped video files stand in for the live drone feed. Once the pipeline works end-to-end on mock data, swapping the mock stream for a real 5G/RTSP drone feed is a drop-in change (Phase 7 covers that transition).

**Guiding principle:** build the system around a *stream source interface*. Whether frames come from an mp4 file on disk, a webcam, or an actual drone's RTSP feed, the rest of the pipeline (AI engine → alerts → dashboard) never needs to know the difference. This is what makes the mock-first approach safe to build on top of later.

---

## Phase 0 — Project Setup & Repo Scaffolding
**Goal:** A clean, runnable skeleton matching the structure in the README.

- [ ] Initialize repo with the folder structure from README (`ai-model/`, `backend/`, `dashboard/`, `thermal-streaming/`, `drone-control/` stubbed for later)
- [ ] Set up Python virtual environment, `requirements.txt` (opencv-python, ultralytics, torch, fastapi, uvicorn, pymongo/firebase-admin, python-dotenv, websockets)
- [ ] Set up `dashboard/` with a React app (Vite recommended over CRA) + Chart.js/Leaflet.js deps
- [ ] Add `.env.example`, `.gitignore`, base `README` updates
- [ ] Set up basic CI (lint + smoke test) if desired

**Deliverable:** empty-but-runnable repo; `python app.py` and `npm run dev` both boot without errors.

---

## Phase 1 — Dataset Acquisition & Preprocessing
**Goal:** FLIR ADAS dataset downloaded, understood, and converted into a training-ready format.

- [ ] Download FLIR ADAS (Free/Reflex or ADAS v2, whichever license you have access to)
- [ ] Explore folder structure — thermal 8-bit/16-bit images, paired annotations (COCO-style JSON or YOLO txt depending on version)
- [ ] Write `ai-model/dataset/prepare_dataset.py`:
  - Convert annotations to YOLO format if not already (`class x_center y_center w h`, normalized)
  - Split into train/val/test (e.g. 80/10/10)
  - Filter to the classes relevant to border security: `person`, `car`, `bicycle` (drop irrelevant classes like `dog`, `traffic light` if present)
- [ ] Basic dataset stats notebook: class balance, image resolution, sample visualizations with bounding boxes drawn

**Deliverable:** `ai-model/dataset/{train,val,test}/{images,labels}` ready for YOLO training + a stats report.

---

## Phase 2 — Mock Thermal Video Feed Generator
**Goal:** Simulate a "live drone feed" using static files, so every downstream module can be built/tested without hardware.

- [ ] `thermal-streaming/mock_stream.py`:
  - Reads a folder of thermal videos/image sequences (can be built from FLIR ADAS images stitched into an mp4, or any sample thermal video you find/create) and loops them at a configurable FPS
  - Exposes the feed via **one of**: local RTSP server (using `mediamtx`/`ffmpeg`), or a simple frame-by-frame generator/WebSocket — pick RTSP if you want the closest simulation to a real drone gimbal+5G module
  - Adds a `--simulate-latency` flag to artificially delay frames, mimicking 5G network jitter for later latency testing
- [ ] Optionally inject synthetic anomalies (draw a moving "hot blob" over a background clip) so you have labeled ground-truth events to test alerting end-to-end even before the model is trained
- [ ] Document how to point this at any new mp4 dropped into `thermal-streaming/sample_videos/`

**Deliverable:** running `python mock_stream.py` serves a looping thermal video that looks, to the rest of the system, like a live drone feed.

---

## Phase 3 — AI Model Development
**Goal:** Two complementary detectors, matching the README's approach.

### 3a. Supervised detector (YOLOv8 fine-tune)
- [ ] Fine-tune YOLOv8n/s on the prepared FLIR ADAS data (`ai-model/train.py`)
- [ ] Track experiments (simple CSV/W&B) — mAP@0.5, precision/recall per class
- [ ] Export best checkpoint to `ai-model/weights/best.pt`

### 3b. Unsupervised anomaly detector (autoencoder)
- [ ] Train a convolutional autoencoder on "normal" thermal frames only (empty scenes, routine patterns)
- [ ] Use reconstruction error as an anomaly score; calibrate a threshold on a held-out normal-only validation set
- [ ] This catches anomalies YOLO wasn't trained to name (e.g. unusual heat blobs, unattended objects)

### 3c. Inference wrapper
- [ ] `ai-model/inference.py`: single function `analyze_frame(frame) -> {detections: [...], anomaly_score: float, is_anomaly: bool}` combining both models
- [ ] Confidence thresholding + simple temporal smoothing (require N consecutive frames before firing an alert, to cut false positives)

**Deliverable:** a single importable inference module, unit-tested against a handful of saved sample frames with known expected outputs.

---

## Phase 4 — Inference Pipeline Integration
**Goal:** Wire the mock stream (Phase 2) into the model (Phase 3), continuously.

- [ ] `ai-model/pipeline.py`: pulls frames from the mock stream source, runs `analyze_frame`, timestamps + geo-tags each result (geo-tag can be mocked with a fixed/simulated GPS path for now)
- [ ] Buffer + log every result (even non-anomalous) at a low sample rate for later analysis; log every anomaly at full detail
- [ ] Measure and log per-frame inference latency — this becomes your baseline before adding real network latency

**Deliverable:** running the pipeline against the mock stream prints/logs detections and anomaly flags in real time.

---

## Phase 5 — Backend API & Alert System
**Goal:** Turn pipeline output into a queryable service with alerting.

- [ ] FastAPI app (`backend/app.py`) with endpoints:
  - `GET /anomalies` (paginated, filterable by time/location/type)
  - `GET /anomalies/{id}` (detail + thumbnail frame)
  - `WS /live` (push live detections/anomalies to connected dashboard clients)
  - `POST /stream/config` (point the backend at a different mock video / later, a real RTSP URL)
- [ ] Persistence: MongoDB (or Firebase) collection for anomaly events: `{timestamp, lat, lon, type, confidence, frame_ref, reviewed: bool}`
- [ ] Alert dispatch module (`backend/alerts.py`): on new anomaly, push to dashboard via WebSocket + stub for SMS/email (e.g. Twilio/SendGrid, can be a no-op/log-only in this phase)
- [ ] Basic auth/token for the API (even a simple API key is fine at this stage)

**Deliverable:** `python app.py` running, `/anomalies` returning real logged events generated from the mock stream, alerts visibly firing.

---

## Phase 6 — Live Monitoring Dashboard
**Goal:** Operator-facing UI, matching the README's dashboard vision.

- [ ] React app consuming the WebSocket + REST API
- [ ] Live view: current frame thumbnail (or embedded video player pointed at the mock RTSP stream) + real-time anomaly feed list
- [ ] Map view (Leaflet.js) plotting geo-tagged anomalies (mock GPS path from Phase 4)
- [ ] Analytics view (Chart.js): anomaly counts over time, by type, false-positive marking (operator can flag a logged anomaly as false positive, feeding back into future model evaluation)
- [ ] Alert banner / notification toast on new incoming anomaly via WebSocket

**Deliverable:** a working dashboard showing the full loop — mock video → detection → alert → visualized on screen — with no hardware involved.

---

## Phase 7 — Evaluation on Mock Data
**Goal:** Fill in the README's Results table with real numbers, using mock data as the testbed.

- [ ] Run YOLO eval on held-out FLIR ADAS test split → Detection Accuracy / mAP, per-class precision-recall
- [ ] Run the full pipeline against a curated set of mock videos (including the synthetic-anomaly clips from Phase 2) → measure False Positive Rate and False Negative Rate end-to-end (not just at the model level)
- [ ] Measure Average Latency: frame capture → inference → alert appears on dashboard, using the `--simulate-latency` flag from Phase 2 to test under different simulated network conditions (e.g. 20ms, 100ms, 300ms)
- [ ] Write up results, update README's Results table

**Deliverable:** filled-in metrics table, a short evaluation report with sample detection screenshots/GIFs.

---

## Phase 8 — (Future) Hardware/5G Integration
**Not part of the software-only phase, but this is why Phase 2's abstraction matters.**

- [ ] Replace `mock_stream.py`'s source with the real drone's 5G-transmitted RTSP/WebRTC feed — same interface, no changes needed downstream
- [ ] Integrate real GPS module output instead of the simulated flight path
- [ ] Field-test latency numbers against Phase 7's simulated baseline
- [ ] Optional: edge deployment of a lighter model (YOLOv8n / quantized) directly on drone companion computer, per the README's "Future Scope"

---

## Suggested Order & Rough Timeline (solo/small team, part-time)

| Phase | Focus | Est. Duration |
|---|---|---|
| 0 | Setup | 1–2 days |
| 1 | Dataset prep | 3–5 days |
| 2 | Mock stream | 2–3 days |
| 3 | Model training | 1–2 weeks |
| 4 | Pipeline integration | 3–5 days |
| 5 | Backend + alerts | 1 week |
| 6 | Dashboard | 1–1.5 weeks |
| 7 | Evaluation | 3–5 days |

Phases 3 and 6 can run in parallel if you have more than one person (one on ML, one on backend/frontend).

## Notes on Using FLIR ADAS Specifically

- Confirm which FLIR ADAS release you have (v1 "Free" vs v2 "ADAS") — annotation format and class list differ slightly between them.
- FLIR images are often 8-bit AGC-processed JPEGs rather than raw 16-bit radiometric data; if you want raw thermal values for the autoencoder's reconstruction-error approach, check whether the raw TIFF/radiometric files are included in your download, since AGC-processed images compress the temperature range and can hide subtle heat-signature anomalies.
- FLIR ADAS is captured from a vehicle-mounted (roughly ground-level, forward-facing) camera, not an aerial/downward-facing one — expect a domain gap versus real drone footage (different scale, angle, background clutter). It's a solid dataset to prove the pipeline and get a baseline detector, but flag this gap explicitly in your writeup, and consider augmenting later with any aerial/nadir thermal data you can find or simulate (e.g. via aggressive crop/rotate augmentation to mimic a top-down view).

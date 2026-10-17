"""
Crowd Counting web API.

Run from project root:
  uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Project root = parent of web/
ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from People_Detection import (  # noqa: E402
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    WEBCAM_WEIGHTS,
    CrowdPeopleCounter,
)

DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
THUMB_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "history.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

for path in (UPLOAD_DIR, OUTPUT_DIR, THUMB_DIR, STATIC_DIR):
    path.mkdir(parents=True, exist_ok=True)

_counter_lock = threading.Lock()
_counter: Optional[CrowdPeopleCounter] = None
_webcam_lock = threading.Lock()
_webcam_counter: Optional[CrowdPeopleCounter] = None
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def get_counter() -> CrowdPeopleCounter:
    global _counter
    with _counter_lock:
        if _counter is None:
            _counter = CrowdPeopleCounter()
        return _counter


def get_webcam_counter() -> CrowdPeopleCounter:
    global _webcam_counter
    with _webcam_lock:
        if _webcam_counter is None:
            _webcam_counter = CrowdPeopleCounter(WEBCAM_WEIGHTS, smooth_window=3)
        return _webcam_counter


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, get_webcam_counter)
    yield


app = FastAPI(title="Crowd Counting Dashboard", lifespan=lifespan)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                people_count INTEGER,
                avg_count REAL,
                max_count INTEGER,
                frame_count INTEGER,
                output_url TEXT,
                thumb_url TEXT,
                error TEXT
            )
            """
        )
        conn.commit()


init_db()


class RecordOut(BaseModel):
    id: str
    created_at: str
    source_type: str
    filename: str
    status: str
    people_count: Optional[int] = None
    avg_count: Optional[float] = None
    max_count: Optional[int] = None
    frame_count: Optional[int] = None
    output_url: Optional[str] = None
    thumb_url: Optional[str] = None
    error: Optional[str] = None


class WebcamSessionEnd(BaseModel):
    people_count: int
    avg_count: float
    max_count: int
    frame_count: int
    filename: str = "webcam-session"


def row_to_record(row: sqlite3.Row) -> RecordOut:
    return RecordOut(**dict(row))


def insert_record(record: Dict[str, Any]) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO records (
                id, created_at, source_type, filename, status,
                people_count, avg_count, max_count, frame_count,
                output_url, thumb_url, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["created_at"],
                record["source_type"],
                record["filename"],
                record["status"],
                record.get("people_count"),
                record.get("avg_count"),
                record.get("max_count"),
                record.get("frame_count"),
                record.get("output_url"),
                record.get("thumb_url"),
                record.get("error"),
            ),
        )
        conn.commit()


def update_record(record_id: str, fields: Dict[str, Any]) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    values = list(fields.values()) + [record_id]
    with db_connect() as conn:
        conn.execute(f"UPDATE records SET {cols} WHERE id=?", values)
        conn.commit()


def save_thumb(image_bgr: np.ndarray, record_id: str) -> str:
    thumb_path = THUMB_DIR / f"{record_id}.jpg"
    h, w = image_bgr.shape[:2]
    scale = min(480 / max(w, 1), 320 / max(h, 1), 1.0)
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(thumb_path), resized)
    return f"/media/thumbs/{record_id}.jpg"


def process_image_job(record_id: str, upload_path: Path) -> None:
    try:
        with _jobs_lock:
            _jobs[record_id] = {"status": "processing", "progress": 10}

        image = cv2.imread(str(upload_path))
        if image is None:
            raise RuntimeError("Could not read image")

        counter = get_counter()
        with _counter_lock:
            counter.count_history.clear()
            annotated, raw_count, smooth_count = counter.process_frame(image)

        out_name = f"{record_id}{upload_path.suffix.lower()}"
        out_path = OUTPUT_DIR / out_name
        cv2.imwrite(str(out_path), annotated)
        thumb_url = save_thumb(annotated, record_id)

        update_record(
            record_id,
            {
                "status": "completed",
                "people_count": int(smooth_count),
                "avg_count": float(smooth_count),
                "max_count": int(smooth_count),
                "frame_count": 1,
                "output_url": f"/media/outputs/{out_name}",
                "thumb_url": thumb_url,
                "error": None,
            },
        )
        with _jobs_lock:
            _jobs[record_id] = {"status": "completed", "progress": 100, "people_count": smooth_count}
    except Exception as exc:  # noqa: BLE001 — surface job failure to UI
        update_record(record_id, {"status": "failed", "error": str(exc)})
        with _jobs_lock:
            _jobs[record_id] = {"status": "failed", "error": str(exc)}


def reencode_for_browser(src_path: Path, dest_path: Path) -> bool:
    """Convert OpenCV output to H.264 so Chrome/Safari can play it."""
    try:
        import imageio_ffmpeg
    except ImportError:
        return False

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dest_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0


def process_video_job(record_id: str, upload_path: Path, frame_stride: int = 5) -> None:
    try:
        with _jobs_lock:
            _jobs[record_id] = {"status": "processing", "progress": 5}

        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        out_fps = max(1, fps // max(frame_stride, 1))

        raw_out = OUTPUT_DIR / f"{record_id}_raw.mp4"
        final_out = OUTPUT_DIR / f"{record_id}.mp4"
        writer = cv2.VideoWriter(
            str(raw_out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (width, height),
        )

        counter = get_counter()
        counts: List[int] = []
        thumb_url = None
        idx = 0

        with _counter_lock:
            counter.count_history.clear()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_stride != 0:
                    idx += 1
                    continue

                annotated, _, smooth_count = counter.process_frame(frame)
                counts.append(int(smooth_count))
                writer.write(annotated)

                if thumb_url is None:
                    thumb_url = save_thumb(annotated, record_id)

                idx += 1
                if total > 0:
                    progress = min(90, int(90 * idx / total))
                    with _jobs_lock:
                        _jobs[record_id] = {
                            "status": "processing",
                            "progress": progress,
                            "people_count": smooth_count,
                        }

        cap.release()
        writer.release()

        if not counts:
            raise RuntimeError("No frames processed")

        with _jobs_lock:
            _jobs[record_id] = {"status": "processing", "progress": 92, "people_count": int(np.median(counts))}

        if reencode_for_browser(raw_out, final_out):
            raw_out.unlink(missing_ok=True)
            out_name = final_out.name
        else:
            # Fallback: keep OpenCV file (may not play in all browsers).
            raw_out.replace(final_out)
            out_name = final_out.name

        people_count = int(np.median(counts))
        update_record(
            record_id,
            {
                "status": "completed",
                "people_count": people_count,
                "avg_count": float(np.mean(counts)),
                "max_count": int(np.max(counts)),
                "frame_count": len(counts),
                "output_url": f"/media/outputs/{out_name}",
                "thumb_url": thumb_url,
                "error": None,
            },
        )
        with _jobs_lock:
            _jobs[record_id] = {
                "status": "completed",
                "progress": 100,
                "people_count": people_count,
            }
    except Exception as exc:  # noqa: BLE001
        update_record(record_id, {"status": "failed", "error": str(exc)})
        with _jobs_lock:
            _jobs[record_id] = {"status": "failed", "error": str(exc)}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/records", response_model=List[RecordOut])
def list_records() -> List[RecordOut]:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM records ORDER BY datetime(created_at) DESC"
        ).fetchall()
    return [row_to_record(r) for r in rows]


@app.get("/api/records/{record_id}", response_model=RecordOut)
def get_record(record_id: str) -> RecordOut:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return row_to_record(row)


@app.get("/api/jobs/{record_id}")
def get_job(record_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(record_id)
    if job is None:
        with db_connect() as conn:
            row = conn.execute(
                "SELECT status, people_count, error FROM records WHERE id=?",
                (record_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "status": row["status"],
            "progress": 100 if row["status"] == "completed" else 0,
            "people_count": row["people_count"],
            "error": row["error"],
        }
    return job


@app.delete("/api/records/{record_id}")
def delete_record(record_id: str) -> Dict[str, str]:
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Record not found")
        conn.execute("DELETE FROM records WHERE id=?", (record_id,))
        conn.commit()

    for folder in (UPLOAD_DIR, OUTPUT_DIR, THUMB_DIR):
        for path in folder.glob(f"{record_id}*"):
            path.unlink(missing_ok=True)
    return {"status": "deleted"}


@app.post("/api/upload", response_model=RecordOut)
async def upload_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> RecordOut:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        source_type = "image"
    elif suffix in VIDEO_EXTENSIONS:
        source_type = "video"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'",
        )

    record_id = uuid.uuid4().hex
    upload_path = UPLOAD_DIR / f"{record_id}{suffix}"
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    record = {
        "id": record_id,
        "created_at": utc_now(),
        "source_type": source_type,
        "filename": file.filename,
        "status": "queued",
        "people_count": None,
        "avg_count": None,
        "max_count": None,
        "frame_count": None,
        "output_url": None,
        "thumb_url": None,
        "error": None,
    }
    insert_record(record)
    with _jobs_lock:
        _jobs[record_id] = {"status": "queued", "progress": 0}

    if source_type == "image":
        background_tasks.add_task(process_image_job, record_id, upload_path)
    else:
        background_tasks.add_task(process_video_job, record_id, upload_path)

    return RecordOut(**record)


@app.post("/api/webcam/session", response_model=RecordOut)
async def save_webcam_session(payload: WebcamSessionEnd) -> RecordOut:
    record_id = uuid.uuid4().hex
    record = {
        "id": record_id,
        "created_at": utc_now(),
        "source_type": "webcam",
        "filename": payload.filename or "webcam-session",
        "status": "completed",
        "people_count": int(payload.people_count),
        "avg_count": float(payload.avg_count),
        "max_count": int(payload.max_count),
        "frame_count": int(payload.frame_count),
        "output_url": None,
        "thumb_url": None,
        "error": None,
    }
    insert_record(record)
    return RecordOut(**record)


@app.websocket("/ws/webcam")
async def webcam_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "status", "message": "Loading model…"}))
    loop = asyncio.get_running_loop()
    counter = await loop.run_in_executor(None, get_webcam_counter)
    with _webcam_lock:
        counter.count_history.clear()
    await websocket.send_text(json.dumps({"type": "ready"}))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None:
                text = message.get("text")
                if text == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid frame"})
                )
                continue

            height, width = frame.shape[:2]

            def _count_frame():
                with _webcam_lock:
                    return counter.detect_live(frame, imgsz=320)

            boxes, scores, raw_count, smooth_count = await loop.run_in_executor(
                None, _count_frame
            )

            payload = {
                "type": "count",
                "people_count": int(smooth_count),
                "raw_count": int(raw_count),
                "frame_width": int(width),
                "frame_height": int(height),
                "boxes": [
                    [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1), round(score, 2)]
                    for (x1, y1, x2, y2), score in zip(boxes, scores)
                ],
                "ts": time.time(),
            }
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        return


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/media/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/media/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/media/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

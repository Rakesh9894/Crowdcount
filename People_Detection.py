"""
Improved YOLO people counting for dense crowds.

Techniques used (CSRNet can replace this later):
- Stronger detector: YOLO11m (person class only)
- SAHI-style tiled inference with overlap + NMS (finds small/distant people)
- Higher per-tile resolution
- Temporal median smoothing so the on-screen count is stable
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

# --- Tunables ---
# Best practical YOLO for crowded scenes on a laptop. Use yolo11l.pt / yolo11x.pt
# for max accuracy (slower). CSRNet can replace this path later.
MODEL_WEIGHTS = "yolo11m.pt"
WEBCAM_WEIGHTS = "yolo11n.pt"
PERSON_CLASS_ID = 0
# Higher conf cuts head/torso splits and TV-screen false people.
CONF_THRESHOLD = 0.28
IOU_THRESHOLD = 0.50
# Aggressive merge for SAHI tile duplicates (same person across overlapping tiles).
TILE_NMS_IOU = 0.15
# Drop a box if most of it sits inside a higher-scoring box (head inside body).
CONTAINMENT_THRESHOLD = 0.65
FULL_IMGSZ = 1280
TILE_IMGSZ = 960
# Large frames: run full + tiled, then pick the safer result.
TILE_TRIGGER_PX = 1600
# Prefer tiled only when it finds substantially more people than full-frame.
TILE_GAIN_RATIO = 1.5
TILE_GRID = (3, 3)
TILE_OVERLAP = 0.20
SMOOTH_WINDOW = 9  # median over last N frame counts


def nms_boxes(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = IOU_THRESHOLD,
) -> List[int]:
    """Greedy NMS. boxes are [x1, y1, x2, y2]. Returns kept indices."""
    if not boxes:
        return []

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes_arr.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores_arr.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return keep


def suppress_contained_boxes(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    contain_threshold: float = CONTAINMENT_THRESHOLD,
) -> Tuple[List[List[float]], List[float]]:
    """Remove smaller boxes mostly contained in a larger, higher-score box."""
    if not boxes:
        return [], []

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    areas = (boxes_arr[:, 2] - boxes_arr[:, 0]).clip(min=0) * (
        boxes_arr[:, 3] - boxes_arr[:, 1]
    ).clip(min=0)
    order = scores_arr.argsort()[::-1]
    suppressed = set()

    for oi, i in enumerate(order):
        if i in suppressed:
            continue
        for j in order[oi + 1 :]:
            if j in suppressed:
                continue
            # How much of the smaller box is inside the larger one?
            if areas[i] >= areas[j]:
                large, small = int(i), int(j)
            else:
                large, small = int(j), int(i)
            xx1 = max(boxes_arr[large, 0], boxes_arr[small, 0])
            yy1 = max(boxes_arr[large, 1], boxes_arr[small, 1])
            xx2 = min(boxes_arr[large, 2], boxes_arr[small, 2])
            yy2 = min(boxes_arr[large, 3], boxes_arr[small, 3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            if areas[small] <= 0:
                suppressed.add(small)
                continue
            if inter / areas[small] >= contain_threshold:
                suppressed.add(small)

    keep = [int(i) for i in order if i not in suppressed]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def merge_detections(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> Tuple[List[List[float]], List[float]]:
    """NMS then containment cleanup — kills tile duplicates and head/body splits."""
    keep = nms_boxes(boxes, scores, iou_threshold)
    nmsed_boxes = [boxes[i] for i in keep]
    nmsed_scores = [scores[i] for i in keep]
    return suppress_contained_boxes(nmsed_boxes, nmsed_scores)


def iter_tiles(
    width: int,
    height: int,
    grid: Tuple[int, int] = TILE_GRID,
    overlap: float = TILE_OVERLAP,
) -> List[Tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) tiles covering the frame with overlap."""
    cols, rows = grid
    if cols < 1 or rows < 1:
        raise ValueError("TILE_GRID must be at least 1x1")

    tile_w = width / cols
    tile_h = height / rows
    pad_w = int(tile_w * overlap)
    pad_h = int(tile_h * overlap)

    tiles: List[Tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(c * tile_w) - pad_w)
            y1 = max(0, int(r * tile_h) - pad_h)
            x2 = min(width, int((c + 1) * tile_w) + pad_w)
            y2 = min(height, int((r + 1) * tile_h) + pad_h)
            tiles.append((x1, y1, x2, y2))
    return tiles


class CrowdPeopleCounter:
    def __init__(
        self,
        weights: str = MODEL_WEIGHTS,
        smooth_window: int = SMOOTH_WINDOW,
    ) -> None:
        print(f"[INFO] Loading model: {weights}")
        self.model = YOLO(weights)
        self.count_history: Deque[int] = deque(maxlen=smooth_window)

    def _predict_full(
        self,
        frame: np.ndarray,
        imgsz: int = FULL_IMGSZ,
    ) -> Tuple[List[List[float]], List[float]]:
        results = self.model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=imgsz,
            verbose=False,
        )[0]
        boxes: List[List[float]] = []
        scores: List[float] = []
        if results.boxes is None:
            return boxes, scores
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append([x1, y1, x2, y2])
            scores.append(float(box.conf[0]))
        return merge_detections(boxes, scores, IOU_THRESHOLD)

    def _predict_tiled(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        height, width = frame.shape[:2]
        all_boxes: List[List[float]] = []
        all_scores: List[float] = []

        for x1, y1, x2, y2 in iter_tiles(width, height):
            tile = frame[y1:y2, x1:x2]
            results = self.model.predict(
                tile,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=TILE_IMGSZ,
                verbose=False,
            )[0]
            if results.boxes is None:
                continue
            for box in results.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(box.conf[0]))

        return merge_detections(all_boxes, all_scores, TILE_NMS_IOU)

    def detect_people(self, frame: np.ndarray) -> Tuple[List[List[float]], List[float]]:
        """
        Adaptive strategy:
        1) Full-frame first (correct for rooms / meetings — avoids tile double-counts).
        2) On large frames, also run tiled SAHI. Use tiled only if it finds
           substantially more people (true dense crowd with tiny figures).
        """
        height, width = frame.shape[:2]
        full_boxes, full_scores = self._predict_full(frame)

        if max(width, height) < TILE_TRIGGER_PX:
            return full_boxes, full_scores

        tiled_boxes, tiled_scores = self._predict_tiled(frame)
        if len(full_boxes) == 0 and tiled_boxes:
            return tiled_boxes, tiled_scores
        if len(tiled_boxes) >= max(1, int(len(full_boxes) * TILE_GAIN_RATIO)):
            return tiled_boxes, tiled_scores
        return full_boxes, full_scores

    def annotate(
        self,
        frame: np.ndarray,
        boxes: Sequence[Sequence[float]],
        scores: Sequence[float],
        raw_count: int,
        smooth_count: int,
        mode: str = "full",
    ) -> np.ndarray:
        annotated = frame.copy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))
            cv2.rectangle(annotated, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{score:.2f}",
                (p1[0], max(20, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            annotated,
            f"People Count: {smooth_count}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Raw: {raw_count}  |  YOLO11m adaptive",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def detect_live(
        self,
        frame: np.ndarray,
        imgsz: int = 320,
    ) -> Tuple[List[List[float]], List[float], int, int]:
        """Fast single-scale detect for webcam. Returns boxes, scores, raw, smooth."""
        boxes, scores = self._predict_full(frame, imgsz=imgsz)
        raw_count = len(boxes)
        self.count_history.append(raw_count)
        smooth_count = int(np.median(self.count_history)) if self.count_history else raw_count
        return boxes, scores, raw_count, smooth_count

    def process_frame(
        self,
        frame: np.ndarray,
        fast: bool = False,
    ) -> Tuple[np.ndarray, int, int]:
        if fast:
            boxes, scores = self._predict_full(frame, imgsz=640)
        else:
            boxes, scores = self.detect_people(frame)
        raw_count = len(boxes)
        self.count_history.append(raw_count)
        smooth_count = int(np.median(self.count_history))
        annotated = self.annotate(frame, boxes, scores, raw_count, smooth_count)
        return annotated, raw_count, smooth_count


def process_video(input_path: str, output_video_path: str, counter: CrowdPeopleCounter) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {input_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print(
        f"[INFO] Processing {frame_width}x{frame_height} video "
        f"({total_frames} frames). Press 'q' to stop early."
    )

    frame_idx = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        output_frame, raw_count, smooth_count = counter.process_frame(frame)
        out.write(output_frame)
        cv2.imshow("Object Detection and People Counting", output_frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[INFO] Frame {frame_idx}/{total_frames}  raw={raw_count}  count={smooth_count}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved annotated video to: {output_video_path}")


def process_image(
    input_path: str,
    output_image_path: Optional[str],
    counter: CrowdPeopleCounter,
) -> None:
    image = cv2.imread(input_path)
    if image is None:
        print(f"[ERROR] Could not open image: {input_path}")
        return

    counter.count_history.clear()
    output_frame, raw_count, smooth_count = counter.process_frame(image)
    print(f"[INFO] People count: {smooth_count} (raw={raw_count})")

    if output_image_path:
        ok = cv2.imwrite(output_image_path, output_frame)
        if ok:
            print(f"[INFO] Saved annotated image to: {output_image_path}")
        else:
            print(f"[ERROR] Failed to save image to: {output_image_path}")

    cv2.imshow("Object Detection and People Counting", output_frame)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam(output_video_path: Optional[str], counter: CrowdPeopleCounter) -> None:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print(
            'macOS fix: System Settings → Privacy & Security → Camera → '
            'turn ON "Terminal" or "Cursor".'
        )
        print("Then fully quit and reopen that app, and run the script again.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print('Webcam started. Press "q" to quit.')

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            print("[ERROR] Failed to read from webcam.")
            print(
                "Allow camera access when macOS prompts you, or enable it in "
                "System Settings → Privacy & Security → Camera."
            )
            break

        output_frame, _, _ = counter.process_frame(frame)
        if out is not None:
            out.write(output_frame)

        cv2.imshow("Object Detection and People Counting", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    counter = CrowdPeopleCounter(MODEL_WEIGHTS)
    source = input("Enter image/video path, or type 'webcam': ").strip()

    if source.lower() == "webcam":
        output_path = input("Enter output path to save (or press Enter to skip): ").strip()
        process_webcam(output_path or None, counter)
    else:
        suffix = Path(source).suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            output_path = input(
                "Enter output image path (or press Enter to skip saving): "
            ).strip()
            process_image(source, output_path or None, counter)
        elif suffix in VIDEO_EXTENSIONS or suffix == "":
            output_path = input("Enter the path to save the output video: ").strip()
            if not output_path:
                print("[ERROR] Output path is required for video mode.")
            else:
                process_video(source, output_path, counter)
        else:
            print(
                f"[ERROR] Unsupported file type '{suffix}'. "
                f"Use an image ({', '.join(sorted(IMAGE_EXTENSIONS))}), "
                f"a video ({', '.join(sorted(VIDEO_EXTENSIONS))}), or 'webcam'."
            )

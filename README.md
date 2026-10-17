# CrowdCount

Browser dashboard for people counting in photos, videos, and live webcam using YOLO11 with adaptive tiled inference.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/AkashRajpuria/CrowdCount?quickstart=1)

## Features

- Upload **photo** or **video** and get a people count
- **Live webcam** counting in the browser
- Dashboard history with preview / **Play** for annotated results
- Adaptive detection: full-frame for rooms, tiled SAHI-style inference for dense crowds

## Use it (anyone)

### Option A — GitHub Codespaces (no local install)

1. Open the repo: https://github.com/AkashRajpuria/CrowdCount  
2. Click **Code → Codespaces → Create codespace on main** (or the badge above)  
3. In the Codespace terminal:

```bash
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

4. Open the forwarded **8000** port in the browser

### Option B — Run locally

```bash
git clone https://github.com/AkashRajpuria/CrowdCount.git
cd CrowdCount
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn web.app:app --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000**

On first run, Ultralytics downloads `yolo11m.pt` automatically.

### macOS camera / webcam

Allow **Camera** for your browser (and Terminal if testing CLI) under  
**System Settings → Privacy & Security → Camera**.

## Docker

```bash
docker build -t crowdcount .
docker run --rm -p 8000:8000 crowdcount
```

Then open **http://127.0.0.1:8000**

## Project layout

| Path | Description |
|------|-------------|
| `web/app.py` | FastAPI backend + dashboard APIs |
| `web/static/` | Browser UI |
| `People_Detection.py` | YOLO11 adaptive people counter |
| `human-counting-project-code.py` | Original HOG/SVM CLI demo |

## API (optional)

- `GET /api/records` — dashboard history
- `POST /api/upload` — upload image/video
- `WS /ws/webcam` — live annotated frames + counts

## Notes

- Dense atrium scenes may still undercount vs density models (CSRNet); YOLO is a strong practical baseline.
- Video processing samples frames for speed; outputs are re-encoded to H.264 for browser playback.
- This app needs a machine with Python + enough RAM/CPU (or GPU). Hosted free tiers may be slow or time out on large videos.

## License

MIT

const recordsBody = document.getElementById("recordsBody");
const refreshBtn = document.getElementById("refreshBtn");
const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const uploadMeta = document.getElementById("uploadMeta");
const selectedFile = document.getElementById("selectedFile");
const uploadBtn = document.getElementById("uploadBtn");
const uploadProgress = document.getElementById("uploadProgress");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const startCamBtn = document.getElementById("startCamBtn");
const stopCamBtn = document.getElementById("stopCamBtn");
const webcamVideo = document.getElementById("webcamVideo");
const webcamOverlay = document.getElementById("webcamOverlay");
const liveCount = document.getElementById("liveCount");
const statSessions = document.getElementById("statSessions");
const statLatest = document.getElementById("statLatest");
const mediaModal = document.getElementById("mediaModal");
const mediaModalTitle = document.getElementById("mediaModalTitle");
const mediaModalMeta = document.getElementById("mediaModalMeta");
const mediaPlayer = document.getElementById("mediaPlayer");
const mediaImage = document.getElementById("mediaImage");

let selectedUpload = null;
let pollTimers = new Map();
let mediaStream = null;
let socket = null;
let sendingFrame = false;
let webcamReady = false;
let webcamStats = { counts: [], max: 0, frames: 0 };
let recordsCache = [];
let frameTimeout = null;

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

function formatTime(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function isVideoOutput(url) {
  return /\.(mp4|webm|mov|avi|mkv)(\?|$)/i.test(url || "");
}

function openMediaPreview(record) {
  if (!record?.output_url) return;

  mediaModalTitle.textContent = record.filename || "Preview";
  mediaModalMeta.textContent = `${record.source_type} · ${record.people_count ?? "—"} people · ${formatTime(record.created_at)}`;

  if (isVideoOutput(record.output_url) || record.source_type === "video") {
    mediaImage.hidden = true;
    mediaImage.removeAttribute("src");
    mediaPlayer.hidden = false;
    mediaPlayer.src = record.output_url;
    mediaPlayer.load();
    mediaPlayer.play().catch(() => {});
  } else {
    mediaPlayer.pause();
    mediaPlayer.removeAttribute("src");
    mediaPlayer.hidden = true;
    mediaImage.hidden = false;
    mediaImage.src = record.output_url;
  }

  mediaModal.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeMediaPreview() {
  mediaPlayer.pause();
  mediaPlayer.removeAttribute("src");
  mediaImage.removeAttribute("src");
  mediaModal.hidden = true;
  document.body.style.overflow = "";
}

mediaModal.querySelectorAll("[data-close-modal]").forEach((el) => {
  el.addEventListener("click", closeMediaPreview);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !mediaModal.hidden) {
    closeMediaPreview();
  }
});

function setProgress(pct, text) {
  uploadProgress.hidden = false;
  progressBar.style.setProperty("--p", `${Math.max(4, Math.min(100, pct))}%`);
  progressText.textContent = text;
}

async function fetchRecords() {
  const res = await fetch("/api/records");
  const records = await res.json();
  recordsCache = records;
  renderRecords(records);
  return records;
}

function renderRecords(records) {
  statSessions.textContent = String(records.length);
  const latestDone = records.find((r) => r.status === "completed" && r.people_count != null);
  statLatest.textContent = latestDone ? String(latestDone.people_count) : "—";

  if (!records.length) {
    recordsBody.innerHTML =
      '<tr class="empty-row"><td colspan="7">No uploads yet. Add a photo, video, or webcam session.</td></tr>';
    return;
  }

  recordsBody.innerHTML = records
    .map((r) => {
      const canPlay = Boolean(r.output_url);
      const thumbInner = r.thumb_url
        ? `<img class="thumb" src="${r.thumb_url}" alt="" />`
        : `<div class="thumb"></div>`;
      const thumb = canPlay
        ? `<button class="thumb-btn" type="button" data-play="${r.id}" title="Play / preview">${thumbInner}</button>`
        : thumbInner;
      const avgMax =
        r.avg_count != null
          ? `${Number(r.avg_count).toFixed(1)} / ${r.max_count ?? "—"}`
          : "—";
      const playLabel =
        r.source_type === "video" || isVideoOutput(r.output_url) ? "Play" : "View";
      const playBtn = canPlay
        ? `<button class="btn primary" type="button" data-play="${r.id}">${playLabel}</button>`
        : "";
      return `
        <tr data-id="${r.id}">
          <td>${thumb}</td>
          <td><span class="pill ${r.source_type}">${r.source_type}</span></td>
          <td>
            <div><strong>${escapeHtml(r.filename)}</strong></div>
            <div style="color:var(--muted);font-size:0.8rem">${formatTime(r.created_at)}</div>
          </td>
          <td><span class="count-num">${r.people_count ?? "—"}</span></td>
          <td>${avgMax}</td>
          <td><span class="pill ${r.status}">${r.status}</span>${
            r.error ? `<div style="color:var(--danger);font-size:0.75rem">${escapeHtml(r.error)}</div>` : ""
          }</td>
          <td>
            <div class="row-actions">
              ${playBtn}
              <button class="btn danger" data-delete="${r.id}" type="button">Delete</button>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  recordsBody.querySelectorAll("[data-play]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-play");
      const record = recordsCache.find((item) => item.id === id);
      openMediaPreview(record);
    });
  });

  recordsBody.querySelectorAll("[data-delete]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-delete");
      await fetch(`/api/records/${id}`, { method: "DELETE" });
      fetchRecords();
    });
  });

  records
    .filter((r) => r.status === "queued" || r.status === "processing")
    .forEach((r) => watchJob(r.id));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function watchJob(id) {
  if (pollTimers.has(id)) return;
  const timer = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${id}`);
      const job = await res.json();
      if (job.status === "processing" || job.status === "queued") {
        setProgress(
          job.progress ?? 10,
          `Processing… ${job.people_count != null ? `count ${job.people_count}` : ""}`
        );
      }
      if (job.status === "completed" || job.status === "failed") {
        clearInterval(timer);
        pollTimers.delete(id);
        uploadProgress.hidden = job.status !== "failed";
        if (job.status === "completed") {
          setProgress(100, `Done — ${job.people_count} people`);
          setTimeout(() => {
            uploadProgress.hidden = true;
          }, 1500);
        } else {
          setProgress(100, `Failed: ${job.error || "unknown error"}`);
        }
        fetchRecords();
      }
    } catch {
      // keep polling
    }
  }, 1200);
  pollTimers.set(id, timer);
}

function onFileChosen(file) {
  if (!file) return;
  selectedUpload = file;
  selectedFile.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
  uploadMeta.hidden = false;
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => onFileChosen(fileInput.files?.[0]));

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  onFileChosen(file);
});

uploadBtn.addEventListener("click", async () => {
  if (!selectedUpload) return;
  const form = new FormData();
  form.append("file", selectedUpload);
  uploadBtn.disabled = true;
  setProgress(5, "Uploading…");

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Upload failed");
    }
    const record = await res.json();
    setProgress(15, "Queued for counting…");
    selectedUpload = null;
    uploadMeta.hidden = true;
    fileInput.value = "";
    await fetchRecords();
    watchJob(record.id);
  } catch (err) {
    setProgress(100, err.message || "Upload failed");
  } finally {
    uploadBtn.disabled = false;
  }
});

function drawOverlay(msg) {
  if (!webcamOverlay || !webcamVideo.videoWidth) return;
  const rect = webcamVideo.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  webcamOverlay.width = Math.round(rect.width * dpr);
  webcamOverlay.height = Math.round(rect.height * dpr);
  const ctx = webcamOverlay.getContext("2d");
  ctx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);

  const srcW = msg.frame_width || 1;
  const srcH = msg.frame_height || 1;
  const scaleX = webcamOverlay.width / srcW;
  const scaleY = webcamOverlay.height / srcH;

  ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = "#22c55e";
  ctx.font = `${12 * dpr}px sans-serif`;
  ctx.fillStyle = "#22c55e";
  (msg.boxes || []).forEach((box) => {
    const [x1, y1, x2, y2, score] = box;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;
    ctx.strokeRect(x, y, w, h);
    if (score != null) {
      ctx.fillText(Number(score).toFixed(2), x + 4, Math.max(14 * dpr, y - 4));
    }
  });
}

function markFrameDone() {
  sendingFrame = false;
  if (frameTimeout) {
    clearTimeout(frameTimeout);
    frameTimeout = null;
  }
}

async function startWebcam() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }

  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  webcamVideo.srcObject = mediaStream;
  webcamVideo.hidden = false;
  liveCount.textContent = "…";
  webcamStats = { counts: [], max: 0, frames: 0 };
  sendingFrame = false;
  webcamReady = false;
  if (webcamOverlay) {
    const ctx = webcamOverlay.getContext("2d");
    ctx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
  }

  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws/webcam`);
  socket.binaryType = "arraybuffer";

  socket.addEventListener("open", () => {
    startCamBtn.disabled = true;
    stopCamBtn.disabled = false;
    liveCount.textContent = "load";
    pumpWebcamFrames();
  });

  socket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") {
      markFrameDone();
      return;
    }
    const msg = JSON.parse(event.data);
    if (msg.type === "status") {
      liveCount.textContent = "…";
      return;
    }
    if (msg.type === "ready") {
      webcamReady = true;
      liveCount.textContent = "0";
      return;
    }
    if (msg.type === "error") {
      markFrameDone();
      return;
    }
    if (msg.type === "count") {
      liveCount.textContent = String(msg.people_count);
      webcamStats.counts.push(msg.people_count);
      webcamStats.max = Math.max(webcamStats.max, msg.people_count);
      webcamStats.frames += 1;
      drawOverlay(msg);
      markFrameDone();
    }
  });

  socket.addEventListener("close", () => {
    webcamReady = false;
    markFrameDone();
    startCamBtn.disabled = false;
    stopCamBtn.disabled = true;
  });
}

function pumpWebcamFrames() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  const tick = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (webcamReady && !sendingFrame && webcamVideo.videoWidth > 0) {
      const maxW = 320;
      const scale = Math.min(1, maxW / webcamVideo.videoWidth);
      canvas.width = Math.round(webcamVideo.videoWidth * scale);
      canvas.height = Math.round(webcamVideo.videoHeight * scale);
      ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (!blob || !socket || socket.readyState !== WebSocket.OPEN) return;
          sendingFrame = true;
          frameTimeout = setTimeout(() => {
            sendingFrame = false;
          }, 4000);
          blob.arrayBuffer().then((buf) => socket.send(buf));
        },
        "image/jpeg",
        0.5
      );
    }
    setTimeout(tick, 250);
  };
  tick();
}

async function stopWebcam() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.close();
  }
  socket = null;
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  webcamVideo.srcObject = null;
  if (webcamOverlay) {
    const ctx = webcamOverlay.getContext("2d");
    ctx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
  }
  startCamBtn.disabled = false;
  stopCamBtn.disabled = true;

  if (webcamStats.frames > 0) {
    const avg =
      webcamStats.counts.reduce((a, b) => a + b, 0) / webcamStats.counts.length;
    const people = Math.round(
      webcamStats.counts.slice().sort((a, b) => a - b)[
        Math.floor(webcamStats.counts.length / 2)
      ]
    );
    await fetch("/api/webcam/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        people_count: people,
        avg_count: avg,
        max_count: webcamStats.max,
        frame_count: webcamStats.frames,
        filename: `webcam-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}`,
      }),
    });
    fetchRecords();
  }
}

startCamBtn.addEventListener("click", () => {
  startWebcam().catch((err) => {
    alert(err.message || "Could not access webcam");
  });
});
stopCamBtn.addEventListener("click", stopWebcam);
refreshBtn.addEventListener("click", fetchRecords);

fetchRecords();

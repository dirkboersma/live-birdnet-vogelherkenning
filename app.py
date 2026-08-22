import csv
import json
import os
import queue
import re
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal
except Exception as exc:  # pragma: no cover - shown in the web UI at runtime
    plt = None
    signal = None
    SPECTROGRAM_IMPORT_ERROR = exc
else:
    SPECTROGRAM_IMPORT_ERROR = None

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - shown in the web UI at runtime
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc
else:
    SOUNDDEVICE_IMPORT_ERROR = None

try:
    from birdnetlib import RecordingBuffer
    from birdnetlib.analyzer import Analyzer
except Exception as exc:  # pragma: no cover - shown in the web UI at runtime
    RecordingBuffer = None
    Analyzer = None
    BIRDNET_IMPORT_ERROR = exc
else:
    BIRDNET_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
CLIPS_DIR = DATA_DIR / "detections"
SPECTROGRAMS_DIR = DATA_DIR / "spectrograms"
UPLOADS_DIR = DATA_DIR / "uploads"
LIVE_CHUNKS_DIR = DATA_DIR / "live-chunks"
CONFIG_DIR = BASE_DIR / "config"
DUTCH_NAMES_CSV = CONFIG_DIR / "dutch_names.csv"

for directory in (RECORDINGS_DIR, CLIPS_DIR, SPECTROGRAMS_DIR, UPLOADS_DIR, LIVE_CHUNKS_DIR, CONFIG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


SAMPLE_RATE = env_int("BIRDNET_SAMPLE_RATE", 48_000)
CHANNELS = 1
ANALYSIS_SECONDS = env_float("BIRDNET_ANALYSIS_SECONDS", 3.0)
MIN_CONFIDENCE = env_float("BIRDNET_MIN_CONFIDENCE", 0.60)
PRE_ROLL_SECONDS = env_float("BIRDNET_PRE_ROLL_SECONDS", 1.0)
POST_ROLL_SECONDS = env_float("BIRDNET_POST_ROLL_SECONDS", 1.0)
RING_BUFFER_SECONDS = env_float("BIRDNET_RING_BUFFER_SECONDS", 90.0)
LIVE_SPECTROGRAM_SECONDS = env_float("BIRDNET_LIVE_SPECTROGRAM_SECONDS", 5.0)
SPECTROGRAM_MAX_FREQ = env_float("BIRDNET_SPECTROGRAM_MAX_FREQ", 12_000.0)
BIRDNET_MODEL_VERSION = os.getenv("BIRDNET_MODEL_VERSION", "2.4")
FULL_RECORD_FORMAT = os.getenv("BIRDNET_FULL_RECORD_FORMAT", "mp3").lower()
FULL_RECORD_MP3_BITRATE = os.getenv("BIRDNET_FULL_RECORD_MP3_BITRATE", "128k")
AUDIO_FORMATS = {
    "wav": {"extension": "wav", "mime": "audio/wav"},
    "mp3": {"extension": "mp3", "mime": "audio/mpeg"},
    "mp4": {"extension": "m4a", "mime": "audio/mp4"},
    "m4a": {"extension": "m4a", "mime": "audio/mp4"},
}
if FULL_RECORD_FORMAT not in AUDIO_FORMATS:
    FULL_RECORD_FORMAT = "mp3"
ALLOWED_UPLOAD_EXTENSIONS = {".mp3", ".mp4", ".m4a", ".mp4a", ".wav", ".aiff", ".aif", ".webm", ".ogg"}
MAX_LIVE_CHUNK_BYTES = env_int("BIRDNET_MAX_LIVE_CHUNK_BYTES", 3 * 1024 * 1024)
LIVE_CHUNK_QUEUE_SIZE = env_int("BIRDNET_LIVE_CHUNK_QUEUE_SIZE", 2)

# Standaard op het geografisch midden van Nederland. Pas dit lokaal aan voor
# betere resultaten, bijvoorbeeld: BIRDNET_LAT=52.37 BIRDNET_LON=4.90.
BIRDNET_LAT = env_float("BIRDNET_LAT", 52.1326)
BIRDNET_LON = env_float("BIRDNET_LON", 5.2913)


INDEX_HTML = """
<!doctype html>
<html lang="nl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live vogelgeluiden</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8f5;
      --panel: #ffffff;
      --ink: #1d2420;
      --muted: #68736d;
      --line: #dce5df;
      --accent: #176b4d;
      --accent-2: #c36b26;
      --bad: #a33a32;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(980px, calc(100% - 32px));
      margin: 32px auto;
    }
    header {
      display: flex;
      gap: 16px;
      justify-content: space-between;
      align-items: flex-end;
      margin-bottom: 20px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: 30px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      margin-bottom: 16px;
    }
    button {
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 0 14px;
      font: inherit;
      font-weight: 650;
      background: #fff;
      color: var(--ink);
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    button.warning {
      color: #fff;
      background: var(--accent-2);
      border-color: var(--accent-2);
    }
    button:disabled {
      opacity: .5;
      cursor: not-allowed;
    }
    .upload-form {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    input[type="file"] {
      max-width: 260px;
      font: inherit;
      color: var(--muted);
    }
    .status {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .metric b {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .metric span {
      display: block;
      font-size: 18px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .panel {
      padding: 0;
      overflow: hidden;
    }
    .panel h2 {
      font-size: 16px;
      margin: 0;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
      background: #fbfcfb;
    }
    tr:last-child td { border-bottom: 0; }
    .bird-thumb {
      width: 100px;
      height: 100px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      object-fit: contain;
    }
    .spectrogram-thumb {
      width: 180px;
      height: 100px;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #141414;
      object-fit: cover;
    }
    .spectrogram-live {
      width: 100%;
      height: 260px;
      display: block;
      background: #101010;
    }
    .panel-note {
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }
    .bird-name {
      font-weight: 700;
    }
    .empty {
      padding: 28px 14px;
      color: var(--muted);
    }
    .error {
      color: var(--bad);
      font-weight: 700;
    }
    a { color: var(--accent); font-weight: 650; }
    @media (max-width: 760px) {
      header { display: block; }
      .toolbar { align-items: stretch; }
      .upload-form { width: 100%; }
      button { flex: 1; }
      .status { grid-template-columns: 1fr 1fr; }
      th:nth-child(6), td:nth-child(6) { display: none; }
      .spectrogram-live { height: 190px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Live vogelgeluiden</h1>
        <p>Herken vogelgeluiden met de microfoon van je browser.</p>
      </div>
    </header>

    <section class="toolbar">
      <button id="browserStart" class="primary">Start browsermicrofoon</button>
      <button id="browserStop" class="warning" disabled>Stop browsermicrofoon</button>
      <button id="refresh">Ververs status</button>
      <form id="uploadForm" class="upload-form">
        <input id="uploadFile" name="audio_file" type="file" accept=".mp3,.mp4,.m4a,.mp4a,.wav,.aiff,.aif,audio/*">
        <button id="uploadButton" type="submit">Analyseer bestand</button>
      </form>
      <span id="message"></span>
    </section>

    <section class="panel" style="margin-bottom: 16px;">
      <h2>Browsermicrofoon</h2>
      <p>Elke drie seconden verstuurt de browser een compact audiofragment. Chrome en Firefox gebruiken waar mogelijk WebM/Opus; Safari gebruikt een geschikt eigen formaat. De server zet het fragment tijdelijk om voor BirdNET en verwijdert het daarna.</p>
      <details style="margin-top: 12px;">
        <summary>Lokale microfoon van de server</summary>
        <p style="margin: 8px 0;">Alleen voor gebruik op de computer waarop de app draait.</p>
        <button id="start">Start lokale opname</button>
        <button id="stop" class="warning" disabled>Stop lokale opname</button>
      </details>
    </section>

    <section class="status">
      <div class="metric"><b>Status</b><span id="state">-</span></div>
      <div class="metric"><b>Min. zekerheid</b><span id="confidence">-</span></div>
      <div class="metric"><b>Volledige opname</b><span id="recording">-</span></div>
      <div class="metric"><b>Invoerapparaat</b><span id="inputDevice">-</span></div>
      <div class="metric"><b>Vóór MP3</b><span id="wavSize">-</span></div>
      <div class="metric"><b>Na MP3</b><span id="compressedSize">-</span></div>
      <div class="metric"><b>Uploadanalyse</b><span id="uploadState">-</span></div>
      <div class="metric"><b>Browseranalyse</b><span id="browserState">-</span></div>
      <div class="metric"><b>Locatie</b><span id="location">-</span></div>
    </section>

    <section class="panel" style="margin-bottom: 16px;">
      <h2>Live spectrogram</h2>
      <div id="spectrogramState" class="panel-note">Start de opname om live frequenties te zien.</div>
      <canvas id="liveSpectrogram" class="spectrogram-live" width="900" height="260"></canvas>
    </section>

    <section class="panel">
      <h2>Live detecties</h2>
      <table>
        <thead>
          <tr>
            <th style="width: 160px;">Tijd</th>
            <th style="width: 128px;">Afbeelding</th>
            <th style="width: 210px;">Spectrogram</th>
            <th>Vogelnaam</th>
            <th style="width: 120px;">Zekerheid</th>
            <th>Fragment</th>
          </tr>
        </thead>
        <tbody id="detections">
          <tr><td colspan="6" class="empty">Nog geen detecties.</td></tr>
        </tbody>
      </table>
    </section>
  </main>

  <script>
    const els = {
      browserStart: document.querySelector("#browserStart"),
      browserStop: document.querySelector("#browserStop"),
      start: document.querySelector("#start"),
      stop: document.querySelector("#stop"),
      refresh: document.querySelector("#refresh"),
      uploadForm: document.querySelector("#uploadForm"),
      uploadFile: document.querySelector("#uploadFile"),
      uploadButton: document.querySelector("#uploadButton"),
      message: document.querySelector("#message"),
      state: document.querySelector("#state"),
      confidence: document.querySelector("#confidence"),
      recording: document.querySelector("#recording"),
      inputDevice: document.querySelector("#inputDevice"),
      wavSize: document.querySelector("#wavSize"),
      compressedSize: document.querySelector("#compressedSize"),
      uploadState: document.querySelector("#uploadState"),
      browserState: document.querySelector("#browserState"),
      location: document.querySelector("#location"),
      detections: document.querySelector("#detections"),
      liveSpectrogram: document.querySelector("#liveSpectrogram"),
      spectrogramState: document.querySelector("#spectrogramState")
    };

    let hasRows = false;
    let spectrogramRequestInFlight = false;
    let browserRecorder = null;
    let browserStream = null;
    let browserSequence = 0;
    let browserRecordingActive = false;
    let browserSegmentTimer = null;
    const browserUploads = new Set();

    function setMessage(text, isError = false) {
      els.message.textContent = text || "";
      els.message.className = isError ? "error" : "";
    }

    async function postJson(path) {
      const response = await fetch(path, { method: "POST" });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || "Onbekende fout");
      }
      return data;
    }

    async function uploadAudioFile() {
      if (!els.uploadFile.files.length) {
        setMessage("Kies eerst een audiobestand.", true);
        return;
      }
      const formData = new FormData();
      formData.append("audio_file", els.uploadFile.files[0]);
      els.uploadButton.disabled = true;
      setMessage("Upload wordt gestart...");
      try {
        const response = await fetch("/api/upload", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok || data.ok === false) {
          throw new Error(data.error || "Uploadanalyse kon niet worden gestart.");
        }
        setMessage(data.message || "Uploadanalyse gestart.");
      } catch (error) {
        setMessage(error.message, true);
      } finally {
        els.uploadButton.disabled = false;
      }
    }

    async function refreshStatus() {
      const response = await fetch("/api/status");
      const data = await response.json();
      els.state.textContent = data.running ? "Opname actief" : "Gestopt";
      els.confidence.textContent = `${Math.round(data.min_confidence * 100)}%`;
      els.location.textContent = `${data.lat}, ${data.lon}`;
      els.inputDevice.textContent = data.input_device?.name || "-";
      els.wavSize.textContent = formatMb(data.wav_before_compression_mb);
      els.compressedSize.textContent = formatMb(data.compressed_recording_mb);
      els.uploadState.textContent = data.upload_running ? `Bezig: ${data.upload_name}` : "Geen";
      const browserParts = [];
      if (data.browser_live_processing) browserParts.push("analyseren");
      if (data.browser_live_pending) browserParts.push(`wachtrij: ${data.browser_live_pending}`);
      els.browserState.textContent = browserParts.length ? browserParts.join(" · ") : "Geen";
      els.recording.innerHTML = data.recording_url
        ? `<a href="${data.recording_url}">${data.recording_name}</a>`
        : `${String(data.recording_format || "").toUpperCase()} na stoppen`;
      els.start.disabled = data.running;
      els.stop.disabled = !data.running;
      if (data.error) setMessage(data.error, true);
    }

    function browserAudioOptions() {
      const supportedTypes = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4;codecs=mp4a.40.2",
        "audio/mp4"
      ];
      const mimeType = supportedTypes.find((type) => MediaRecorder.isTypeSupported(type));
      return mimeType ? { mimeType } : {};
    }

    function browserAudioExtension(mimeType) {
      return mimeType.includes("webm") ? "webm" : "m4a";
    }

    async function uploadBrowserChunk(blob, mimeType) {
      if (!blob.size) return;
      const detectedMimeType = blob.type || mimeType || "audio/webm";
      const extension = browserAudioExtension(detectedMimeType);
      const formData = new FormData();
      formData.append("audio_file", blob, `browser-${Date.now()}-${++browserSequence}.${extension}`);

      const task = (async () => {
        const response = await fetch("/api/live-chunk", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok || data.ok === false) {
          throw new Error(data.error || "Browserfragment kon niet worden verstuurd.");
        }
        if (data.message) setMessage(data.message);
      })();
      browserUploads.add(task);
      try {
        await task;
      } catch (error) {
        setMessage(error.message, true);
      } finally {
        browserUploads.delete(task);
      }
    }

    function stopBrowserSegment() {
      window.clearTimeout(browserSegmentTimer);
      browserSegmentTimer = null;
      if (browserRecorder?.state === "recording") browserRecorder.stop();
    }

    function finishBrowserMicrophone() {
      browserStream?.getTracks().forEach((track) => track.stop());
      browserStream = null;
      browserRecorder = null;
      Promise.allSettled([...browserUploads]).then(() => {
        els.browserStart.disabled = false;
        els.browserStop.disabled = true;
        setMessage("Browsermicrofoon gestopt.");
        refreshStatus();
      });
    }

    function startBrowserSegment() {
      if (!browserRecordingActive || !browserStream) return;
      const segmentRecorder = new MediaRecorder(browserStream, browserAudioOptions());
      browserRecorder = segmentRecorder;
      const mimeType = segmentRecorder.mimeType;
      segmentRecorder.addEventListener("dataavailable", (event) => uploadBrowserChunk(event.data, mimeType));
      segmentRecorder.addEventListener("stop", () => {
        if (browserRecorder === segmentRecorder) browserRecorder = null;
        if (browserRecordingActive) {
          window.setTimeout(startBrowserSegment, 0);
        } else {
          finishBrowserMicrophone();
        }
      });
      segmentRecorder.start();
      browserSegmentTimer = window.setTimeout(stopBrowserSegment, 3000);
    }

    async function startBrowserMicrophone() {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error("Deze browser ondersteunt geen opname via de microfoon.");
      }
      browserStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
      });
      browserRecordingActive = true;
      startBrowserSegment();
      els.browserStart.disabled = true;
      els.browserStop.disabled = false;
      setMessage("Browsermicrofoon actief; elk fragment duurt drie seconden.");
    }

    els.browserStart.addEventListener("click", async () => {
      try {
        await startBrowserMicrophone();
      } catch (error) {
        browserStream?.getTracks().forEach((track) => track.stop());
        browserStream = null;
        browserRecorder = null;
        browserRecordingActive = false;
        setMessage(error.message, true);
      }
    });

    els.browserStop.addEventListener("click", () => {
      browserRecordingActive = false;
      if (browserRecorder?.state === "recording") {
        stopBrowserSegment();
      } else {
        finishBrowserMicrophone();
      }
    });

    function formatMb(value) {
      if (value === null || value === undefined) return "-";
      return `${Number(value).toFixed(2)} MB`;
    }

    function escapeHtml(value) {
      return String(value || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function birdPlaceholder(name) {
      const label = encodeURIComponent(name || "Vogel");
      return `data:image/svg+xml;charset=UTF-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 100 100' role='img' aria-label='${label}'%3E%3Crect width='100' height='100' fill='white'/%3E%3Cpath d='M24 60c8-20 27-31 48-23 8 3 13 7 18 14-12-1-19 1-26 8-10 10-25 16-40 1Z' fill='black'/%3E%3Ccircle cx='69' cy='40' r='3' fill='white'/%3E%3Cpath d='M82 42l13-5-7 11Z' fill='black'/%3E%3Cpath d='M39 63l-7 22M50 64l3 21' stroke='black' stroke-width='5' stroke-linecap='round'/%3E%3Cpath d='M18 63c9 1 17 0 25-4' stroke='black' stroke-width='6' stroke-linecap='round' fill='none'/%3E%3C/svg%3E`;
    }

    function colorForDb(value, minDb, maxDb) {
      const t = Math.max(0, Math.min(1, (value - minDb) / (maxDb - minDb)));
      const stops = [
        [18, 18, 24],
        [64, 20, 94],
        [154, 48, 90],
        [236, 114, 38],
        [252, 220, 92]
      ];
      const scaled = t * (stops.length - 1);
      const index = Math.min(stops.length - 2, Math.floor(scaled));
      const local = scaled - index;
      return stops[index].map((start, channel) => {
        const end = stops[index + 1][channel];
        return Math.round(start + (end - start) * local);
      });
    }

    function drawSpectrogram(canvas, spectrogram) {
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      const matrix = spectrogram.db || [];
      const rows = matrix.length;
      const cols = rows ? matrix[0].length : 0;
      if (!rows || !cols) {
        ctx.fillStyle = "#101010";
        ctx.fillRect(0, 0, width, height);
        return;
      }

      const plotLeft = 48;
      const plotRight = 10;
      const plotTop = 10;
      const plotBottom = 28;
      const plotWidth = width - plotLeft - plotRight;
      const plotHeight = height - plotTop - plotBottom;
      const cellWidth = Math.ceil(plotWidth / cols);
      const cellHeight = Math.ceil(plotHeight / rows);

      ctx.fillStyle = "#101010";
      ctx.fillRect(0, 0, width, height);

      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const [r, g, b] = colorForDb(matrix[y][x], spectrogram.min_db, spectrogram.max_db);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(
            plotLeft + x * (plotWidth / cols),
            plotTop + (rows - y - 1) * (plotHeight / rows),
            cellWidth,
            cellHeight
          );
        }
      }

      ctx.strokeStyle = "rgba(255,255,255,.35)";
      ctx.lineWidth = 1;
      ctx.strokeRect(plotLeft, plotTop, plotWidth, plotHeight);
      ctx.fillStyle = "rgba(255,255,255,.75)";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("kHz", 12, 18);
      ctx.fillText("0", 26, plotTop + plotHeight);
      const maxFrequency = spectrogram.frequencies.at(-1) || 0;
      ctx.fillText(String(Math.round(maxFrequency / 1000)), 22, plotTop + 10);
      ctx.fillText("tijd", plotLeft + plotWidth - 30, height - 8);
    }

    async function refreshLiveSpectrogram() {
      if (spectrogramRequestInFlight) return;
      spectrogramRequestInFlight = true;
      try {
        const response = await fetch("/api/spectrogram/latest");
        const text = await response.text();
        const data = text ? JSON.parse(text) : { ok: false, error: "Geen spectrogram-response ontvangen." };
        if (!response.ok || data.ok === false) {
          els.spectrogramState.textContent = data.error || "Nog geen spectrogram beschikbaar.";
          drawSpectrogram(els.liveSpectrogram, { db: [] });
          return;
        }
        els.spectrogramState.textContent = `Laatste ${data.seconds}s audio, ${data.frequencies.length} frequentiebanen.`;
        drawSpectrogram(els.liveSpectrogram, data);
      } catch (error) {
        els.spectrogramState.textContent = error.message;
      } finally {
        spectrogramRequestInFlight = false;
      }
    }

    function addDetection(item) {
      if (!hasRows) {
        els.detections.innerHTML = "";
        hasRows = true;
      }
      const row = document.createElement("tr");
      const confidence = Math.round(item.confidence * 1000) / 10;
      const birdName = escapeHtml(item.dutch_name);
      const detectedAt = escapeHtml(item.detected_at);
      const clipName = escapeHtml(item.clip_name);
      row.innerHTML = `
        <td>${detectedAt}</td>
        <td><img class="bird-thumb" src="${birdPlaceholder(item.dutch_name)}" width="100" height="100" alt="${birdName}"></td>
        <td>${item.spectrogram_url ? `<img class="spectrogram-thumb" src="${item.spectrogram_url}" width="180" height="100" alt="Spectrogram ${birdName}">` : "-"}</td>
        <td class="bird-name">${birdName}</td>
        <td>${confidence}%</td>
        <td>${item.clip_url ? `<a href="${item.clip_url}">${clipName}</a>` : "-"}</td>
      `;
      els.detections.prepend(row);
    }

    els.start.addEventListener("click", async () => {
      setMessage("Opname wordt gestart...");
      try {
        const data = await postJson("/api/start");
        setMessage(data.message || "Opname gestart.");
      } catch (error) {
        setMessage(error.message, true);
      }
      refreshStatus();
    });

    els.stop.addEventListener("click", async () => {
      setMessage("Opname wordt gestopt...");
      try {
        const data = await postJson("/api/stop");
        setMessage(data.message || "Opname gestopt.");
      } catch (error) {
        setMessage(error.message, true);
      }
      refreshStatus();
    });

    els.refresh.addEventListener("click", refreshStatus);
    els.uploadForm.addEventListener("submit", (event) => {
      event.preventDefault();
      uploadAudioFile();
    });
    setInterval(refreshLiveSpectrogram, 500);

    const events = new EventSource("/events");
    events.addEventListener("status", (event) => {
      const data = JSON.parse(event.data);
      if (data.message) setMessage(data.message, data.level === "error");
      refreshStatus();
    });
    events.addEventListener("detection", (event) => {
      addDetection(JSON.parse(event.data));
      refreshStatus();
    });
    events.onerror = () => setMessage("Live verbinding wordt opnieuw opgebouwd...", true);

    refreshStatus();
    refreshLiveSpectrogram();
  </script>
</body>
</html>
"""


def slugify_filename(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^\w.-]+", "", value, flags=re.UNICODE)
    return value.strip("._-") or "onbekende_vogel"


def unique_path(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())


def read_wav_int16(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples, sample_rate


def convert_audio_to_analysis_wav(input_path: Path, output_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip().splitlines()
        if detail:
            print(f"FFmpeg-conversiefout: {detail[-1]}")
        raise RuntimeError("FFmpeg kon dit audiofragment niet lezen.") from exc


def compressed_recording_path(temp_wav_path: Path) -> Path:
    extension = AUDIO_FORMATS[FULL_RECORD_FORMAT]["extension"]
    return temp_wav_path.with_name(temp_wav_path.name.replace(".tmp.wav", f".{extension}"))


def file_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return path.stat().st_size


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / (1024 * 1024), 3)


def default_input_device_info() -> dict[str, Any]:
    if sd is None:
        return {"name": "Onbekend", "index": None, "sample_rate": SAMPLE_RATE}
    try:
        default_device = sd.default.device
        try:
            input_index = default_device[0]
        except TypeError:
            input_index = default_device
        device = sd.query_devices(input_index, kind="input")
        return {
            "name": str(device.get("name") or "Onbekend"),
            "index": int(device.get("index") if device.get("index") is not None else input_index),
            "sample_rate": int(device.get("default_samplerate") or SAMPLE_RATE),
            "channels": int(device.get("max_input_channels") or CHANNELS),
        }
    except Exception as exc:
        return {
            "name": f"Onbekend ({exc})",
            "index": None,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
        }


def compress_full_recording(temp_wav_path: Path, output_path: Path) -> None:
    if FULL_RECORD_FORMAT == "wav":
        temp_wav_path.replace(output_path)
        return

    if FULL_RECORD_FORMAT == "mp3":
        codec_args = ["-codec:a", "libmp3lame", "-b:a", FULL_RECORD_MP3_BITRATE]
    else:
        codec_args = ["-codec:a", "aac", "-b:a", FULL_RECORD_MP3_BITRATE]

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temp_wav_path),
        *codec_args,
        str(output_path),
    ]
    subprocess.run(command, check=True)
    temp_wav_path.unlink(missing_ok=True)


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples[:, 0]
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767).astype(np.int16)


def int16_to_float32(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=np.float32) / 32768.0


def spectrogram_data(
    samples: np.ndarray,
    sample_rate: int,
    max_frequency: float = SPECTROGRAM_MAX_FREQ,
    max_time_bins: int = 240,
    max_frequency_bins: int = 160,
) -> dict[str, Any]:
    if signal is None:
        raise RuntimeError(f"Spectrogram-libraries ontbreken: {SPECTROGRAM_IMPORT_ERROR}")

    samples = np.asarray(samples).reshape(-1)
    if samples.dtype != np.float32 and samples.dtype != np.float64:
        samples = int16_to_float32(samples)
    samples = samples.astype(np.float32)
    samples = samples - float(np.mean(samples))

    if len(samples) < 512:
        raise ValueError("Niet genoeg audio beschikbaar voor een spectrogram.")

    nperseg = min(1024, len(samples))
    noverlap = min(768, nperseg // 2)
    frequencies, times, spectrum = signal.spectrogram(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )

    frequency_mask = frequencies <= max_frequency
    frequencies = frequencies[frequency_mask]
    spectrum = spectrum[frequency_mask, :]
    db = 20 * np.log10(spectrum + 1e-10)
    db = np.clip(db, -100, -20)

    if db.shape[0] > max_frequency_bins:
        frequency_indexes = np.linspace(0, db.shape[0] - 1, max_frequency_bins).astype(int)
        db = db[frequency_indexes, :]
        frequencies = frequencies[frequency_indexes]

    if db.shape[1] > max_time_bins:
        time_indexes = np.linspace(0, db.shape[1] - 1, max_time_bins).astype(int)
        db = db[:, time_indexes]
        times = times[time_indexes]

    return {
        "times": np.round(times, 3).tolist(),
        "frequencies": np.round(frequencies, 1).tolist(),
        "db": np.round(db, 1).tolist(),
        "min_db": -100,
        "max_db": -20,
    }


def save_spectrogram_png(path: Path, samples: np.ndarray, sample_rate: int, title: str) -> None:
    if plt is None:
        raise RuntimeError(f"Spectrogram-libraries ontbreken: {SPECTROGRAM_IMPORT_ERROR}")

    data = spectrogram_data(
        samples,
        sample_rate,
        max_time_bins=360,
        max_frequency_bins=220,
    )
    matrix = np.asarray(data["db"], dtype=np.float32)
    extent = [
        min(data["times"] or [0]),
        max(data["times"] or [0]),
        min(data["frequencies"] or [0]) / 1000,
        max(data["frequencies"] or [0]) / 1000,
    ]

    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=120)
    ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=data["min_db"],
        vmax=data["max_db"],
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Tijd (s)")
    ax.set_ylabel("Frequentie (kHz)")
    fig.tight_layout()
    fig.savefig(path, format="png")
    plt.close(fig)


class EventBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[tuple[str, dict[str, Any]]]] = []

    def subscribe(self) -> queue.Queue[tuple[str, dict[str, Any]]]:
        client_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(maxsize=100)
        with self._lock:
            self._clients.append(client_queue)
        return client_queue

    def unsubscribe(self, client_queue: queue.Queue[tuple[str, dict[str, Any]]]) -> None:
        with self._lock:
            if client_queue in self._clients:
                self._clients.remove(client_queue)

    def publish(self, event: str, payload: dict[str, Any]) -> None:
        with self._lock:
            clients = list(self._clients)
        for client_queue in clients:
            try:
                client_queue.put_nowait((event, payload))
            except queue.Full:
                pass


class SampleRing:
    def __init__(self, sample_rate: int, seconds: float) -> None:
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * seconds)
        self._chunks: deque[tuple[int, int, np.ndarray]] = deque()
        self._start = 0
        self._end = 0
        self._lock = threading.Lock()

    @property
    def end_index(self) -> int:
        with self._lock:
            return self._end

    def append(self, samples: np.ndarray) -> tuple[int, int]:
        samples = np.asarray(samples, dtype=np.int16).reshape(-1).copy()
        with self._lock:
            start = self._end
            end = start + len(samples)
            self._chunks.append((start, end, samples))
            self._end = end
            while self._chunks and self._end - self._chunks[0][0] > self.max_samples:
                self._chunks.popleft()
            self._start = self._chunks[0][0] if self._chunks else self._end
            return start, end

    def extract(self, start: int, end: int) -> np.ndarray | None:
        with self._lock:
            if start < self._start or end > self._end:
                return None
            parts: list[np.ndarray] = []
            for chunk_start, chunk_end, chunk in self._chunks:
                if chunk_end <= start:
                    continue
                if chunk_start >= end:
                    break
                offset_start = max(start, chunk_start) - chunk_start
                offset_end = min(end, chunk_end) - chunk_start
                parts.append(chunk[offset_start:offset_end])
            if not parts:
                return np.array([], dtype=np.int16)
            return np.concatenate(parts)

    def latest(self, seconds: float) -> np.ndarray | None:
        with self._lock:
            end = self._end
            start = max(self._start, end - int(seconds * self.sample_rate))
        return self.extract(start, end)


class DutchNameResolver:
    def __init__(self, csv_path: Path) -> None:
        self._names: dict[str, str] = {}
        self._load_csv(csv_path)
        self._load_label_file_from_env()

    def resolve(self, detection: dict[str, Any]) -> str:
        scientific = str(detection.get("scientific_name") or "").strip()
        if scientific and scientific in self._names:
            return self._names[scientific]

        label = str(detection.get("label") or "")
        if "_" in label:
            label_scientific, common = label.split("_", 1)
            if label_scientific in self._names:
                return self._names[label_scientific]
            if common:
                return common

        return str(detection.get("common_name") or scientific or "Onbekende vogel")

    def _load_csv(self, csv_path: Path) -> None:
        if not csv_path.exists():
            return
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                scientific = (row.get("scientific_name") or "").strip()
                dutch = (row.get("dutch_name") or "").strip()
                if scientific and dutch:
                    self._names[scientific] = dutch

    def _load_label_file_from_env(self) -> None:
        label_path = os.getenv("BIRDNET_LABELS_NL")
        if not label_path:
            return
        path = Path(label_path).expanduser()
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if "_" not in line:
                    continue
                scientific, dutch = line.split("_", 1)
                if scientific and dutch:
                    self._names.setdefault(scientific.strip(), dutch.strip())


class BirdNetService:
    def __init__(self, resolver: DutchNameResolver) -> None:
        self.resolver = resolver
        self._analyzer = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str | None]:
        if BIRDNET_IMPORT_ERROR:
            return False, f"birdnetlib kan niet worden geladen: {BIRDNET_IMPORT_ERROR}"
        return True, None

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        date: datetime,
    ) -> list[dict[str, Any]]:
        if RecordingBuffer is None or Analyzer is None:
            raise RuntimeError(f"birdnetlib is niet beschikbaar: {BIRDNET_IMPORT_ERROR}")

        with self._lock:
            if self._analyzer is None:
                try:
                    self._analyzer = Analyzer(version=BIRDNET_MODEL_VERSION)
                except TypeError:
                    self._analyzer = Analyzer()

            recording = RecordingBuffer(
                self._analyzer,
                samples,
                sample_rate,
                lat=BIRDNET_LAT,
                lon=BIRDNET_LON,
                date=date,
                min_conf=MIN_CONFIDENCE,
            )
            recording.analyze()

        detections = []
        for detection in recording.detections:
            confidence = float(detection.get("confidence") or 0)
            if confidence < MIN_CONFIDENCE:
                continue
            detection = dict(detection)
            detection["dutch_name"] = self.resolver.resolve(detection)
            detections.append(detection)
        return detections


@dataclass
class AnalysisJob:
    block_start_sample: int
    samples: np.ndarray
    session_started_at: datetime


@dataclass
class BrowserChunkJob:
    upload_path: Path
    received_at: datetime


class RecorderService:
    def __init__(self, broker: EventBroker, birdnet: BirdNetService) -> None:
        self.broker = broker
        self.birdnet = birdnet
        self._lock = threading.Lock()
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._analysis_queue: queue.Queue[AnalysisJob | None] = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._analysis_thread: threading.Thread | None = None
        self._running = False
        self._error: str | None = None
        self._recording_path: Path | None = None
        self._recording_temp_path: Path | None = None
        self._input_device: dict[str, Any] | None = None
        self._temp_recording_bytes: int | None = None
        self._pre_compression_bytes: int | None = None
        self._compressed_recording_bytes: int | None = None
        self._session_started_at: datetime | None = None
        self._upload_running = False
        self._upload_name: str | None = None
        self._browser_chunk_queue: queue.Queue[BrowserChunkJob] = queue.Queue(maxsize=LIVE_CHUNK_QUEUE_SIZE)
        self._browser_chunk_thread = threading.Thread(target=self._browser_chunk_loop, daemon=True)
        self._browser_chunk_thread.start()
        self._browser_live_processing = False
        self._browser_live_error: str | None = None
        self._ring = SampleRing(SAMPLE_RATE, RING_BUFFER_SECONDS)

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._running:
                return {"ok": True, "message": "Opname loopt al."}
            if sd is None:
                return {
                    "ok": False,
                    "error": f"sounddevice kan niet worden geladen: {SOUNDDEVICE_IMPORT_ERROR}",
                }
            ok, error = self.birdnet.available()
            if not ok:
                return {"ok": False, "error": error}

            self._stop_event.clear()
            self._error = None
            self._ring = SampleRing(SAMPLE_RATE, RING_BUFFER_SECONDS)
            self._audio_queue = queue.Queue(maxsize=100)
            self._analysis_queue = queue.Queue(maxsize=4)
            self._session_started_at = datetime.now()
            self._input_device = default_input_device_info()
            self._temp_recording_bytes = None
            self._pre_compression_bytes = None
            self._compressed_recording_bytes = None
            temp_name = self._session_started_at.strftime("%Y%m%d-%H%M%S-full-record.tmp.wav")
            self._recording_temp_path = unique_path(RECORDINGS_DIR, temp_name)
            self._recording_path = compressed_recording_path(self._recording_temp_path)
            self._running = True

            self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self._analysis_thread.start()
            self._recording_thread.start()

        self.broker.publish("status", {"message": "Opname gestart.", "level": "info"})
        return {"ok": True, "message": "Opname gestart."}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._running:
                return {"ok": True, "message": "Opname is al gestopt."}
            self._stop_event.set()
            recording_thread = self._recording_thread
            analysis_thread = self._analysis_thread

        if recording_thread:
            recording_thread.join(timeout=8)
        try:
            self._analysis_queue.put_nowait(None)
        except queue.Full:
            pass
        if analysis_thread:
            analysis_thread.join(timeout=8)

        with self._lock:
            self._running = False

        self.broker.publish("status", {"message": "Opname gestopt.", "level": "info"})
        return {"ok": True, "message": "Opname gestopt."}

    def status(self) -> dict[str, Any]:
        with self._lock:
            recording_path = self._recording_path
            recording_temp_path = self._recording_temp_path
            running = self._running
            error = self._error
            input_device = self._input_device
            temp_recording_bytes = self._temp_recording_bytes
            pre_compression_bytes = self._pre_compression_bytes
            compressed_recording_bytes = self._compressed_recording_bytes
            upload_running = self._upload_running
            upload_name = self._upload_name
            browser_live_processing = self._browser_live_processing
            browser_live_error = self._browser_live_error
            browser_live_pending = self._browser_chunk_queue.qsize()
        visible_recording_path = recording_path if recording_path and recording_path.exists() else None
        temp_size = file_size_bytes(recording_temp_path)
        wav_before_compression_bytes = pre_compression_bytes or temp_recording_bytes or temp_size
        final_compressed_bytes = compressed_recording_bytes or file_size_bytes(visible_recording_path)
        visible_input_device = input_device or default_input_device_info()

        return {
            "running": running,
            "error": error,
            "sample_rate": SAMPLE_RATE,
            "min_confidence": MIN_CONFIDENCE,
            "lat": BIRDNET_LAT,
            "lon": BIRDNET_LON,
            "input_device": visible_input_device,
            "recording_format": FULL_RECORD_FORMAT,
            "recording_name": visible_recording_path.name if visible_recording_path else None,
            "recording_url": f"/recordings/{visible_recording_path.name}" if visible_recording_path else None,
            "wav_before_compression_bytes": wav_before_compression_bytes,
            "wav_before_compression_mb": bytes_to_mb(wav_before_compression_bytes),
            "compressed_recording_bytes": final_compressed_bytes,
            "compressed_recording_mb": bytes_to_mb(final_compressed_bytes),
            "upload_running": upload_running,
            "upload_name": upload_name,
            "browser_live_processing": browser_live_processing,
            "browser_live_pending": browser_live_pending,
            "browser_live_error": browser_live_error,
        }

    def _recording_loop(self) -> None:
        assert self._recording_temp_path is not None
        assert self._recording_path is not None
        assert self._session_started_at is not None

        analysis_frames = int(ANALYSIS_SECONDS * SAMPLE_RATE)
        pending = np.array([], dtype=np.int16)
        next_block_start = 0
        written_audio_bytes = 0

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            if status:
                self.broker.publish("status", {"message": str(status), "level": "error"})
            try:
                self._audio_queue.put_nowait(indata.copy())
            except queue.Full:
                self.broker.publish(
                    "status",
                    {"message": "Audio-buffer is vol; enkele samples zijn overgeslagen.", "level": "error"},
                )

        try:
            with wave.open(str(self._recording_temp_path), "wb") as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)

                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    callback=callback,
                    blocksize=int(SAMPLE_RATE * 0.25),
                ):
                    while not self._stop_event.is_set():
                        try:
                            float_chunk = self._audio_queue.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        int_chunk = float_to_int16(float_chunk)
                        chunk_bytes = int_chunk.tobytes()
                        wav_file.writeframes(chunk_bytes)
                        written_audio_bytes += len(chunk_bytes)
                        with self._lock:
                            self._temp_recording_bytes = 44 + written_audio_bytes
                        self._ring.append(int_chunk)
                        pending = np.concatenate((pending, int_chunk))

                        while len(pending) >= analysis_frames:
                            block = pending[:analysis_frames]
                            pending = pending[analysis_frames:]
                            job = AnalysisJob(
                                block_start_sample=next_block_start,
                                samples=int16_to_float32(block),
                                session_started_at=self._session_started_at,
                            )
                            next_block_start += analysis_frames
                            try:
                                self._analysis_queue.put_nowait(job)
                            except queue.Full:
                                self.broker.publish(
                                    "status",
                                    {
                                        "message": "BirdNET is nog bezig; analyseblok overgeslagen.",
                                        "level": "error",
                                    },
                                )
            pre_compression_bytes = file_size_bytes(self._recording_temp_path)
            with self._lock:
                self._pre_compression_bytes = pre_compression_bytes
            compress_full_recording(self._recording_temp_path, self._recording_path)
            with self._lock:
                self._compressed_recording_bytes = file_size_bytes(self._recording_path)
        except Exception as exc:
            with self._lock:
                self._error = f"Microfoon/opnamefout: {exc}"
                self._running = False
            self._stop_event.set()
            self.broker.publish("status", {"message": self._error, "level": "error"})
        finally:
            with self._lock:
                self._running = False

    def _analysis_loop(self) -> None:
        while True:
            job = self._analysis_queue.get()
            if job is None:
                return
            try:
                self._handle_analysis_job(job)
            except Exception as exc:
                with self._lock:
                    self._error = f"BirdNET-analysefout: {exc}"
                self.broker.publish("status", {"message": self._error, "level": "error"})

    def _handle_analysis_job(self, job: AnalysisJob) -> None:
        block_start_seconds = job.block_start_sample / SAMPLE_RATE
        block_datetime = job.session_started_at + timedelta(seconds=block_start_seconds)
        detections = self.birdnet.analyze(job.samples, SAMPLE_RATE, block_datetime)

        for detection in detections:
            start_seconds = float(detection.get("start_time") or 0)
            end_seconds = float(detection.get("end_time") or ANALYSIS_SECONDS)
            abs_start = max(
                0,
                job.block_start_sample + int((start_seconds - PRE_ROLL_SECONDS) * SAMPLE_RATE),
            )
            abs_end = job.block_start_sample + int((end_seconds + POST_ROLL_SECONDS) * SAMPLE_RATE)

            clip_samples = self._wait_for_clip(abs_start, abs_end)
            if clip_samples is None or len(clip_samples) == 0:
                self.broker.publish(
                    "status",
                    {"message": "Fragment kon niet uit de rolling buffer worden gelezen.", "level": "error"},
                )
                continue

            detected_at = job.session_started_at + timedelta(
                seconds=(job.block_start_sample / SAMPLE_RATE) + start_seconds
            )
            self._save_and_publish_detection(detection, clip_samples, detected_at, "live")

    def _wait_for_clip(self, start: int, end: int) -> np.ndarray | None:
        for _ in range(30):
            clip = self._ring.extract(start, end)
            if clip is not None:
                return clip
            if self._stop_event.is_set():
                return None
            time.sleep(0.1)
        return None

    def latest_spectrogram(self) -> dict[str, Any]:
        samples = self._ring.latest(LIVE_SPECTROGRAM_SECONDS)
        if samples is None or len(samples) < SAMPLE_RATE:
            return {
                "ok": False,
                "error": "Nog niet genoeg live audio beschikbaar voor een spectrogram.",
            }
        data = spectrogram_data(samples, SAMPLE_RATE)
        data["ok"] = True
        data["seconds"] = round(len(samples) / SAMPLE_RATE, 2)
        return data

    def start_upload_analysis(self, upload_path: Path) -> dict[str, Any]:
        ok, error = self.birdnet.available()
        if not ok:
            return {"ok": False, "error": error}
        with self._lock:
            if self._upload_running:
                return {"ok": False, "error": f"Er loopt al een uploadanalyse: {self._upload_name}"}
            self._upload_running = True
            self._upload_name = upload_path.name

        thread = threading.Thread(target=self._upload_analysis_loop, args=(upload_path,), daemon=True)
        thread.start()
        self.broker.publish(
            "status",
            {"message": f"Uploadanalyse gestart: {upload_path.name}", "level": "info"},
        )
        return {"ok": True, "message": f"Upload opgeslagen en analyse gestart: {upload_path.name}"}

    def _upload_analysis_loop(self, upload_path: Path) -> None:
        analysis_wav_path = unique_path(UPLOADS_DIR, f"{upload_path.stem}-analysis.wav")
        detection_count = 0
        try:
            convert_audio_to_analysis_wav(upload_path, analysis_wav_path)
            samples, sample_rate = read_wav_int16(analysis_wav_path)
            if sample_rate != SAMPLE_RATE:
                raise RuntimeError(f"Onverwachte samplerate na conversie: {sample_rate}")

            analysis_frames = int(ANALYSIS_SECONDS * SAMPLE_RATE)
            session_started_at = datetime.now()
            total_blocks = max(1, int(np.ceil(len(samples) / analysis_frames)))

            with self._lock:
                if not self._running:
                    self._ring = SampleRing(SAMPLE_RATE, RING_BUFFER_SECONDS)

            for block_index in range(total_blocks):
                block_start_sample = block_index * analysis_frames
                block = samples[block_start_sample : block_start_sample + analysis_frames]
                if len(block) < SAMPLE_RATE:
                    continue
                if len(block) < analysis_frames:
                    block = np.pad(block, (0, analysis_frames - len(block)))

                with self._lock:
                    live_recording_running = self._running
                if not live_recording_running:
                    self._ring.append(block[:analysis_frames])

                block_datetime = session_started_at + timedelta(seconds=block_start_sample / SAMPLE_RATE)
                detections = self.birdnet.analyze(int16_to_float32(block), SAMPLE_RATE, block_datetime)

                for detection in detections:
                    start_seconds = float(detection.get("start_time") or 0)
                    end_seconds = float(detection.get("end_time") or ANALYSIS_SECONDS)
                    abs_start = max(
                        0,
                        block_start_sample + int((start_seconds - PRE_ROLL_SECONDS) * SAMPLE_RATE),
                    )
                    abs_end = min(
                        len(samples),
                        block_start_sample + int((end_seconds + POST_ROLL_SECONDS) * SAMPLE_RATE),
                    )
                    clip_samples = samples[abs_start:abs_end]
                    if len(clip_samples) == 0:
                        continue
                    detected_at = session_started_at + timedelta(
                        seconds=(block_start_sample / SAMPLE_RATE) + start_seconds
                    )
                    self._save_and_publish_detection(detection, clip_samples, detected_at, upload_path.name)
                    detection_count += 1

            self.broker.publish(
                "status",
                {
                    "message": f"Uploadanalyse klaar: {detection_count} detecties in {upload_path.name}",
                    "level": "info",
                },
            )
        except Exception as exc:
            self.broker.publish(
                "status",
                {"message": f"Uploadanalyse mislukt: {exc}", "level": "error"},
            )
        finally:
            analysis_wav_path.unlink(missing_ok=True)
            with self._lock:
                self._upload_running = False
                self._upload_name = None

    def start_browser_chunk_analysis(self, upload_path: Path) -> dict[str, Any]:
        ok, error = self.birdnet.available()
        if not ok:
            return {"ok": False, "error": error}
        with self._lock:
            if self._running:
                return {
                    "ok": False,
                    "error": "Stop eerst de lokale servermicrofoon voordat je de browsermicrofoon gebruikt.",
                }
            self._browser_live_error = None
        try:
            self._browser_chunk_queue.put_nowait(BrowserChunkJob(upload_path, datetime.now()))
        except queue.Full:
            upload_path.unlink(missing_ok=True)
            return {
                "ok": True,
                "message": "BirdNET is nog bezig; dit browserfragment is overgeslagen.",
                "skipped": True,
            }
        return {"ok": True, "message": "Browserfragment ontvangen."}

    def _browser_chunk_loop(self) -> None:
        while True:
            job = self._browser_chunk_queue.get()
            analysis_wav_path = unique_path(LIVE_CHUNKS_DIR, f"{job.upload_path.stem}-analysis.wav")
            try:
                with self._lock:
                    self._browser_live_processing = True
                    self._browser_live_error = None

                convert_audio_to_analysis_wav(job.upload_path, analysis_wav_path)
                samples, sample_rate = read_wav_int16(analysis_wav_path)
                if sample_rate != SAMPLE_RATE:
                    raise RuntimeError(f"Onverwachte samplerate na conversie: {sample_rate}")
                if len(samples) < SAMPLE_RATE:
                    raise RuntimeError("Browserfragment is korter dan één seconde.")

                analysis_frames = int(ANALYSIS_SECONDS * SAMPLE_RATE)
                samples = samples[:analysis_frames]
                if len(samples) < analysis_frames:
                    samples = np.pad(samples, (0, analysis_frames - len(samples)))

                with self._lock:
                    self._ring.append(samples)
                detections = self.birdnet.analyze(int16_to_float32(samples), SAMPLE_RATE, job.received_at)
                for detection in detections:
                    start_seconds = float(detection.get("start_time") or 0)
                    end_seconds = float(detection.get("end_time") or ANALYSIS_SECONDS)
                    start = max(0, int((start_seconds - PRE_ROLL_SECONDS) * SAMPLE_RATE))
                    end = min(len(samples), int((end_seconds + POST_ROLL_SECONDS) * SAMPLE_RATE))
                    clip_samples = samples[start:end]
                    if len(clip_samples):
                        detected_at = job.received_at + timedelta(seconds=start_seconds)
                        self._save_and_publish_detection(detection, clip_samples, detected_at, "browsermicrofoon")
            except Exception as exc:
                message = f"Browseranalyse mislukt: {exc}"
                with self._lock:
                    self._browser_live_error = message
                self.broker.publish("status", {"message": message, "level": "error"})
            finally:
                job.upload_path.unlink(missing_ok=True)
                analysis_wav_path.unlink(missing_ok=True)
                with self._lock:
                    self._browser_live_processing = False

    def _save_and_publish_detection(
        self,
        detection: dict[str, Any],
        clip_samples: np.ndarray,
        detected_at: datetime,
        source: str,
    ) -> None:
        dutch_name = str(detection["dutch_name"])
        filename = f"{detected_at.strftime('%Y%m%d-%H%M%S')}-{slugify_filename(dutch_name)}.wav"
        clip_path = unique_path(CLIPS_DIR, filename)
        write_wav(clip_path, clip_samples, SAMPLE_RATE)
        spectrogram_path = unique_path(SPECTROGRAMS_DIR, f"{clip_path.stem}.png")
        try:
            save_spectrogram_png(spectrogram_path, clip_samples, SAMPLE_RATE, dutch_name)
            spectrogram_url = f"/spectrograms/{spectrogram_path.name}"
        except Exception as exc:
            spectrogram_url = None
            self.broker.publish(
                "status",
                {"message": f"Spectrogram kon niet worden gemaakt: {exc}", "level": "error"},
            )

        payload = {
            "detected_at": detected_at.strftime("%Y-%m-%d %H:%M:%S"),
            "dutch_name": dutch_name,
            "confidence": float(detection.get("confidence") or 0),
            "scientific_name": detection.get("scientific_name"),
            "clip_name": clip_path.name,
            "clip_url": f"/clips/{clip_path.name}",
            "spectrogram_url": spectrogram_url,
            "source": source,
        }
        self.broker.publish("detection", payload)


app = Flask(__name__)
broker = EventBroker()
name_resolver = DutchNameResolver(DUTCH_NAMES_CSV)
birdnet_service = BirdNetService(name_resolver)
recorder = RecorderService(broker, birdnet_service)


@app.get("/")
def index() -> str:
    return INDEX_HTML


@app.post("/api/start")
def api_start() -> tuple[Response, int] | Response:
    result = recorder.start()
    status = 200 if result.get("ok") else 500
    return jsonify(result), status


@app.post("/api/stop")
def api_stop() -> Response:
    return jsonify(recorder.stop())


@app.post("/api/upload")
def api_upload() -> tuple[Response, int] | Response:
    uploaded_file = request.files.get("audio_file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "Geen audiobestand ontvangen."}), 400

    original_name = secure_filename(uploaded_file.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        return jsonify({"ok": False, "error": f"Niet ondersteund audioformaat. Gebruik: {allowed}"}), 400

    stem = slugify_filename(Path(original_name).stem)
    upload_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-upload-{stem}{extension}"
    upload_path = unique_path(UPLOADS_DIR, upload_name)
    uploaded_file.save(upload_path)

    result = recorder.start_upload_analysis(upload_path)
    status = 200 if result.get("ok") else 409
    return jsonify(result), status


@app.post("/api/live-chunk")
def api_live_chunk() -> tuple[Response, int] | Response:
    uploaded_file = request.files.get("audio_file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "Geen browser-audio ontvangen."}), 400

    original_name = secure_filename(uploaded_file.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in {".webm", ".mp4", ".m4a", ".ogg"}:
        return jsonify({"ok": False, "error": "Browseraudio moet WebM, M4A, MP4 of Ogg zijn."}), 400
    if request.content_length and request.content_length > MAX_LIVE_CHUNK_BYTES + 64 * 1024:
        return jsonify({"ok": False, "error": "Browserfragment is te groot."}), 413

    upload_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-browser{extension}"
    upload_path = unique_path(LIVE_CHUNKS_DIR, upload_name)
    uploaded_file.save(upload_path)
    if upload_path.stat().st_size > MAX_LIVE_CHUNK_BYTES:
        upload_path.unlink(missing_ok=True)
        return jsonify({"ok": False, "error": "Browserfragment is te groot."}), 413

    result = recorder.start_browser_chunk_analysis(upload_path)
    if not result.get("ok"):
        upload_path.unlink(missing_ok=True)
    status = 200 if result.get("ok") else 409
    return jsonify(result), status


@app.get("/api/status")
def api_status() -> Response:
    return jsonify(recorder.status())


@app.get("/api/devices")
def api_devices() -> Response:
    if sd is None:
        return jsonify({"ok": False, "error": str(SOUNDDEVICE_IMPORT_ERROR), "devices": []}), 500
    devices = sd.query_devices()
    return jsonify({"ok": True, "devices": [dict(device) for device in devices]})


@app.get("/api/spectrogram/latest")
def api_latest_spectrogram() -> tuple[Response, int] | Response:
    result = recorder.latest_spectrogram()
    status = 200 if result.get("ok") else 404
    return jsonify(result), status


@app.get("/recordings/<path:filename>")
def recordings(filename: str) -> Response:
    return send_file_from(RECORDINGS_DIR, filename)


@app.get("/clips/<path:filename>")
def clips(filename: str) -> Response:
    return send_file_from(CLIPS_DIR, filename)


@app.get("/uploads/<path:filename>")
def uploads(filename: str) -> Response:
    return send_file_from(UPLOADS_DIR, filename)


@app.get("/spectrograms/<path:filename>")
def spectrograms(filename: str) -> Response:
    return send_file_from(SPECTROGRAMS_DIR, filename)


def send_file_from(directory: Path, filename: str) -> Response:
    return send_from_directory(directory, filename, as_attachment=False)


@app.get("/events")
def events() -> Response:
    client_queue = broker.subscribe()

    def stream():
        try:
            yield "event: status\ndata: {\"message\":\"Live verbinding actief.\"}\n\n"
            while True:
                try:
                    event, payload = client_queue.get(timeout=15)
                    yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            broker.unsubscribe(client_queue)

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=env_int("BIRDNET_PORT", 5055), threaded=True)

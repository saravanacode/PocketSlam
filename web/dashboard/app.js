"use strict";

// Laptop dashboard. One WebSocket carries:
//   binary  -> a JPEG camera frame (newest only)
//   text    -> {"type": "imu", "samples": [["g"|"a", t_s, x, y, z], ...]}
//              {"type": "status", "phone_url": ..., "session": {...} | null}

const WINDOW_S = 10;
const $ = (id) => document.getElementById(id);
const ui = {
  strip: $("strip"),
  lampLabel: $("lamp-label"),
  timecode: $("timecode"),
  file: $("file"),
  camera: document.querySelector(".camera"),
  frame: $("frame"),
  phoneUrl: $("phone-url"),
  phoneUrlSide: $("phone-url-side"),
  device: $("device"),
  res: $("res"),
  fps: $("fps"),
  recorded: $("recorded"),
  dropped: $("dropped"),
  imuCount: $("imu-count"),
  intrinsics: $("intrinsics"),
  gyroRate: $("gyro-rate"),
  cameraFps: $("camera-fps"),
  sentFps: $("sent-fps"),
  recvFps: $("recv-fps"),
  bitrate: $("bitrate"),
  encoder: $("encoder"),
  rtt: $("rtt"),
  lag: $("lag"),
  skipped: $("skipped"),
  demoButton: $("demo-button"),
  distPanel: $("dist-panel"),
  distValue: $("dist-value"),
  distDetail: $("dist-detail"),
  distPlot: $("dist-plot"),
  distRecal: $("dist-recal"),
  distStop: $("dist-stop"),
  calButton: $("cal-button"),
  calPanel: $("cal-panel"),
  calHint: $("cal-hint"),
  calResult: $("cal-result"),
  recButton: $("rec-button"),
  recLabel: $("rec-label"),
  accelRate: $("accel-rate"),
};
const frameCtx = ui.frame.getContext("2d");

const series = { g: [], a: [] }; // [t, x, y, z]
let session = null;
let frameTimes = [];
let lastFrameAt = 0;
let drawing = false;

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// ---- Camera ---------------------------------------------------------------

async function showFrame(blob) {
  if (drawing) return; // still decoding the previous frame; newest wins next time
  drawing = true;
  try {
    const bitmap = await createImageBitmap(blob);
    if (ui.frame.width !== bitmap.width || ui.frame.height !== bitmap.height) {
      ui.frame.width = bitmap.width;
      ui.frame.height = bitmap.height;
    }
    frameCtx.drawImage(bitmap, 0, 0);
    bitmap.close();
    drawMarkerOutline();
    const now = performance.now();
    lastFrameAt = now;
    frameTimes.push(now);
    while (frameTimes.length && now - frameTimes[0] > 1000) frameTimes.shift();
    ui.camera.classList.add("live");
  } finally {
    drawing = false;
  }
}

// ---- IMU traces -----------------------------------------------------------

function addImu(samples) {
  for (const [k, t, x, y, z] of samples) series[k]?.push([t, x, y, z]);
  for (const k of ["g", "a"]) {
    const s = series[k];
    if (!s.length) continue;
    const cutoff = s[s.length - 1][0] - WINDOW_S;
    let drop = 0;
    while (drop < s.length && s[drop][0] < cutoff) drop++;
    if (drop) s.splice(0, drop);
  }
}

function rate(s) {
  if (s.length < 2) return 0;
  const end = s[s.length - 1][0];
  let n = 0;
  for (let i = s.length - 1; i >= 0 && end - s[i][0] <= 1; i--) n++;
  return n;
}

function plot(canvas, s, minRange) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const end = s.length ? s[s.length - 1][0] : 0;
  let peak = minRange;
  for (const p of s) peak = Math.max(peak, Math.abs(p[1]), Math.abs(p[2]), Math.abs(p[3]));
  const range = peak * 1.1;
  const yOf = (v) => h / 2 - (v / range) * (h / 2 - 4);
  const xOf = (t) => w - ((end - t) / WINDOW_S) * w;

  // Grid: one line per second, zero line, and the scale at the top left.
  ctx.strokeStyle = cssVar("--grid");
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < WINDOW_S; i++) {
    const x = Math.round((i / WINDOW_S) * w) + 0.5;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  ctx.stroke();
  ctx.strokeStyle = cssVar("--line");
  ctx.beginPath();
  ctx.moveTo(0, Math.round(h / 2) + 0.5);
  ctx.lineTo(w, Math.round(h / 2) + 0.5);
  ctx.stroke();
  ctx.fillStyle = cssVar("--ink-soft");
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(`±${range.toFixed(range < 10 ? 2 : 1)}`, 4, 13);

  if (s.length < 2) return;
  ["--x", "--y", "--z"].forEach((color, axis) => {
    ctx.strokeStyle = cssVar(color);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    s.forEach((p, i) => {
      const x = xOf(p[0]);
      const y = yOf(p[axis + 1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
}

// ---- Status ---------------------------------------------------------------

function pad(n, width = 2) {
  return String(Math.floor(n)).padStart(width, "0");
}

function timecode(seconds) {
  const s = Math.max(0, seconds);
  return `${pad(s / 3600)}:${pad((s % 3600) / 60)}:${pad(s % 60)}.${Math.floor((s * 10) % 10)}`;
}

function applyStatus(msg) {
  phoneUrl = msg.phone_url;
  ui.phoneUrl.textContent = msg.phone_url;
  ui.phoneUrlSide.textContent = msg.phone_url.replace(/^https?:\/\//, "");
  ui.phoneUrlSide.title = msg.phone_url;
  session = msg.session;
  const s = session;
  ui.lag.textContent = `${Math.round(msg.loop_lag_ms ?? 0)} ms`;
  applyLink(s?.link, s?.recv_fps);
  setRecButton(Boolean(s?.clip), Boolean(s));
  ui.calButton.disabled = !s && !calibration?.active;
  applyCalibration(msg.calibration);
  ui.demoButton.disabled = !s && demoStage === "idle";
  if (msg.demo) applyDistance(msg.demo.distance, msg.demo.stage);
  measuredFov = msg.focal_ratio ? (2 * Math.atan(0.5 / msg.focal_ratio) * 180) / Math.PI : null;
  if (!s) {
    ui.strip.dataset.state = "idle";
    ui.lampLabel.textContent = "Waiting for phone";
    ui.file.textContent = lastJobText(msg.jobs) ?? "No clip yet";
    ui.file.title = "";
    for (const el of [ui.device, ui.res, ui.recorded, ui.dropped, ui.imuCount, ui.intrinsics, ui.skipped]) el.textContent = "–";
    return;
  }
  const clip = s.clip;
  ui.strip.dataset.state = clip ? "live" : "streaming";
  ui.lampLabel.textContent = clip ? `Recording clip, ${clip.frames} frames` : "Streaming. Press Record to save a clip";
  const file = clip?.path ?? s.recording;
  ui.file.textContent = (file && file.split("/").pop()) ?? lastJobText(msg.jobs) ?? "No clip yet";
  ui.file.title = file ?? "";
  ui.device.textContent = shortDevice(s.device);
  ui.device.title = s.device ?? "";
  ui.res.textContent = s.width ? `${s.width}×${s.height}` : "–";
  ui.recorded.textContent = clip ? clip.frames.toLocaleString() : "–";
  ui.dropped.textContent = s.record_dropped.toLocaleString();
  ui.imuCount.textContent = s.imu_received.toLocaleString();
  ui.skipped.textContent = (s.frames_skipped ?? 0).toLocaleString();
  ui.intrinsics.textContent = measuredFov
    ? `Measured, ${measuredFov.toFixed(0)}° view`
    : "Not calibrated yet";
}

const fmt = (v, unit, digits = 0) => (v === null || v === undefined ? "–" : `${Number(v).toFixed(digits)} ${unit}`);

// The chain camera -> phone encoder -> network -> laptop, one number per hop,
// so a frame-rate drop points at the hop that loses frames.
function applyLink(link, recvFps) {
  ui.recvFps.textContent = recvFps === undefined ? "–" : fmt(recvFps, "fps");
  if (!link) {
    for (const el of [ui.cameraFps, ui.sentFps, ui.bitrate, ui.encoder, ui.rtt]) el.textContent = "–";
    return;
  }
  ui.cameraFps.textContent = fmt(link.camera_fps, "fps");
  const limited = link.quality_limitation && link.quality_limitation !== "none" ? `, limited by ${link.quality_limitation}` : "";
  ui.sentFps.textContent = `${fmt(link.sent_fps, "fps")}${limited}`;
  ui.sentFps.title = ui.sentFps.textContent;
  ui.bitrate.textContent = link.target_kbps
    ? `${Math.round(link.target_kbps)} of ${link.available_kbps ? Math.round(link.available_kbps) : "?"} kbps`
    : "–";
  const codec = link.codec ? link.codec.replace("video/", "") : "";
  ui.encoder.textContent = [codec, link.encoder].filter(Boolean).join(", ") || "–";
  ui.encoder.title = ui.encoder.textContent;
  ui.rtt.textContent = fmt(link.rtt_ms, "ms");
}

// ---- Live camera calibration ----------------------------------------------

let calibration = null;
let phoneUrl = "";
let calDismissed = true; // the panel stays hidden until the user starts a calibration

// Outline the detected marker on the live picture while calibrating.
function drawMarkerOutline() {
  const c = calibration?.active ? calibration.corners : distance?.active ? distance.corners : null;
  if (!c) return;
  const w = ui.frame.width, h = ui.frame.height;
  frameCtx.lineWidth = Math.max(2, w / 240);
  frameCtx.strokeStyle = "#40e6ff";
  frameCtx.beginPath();
  c.forEach(([x, y], i) => (i ? frameCtx.lineTo(x * w, y * h) : frameCtx.moveTo(x * w, y * h)));
  frameCtx.closePath();
  frameCtx.stroke();
}

function applyCalibration(cal) {
  if (!cal) return;
  calibration = cal;
  const show = cal.active || (!calDismissed && (cal.saved || cal.error));
  ui.calPanel.hidden = !show;
  ui.calButton.setAttribute("aria-pressed", String(cal.active));
  ui.calButton.textContent = cal.active ? "Stop calibrating" : "Calibrate camera";
  if (!show) return;
  ui.calHint.textContent = cal.hint;
  for (const li of ui.calPanel.querySelectorAll(".cal-sides li")) {
    const n = cal.sides[li.dataset.side] ?? 0;
    const full = n >= cal.sides_needed;
    // One sideways and one vertical direction are enough, so a direction can
    // be left empty when calibration saves. Say so instead of showing 0/3.
    li.querySelector("span").textContent = cal.saved && !full ? "not needed" : `${Math.min(n, cal.sides_needed)}/${cal.sides_needed}`;
    li.classList.toggle("done", full);
    li.classList.toggle("skipped", cal.saved && !full);
  }
  const focal = cal.focal_px
    ? `Focal length ${Math.round(cal.focal_px)} px, field of view ${cal.fov_deg}°, uncertainty ${(cal.rel_std * 100).toFixed(1)}% (saves below 5%).`
    : `Focal length appears after ${cal.views_needed} views.`;
  // No detection at all usually means the marker has no white border around it.
  const tip = cal.active && !cal.marker_visible && phoneUrl
    ? ` Not detected? The black square needs a white border. On the other screen, open ${phoneUrl}/marker.`
    : "";
  ui.calResult.textContent = `${cal.views} views. ${focal}${tip}`;
}

ui.calButton.addEventListener("click", async () => {
  const on = ui.calButton.getAttribute("aria-pressed") !== "true";
  calDismissed = !on;
  ui.calButton.disabled = true;
  try {
    const res = await fetch("/api/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ on }),
    });
    if (res.ok) {
      applyCalibration(await res.json());
    } else {
      ui.calPanel.hidden = false;
      ui.calHint.textContent = await res.text();
      ui.calResult.textContent = "";
    }
  } catch {
    ui.calPanel.hidden = false;
    ui.calHint.textContent = "Could not reach the laptop server.";
  } finally {
    ui.calButton.disabled = false;
  }
});

// ---- Distance demo ---------------------------------------------------------

let distance = null;
let demoStage = "idle";

function formatDistance(m) {
  if (m === null || m === undefined) return "–";
  return m < 1 ? `${(m * 100).toFixed(1)} cm` : `${m.toFixed(2)} m`;
}

function applyDistance(d, stage) {
  if (!d) return;
  distance = d;
  demoStage = stage ?? demoStage;
  const measuring = d.active;
  ui.distPanel.hidden = !measuring;
  ui.demoButton.setAttribute("aria-pressed", String(demoStage !== "idle"));
  ui.demoButton.textContent = demoStage === "idle" ? "Distance demo" : "Stop demo";
  if (!measuring) return;
  ui.distPanel.classList.toggle("lost", !d.marker_visible);
  ui.distValue.textContent = formatDistance(d.distance_m);
  if (d.distance_m === null) {
    ui.distDetail.textContent = "Point the phone at the marker.";
  } else if (!d.marker_visible) {
    ui.distDetail.textContent = "Marker not in view. Showing the last reading.";
  } else {
    const err = d.error_m < 0.01 ? `${(d.error_m * 1000).toFixed(0)} mm` : `${(d.error_m * 100).toFixed(1)} cm`;
    ui.distDetail.textContent =
      `Within about ${err}. Viewing angle ${Math.round(d.tilt_deg)}°. ` +
      `${Math.round(d.rate_hz)} readings a second. Seen so far: ${formatDistance(d.min_m)} to ${formatDistance(d.max_m)}.`;
  }
  plotDistance(d.history);
}

// Distance over the last 30 seconds, with a line per metre.
function plotDistance(history) {
  const canvas = ui.distPlot;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!history?.length) return;
  const span = 30;
  const end = history[history.length - 1][0];
  const top = Math.max(1, Math.ceil(Math.max(...history.map((p) => p[1])) * 1.15 * 2) / 2);
  const xOf = (t) => w - ((end - t) / span) * w;
  const yOf = (m) => h - 4 - (m / top) * (h - 22);
  ctx.font = "12px system-ui, sans-serif";
  ctx.lineWidth = 1;
  const step = top > 4 ? 1 : 0.5;
  for (let m = 0; m <= top + 1e-9; m += step) {
    const y = Math.round(yOf(m)) + 0.5;
    ctx.strokeStyle = "rgb(255 255 255 / 0.14)";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.fillStyle = "#aeb9c8";
    ctx.fillText(`${m} m`, 4, y - 3);
  }
  ctx.strokeStyle = "#40e6ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  history.forEach(([t, m], i) => (i ? ctx.lineTo(xOf(t), yOf(m)) : ctx.moveTo(xOf(t), yOf(m))));
  ctx.stroke();
}

async function setDemo(on, recalibrate = false) {
  ui.demoButton.disabled = true;
  try {
    const res = await fetch("/api/demo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ on, recalibrate }),
    });
    if (res.ok) {
      const snap = await res.json();
      applyDistance(snap.distance, snap.stage);
      // The calibration stage reuses the calibration panel; make sure it shows.
      if (snap.stage === "calibrating") calDismissed = false;
    } else {
      ui.calPanel.hidden = false;
      ui.calHint.textContent = await res.text();
      ui.calResult.textContent = "";
    }
  } catch {
    ui.calPanel.hidden = false;
    ui.calHint.textContent = "Could not reach the laptop server.";
  } finally {
    ui.demoButton.disabled = false;
  }
}

ui.demoButton.addEventListener("click", () => setDemo(demoStage === "idle"));
ui.distStop.addEventListener("click", () => setDemo(false));
ui.distRecal.addEventListener("click", () => setDemo(true, true));

// ---- Record button ---------------------------------------------------------

let recBusy = false;
let measuredFov = null;

function setRecButton(recording, phoneConnected) {
  ui.recButton.setAttribute("aria-pressed", String(recording));
  ui.recLabel.textContent = recording ? "Stop" : "Record";
  ui.recButton.disabled = recBusy || !phoneConnected;
  ui.recButton.title = phoneConnected ? "" : "Connect a phone first";
}

ui.recButton.addEventListener("click", async () => {
  const on = ui.recButton.getAttribute("aria-pressed") !== "true";
  recBusy = true;
  ui.recButton.disabled = true;
  try {
    const res = await fetch("/api/record", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ on }),
    });
    if (!res.ok) ui.lampLabel.textContent = await res.text();
  } catch {
    ui.lampLabel.textContent = "Could not reach the laptop server";
  } finally {
    recBusy = false;
  }
});

function lastJobText(jobs) {
  const job = jobs?.[0];
  if (!job) return null;
  return {
    queued: `${job.name}: waiting to reconstruct`,
    running: `${job.name}: reconstructing 3D scan…`,
    done: `${job.name}: 3D scan ready`,
    failed: `${job.name}: reconstruction failed`,
    skipped: `${job.name}: too short to reconstruct`,
  }[job.state];
}

function shortDevice(ua) {
  if (!ua) return "–";
  const android = ua.match(/Android [\d.]+; ([^;)]+)/);
  if (android) return android[1].replace(/ Build.*/, "");
  if (/iPhone/.test(ua)) return "iPhone";
  return ua.slice(0, 40);
}

function tick() {
  const now = performance.now();
  if (now - lastFrameAt > 2000) {
    ui.camera.classList.remove("live");
    frameTimes = [];
  }
  ui.fps.textContent = frameTimes.length ? `${frameTimes.length} fps` : "–";
  const clipStart = session?.clip?.started_at;
  ui.timecode.textContent = clipStart ? timecode(Date.now() / 1000 - clipStart) : timecode(0);
  ui.gyroRate.textContent = series.g.length ? `${rate(series.g)} Hz` : "–";
  ui.accelRate.textContent = series.a.length ? `${rate(series.a)} Hz` : "–";
  plot($("gyro-plot"), series.g, 0.5);
  plot($("accel-plot"), series.a, 12);
  requestAnimationFrame(tick);
}

// ---- Connection -----------------------------------------------------------

function connect(delay = 500) {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.addEventListener("open", () => { delay = 500; });
  ws.addEventListener("message", (e) => {
    if (typeof e.data !== "string") {
      showFrame(e.data);
      return;
    }
    const msg = JSON.parse(e.data);
    if (msg.type === "imu") addImu(msg.samples);
    else if (msg.type === "status") applyStatus(msg);
    else if (msg.type === "calibration") applyCalibration(msg);
    else if (msg.type === "distance") applyDistance(msg, msg.stage);
  });
  ws.addEventListener("close", () => {
    ui.strip.dataset.state = "idle";
    ui.lampLabel.textContent = "Dashboard lost the laptop server. Reconnecting";
    setTimeout(() => connect(Math.min(delay * 2, 5000)), delay);
  });
}

connect();
requestAnimationFrame(tick);

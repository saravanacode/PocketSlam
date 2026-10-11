"use strict";

// Phone client: stream the rear camera over WebRTC and the gyroscope and
// accelerometer over a DataChannel. Plain JS, no build step.
//
// IMU wire format (see transport/webrtc.py):
//   {"type": "imu", "samples": [["g"|"a", t_ms, x, y, z], ...]}
// g = rad/s, a = m/s^2 including gravity, t_ms = phone Unix time in ms.

const IMU_HZ = 100;
const FLUSH_MS = 50;
const MAX_BATCH = 400;

// Capture settings. Override with URL parameters for quick experiments, e.g.
//   /?w=640&h=480&fps=30&pref=balanced&kbps=4000&start_kbps=3000&min_kbps=1500
// pref is the WebRTC degradationPreference: what the encoder gives up when
// bandwidth or CPU runs short (maintain-resolution drops frames).
const query = new URLSearchParams(location.search);
const num = (key, fallback) => Number(query.get(key)) || fallback;
const CAPTURE = {
  width: num("w", 1280),
  height: num("h", 720),
  fps: num("fps", 30),
  pref: query.get("pref") || "maintain-resolution",
  maxBitrate: num("kbps", 4000) * 1000,
  // Chrome starts video near 300 kbps and ramps up only as the receiver
  // reports bandwidth; aiortc reports it slowly, which cost ~2 minutes of
  // low frame rate. On a LAN, start high and keep a floor.
  startKbps: num("start_kbps", 3000),
  minKbps: num("min_kbps", 1500),
};

// Add Chrome's start/min bitrate hints to every video codec in the answer.
// aiortc writes no a=fmtp line for VP8, so add one where it is missing.
function withBitrateHints(sdp) {
  const hints = `x-google-start-bitrate=${CAPTURE.startKbps};x-google-min-bitrate=${CAPTURE.minKbps}`;
  const lines = sdp.split("\r\n");
  const withFmtp = new Set(lines.map((l) => l.match(/^a=fmtp:(\d+) /)?.[1]).filter(Boolean));
  const out = [];
  let inVideo = false;
  const videoPts = new Set();
  for (const line of lines) {
    if (line.startsWith("m=")) inVideo = line.startsWith("m=video");
    const rtpmap = inVideo && line.match(/^a=rtpmap:(\d+) (VP8|VP9|H264|AV1)\//);
    const fmtp = inVideo && line.match(/^a=fmtp:(\d+) /);
    if (rtpmap) videoPts.add(rtpmap[1]);
    if (fmtp && videoPts.has(fmtp[1])) {
      out.push(`${line};${hints}`);
      continue;
    }
    out.push(line);
    if (rtpmap && !withFmtp.has(rtpmap[1])) out.push(`a=fmtp:${rtpmap[1]} ${hints}`);
  }
  return out.join("\r\n");
}

const $ = (id) => document.getElementById(id);
const ui = {
  link: $("link"),
  preview: $("preview"),
  viewfinder: document.querySelector(".viewfinder"),
  hint: $("hint"),
  cam: $("cam"),
  fps: $("fps"),
  gyro: $("gyro"),
  accel: $("accel"),
  api: $("api"),
  toggle: $("toggle"),
  toggleLabel: $("toggle-label"),
  record: $("record"),
  recordLabel: $("record-label"),
  clip: $("clip"),
  disconnect: $("disconnect"),
  guide: $("guide"),
  message: $("message"),
};

let pc = null;
let stream = null;
let channel = null;
let wakeLock = null;
let stopSensors = null;
let timers = [];
let pending = [];
const counts = { g: 0, a: 0 };

function setLink(text, state) {
  ui.link.textContent = text;
  ui.link.dataset.state = state;
}

function setMessage(text, ok = false) {
  ui.message.textContent = text;
  ui.message.classList.toggle("ok", ok);
}

// ---- Clip recording (the laptop writes the file) -------------------------

let recording = false;
let clipStartedAt = 0;
let recordTimeout = null;

function showRecording(on) {
  recording = on;
  clearTimeout(recordTimeout);
  ui.record.disabled = false;
  ui.record.setAttribute("aria-pressed", String(on));
  ui.recordLabel.textContent = on ? "Stop recording" : "Record";
  ui.clip.hidden = !on;
  if (on) clipStartedAt = performance.now();
}

function requestRecording(on) {
  if (!channel || channel.readyState !== "open") return;
  ui.record.disabled = true;
  channel.send(JSON.stringify({ type: "record", on }));
  // The laptop confirms with a record_state message; re-enable if it does not.
  recordTimeout = setTimeout(() => {
    ui.record.disabled = false;
    setMessage("The laptop did not answer the Record button. Is the server up to date?");
  }, 4000);
}

function onLaptopMessage(event) {
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }
  if (msg.type === "calibration") {
    // The laptop is calibrating the camera: show its guidance where the user is looking.
    ui.guide.hidden = !(msg.active || msg.saved);
    ui.guide.textContent = msg.active ? `${msg.hint} (${msg.views} views)` : msg.hint;
    if (msg.saved) setTimeout(() => { ui.guide.hidden = true; }, 6000);
    return;
  }
  if (msg.type === "distance") {
    // Distance demo: show the reading large, where the user is looking while walking back.
    ui.guide.hidden = !msg.active;
    const shown = msg.visible && msg.distance_m !== null;
    ui.guide.classList.toggle("reading", msg.active);
    ui.guide.classList.toggle("lost", !shown);
    if (shown) {
      const m = msg.distance_m;
      ui.guide.textContent = m < 1 ? `${(m * 100).toFixed(1)} cm` : `${m.toFixed(2)} m`;
    } else {
      ui.guide.textContent = "Point the phone at the marker.";
    }
    if (!msg.active) ui.guide.classList.remove("reading", "lost");
    return;
  }
  if (msg.type !== "record_state") return;
  showRecording(Boolean(msg.on));
  if (msg.on) {
    setMessage("");
  } else if (msg.saved) {
    const secs = msg.saved.seconds.toFixed(1);
    setMessage(`Clip saved (${secs} s, ${msg.saved.frames} frames). The laptop is turning it into a 3D scan.`, true);
  }
}

function tickClip() {
  if (!recording) return;
  const s = (performance.now() - clipStartedAt) / 1000;
  ui.clip.textContent = `Recording ${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}.${Math.floor((s * 10) % 10)}`;
}

// Sensor timestamps are relative to performance.timeOrigin; send Unix ms.
const toUnixMs = (t) => performance.timeOrigin + t;

function pushSample(kind, t, x, y, z) {
  counts[kind] += 1;
  if (pending.length < MAX_BATCH) pending.push([kind, Math.round(toUnixMs(t) * 1000) / 1000, x, y, z]);
}

function flushImu() {
  if (!channel || channel.readyState !== "open" || pending.length === 0) return;
  if (channel.bufferedAmount > 256 * 1024) return; // link congested: hold, MAX_BATCH caps memory
  channel.send(JSON.stringify({ type: "imu", samples: pending }));
  pending = [];
}

async function genericSensorsAvailable() {
  if (!("Gyroscope" in window) || !("Accelerometer" in window)) return false;
  try {
    const results = await Promise.all(
      ["gyroscope", "accelerometer"].map((name) => navigator.permissions.query({ name })),
    );
    return results.every((r) => r.state !== "denied");
  } catch {
    return false;
  }
}

// Generic Sensor API (Chrome on Android): exact rates and per-reading timestamps.
// If a sensor cannot start (no hardware, blocked), switch to DeviceMotion.
function startGenericSensors() {
  const gyro = new Gyroscope({ frequency: IMU_HZ });
  const accel = new Accelerometer({ frequency: IMU_HZ });
  let fallbackStop = null;
  gyro.addEventListener("reading", () => pushSample("g", gyro.timestamp, gyro.x, gyro.y, gyro.z));
  accel.addEventListener("reading", () => pushSample("a", accel.timestamp, accel.x, accel.y, accel.z));
  for (const s of [gyro, accel]) {
    s.addEventListener("error", (e) => {
      console.warn("Generic Sensor error", e.error);
      if (fallbackStop) return;
      gyro.stop();
      accel.stop();
      fallbackStop = "DeviceMotionEvent" in window ? startDeviceMotion() : () => {};
      if (!("DeviceMotionEvent" in window)) ui.api.textContent = "No motion sensors found. Streaming camera only.";
    });
    s.start();
  }
  ui.api.textContent = `Motion sensors: Generic Sensor API at ${IMU_HZ} Hz.`;
  return () => {
    gyro.stop();
    accel.stop();
    fallbackStop?.();
  };
}

// DeviceMotion fallback: rotationRate is in deg/s and alpha/beta/gamma are
// rotations about z/x/y, so remap and convert to rad/s.
function startDeviceMotion() {
  const DEG = Math.PI / 180;
  const onMotion = (e) => {
    const r = e.rotationRate;
    const a = e.accelerationIncludingGravity;
    if (r && r.alpha !== null) pushSample("g", e.timeStamp, r.beta * DEG, r.gamma * DEG, r.alpha * DEG);
    if (a && a.x !== null) pushSample("a", e.timeStamp, a.x, a.y, a.z);
  };
  window.addEventListener("devicemotion", onMotion);
  ui.api.textContent = "Motion sensors: DeviceMotion events (browser-chosen rate).";
  return () => window.removeEventListener("devicemotion", onMotion);
}

async function startSensors() {
  // iOS asks for motion permission; Android Chrome grants it on HTTPS.
  if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
    if ((await DeviceMotionEvent.requestPermission()) !== "granted") throw new Error("Motion sensor access was denied.");
  }
  if (await genericSensorsAvailable()) {
    try {
      return startGenericSensors();
    } catch (err) {
      console.warn("Generic Sensor API failed, using DeviceMotion", err);
    }
  }
  if ("DeviceMotionEvent" in window) return startDeviceMotion();
  ui.api.textContent = "This browser has no motion sensors. Streaming camera only.";
  return () => {};
}

// aiortc does not support trickle ICE, so send the offer after gathering ends.
function iceGatheringDone(peer) {
  if (peer.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      if (peer.iceGatheringState === "complete") {
        peer.removeEventListener("icegatheringstatechange", check);
        resolve();
      }
    };
    peer.addEventListener("icegatheringstatechange", check);
  });
}

async function tuneSender(sender) {
  // SLAM needs sharp frames more than smooth motion: keep resolution, drop fps.
  const params = sender.getParameters();
  params.degradationPreference = CAPTURE.pref;
  if (params.encodings && params.encodings[0]) params.encodings[0].maxBitrate = CAPTURE.maxBitrate;
  try {
    await sender.setParameters(params);
  } catch (err) {
    console.warn("setParameters failed", err);
  }
}

function startRateTimer() {
  let last = { g: 0, a: 0 };
  timers.push(setInterval(() => {
    ui.gyro.textContent = `${counts.g - last.g} Hz`;
    ui.accel.textContent = `${counts.a - last.a} Hz`;
    last = { ...counts };
  }, 1000));
  timers.push(setInterval(reportLinkStats, 1000));
  timers.push(setInterval(flushImu, FLUSH_MS));
}

// Collect the sender's view of the link once a second, show the headline on
// the phone and send everything to the laptop as {"type": "stats", ...}.
let lastOutbound = null;
async function reportLinkStats() {
  if (!pc) return;
  const report = await pc.getStats();
  const byId = new Map();
  report.forEach((r) => byId.set(r.id, r));
  const out = { type: "stats" };
  report.forEach((r) => {
    if (r.type === "media-source" && r.kind === "video") {
      out.camera_fps = r.framesPerSecond;
    } else if (r.type === "outbound-rtp" && r.kind === "video") {
      out.sent_fps = r.framesPerSecond;
      out.width = r.frameWidth;
      out.height = r.frameHeight;
      out.quality_limitation = r.qualityLimitationReason;
      out.encoder = r.encoderImplementation;
      out.codec = byId.get(r.codecId)?.mimeType;
      out.target_kbps = r.targetBitrate ? r.targetBitrate / 1000 : undefined;
      out.nack = r.nackCount;
      out.pli = r.pliCount;
      if (lastOutbound && r.framesEncoded > lastOutbound.framesEncoded) {
        out.encode_ms = (1000 * (r.totalEncodeTime - lastOutbound.totalEncodeTime)) /
          (r.framesEncoded - lastOutbound.framesEncoded);
      }
      lastOutbound = r;
    } else if (r.type === "candidate-pair" && r.nominated && r.state === "succeeded") {
      out.available_kbps = r.availableOutgoingBitrate ? r.availableOutgoingBitrate / 1000 : undefined;
      out.rtt_ms = r.currentRoundTripTime !== undefined ? r.currentRoundTripTime * 1000 : undefined;
    }
  });
  if (out.width) ui.cam.textContent = `${out.width}×${out.height}`;
  const limited = out.quality_limitation && out.quality_limitation !== "none" ? `, limited by ${out.quality_limitation}` : "";
  ui.fps.textContent = out.sent_fps !== undefined ? `${Math.round(out.sent_fps)} fps${limited}` : "–";
  if (channel?.readyState === "open") channel.send(JSON.stringify(out));
}

async function start() {
  setMessage("");
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setMessage("Camera needs HTTPS. Open this page with https:// and the laptop's address.");
    return;
  }
  ui.toggle.disabled = true;
  setLink("Starting", "idle");
  try {
    stopSensors = await startSensors();
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: CAPTURE.width },
        height: { ideal: CAPTURE.height },
        frameRate: { ideal: CAPTURE.fps },
      },
    });
    const track = stream.getVideoTracks()[0];
    track.contentHint = "detail";
    ui.preview.srcObject = stream;
    ui.viewfinder.classList.add("live");
    const settings = track.getSettings();
    ui.cam.textContent = `${settings.width}×${settings.height}`;
    // Match the viewfinder to the real frame shape (portrait phones send tall frames).
    if (settings.width && settings.height) ui.viewfinder.style.aspectRatio = `${settings.width} / ${settings.height}`;

    pc = new RTCPeerConnection();
    pc.addEventListener("connectionstatechange", () => {
      const s = pc?.connectionState;
      if (s === "connected") setLink("Streaming", "live");
      else if (s === "failed" || s === "disconnected") {
        setLink("Connection lost", "error");
        setMessage("The laptop stopped answering. Check it is running, then start again.");
      }
    });
    channel = pc.createDataChannel("data");
    channel.addEventListener("message", onLaptopMessage);
    channel.addEventListener("open", async () => {
      // Chrome's user agent says only "K"; ask for the real model so calibration is per phone.
      let model;
      try {
        model = (await navigator.userAgentData?.getHighEntropyValues(["model"]))?.model || undefined;
      } catch {
        model = undefined;
      }
      channel.send(JSON.stringify({
        type: "meta",
        width: settings.width,
        height: settings.height,
        device: navigator.userAgent,
        orientation: screen.orientation?.angle ?? 0,
        model,
      }));
    });
    const sender = pc.addTrack(track, stream);

    await pc.setLocalDescription(await pc.createOffer());
    await iceGatheringDone(pc);
    const resp = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) throw new Error(`The laptop rejected the connection (${resp.status}).`);
    const answer = await resp.json();
    await pc.setRemoteDescription({ sdp: withBitrateHints(answer.sdp), type: answer.type });
    await tuneSender(sender);

    try {
      wakeLock = await navigator.wakeLock?.request("screen");
    } catch {
      // Screen may sleep; streaming still works while it is on.
    }
    startRateTimer();
    ui.toggle.setAttribute("aria-pressed", "true");
    ui.toggle.hidden = true;
    ui.record.hidden = false;
    ui.disconnect.hidden = false;
    timers.push(setInterval(tickClip, 100));
  } catch (err) {
    setMessage(err.message || String(err));
    setLink("Not connected", "error");
    stop();
  } finally {
    ui.toggle.disabled = false;
  }
}

function stop() {
  timers.forEach(clearInterval);
  timers = [];
  flushImu();
  stopSensors?.();
  stopSensors = null;
  channel?.close();
  channel = null;
  pc?.close();
  pc = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  wakeLock?.release();
  wakeLock = null;
  pending = [];
  lastOutbound = null;
  ui.preview.srcObject = null;
  ui.viewfinder.classList.remove("live");
  ui.viewfinder.style.aspectRatio = "";
  ui.guide.hidden = true;
  ui.toggle.setAttribute("aria-pressed", "false");
  ui.toggleLabel.textContent = "Start streaming";
  ui.toggle.hidden = false;
  ui.record.hidden = true;
  ui.disconnect.hidden = true;
  showRecording(false);
  for (const el of [ui.fps, ui.gyro, ui.accel]) el.textContent = "–";
}

ui.record.addEventListener("click", () => requestRecording(!recording));
ui.disconnect.addEventListener("click", () => {
  const wasRecording = recording;
  stop();
  setLink("Not connected", "idle");
  if (wasRecording) setMessage("Streaming stopped. The clip so far was saved on the laptop.", true);
});

ui.toggle.addEventListener("click", () => {
  if (ui.toggle.getAttribute("aria-pressed") === "true") {
    stop();
    setLink("Not connected", "idle");
  } else {
    start();
  }
});

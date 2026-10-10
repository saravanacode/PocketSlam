"use strict";

// Phone client: capture the rear camera and stream it to the host over WebRTC.
// Plain JS, no build step, so it runs on any modern mobile browser.

const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const preview = document.getElementById("preview");
const statusEl = document.getElementById("status");

let pc = null;
let stream = null;

function setStatus(text) {
  statusEl.textContent = text;
}

// aiortc does not support trickle ICE, so send the offer only after gathering ends.
function waitForIceGathering(peer) {
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

async function start() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera needs HTTPS. Open this page with https:// on the host's LAN address.");
    return;
  }
  startBtn.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
    });
    preview.srcObject = stream;
    const track = stream.getVideoTracks()[0];
    // The browser may not honour the request; report what it actually gives.
    const settings = track.getSettings();

    pc = new RTCPeerConnection();
    pc.addEventListener("connectionstatechange", () => setStatus(`Connection: ${pc.connectionState}`));
    const meta = pc.createDataChannel("meta");
    meta.addEventListener("open", () => {
      meta.send(JSON.stringify({
        type: "meta",
        width: settings.width,
        height: settings.height,
        device: navigator.userAgent,
      }));
    });
    pc.addTrack(track, stream);

    await pc.setLocalDescription(await pc.createOffer());
    await waitForIceGathering(pc);
    const resp = await fetch("/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) throw new Error(`offer rejected: ${resp.status} ${await resp.text()}`);
    const answer = await resp.json();
    await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
    setStatus(`Streaming ${settings.width}x${settings.height} @ ${settings.frameRate ?? "?"} fps\nSession ${answer.session_id}`);
    stopBtn.disabled = false;
  } catch (err) {
    setStatus(`Error: ${err.message}`);
    stop();
  }
}

function stop() {
  pc?.close();
  pc = null;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  preview.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", () => {
  stop();
  setStatus("Stopped.");
});

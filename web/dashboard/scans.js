"use strict";

// 3D scans tab: lists reconstructions (GET /api/scans) and shows one point
// cloud with an orbit camera. Plain WebGL2, no libraries.
//
// World frame from pocketslam-recon: metres, z up, camera poses camera-to-world.

(() => {
  const $ = (id) => document.getElementById(id);
  const ui = {
    viewLive: $("view-live"),
    viewScans: $("view-scans"),
    tabLive: $("tab-live"),
    tabScans: $("tab-scans"),
    items: $("scan-items"),
    jobs: $("job-items"),
    hint: $("scan-hint"),
    refresh: $("scan-refresh"),
    viewer: document.querySelector(".viewer"),
    canvas: $("viewer"),
    title: $("scan-title"),
    reset: $("tool-reset"),
    measure: $("tool-measure"),
    path: $("tool-path"),
    size: $("tool-size"),
    readout: $("measure-readout"),
    gridLabel: $("grid-label"),
  };

  // ---- Tabs ---------------------------------------------------------------

  function showView() {
    const scans = location.hash === "#scans";
    ui.viewLive.hidden = scans;
    ui.viewScans.hidden = !scans;
    ui.tabLive.toggleAttribute("aria-current", !scans);
    ui.tabScans.toggleAttribute("aria-current", scans);
    if (scans) {
      ui.tabScans.setAttribute("aria-current", "page");
      if (!listed) loadList();
      pollJobs();
      viewer.resize();
    } else {
      ui.tabLive.setAttribute("aria-current", "page");
    }
  }
  window.addEventListener("hashchange", showView);

  // ---- Small matrix helpers (column-major, like WebGL) ----------------------

  const mat4 = {
    perspective(fovy, aspect, near, far) {
      const f = 1 / Math.tan(fovy / 2);
      const nf = 1 / (near - far);
      return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
    },
    lookAt(eye, target, up) {
      const z = norm(sub(eye, target));
      const x = norm(cross(up, z));
      const y = cross(z, x);
      return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
      ]);
    },
    multiply(a, b) {
      const out = new Float32Array(16);
      for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
          let s = 0;
          for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
          out[c * 4 + r] = s;
        }
      }
      return out;
    },
  };
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => scale(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1));

  // ---- File parsing -------------------------------------------------------

  // Binary little-endian PLY with float x/y/z and uchar red/green/blue.
  function parsePly(buffer) {
    const bytes = new Uint8Array(buffer);
    const marker = new TextEncoder().encode("end_header\n");
    let headerEnd = -1;
    for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
      if (marker.every((b, j) => bytes[i + j] === b)) { headerEnd = i + marker.length; break; }
    }
    if (headerEnd < 0) throw new Error("not a PLY file");
    const header = new TextDecoder().decode(bytes.subarray(0, headerEnd));
    if (!header.includes("binary_little_endian")) throw new Error("only binary little-endian PLY is supported");
    const count = Number(header.match(/element vertex (\d+)/)[1]);
    const sizes = { float: 4, float32: 4, uchar: 1, uint8: 1, double: 8 };
    const offsets = {};
    let stride = 0;
    for (const [, type, name] of header.matchAll(/property (\w+) (\w+)/g)) {
      offsets[name] = { offset: stride, type };
      stride += sizes[type] ?? 4;
    }
    const view = new DataView(buffer, headerEnd);
    const positions = new Float32Array(count * 3);
    const colors = new Uint8Array(count * 3);
    const hasColor = "red" in offsets;
    for (let i = 0; i < count; i++) {
      const base = i * stride;
      positions[i * 3] = view.getFloat32(base + offsets.x.offset, true);
      positions[i * 3 + 1] = view.getFloat32(base + offsets.y.offset, true);
      positions[i * 3 + 2] = view.getFloat32(base + offsets.z.offset, true);
      if (hasColor) {
        colors[i * 3] = view.getUint8(base + offsets.red.offset);
        colors[i * 3 + 1] = view.getUint8(base + offsets.green.offset);
        colors[i * 3 + 2] = view.getUint8(base + offsets.blue.offset);
      } else {
        colors.fill(200, i * 3, i * 3 + 3);
      }
    }
    return { positions, colors, count };
  }

  // TUM lines: "t tx ty tz qx qy qz qw" -> camera centres.
  function parseTum(text) {
    return text.trim().split("\n").filter(Boolean).map((line) => line.trim().split(/\s+/).slice(1, 4).map(Number));
  }

  function percentile(values, p) {
    const sorted = Float32Array.from(values).sort();
    return sorted[Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))];
  }

  // ---- Viewer -------------------------------------------------------------

  const viewer = (() => {
    const gl = ui.canvas.getContext("webgl2", { antialias: true });
    if (!gl) {
      ui.viewer.querySelector(".viewer-empty h2").textContent = "This browser has no WebGL2";
      return { load() {}, resize() {}, render() {} };
    }

    const VS = `#version 300 es
      in vec3 aPos; in vec3 aColor;
      uniform mat4 uMvp; uniform float uSize; uniform float uScale;
      out vec3 vColor;
      void main() {
        gl_Position = uMvp * vec4(aPos, 1.0);
        gl_PointSize = clamp(uSize * uScale / gl_Position.w, 1.0, 24.0);
        vColor = aColor;
      }`;
    const FS = `#version 300 es
      precision mediump float;
      in vec3 vColor; uniform float uRound; out vec4 outColor;
      void main() {
        if (uRound > 0.5 && length(gl_PointCoord - 0.5) > 0.5) discard;
        outColor = vec4(vColor, 1.0);
      }`;

    function compile(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    const loc = {
      pos: gl.getAttribLocation(program, "aPos"),
      color: gl.getAttribLocation(program, "aColor"),
      mvp: gl.getUniformLocation(program, "uMvp"),
      size: gl.getUniformLocation(program, "uSize"),
      scale: gl.getUniformLocation(program, "uScale"),
      round: gl.getUniformLocation(program, "uRound"),
    };

    // A drawable: one VAO with positions (float) and colours (normalised bytes or floats).
    function makeMesh(positions, colors, normalized) {
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const pb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pb);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc.pos);
      gl.vertexAttribPointer(loc.pos, 3, gl.FLOAT, false, 0, 0);
      const cb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, cb);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc.color);
      gl.vertexAttribPointer(loc.color, 3, normalized ? gl.UNSIGNED_BYTE : gl.FLOAT, normalized, 0, 0);
      gl.bindVertexArray(null);
      return { vao, count: positions.length / 3, buffers: [pb, cb] };
    }
    function freeMesh(mesh) {
      if (!mesh) return;
      gl.deleteVertexArray(mesh.vao);
      mesh.buffers.forEach((b) => gl.deleteBuffer(b));
    }
    const solid = (n, rgb) => {
      const c = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) c.set(rgb, i * 3);
      return c;
    };

    const state = {
      cloud: null, grid: null, path: null, pathDots: null, measureLine: null, measureDots: null, marker: null,
      points: null, target: [0, 0, 0], yaw: -Math.PI / 2, pitch: 0.5, distance: 3, home: null,
      fov: (55 * Math.PI) / 180, gridStep: 0.5, showPath: true, measuring: false, picks: [], dirty: false,
    };

    function buildGrid(minXY, maxXY, z, step) {
      const lines = [];
      const x0 = Math.floor(minXY[0] / step) * step, x1 = Math.ceil(maxXY[0] / step) * step;
      const y0 = Math.floor(minXY[1] / step) * step, y1 = Math.ceil(maxXY[1] / step) * step;
      for (let x = x0; x <= x1 + 1e-9; x += step) lines.push(x, y0, z, x, y1, z);
      for (let y = y0; y <= y1 + 1e-9; y += step) lines.push(x0, y, z, x1, y, z);
      const pos = new Float32Array(lines);
      return makeMesh(pos, solid(pos.length / 3, [0.36, 0.42, 0.5]), false);
    }

    function load(cloud, cameras, metric, markerCorners) {
      [state.cloud, state.grid, state.path, state.pathDots, state.measureLine, state.measureDots, state.marker].forEach(freeMesh);
      // Outline of the fiducial marker that set the scale, if this scan had one.
      state.marker = markerCorners?.length === 4
        ? makeMesh(new Float32Array(markerCorners.flat()), solid(4, [0.25, 0.9, 1.0]), false)
        : null;
      state.cloud = makeMesh(cloud.positions, cloud.colors, true);
      state.points = cloud.positions;
      const xs = [], ys = [], zs = [];
      const n = cloud.count;
      const stepSample = Math.max(1, Math.floor(n / 20000));
      for (let i = 0; i < n; i += stepSample) {
        xs.push(cloud.positions[i * 3]); ys.push(cloud.positions[i * 3 + 1]); zs.push(cloud.positions[i * 3 + 2]);
      }
      const lo = [percentile(xs, 2), percentile(ys, 2), percentile(zs, 2)];
      const hi = [percentile(xs, 98), percentile(ys, 98), percentile(zs, 98)];
      const center = scale(add(lo, hi), 0.5);
      const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.1);
      state.gridStep = extent > 6 ? 1 : extent > 1.5 ? 0.5 : 0.1;
      state.grid = buildGrid(lo, hi, lo[2], state.gridStep);
      state.metric = metric;
      ui.gridLabel.textContent = metric
        ? `Grid square ${state.gridStep >= 1 ? state.gridStep + " m" : state.gridStep * 100 + " cm"}`
        : `Grid square ${state.gridStep} model units`;
      if (cameras.length > 1) {
        const pos = new Float32Array(cameras.flat());
        state.path = makeMesh(pos, solid(cameras.length, [0.86, 0.65, 0.2]), false);
        state.pathDots = makeMesh(pos, solid(cameras.length, [1.0, 0.8, 0.35]), false);
      } else {
        state.path = state.pathDots = null;
      }
      state.home = { target: center, distance: extent * 1.4, yaw: -Math.PI / 2, pitch: 0.45 };
      // Start behind the first camera when we know where it was.
      if (cameras.length) {
        const c = cameras[0];
        const d = sub(c, center);
        state.home.yaw = Math.atan2(d[1], d[0]);
        state.home.pitch = Math.max(0.15, Math.min(1.2, Math.atan2(d[2], Math.hypot(d[0], d[1])) + 0.25));
      }
      resetView();
      clearMeasure();
      ui.viewer.classList.add("loaded");
    }

    function resetView() {
      if (!state.home) return;
      Object.assign(state, { ...state.home, target: [...state.home.target] });
      requestRender();
    }

    function eye() {
      const cp = Math.cos(state.pitch);
      return add(state.target, scale([cp * Math.cos(state.yaw), cp * Math.sin(state.yaw), Math.sin(state.pitch)], state.distance));
    }

    function matrices() {
      const w = ui.canvas.width, h = ui.canvas.height;
      const proj = mat4.perspective(state.fov, w / h, Math.max(0.005, state.distance / 500), state.distance * 50 + 50);
      const view = mat4.lookAt(eye(), state.target, [0, 0, 1]);
      return { mvp: mat4.multiply(proj, view), w, h };
    }

    function draw(mesh, mode, size, round) {
      if (!mesh) return;
      gl.uniform1f(loc.size, size);
      gl.uniform1f(loc.round, round ? 1 : 0);
      gl.bindVertexArray(mesh.vao);
      gl.drawArrays(mode, 0, mesh.count);
    }

    function render() {
      state.dirty = false;
      const { mvp, h } = matrices();
      gl.viewport(0, 0, ui.canvas.width, ui.canvas.height);
      gl.clearColor(0.114, 0.145, 0.2, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (!state.cloud) return;
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(program);
      gl.uniformMatrix4fv(loc.mvp, false, mvp);
      // Points about 1 cm across in the world, scaled by the slider.
      const pxPerUnitAtDepth1 = h / (2 * Math.tan(state.fov / 2));
      gl.uniform1f(loc.scale, pxPerUnitAtDepth1 * 0.01);
      draw(state.grid, gl.LINES, 1, false);
      draw(state.cloud, gl.POINTS, Number(ui.size.value), false);
      if (state.showPath) {
        gl.uniform1f(loc.scale, pxPerUnitAtDepth1);
        draw(state.path, gl.LINE_STRIP, 1, false);
        draw(state.pathDots, gl.POINTS, 0.025, true);
      }
      gl.disable(gl.DEPTH_TEST);
      gl.uniform1f(loc.scale, pxPerUnitAtDepth1);
      draw(state.marker, gl.LINE_LOOP, 1, false);
      draw(state.marker, gl.POINTS, 0.012, true);
      draw(state.measureLine, gl.LINES, 1, false);
      draw(state.measureDots, gl.POINTS, 0.02, true);
    }

    function requestRender() {
      if (!state.dirty) {
        state.dirty = true;
        requestAnimationFrame(render);
      }
    }

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(ui.canvas.clientWidth * dpr));
      const h = Math.max(1, Math.round(ui.canvas.clientHeight * dpr));
      if (ui.canvas.width !== w || ui.canvas.height !== h) {
        ui.canvas.width = w;
        ui.canvas.height = h;
      }
      requestRender();
    }
    new ResizeObserver(resize).observe(ui.canvas);

    // Nearest cloud point to a screen position (within 14 px), or null.
    function pick(clientX, clientY) {
      if (!state.points) return null;
      const rect = ui.canvas.getBoundingClientRect();
      const dpr = ui.canvas.width / rect.width;
      const sx = (clientX - rect.left) * dpr, sy = (clientY - rect.top) * dpr;
      const { mvp, w, h } = matrices();
      const p = state.points;
      let best = null, bestScore = Infinity;
      const radius = 14 * dpr;
      for (let i = 0; i < p.length; i += 3) {
        const x = p[i], y = p[i + 1], z = p[i + 2];
        const cw = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
        if (cw <= 0) continue;
        const px = ((mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12]) / cw * 0.5 + 0.5) * w;
        const py = (1 - ((mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13]) / cw * 0.5 + 0.5)) * h;
        const d2 = (px - sx) ** 2 + (py - sy) ** 2;
        if (d2 > radius * radius) continue;
        const score = d2 / (radius * radius) + cw / state.distance; // prefer near points under the cursor
        if (score < bestScore) { bestScore = score; best = [x, y, z]; }
      }
      return best;
    }

    function clearMeasure() {
      state.picks = [];
      freeMesh(state.measureLine); freeMesh(state.measureDots);
      state.measureLine = state.measureDots = null;
      ui.readout.textContent = "";
      requestRender();
    }

    function addPick(point) {
      if (state.picks.length === 2) state.picks = [];
      state.picks.push(point);
      freeMesh(state.measureLine); freeMesh(state.measureDots);
      const pos = new Float32Array(state.picks.flat());
      state.measureDots = makeMesh(pos, solid(state.picks.length, [1, 1, 1]), false);
      state.measureLine = state.picks.length === 2 ? makeMesh(pos, solid(2, [1, 1, 1]), false) : null;
      if (state.picks.length === 2) {
        const d = Math.hypot(...sub(state.picks[0], state.picks[1]));
        ui.readout.textContent = state.metric ? `${d < 1 ? (d * 100).toFixed(1) + " cm" : d.toFixed(3) + " m"}` : `${d.toFixed(3)} model units (no metric scale)`;
      } else {
        ui.readout.textContent = "Click the second point";
      }
      requestRender();
    }

    // ---- Input ------------------------------------------------------------

    let drag = null;
    ui.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    ui.canvas.addEventListener("pointerdown", (e) => {
      ui.canvas.focus();
      ui.canvas.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey, moved: false };
    });
    ui.canvas.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) { drag.moved = true; ui.canvas.classList.add("dragging"); }
      drag.x = e.clientX; drag.y = e.clientY;
      if (drag.pan) {
        const f = norm(sub(state.target, eye()));
        const right = norm(cross(f, [0, 0, 1]));
        const up = cross(right, f);
        const k = (state.distance * 2 * Math.tan(state.fov / 2)) / ui.canvas.clientHeight;
        state.target = add(state.target, add(scale(right, -dx * k), scale(up, dy * k)));
      } else {
        state.yaw -= dx * 0.008;
        state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch + dy * 0.008));
      }
      requestRender();
    });
    ui.canvas.addEventListener("pointerup", (e) => {
      ui.canvas.classList.remove("dragging");
      if (drag && !drag.moved && state.measuring && e.button === 0) {
        const p = pick(e.clientX, e.clientY);
        if (p) addPick(p);
      }
      drag = null;
    });
    ui.canvas.addEventListener("dblclick", (e) => {
      const p = pick(e.clientX, e.clientY);
      if (p) { state.target = p; state.distance *= 0.6; requestRender(); }
    });
    ui.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      state.distance = Math.max(0.02, Math.min(500, state.distance * Math.exp(e.deltaY * 0.0012)));
      requestRender();
    }, { passive: false });
    ui.canvas.addEventListener("keydown", (e) => {
      const f = norm(sub(state.target, eye()));
      const flat = norm([f[0], f[1], 0]);
      const right = norm(cross(flat, [0, 0, 1]));
      const step = state.distance * 0.06;
      const moves = { w: flat, s: scale(flat, -1), d: right, a: scale(right, -1), e: [0, 0, 1], q: [0, 0, -1] };
      const m = moves[e.key.toLowerCase()];
      if (m) { state.target = add(state.target, scale(m, step)); requestRender(); e.preventDefault(); }
      if (e.key === "Escape") clearMeasure();
    });

    ui.reset.addEventListener("click", resetView);
    ui.size.addEventListener("input", requestRender);
    ui.path.addEventListener("click", () => {
      state.showPath = !state.showPath;
      ui.path.setAttribute("aria-pressed", String(state.showPath));
      requestRender();
    });
    ui.measure.addEventListener("click", () => {
      state.measuring = !state.measuring;
      ui.measure.setAttribute("aria-pressed", String(state.measuring));
      ui.canvas.classList.toggle("measuring", state.measuring);
      if (state.measuring) ui.readout.textContent = "Click the first point"; else clearMeasure();
    });

    return { load, resize, render: requestRender };
  })();

  // ---- Scan list ----------------------------------------------------------

  let listed = false;
  let current = null;

  function describe(scan) {
    const when = new Date(scan.created).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    const pts = scan.points ? `${(scan.points / 1000).toFixed(0)}k points` : "";
    const source = {
      marker: "marker scale",
      imu: "IMU scale",
      "metric-depth (calibrated)": "calibrated depth-model scale",
      "metric-depth": "depth-model scale",
      none: "no metric scale",
    }[scan.scale_source] ?? "";
    return { when, detail: [pts, source].filter(Boolean).join(", ") };
  }

  async function loadList(openName = null) {
    listed = true;
    ui.items.replaceChildren();
    let scans = [];
    let res;
    try {
      res = await fetch("/api/scans");
    } catch {
      ui.hint.hidden = false;
      ui.hint.textContent = "Could not reach the laptop server. Check that uv run python -m pocketslam is running.";
      return;
    }
    if (res.status === 404) {
      ui.hint.hidden = false;
      ui.hint.textContent = "The laptop server is older than this page and cannot list scans. Restart it: press Ctrl+C, then run uv run python -m pocketslam.";
      return;
    }
    try {
      if (!res.ok) throw new Error(String(res.status));
      scans = await res.json();
    } catch {
      ui.hint.hidden = false;
      ui.hint.textContent = `The laptop server answered, but its scan list could not be read (status ${res.status}). Restart the server; if it persists, a scan's summary.json is damaged.`;
      return;
    }
    ui.hint.hidden = scans.length > 0;
    for (const scan of scans) {
      const { when, detail } = describe(scan);
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.innerHTML = `<span class="name"></span><span class="meta"></span><span class="meta"></span>`;
      btn.children[0].textContent = scan.name;
      btn.children[1].textContent = when;
      btn.children[2].textContent = detail;
      btn.addEventListener("click", () => open(scan, btn));
      li.append(btn);
      ui.items.append(li);
    }
    const wanted = openName ? [...ui.items.querySelectorAll("button")].find((b) => b.children[0].textContent === openName) : null;
    if (wanted) wanted.click();
    else if (scans.length && !current) ui.items.querySelector("button").click();
  }

  async function open(scan, btn) {
    current = scan.name;
    ui.items.querySelectorAll("button").forEach((b) => b.setAttribute("aria-current", String(b === btn)));
    ui.title.textContent = `Loading ${scan.name}…`;
    try {
      const base = `/api/scans/${encodeURIComponent(scan.name)}`;
      const [plyRes, tumRes, sumRes] = await Promise.all([
        fetch(`${base}/points.ply`),
        fetch(`${base}/trajectory.tum`),
        fetch(`${base}/summary.json`),
      ]);
      if (!plyRes.ok) throw new Error(`point cloud missing (${plyRes.status})`);
      const cloud = parsePly(await plyRes.arrayBuffer());
      const cameras = tumRes.ok ? parseTum(await tumRes.text()) : [];
      const marker = sumRes.ok ? (await sumRes.json()).marker : null;
      if (current !== scan.name) return; // another scan was picked meanwhile
      viewer.load(cloud, cameras, scan.scale_source !== "none", marker?.corners_world);
      const { detail } = describe(scan);
      const markerNote = marker
        ? ` Marker outlined in cyan: ${(marker.size_m * 100).toFixed(1)} cm, seen in ${marker.views_used} views.` +
          (marker.focal_calibrated ? "" : " Camera focal length not measured yet, so distances may be off; run pocketslam-calibrate.")
        : "";
      ui.title.textContent = `${scan.name}. ${detail}.${markerNote}`;
    } catch (err) {
      ui.title.textContent = `Could not open ${scan.name}: ${err.message}`;
    }
  }

  // ---- Reconstruction jobs (clips saved with the Record button) -------------

  const jobStates = new Map();
  let jobTimer = null;
  const JOB_TEXT = {
    queued: "Waiting to reconstruct",
    running: "Reconstructing 3D scan (about a minute)",
    failed: "Reconstruction failed",
    skipped: "Too short to reconstruct",
  };

  async function pollJobs() {
    clearTimeout(jobTimer);
    if (ui.viewScans.hidden) return;
    let jobs = [];
    try {
      const res = await fetch("/api/jobs");
      if (res.ok) jobs = await res.json();
    } catch {
      // Server unreachable; try again on the next tick.
    }
    let finished = null;
    for (const job of jobs) {
      const before = jobStates.get(job.name);
      if (before && before !== "done" && job.state === "done") finished = job.name;
      jobStates.set(job.name, job.state);
    }
    ui.jobs.replaceChildren(
      ...jobs.filter((j) => j.state !== "done").slice(0, 5).map((job) => {
        const li = document.createElement("li");
        li.dataset.state = job.state;
        li.innerHTML = '<span class="job-name"></span><span class="job-state"></span>';
        li.children[0].textContent = job.name;
        li.children[1].textContent = job.message ? `${JOB_TEXT[job.state]}: ${job.message}` : JOB_TEXT[job.state];
        return li;
      }),
    );
    if (finished) {
      current = null;
      await loadList(finished);
    }
    const active = jobs.some((j) => j.state === "queued" || j.state === "running");
    jobTimer = setTimeout(pollJobs, active ? 2000 : 6000);
  }

  ui.refresh.addEventListener("click", () => loadList());
  showView();
})();

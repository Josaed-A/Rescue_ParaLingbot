import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { PLYLoader } from "three/addons/PLYLoader.js";
import { GLTFLoader } from "three/addons/GLTFLoader.js";
import * as GS from "./vendor/gaussian-splats-3d.module.js";

// ---------------------------------------------------------------------------
// Scene / renderer / camera (shared by both navigation modes)
// ---------------------------------------------------------------------------
const wrap = document.getElementById("canvas-wrap");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
wrap.appendChild(renderer.domElement);
const canvas = renderer.domElement;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 5000);
camera.position.set(0, 0, 3);

function onResize() {
  const w = wrap.clientWidth || window.innerWidth;
  const h = wrap.clientHeight || window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener("resize", onResize);

const axesHelper = new THREE.AxesHelper(1);
axesHelper.visible = false;
scene.add(axesHelper);

// ---------------------------------------------------------------------------
// Órbita: la navegación que ya tenía el visor viser (subir/bajar, acercar,
// desplazar de lado a lado) -- se mantiene tal cual como uno de los dos modos.
// ---------------------------------------------------------------------------
const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.rotateSpeed = 0.9;
orbit.zoomSpeed = 1.0;
orbit.panSpeed = 0.9;
orbit.screenSpacePanning = true;

// ---------------------------------------------------------------------------
// Primera persona: mouse-look (pointer lock) + WASD + gamepad, sin las
// restricciones de una órbita alrededor de un punto -- vuelo libre por la nube.
// ---------------------------------------------------------------------------
const fps = {
  active: false,
  yaw: 0,
  pitch: 0,
  keys: new Set(),
  locked: false,
};

function setMode(mode) {
  const wasFps = fps.active;
  fps.active = mode === "fps";
  document.getElementById("mode-orbit").classList.toggle("active", !fps.active);
  document.getElementById("mode-fps").classList.toggle("active", fps.active);
  orbit.enabled = !fps.active;
  document.getElementById("crosshair").style.display = fps.active ? "block" : "none";
  // el marcador queda justo donde se teletransporta la cámara: en primera persona
  // taparía toda la vista, así que sólo se muestra en órbita
  if (camMarker) camMarker.visible = !fps.active;
  if (fps.active && !wasFps) {
    // Al entrar, arrancar mirando hacia donde ya apuntaba la cámara de órbita.
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    lookFrom(dir);
    document.getElementById("pointerlock-hint").style.display = fps.locked ? "none" : "flex";
  }
  if (!fps.active) {
    document.getElementById("pointerlock-hint").style.display = "none";
    if (document.pointerLockElement === canvas) document.exitPointerLock();
    // Al volver a órbita, apuntar el objetivo a un punto delante de la cámara.
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    orbit.target.copy(camera.position).addScaledVector(dir, 5);
  }
}
document.getElementById("mode-orbit").onclick = () => setMode("orbit");
document.getElementById("mode-fps").onclick = () => setMode("fps");

canvas.addEventListener("click", () => {
  if (fps.active && document.pointerLockElement !== canvas) canvas.requestPointerLock();
});
document.getElementById("pointerlock-hint").addEventListener("click", () => {
  if (fps.active) canvas.requestPointerLock();
});
document.addEventListener("pointerlockchange", () => {
  fps.locked = document.pointerLockElement === canvas;
  document.getElementById("pointerlock-hint").style.display =
    fps.active && !fps.locked ? "flex" : "none";
});
document.addEventListener("mousemove", (e) => {
  if (!fps.active || !fps.locked) return;
  const sens = 0.0022;
  fps.yaw -= e.movementX * sens;
  fps.pitch -= e.movementY * sens;
  fps.pitch = THREE.MathUtils.clamp(fps.pitch, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
});
window.addEventListener("keydown", (e) => {
  fps.keys.add(e.code);
  if (e.code === "Escape" && fps.locked) document.exitPointerLock();
});
window.addEventListener("keyup", (e) => fps.keys.delete(e.code));

function applyFpsLook() {
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(fps.pitch, fps.yaw, 0, "YXZ"));
  camera.quaternion.copy(q);
}

// Con Euler(pitch, yaw, 0, "YXZ") la cámara mira hacia
//   (-sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))
// (three.js mira por -Z). Derivar yaw/pitch de una dirección tiene que usar
// esos signos: con el signo cambiado la vista queda 180° girada respecto del
// movimiento, que es justo lo que pasaba al saltar a una cámara.
function lookFrom(dir) {
  const d = dir.clone().normalize();
  fps.pitch = Math.asin(THREE.MathUtils.clamp(d.y, -1, 1));
  fps.yaw = Math.atan2(-d.x, -d.z);
}

// Ejes de movimiento tomados de la cámara ya orientada, no recalculados del
// yaw: así avanzar siempre coincide con lo que se está mirando.
const WORLD_UP = new THREE.Vector3(0, 1, 0);
function moveAxes() {
  const fwd = new THREE.Vector3();
  camera.getWorldDirection(fwd);
  const right = new THREE.Vector3().crossVectors(fwd, WORLD_UP).normalize();
  if (!isFinite(right.x) || right.lengthSq() < 1e-6) right.set(1, 0, 0);
  return { fwd, right };
}

function fpsMove(dt) {
  let speed = parseFloat(document.getElementById("fps-speed").value);
  const sprint = fps.keys.has("ShiftLeft") || fps.keys.has("ShiftRight") || gamepadSprint();
  if (sprint) speed *= 3.0;

  const { fwd: forward, right } = moveAxes();
  const move = new THREE.Vector3();

  if (fps.keys.has("KeyW") || fps.keys.has("ArrowUp")) move.add(forward);
  if (fps.keys.has("KeyS") || fps.keys.has("ArrowDown")) move.sub(forward);
  if (fps.keys.has("KeyD") || fps.keys.has("ArrowRight")) move.add(right);
  if (fps.keys.has("KeyA") || fps.keys.has("ArrowLeft")) move.sub(right);
  if (fps.keys.has("Space")) move.y += 1;
  if (fps.keys.has("ControlLeft") || fps.keys.has("KeyC")) move.y -= 1;

  applyGamepad(dt, forward, right, speed);

  if (move.lengthSq() > 0) {
    move.normalize().multiplyScalar(speed * dt);
    camera.position.add(move);
  }
}

// ---------------------------------------------------------------------------
// Control Xbox (Gamepad API): funciona en 1ª persona haya o no mouse
// capturado -- stick izquierdo mueve, stick derecho mira, gatillos suben/bajan.
// ---------------------------------------------------------------------------
const DEADZONE = 0.15;
function axis(v) { return Math.abs(v) < DEADZONE ? 0 : v; }

function getPad() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  for (const p of pads) if (p) return p;
  return null;
}
function gamepadSprint() {
  const pad = getPad();
  return !!(pad && pad.buttons[0] && pad.buttons[0].pressed); // A
}
let lastStart = false;
function applyGamepad(dt, forward, right, speed) {
  const pad = getPad();
  document.getElementById("gamepad-status").textContent = pad ? pad.id.slice(0, 22) : "no detectado";
  if (!pad) return;
  const lx = axis(pad.axes[0] || 0), ly = axis(pad.axes[1] || 0);
  const rx = axis(pad.axes[2] || 0), ry = axis(pad.axes[3] || 0);
  const rt = pad.buttons[7] ? pad.buttons[7].value : 0;
  const lt = pad.buttons[6] ? pad.buttons[6].value : 0;
  const start = pad.buttons[9] && pad.buttons[9].pressed;

  if (start && !lastStart) setMode(fps.active ? "orbit" : "fps");
  lastStart = !!start;

  if (!fps.active) return;
  const move = new THREE.Vector3();
  move.addScaledVector(forward, -ly);
  move.addScaledVector(right, lx);
  move.y += rt - lt;
  if (move.lengthSq() > 0) camera.position.addScaledVector(move.normalize(), speed * dt * Math.hypot(lx, ly, rt - lt || 0.001));
  fps.yaw -= rx * 2.2 * dt;
  fps.pitch -= ry * 1.8 * dt;
  fps.pitch = THREE.MathUtils.clamp(fps.pitch, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
}

// ---------------------------------------------------------------------------
// Carga de nubes / cámaras
// ---------------------------------------------------------------------------
let loadedGroup = null;
let pointObjects = [];
let cameraPoses = [];   // [{index, c2w:[4][4]}]

let splatViewer = null;
function clearScene() {
  if (loadedGroup) scene.remove(loadedGroup);
  loadedGroup = null;
  pointObjects = [];
  if (splatViewer) {
    try { splatViewer.dispose(); } catch (e) { console.warn(e); }
    splatViewer = null;
  }
}

// ---------------------------------------------------------------------------
// Gaussian Splatting: se dibuja con gaussian-splats-3d (vendor/) metido en la misma
// escena three.js, así hereda la normalización, la órbita, la primera persona y el
// control. La caja para normalizar sale de los centros de las gaussianas (percentiles
// 2-98: las gaussianas flotantes de los bordes no deben decidir la escala).
// ---------------------------------------------------------------------------
async function loadSplat(meta) {
  const dv = new GS.DropInViewer({
    sharedMemoryForWorkers: !!window.crossOriginIsolated,
    gpuAcceleratedSort: false,
    sphericalHarmonicsDegree: 1,
    dynamicScene: false,
    // por defecto la librería "revela" la escena de a poco, radialmente desde el
    // centro: con muchas gaussianas parecía que faltaba media escena
    sceneRevealMode: GS.SceneRevealMode.Instant,
    logLevel: GS.LogLevel.None,
  });
  setStatus(`cargando gaussian splat (${meta.size_mb} MB)...`);
  await dv.addSplatScene(`/data/${meta.file}`, {
    showLoadingUI: false, progressiveLoad: false, format: GS.SceneFormat.Ply,
    splatAlphaRemovalThreshold: 5,
  });
  splatViewer = dv;
  const mesh = dv.viewer.splatMesh;
  const n = mesh.getSplatCount();
  const step = Math.max(1, Math.floor(n / 100000));
  const xs = [], ys = [], zs = [];
  const v = new THREE.Vector3();
  for (let i = 0; i < n; i += step) {
    mesh.getSplatCenter(i, v);
    xs.push(v.x); ys.push(v.y); zs.push(v.z);
  }
  const q = (arr, f) => { arr.sort((a, b) => a - b); return arr[Math.floor(f * (arr.length - 1))]; };
  const lo = new THREE.Vector3(q(xs, 0.02), q(ys, 0.02), q(zs, 0.02));
  const hi = new THREE.Vector3(q(xs, 0.98), q(ys, 0.98), q(zs, 0.98));
  return { object: dv, points: [], box: new THREE.Box3(lo, hi), count: n };
}

function setStatus(msg) { document.getElementById("status").textContent = msg; }

function transformPoint(c2w, x, y, z) {
  return new THREE.Vector3(
    c2w[0][0] * x + c2w[0][1] * y + c2w[0][2] * z + c2w[0][3],
    c2w[1][0] * x + c2w[1][1] * y + c2w[1][2] * z + c2w[1][3],
    c2w[2][0] * x + c2w[2][1] * y + c2w[2][2] * z + c2w[2][3]
  );
}

// Degradado viridis, el mismo que usa el visor del repo original
// (lingbot_map/vis/point_cloud_viewer.py: cmap viridis sobre el índice de frame).
const VIRIDIS = [
  [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142], [33, 145, 140],
  [34, 168, 132], [68, 191, 112], [122, 209, 81], [189, 223, 38], [253, 231, 37],
];
function viridis(t) {
  t = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.min(Math.floor(t), VIRIDIS.length - 2);
  const f = t - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [(a[0] + (b[0] - a[0]) * f) / 255, (a[1] + (b[1] - a[1]) * f) / 255, (a[2] + (b[2] - a[2]) * f) / 255];
}

function buildCameraVisuals(cams) {
  const group = new THREE.Group();
  const positions = cams.map((c) => transformPoint(c.c2w, 0, 0, 0));
  const n = Math.max(cams.length - 1, 1);

  // Trayectoria con el mismo degradado, coloreada por avance del recorrido
  const pathGeo = new THREE.BufferGeometry().setFromPoints(positions);
  const pathCols = new Float32Array(positions.length * 3);
  positions.forEach((_, i) => {
    const c = viridis(i / n);
    pathCols[i * 3] = c[0]; pathCols[i * 3 + 1] = c[1]; pathCols[i * 3 + 2] = c[2];
  });
  pathGeo.setAttribute("color", new THREE.BufferAttribute(pathCols, 3));
  group.add(new THREE.Line(pathGeo, new THREE.LineBasicMaterial({ vertexColors: true })));

  // espaciado típico entre cámaras consecutivas, para escalar las flechas
  const spacings = [];
  for (let i = 1; i < positions.length; i++) spacings.push(positions[i].distanceTo(positions[i - 1]));
  spacings.sort((a, b) => a - b);
  const med = spacings.length ? spacings[Math.floor(spacings.length / 2)] || 0.05 : 0.05;

  // Flechas en vez de pirámides de frustum: los recuadros se superponían y
  // tapaban el mapa. Cada cámara es una flecha corta que apunta a donde mira.
  const len = Math.max(med * 1.6, 1e-4);
  const barb = len * 0.33;
  const verts = [], cols = [];
  const stride = Math.max(1, Math.ceil(cams.length / 220));
  const push = (a, b, c) => {
    verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    cols.push(c[0], c[1], c[2], c[0], c[1], c[2]);
  };
  for (let i = 0; i < cams.length; i += stride) {
    const c2w = cams[i].c2w;
    const o = transformPoint(c2w, 0, 0, 0);
    const tip = transformPoint(c2w, 0, 0, len);
    // ejes locales de la cámara (OpenCV: +X derecha, +Y abajo, +Z adelante)
    const rgt = transformPoint(c2w, 1, 0, 0).sub(o).normalize();
    const dwn = transformPoint(c2w, 0, 1, 0).sub(o).normalize();
    const back = o.clone().sub(tip).normalize().multiplyScalar(barb);
    const col = viridis(i / n);
    push(o, tip, col);                                                   // asta
    push(tip, tip.clone().add(back).add(rgt.clone().multiplyScalar(barb * 0.55)), col);
    push(tip, tip.clone().add(back).add(dwn.clone().multiplyScalar(barb * 0.55)), col);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(cols, 3));
  group.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true })));

  return { group, positions };
}

let camPositions = [];
let camMarker = null;
let lastBox = null, lastDown = null;

// Tamaño canónico: la escala de cada captura es arbitraria (profundidad
// monocular), así que cada mapa llegaba con una escala distinta y se navegaba
// distinto. Todo lo cargado se normaliza a esta diagonal y se centra en el
// origen, de modo que la vista inicial, la velocidad de vuelo, el tamaño de
// punto y los planos de recorte sean IGUALES para todos los mapas.
const CANON_DIAG = 10;

// Orientación: LingBot-Map usa la convención de cámara de OpenCV, donde el eje
// local +Y de cada cámara apunta HACIA ABAJO (al piso). three.js asume +Y arriba,
// así que sin corregir esto el mapa se ve dado vuelta: el piso queda de techo.
// El "abajo" del mundo se saca promediando el eje Y de las cámaras (lo mismo que
// hace compare_route.py para proyectar la ruta); si la nube no trae cámaras se usa
// +Y como abajo, que es lo que produce este pipeline.
let flipExtra = false;            // inversión manual, por si una captura viene rara

function worldDownFrom(cams) {
  const d = new THREE.Vector3(0, 1, 0);
  if (cams && cams.length) {
    const acc = new THREE.Vector3();
    for (const c of cams) acc.add(new THREE.Vector3(c.c2w[0][1], c.c2w[1][1], c.c2w[2][1]));
    if (acc.lengthSq() > 1e-9) d.copy(acc.normalize());
  }
  return d;
}

function normalizeGroup(group, box, downVec) {
  const down = (downVec || new THREE.Vector3(0, 1, 0)).clone().normalize();
  if (flipExtra) down.negate();
  const q = new THREE.Quaternion().setFromUnitVectors(down, new THREE.Vector3(0, -1, 0));
  group.quaternion.copy(q);

  // caja después de rotar: se transforman las 8 esquinas de la caja original
  const rb = new THREE.Box3();
  const mn = box.min, mx = box.max;
  for (let i = 0; i < 8; i++) {
    rb.expandByPoint(new THREE.Vector3(
      i & 1 ? mx.x : mn.x, i & 2 ? mx.y : mn.y, i & 4 ? mx.z : mn.z).applyQuaternion(q));
  }
  const size = new THREE.Vector3(), center = new THREE.Vector3();
  rb.getSize(size); rb.getCenter(center);
  const k = CANON_DIAG / Math.max(size.length(), 1e-6);
  group.scale.setScalar(k);
  group.position.copy(center).multiplyScalar(-k);
  group.updateMatrixWorld(true);
  return new THREE.Box3().setFromCenterAndSize(
    new THREE.Vector3(0, 0, 0), size.clone().multiplyScalar(k));
}

function frameScene(box) {
  const center = new THREE.Vector3();
  box.getCenter(center);
  const size = new THREE.Vector3();
  box.getSize(size);
  const radius = Math.max(size.length() * 0.5, 0.1);
  // Con todo normalizado a CANON_DIAG, estos rangos valen para cualquier mapa.
  const spd = document.getElementById("fps-speed");
  spd.min = 0.05; spd.max = 5; spd.step = 0.05; spd.value = 0.4;
  const ps = document.getElementById("point-size");
  ps.min = 0.0005; ps.max = 0.05; ps.step = 0.0005; ps.value = 0.0035;
  for (const pt of pointObjects) pt.material.size = parseFloat(ps.value);
  if (live.points) live.points.material.size = parseFloat(ps.value);
  camera.position.copy(center).add(new THREE.Vector3(radius, radius * 0.6, radius));
  camera.near = Math.max(radius / 1000, 0.001);
  camera.far = radius * 50;
  camera.updateProjectionMatrix();
  orbit.target.copy(center);
  orbit.update();
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  lookFrom(dir);
}

// ---------------------------------------------------------------------------
// Lector rápido de PLY binario.
// PLYLoader de three.js acumula los vértices en arrays JS normales antes de
// pasarlos a arrays tipados: con 46.7 M de puntos eso revienta ("Invalid array
// length"). Los PLY de este repo tienen un layout fijo y conocido (x,y,z en
// float o double + rgb en uchar), así que se llenan los arrays tipados
// directamente desde el ArrayBuffer. Si el archivo no encaja, se usa PLYLoader.
// ---------------------------------------------------------------------------
function parsePlyFast(buffer) {
  const bytes = new Uint8Array(buffer);
  const probe = new TextDecoder("ascii").decode(bytes.subarray(0, Math.min(4096, bytes.length)));
  const endTag = "end_header\n";
  const endIdx = probe.indexOf(endTag);
  if (endIdx < 0) return null;
  const dataStart = endIdx + endTag.length;          // cabecera ASCII: 1 byte por carácter
  const header = probe.slice(0, endIdx);
  if (!/format\s+binary_little_endian/.test(header)) return null;

  const mCount = header.match(/element\s+vertex\s+(\d+)/);
  if (!mCount) return null;
  const count = parseInt(mCount[1], 10);

  const SIZES = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2,
                  int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
  const props = [];
  for (const line of header.split("\n")) {
    const m = line.match(/^property\s+(\w+)\s+(\w+)\s*$/);
    if (m && SIZES[m[1]] !== undefined) props.push({ type: m[1], name: m[2] });
    else if (/^property\s+list/.test(line)) return null;         // caras: no es una nube simple
  }
  let stride = 0;
  const off = {};
  for (const pr of props) { off[pr.name] = { o: stride, t: pr.type }; stride += SIZES[pr.type]; }
  for (const k of ["x", "y", "z"]) if (!off[k]) return null;
  const xt = off.x.t;
  if (xt !== "float" && xt !== "float32" && xt !== "double" && xt !== "float64") return null;
  if (off.y.t !== xt || off.z.t !== xt) return null;
  const hasColor = off.red && off.green && off.blue &&
                   SIZES[off.red.t] === 1 && SIZES[off.green.t] === 1 && SIZES[off.blue.t] === 1;
  if (buffer.byteLength - dataStart < count * stride) return null;

  const dv = new DataView(buffer, dataStart);
  const pos = new Float32Array(count * 3);
  const col = hasColor ? new Uint8Array(count * 3) : null;
  const isDouble = xt === "double" || xt === "float64";
  const ox = off.x.o, oy = off.y.o, oz = off.z.o;
  const orr = hasColor ? off.red.o : 0, og = hasColor ? off.green.o : 0, ob = hasColor ? off.blue.o : 0;
  for (let i = 0, base = 0, p = 0; i < count; i++, base += stride, p += 3) {
    if (isDouble) {
      pos[p] = dv.getFloat64(base + ox, true);
      pos[p + 1] = dv.getFloat64(base + oy, true);
      pos[p + 2] = dv.getFloat64(base + oz, true);
    } else {
      pos[p] = dv.getFloat32(base + ox, true);
      pos[p + 1] = dv.getFloat32(base + oy, true);
      pos[p + 2] = dv.getFloat32(base + oz, true);
    }
    if (hasColor) {
      col[p] = dv.getUint8(base + orr);
      col[p + 1] = dv.getUint8(base + og);
      col[p + 2] = dv.getUint8(base + ob);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  if (col) geo.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
  geo.computeBoundingBox();
  return geo;
}

function fetchWithProgress(url, label) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", url);
    xhr.responseType = "arraybuffer";
    xhr.onprogress = (e) => {
      if (e.lengthComputable) setStatus(`${label} ${(e.loaded / e.total * 100).toFixed(0)}% (${(e.total / 2 ** 20).toFixed(0)} MB)`);
    };
    xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve(xhr.response) : reject(new Error("HTTP " + xhr.status)));
    xhr.onerror = () => reject(new Error("error de red"));
    xhr.send();
  });
}

async function loadCloud(url, ext) {
  if (ext === "ply") {
    const buf = await fetchWithProgress(url, "cargando nube...");
    setStatus("interpretando la nube...");
    await new Promise((r) => setTimeout(r, 0));     // dejar pintar el estado
    const geometry = parsePlyFast(buf);
    if (geometry) {
      const mat = new THREE.PointsMaterial({
        size: parseFloat(document.getElementById("point-size").value),
        vertexColors: geometry.hasAttribute("color"),
        color: geometry.hasAttribute("color") ? 0xffffff : 0xbfe3ff,
        sizeAttenuation: true,
      });
      const points = new THREE.Points(geometry, mat);
      return { object: points, points: [points], box: geometry.boundingBox,
               count: geometry.attributes.position.count };
    }
    console.warn("layout de PLY no reconocido: se usa PLYLoader");
  }
  return new Promise((resolve, reject) => {
    if (ext === "ply") {
      new PLYLoader().load(
        url,
        (geometry) => {
          geometry.computeBoundingBox();
          const mat = new THREE.PointsMaterial({
            size: parseFloat(document.getElementById("point-size").value),
            vertexColors: geometry.hasAttribute("color"),
            color: geometry.hasAttribute("color") ? 0xffffff : 0xbfe3ff,
            sizeAttenuation: true,
          });
          const points = new THREE.Points(geometry, mat);
          resolve({ object: points, points: [points], box: geometry.boundingBox, count: geometry.attributes.position.count });
        },
        (xhr) => { if (xhr.lengthComputable) setStatus(`cargando nube... ${(xhr.loaded / xhr.total * 100).toFixed(0)}%`); },
        reject
      );
    } else {
      new GLTFLoader().load(
        url,
        (gltf) => {
          const box = new THREE.Box3().setFromObject(gltf.scene);
          const pts = [];
          let count = 0;
          gltf.scene.traverse((o) => {
            if (o.isPoints) {
              o.material = new THREE.PointsMaterial({
                size: parseFloat(document.getElementById("point-size").value),
                vertexColors: !!(o.geometry.hasAttribute && o.geometry.hasAttribute("color")),
                color: 0xbfe3ff,
                sizeAttenuation: true,
              });
              pts.push(o);
              count += o.geometry.attributes.position.count;
            } else if (o.isMesh) {
              // mallas (TSDF): sin luces en la escena, así que color de vértice sin sombreado
              o.material = new THREE.MeshBasicMaterial({
                vertexColors: !!o.geometry.getAttribute("color"),
                color: o.geometry.getAttribute("color") ? 0xffffff : 0xbfe3ff,
                side: THREE.DoubleSide,
              });
              count += o.geometry.index ? o.geometry.index.count / 3 : o.geometry.attributes.position.count / 3;
            }
          });
          resolve({ object: gltf.scene, points: pts, box, count });
        },
        (xhr) => { if (xhr.lengthComputable) setStatus(`cargando mapa... ${(xhr.loaded / xhr.total * 100).toFixed(0)}%`); },
        reject
      );
    }
  });
}

const UNITS = { malla: "triángulos", malla_f: "triángulos", estructura: "triángulos", splat: "gaussianas", splat_f: "gaussianas" };
let currentMap = null;
async function loadMap(meta) {
  if (live.running) { setStatus("hay una sesión en vivo en curso: detenela para cargar otro mapa"); return; }
  currentMap = meta;
  setStatus(`cargando ${meta.file} (${meta.size_mb} MB)...`);
  clearScene();
  camPositions = [];
  cameraPoses = [];
  document.getElementById("cam-slider").max = 0;
  document.getElementById("cam-label").textContent = "sin cámaras cargadas";

  try {
    const { object, points, box, count } = meta.type.startsWith("splat")
      ? await loadSplat(meta) : await loadCloud(`/data/${meta.file}`, meta.ext);
    loadedGroup = new THREE.Group();
    loadedGroup.add(object);
    pointObjects = points;

    if (meta.cameras && document.getElementById("show-cams").checked !== null) {
      const camRes = await fetch(`/data/${meta.cameras}`);
      if (camRes.ok) {
        const camData = await camRes.json();
        cameraPoses = camData.cameras;
        const { group, positions } = buildCameraVisuals(cameraPoses);
        group.visible = document.getElementById("show-cams").checked;
        loadedGroup.add(group);
        camMarker = new THREE.Mesh(
          new THREE.SphereGeometry(box ? box.getSize(new THREE.Vector3()).length() * 0.006 || 0.02 : 0.02, 12, 12),
          new THREE.MeshBasicMaterial({ color: 0xff3d9e })
        );
        camMarker.visible = !fps.active;
        loadedGroup.add(camMarker);
        camPositions = positions;
        document.getElementById("cam-slider").max = Math.max(0, positions.length - 1);
        document.getElementById("cam-label").textContent = `cámara 0 / ${positions.length - 1}`;
        placeCamMarker(0);
      }
    }
    scene.add(loadedGroup);
    lastBox = box; lastDown = worldDownFrom(cameraPoses);
    if (box) frameScene(normalizeGroup(loadedGroup, box, lastDown));
    setStatus(`${meta.file}\n${count.toLocaleString("es")} ${UNITS[meta.type] || "puntos"} · ${meta.size_mb} MB` +
      (cameraPoses.length ? ` · ${cameraPoses.length} cámaras` : ""));
    window.dispatchEvent(new CustomEvent("map-loaded", { detail: meta }));
  } catch (err) {
    console.error(err);
    setStatus(`error cargando ${meta.file}: ${err.message || err}`);
  }
}

function placeCamMarker(i) {
  if (!camMarker || !camPositions[i]) return;
  camMarker.position.copy(camPositions[i]);   // el marcador vive dentro del grupo
  document.getElementById("cam-label").textContent = `cámara ${i} / ${camPositions.length - 1}`;
}

// "Ir a" una cámara: teleporta la vista actual (órbita o 1ª persona) a esa pose.
function teleportTo(i) {
  if (!cameraPoses[i]) return;
  const c2w = cameraPoses[i].c2w;
  // el grupo está normalizado: pasar la pose de local a mundo
  const pos = loadedGroup ? loadedGroup.localToWorld(transformPoint(c2w, 0, 0, 0))
                          : transformPoint(c2w, 0, 0, 0);
  const tip = loadedGroup ? loadedGroup.localToWorld(transformPoint(c2w, 0, 0, 1))
                          : transformPoint(c2w, 0, 0, 1);
  const fwd = tip.sub(pos).normalize();
  camera.position.copy(pos);
  if (fps.active) {
    lookFrom(fwd);
    applyFpsLook();
  } else {
    orbit.target.copy(pos).addScaledVector(fwd, 3);
  }
  placeCamMarker(i);
}

document.getElementById("cam-slider").addEventListener("input", (e) => teleportTo(parseInt(e.target.value)));
document.getElementById("cam-prev").onclick = () => {
  const s = document.getElementById("cam-slider");
  s.value = Math.max(0, parseInt(s.value) - 1);
  teleportTo(parseInt(s.value));
};
document.getElementById("cam-next").onclick = () => {
  const s = document.getElementById("cam-slider");
  s.value = Math.min(parseInt(s.max), parseInt(s.value) + 1);
  teleportTo(parseInt(s.value));
};

document.getElementById("point-size").addEventListener("input", (e) => {
  const v = parseFloat(e.target.value);
  for (const p of pointObjects) p.material.size = v;
});
document.getElementById("show-axes").addEventListener("change", (e) => { axesHelper.visible = e.target.checked; });
document.getElementById("flip-v").addEventListener("change", (e) => {
  flipExtra = e.target.checked;
  const g = (live.framed && live.group) ? live.group : loadedGroup;
  if (g && lastBox) frameScene(normalizeGroup(g, lastBox, lastDown));
});
document.getElementById("show-cams").addEventListener("change", (e) => {
  if (loadedGroup) {
    loadedGroup.children.forEach((c) => { if (c.isGroup) c.visible = e.target.checked; });
  }
});
// ---------------------------------------------------------------------------
// Mapeo en vivo por WebSocket
// El servidor empuja un mensaje binario por frame reconstruido; acá se agregan
// los puntos a un buffer preasignado (sin recrear la geometría) y se extiende
// la trayectoria. La cámara del visor no se mueve sola: se puede seguir
// navegando en órbita o en primera persona mientras el mapa crece.
// ---------------------------------------------------------------------------
const LIVE_MAX_POINTS = 6_000_000;
const live = {
  ws: null, group: null, pos: null, col: null, geo: null, points: null,
  count: 0, frames: 0, path: [], pathGeo: null, pathLine: null, framed: false, sticky: false,
  down: new THREE.Vector3(0, 0, 0),
};

function liveEnsureScene() {
  if (live.group) return;
  live.group = new THREE.Group();
  live.pos = new Float32Array(LIVE_MAX_POINTS * 3);
  live.col = new Uint8Array(LIVE_MAX_POINTS * 3);
  live.geo = new THREE.BufferGeometry();
  const pa = new THREE.BufferAttribute(live.pos, 3);
  const ca = new THREE.BufferAttribute(live.col, 3, true);
  pa.setUsage(THREE.DynamicDrawUsage);
  ca.setUsage(THREE.DynamicDrawUsage);
  live.geo.setAttribute("position", pa);
  live.geo.setAttribute("color", ca);
  live.geo.setDrawRange(0, 0);
  const mat = new THREE.PointsMaterial({ size: 0.01, vertexColors: true, sizeAttenuation: true });
  live.points = new THREE.Points(live.geo, mat);
  live.points.frustumCulled = false;      // la caja crece a cada frame
  live.group.add(live.points);

  live.pathGeo = new THREE.BufferGeometry();
  live.pathGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(20000 * 3), 3));
  live.pathGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(20000 * 3), 3));
  live.pathGeo.setDrawRange(0, 0);
  live.pathLine = new THREE.Line(live.pathGeo, new THREE.LineBasicMaterial({ vertexColors: true }));
  live.pathLine.frustumCulled = false;
  live.group.add(live.pathLine);
  scene.add(live.group);
}

function liveReset() {
  if (!live.group) return;
  live.count = 0; live.frames = 0; live.path = []; live.framed = false; live.sticky = false;
  live.down.set(0, 0, 0);
  live.geo.setDrawRange(0, 0);
  live.pathGeo.setDrawRange(0, 0);
  document.getElementById("live-points").textContent = "0";
}

// Vista del video real que entra al modelo. Llega como mensaje binario con
// frame_idx = 0xFFFFFFFF (ningún frame real usa ese índice).
const PREVIEW_TAG = 0xFFFFFFFF;
let camViewUrl = null;
function liveOnPreview(buf) {
  const view = document.getElementById("cam-view");
  view.style.display = view.classList.contains("full") ? "flex" : "block";
  const blob = new Blob([new Uint8Array(buf, 4)], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  const img = document.getElementById("cam-view-img");
  img.onload = () => { if (camViewUrl) URL.revokeObjectURL(camViewUrl); camViewUrl = url; };
  img.src = url;
}

function camViewSwap() {
  const view = document.getElementById("cam-view");
  const full = !view.classList.contains("full");
  view.classList.toggle("full", full);
  document.body.classList.toggle("cam-full", full);
  view.style.display = full ? "flex" : "block";
  document.getElementById("cam-view-swap").textContent = full ? "volver al render" : "pantalla completa";
  onResize();                      // el canvas cambia de tamaño en los dos sentidos
}

function liveOnFrame(buf) {
  liveEnsureScene();
  const dv = new DataView(buf);
  const frameIdx = dv.getUint32(0, true);
  const n = dv.getUint32(4, true);
  const c2w = [];
  for (let i = 0; i < 16; i++) c2w.push(dv.getFloat32(8 + i * 4, true));
  const posOff = 72;
  const colOff = posOff + n * 12;
  const src = new Float32Array(buf, posOff, n * 3);
  const srcCol = new Uint8Array(buf, colOff, n * 3);

  const room = LIVE_MAX_POINTS - live.count;
  const take = Math.min(n, room);
  if (take > 0) {
    live.pos.set(src.subarray(0, take * 3), live.count * 3);
    live.col.set(srcCol.subarray(0, take * 3), live.count * 3);
    const pa = live.geo.attributes.position, ca = live.geo.attributes.color;
    pa.addUpdateRange(live.count * 3, take * 3); pa.needsUpdate = true;
    ca.addUpdateRange(live.count * 3, take * 3); ca.needsUpdate = true;
    live.count += take;
    live.geo.setDrawRange(0, live.count);
  }

  // c2w viene row-major: la traslación son los elementos 3, 7, 11
  // y la columna 1 (elementos 1, 5, 9) es el eje Y de la cámara = "abajo"
  live.down.add(new THREE.Vector3(c2w[1], c2w[5], c2w[9]));
  const p = new THREE.Vector3(c2w[3], c2w[7], c2w[11]);
  live.path.push(p);
  const k = live.path.length - 1;
  if (k < 20000) {
    const pp = live.pathGeo.attributes.position, pc = live.pathGeo.attributes.color;
    pp.setXYZ(k, p.x, p.y, p.z);
    for (let i = 0; i <= k; i++) {           // el degradado se reescala al avanzar
      const c = viridis(k ? i / k : 0);
      pc.setXYZ(i, c[0], c[1], c[2]);
    }
    pp.needsUpdate = true; pc.needsUpdate = true;
    live.pathGeo.setDrawRange(0, live.path.length);
  }

  live.frames++;
  document.getElementById("live-points").textContent = live.count.toLocaleString("es");

  // encuadrar una sola vez, cuando ya hay geometría suficiente para estimar la escala
  if (!live.framed && live.count > 20000) {
    live.framed = true;
    live.geo.computeBoundingBox();
    clearScene();                     // la sesión en vivo reemplaza lo que hubiera cargado
    lastBox = live.geo.boundingBox.clone();
    lastDown = live.down.clone().normalize();
    frameScene(normalizeGroup(live.group, lastBox, lastDown));
    live.points.material.size = parseFloat(document.getElementById("point-size").value);
  }
}

function liveSetRunning(on) {
  live.running = on;
  document.getElementById("live-dot").classList.toggle("on", on);
  document.getElementById("live-start").disabled = on;
  document.getElementById("live-stop").disabled = !on;
}

function liveConnect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  live.ws = ws;
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {
      const tag = new DataView(ev.data).getUint32(0, true);
      if (tag === PREVIEW_TAG) liveOnPreview(ev.data); else liveOnFrame(ev.data);
      return;
    }
    const m = JSON.parse(ev.data);
    if (m.type === "status") {
      liveSetRunning(!!m.running);
      // "fuente agotada"/"error" son el desenlace de la sesión: no dejar que el
      // siguiente estado ("detenido") los pise antes de que se puedan leer.
      if (!live.sticky) document.getElementById("live-msg").textContent = m.msg || "";
      document.getElementById("live-frames").textContent = m.frames ?? 0;
      document.getElementById("live-rate").textContent = m.fps ? m.fps.toFixed(2) : "—";
      document.getElementById("live-vram").textContent = m.vram_mb ? `${m.vram_mb} MB` : "—";
      const c = m.context;
      document.getElementById("live-ctx-row").style.display = c ? "" : "none";
      if (c) document.getElementById("live-ctx-stat").textContent =
        `${c.sent}/${c.read} enviados` + (c.synth ? `, ${c.synth} sintéticos` : "");
    } else if (m.type === "error") {
      live.sticky = true;
      document.getElementById("live-msg").textContent = "error: " + m.msg;
      liveSetRunning(false);
    } else if (m.type === "done") {
      live.sticky = true;
      document.getElementById("live-msg").textContent = `fuente agotada (${m.frames} frames)`;
    } else if (m.type === "saved" || m.type === "job") {
      window.dispatchEvent(new CustomEvent("server-" + m.type, { detail: m }));
      if (m.type === "saved") {
        live.sticky = true;
        document.getElementById("live-msg").textContent =
          `sesión guardada (${m.frames} frames) en el explorador, como "sin guardar"`;
      }
    } else if (m.type === "stopped") {
      liveSetRunning(false);
      if (document.body.classList.contains("cam-full")) camViewSwap();
      document.getElementById("cam-view").style.display = "none";
    }
  };
  ws.onclose = () => { liveSetRunning(false); setTimeout(liveConnect, 3000); };
  ws.onerror = () => {};
}

document.getElementById("live-start").onclick = async () => {
  liveReset(); liveEnsureScene();
  const body = {
    source: document.getElementById("live-source").value,
    device: parseInt(document.getElementById("live-device").value || "0"),
    path: document.getElementById("live-path").value.trim(),
    fps: parseFloat(document.getElementById("live-fps").value),
    max_frames: parseInt(document.getElementById("live-max").value) || 0,
    points_per_frame: parseInt(document.getElementById("live-pts").value) || 6000,
    conf_percentile: parseFloat(document.getElementById("live-conf").value),
    context: document.getElementById("live-ctx").checked,
    context_synth: document.getElementById("live-ctx").checked && document.getElementById("live-synth").checked,
  };
  if (body.context) body.fps = 0;      // con el analizador, el ritmo lo decide el movimiento
  document.getElementById("live-msg").textContent = "iniciando...";
  try {
    const r = await fetch("/api/live/start", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const j = await r.json();
    if (!j.ok) document.getElementById("live-msg").textContent = j.msg || "no se pudo iniciar";
  } catch (e) {
    document.getElementById("live-msg").textContent = "sin servidor de streaming: " + e.message;
  }
};
document.getElementById("live-stop").onclick = async () => {
  document.getElementById("live-msg").textContent = "deteniendo...";
  try { await fetch("/api/live/stop", { method: "POST" }); } catch (e) { /* sin servidor */ }
};
async function loadDevices() {
  const sel = document.getElementById("live-device");
  sel.innerHTML = "";
  try {
    const r = await fetch("/api/live/devices");
    const { devices } = await r.json();
    const usable = devices.filter((d) => d.usable);
    for (const d of (usable.length ? usable : devices)) {
      const o = document.createElement("option");
      o.value = d.index;
      o.textContent = d.label + (d.usable ? "" : " — no entrega imagen");
      o.disabled = !d.usable;
      sel.appendChild(o);
    }
    if (!devices.length) {
      const o = document.createElement("option");
      o.textContent = "no se detectó ninguna cámara"; o.disabled = true;
      sel.appendChild(o);
    }
  } catch (e) { /* sin servidor */ }
}
document.getElementById("live-source").onchange = (e) => {
  const cam = e.target.value === "webcam";
  document.getElementById("live-path").style.display = cam ? "none" : "";
  document.getElementById("live-device").style.display = cam ? "" : "none";
  if (cam) loadDevices();
};
document.getElementById("cam-view-swap").onclick = camViewSwap;
liveConnect();

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------
// Enganche de depuración: permite inspeccionar cámara y estado desde la consola
// del navegador o desde una prueba automatizada.
window.__viewer = { THREE, camera, fps, scene, getPad: () => getPad(), loadMap, setStatus,
                    current: () => currentMap };
window.dispatchEvent(new Event("viewer-ready"));

const clock = new THREE.Clock();
let fpsAccum = 0, fpsFrames = 0;
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);

  if (fps.active) {
    applyFpsLook();   // orientar primero: fpsMove toma los ejes de la cámara ya orientada
    fpsMove(dt);
  } else {
    orbit.update();
  }
  renderer.render(scene, camera);

  fpsAccum += dt; fpsFrames++;
  if (fpsAccum >= 0.5) {
    document.getElementById("fps-counter").textContent = (fpsFrames / fpsAccum).toFixed(0);
    fpsAccum = 0; fpsFrames = 0;
  }
}
animate();

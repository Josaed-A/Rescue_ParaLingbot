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

// Mando recordado: el último mando con el que se navegó queda guardado en este navegador y se
// prefiere sobre cualquier otro dispositivo que el navegador exponga como "gamepad" (en Linux
// Firefox también muestra acelerómetros y otros dispositivos de entrada).
const PAD_KEY = "visor.gamepad.preferido";
let padPreferred = null;
try { padPreferred = localStorage.getItem(PAD_KEY); } catch (e) { /* sin almacenamiento */ }
const padTrig = {};            // ejes de gatillo que ya se movieron (antes de moverse valen 0, no -1)

function getPad() {
  const pads = (navigator.getGamepads ? Array.from(navigator.getGamepads()) : []).filter(
    (p) => p && p.connected !== false && p.axes.length >= 4 && p.buttons.length >= 8);
  if (!pads.length) return null;
  return pads.find((p) => p.id === padPreferred) || pads[0];
}

// Estado normalizado del mando. Dos mapeos:
//  - "standard" (W3C; Chrome y Firefox con mandos conocidos): ejes LX LY RX RY, gatillos = botones 6/7
//  - el de xpad/evdev en Linux sin remapeo: ejes LX LY LT RX RY RT HATX HATY (gatillos de -1 a 1),
//    botones A B X Y LB RB Back Start Guide LS RS
function readPad() {
  const pad = getPad();
  if (!pad) return null;
  const ax = pad.axes, bt = pad.buttons, b = (k) => !!(bt[k] && bt[k].pressed);
  if (pad.mapping === "standard") {
    return { pad, lx: axis(ax[0] || 0), ly: axis(ax[1] || 0), rx: axis(ax[2] || 0), ry: axis(ax[3] || 0),
             lt: bt[6] ? bt[6].value : 0, rt: bt[7] ? bt[7].value : 0,
             a: b(0), bb: b(1), x: b(2), y: b(3), lb: b(4), rb: b(5), back: b(8), start: b(9),
             any: bt.some((q) => q && q.pressed) };
  }
  const trig = (k) => {
    const v = ax[k] || 0;
    if (v !== 0) padTrig[k] = true;
    return padTrig[k] ? Math.max(0, (v + 1) / 2) : 0;
  };
  return { pad, lx: axis(ax[0] || 0), ly: axis(ax[1] || 0), rx: axis(ax[3] || 0), ry: axis(ax[4] || 0),
           lt: ax.length >= 6 ? trig(2) : 0, rt: ax.length >= 6 ? trig(5) : 0,
           a: b(0), bb: b(1), x: b(2), y: b(3), lb: b(4), rb: b(5), back: b(6), start: b(7),
           any: bt.some((q) => q && q.pressed) };
}

function rememberPad(pad) {
  if (!pad || pad.id === padPreferred) return;
  padPreferred = pad.id;
  try { localStorage.setItem(PAD_KEY, pad.id); } catch (e) { /* sin almacenamiento */ }
}

function gamepadSprint() {
  const st = readPad();
  return !!(st && st.a);
}

window.addEventListener("gamepadconnected", (e) => {
  const st = document.getElementById("gamepad-status");
  if (st) st.textContent = e.gamepad.id.slice(0, 22) + (e.gamepad.id === padPreferred ? " (recordado)" : "");
});
window.addEventListener("gamepaddisconnected", () => {
  const st = document.getElementById("gamepad-status");
  if (st) st.textContent = "desconectado";
});

// Se llama en cada cuadro, en los dos modos: estado del panel, Start alterna órbita / 1ª persona,
// y en órbita el mando también navega (stick izq. gira alrededor del objetivo, stick der. desplaza,
// gatillos acercan/alejan).
let lastStart = false, lastBack = false;
const _sph = new THREE.Spherical(), _off = new THREE.Vector3(), _pan = new THREE.Vector3();
function pollGamepad(dt) {
  const st = readPad();
  const label = document.getElementById("gamepad-status");
  if (!st) { if (label) label.textContent = navigator.getGamepads ? "no detectado (pulsa un botón)" : "no soportado"; return null; }
  if (st.any) rememberPad(st.pad);
  if (label) label.textContent = st.pad.id.split("(")[0].replace(/^[0-9a-f]{4}-[0-9a-f]{4}-/i, "").trim().slice(0, 26) +
    (st.pad.id === padPreferred ? " ✓" : "");
  if (st.start && !lastStart) setMode(fps.active ? "orbit" : "fps");
  lastStart = st.start;
  if (st.back && !lastBack && typeof lastBox !== "undefined" && lastBox) {   // View/Back: volver a encuadrar
    const g = (live.framed && live.group) ? live.group : loadedGroup;
    if (g) frameScene(normalizeGroup(g, lastBox, lastDown));
  }
  lastBack = st.back;
  if (fps.active) return st;
  const moving = st.lx || st.ly || st.rx || st.ry || st.lt > 0.05 || st.rt > 0.05;
  if (!moving) return st;
  if (live.follow && live.running) { live.follow = false; document.getElementById("live-follow").checked = false; }
  _off.copy(camera.position).sub(orbit.target);
  _sph.setFromVector3(_off);
  _sph.theta -= st.lx * 1.8 * dt;
  _sph.phi = THREE.MathUtils.clamp(_sph.phi + st.ly * 1.4 * dt, 0.05, Math.PI - 0.05);
  _sph.radius = THREE.MathUtils.clamp(_sph.radius * Math.exp((st.lt - st.rt) * 1.2 * dt), 0.05, 500);
  _off.setFromSpherical(_sph);
  // desplazamiento en el plano de la vista, proporcional a la distancia
  camera.updateMatrixWorld();
  const r = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
  const u = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1);
  _pan.set(0, 0, 0).addScaledVector(r, st.rx).addScaledVector(u, -st.ry).multiplyScalar(_sph.radius * 0.8 * dt);
  orbit.target.add(_pan);
  camera.position.copy(orbit.target).add(_off);
  return st;
}

function applyGamepad(dt, forward, right, speed) {
  const st = readPad();
  if (!st || !fps.active) return;
  // LB / RB: más lento / más rápido mientras se mantienen
  const k = st.rb ? 2.5 : (st.lb ? 0.35 : 1);
  const move = new THREE.Vector3();
  move.addScaledVector(forward, -st.ly);
  move.addScaledVector(right, st.lx);
  move.y += st.rt - st.lt;
  if (move.lengthSq() > 0) camera.position.addScaledVector(move.normalize(), k * speed * dt * Math.hypot(st.lx, st.ly, st.rt - st.lt || 0.001));
  fps.yaw -= st.rx * 2.2 * dt;
  fps.pitch -= st.ry * 1.8 * dt;
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
        // etapa 16: trayectorias BASIC / STELLA / HYBRID guardadas con la sesión
        for (const [mode, pts] of Object.entries(camData.trayectorias || {})) {
          if (mode === "basic" || !pts.length) continue;      // BASIC ya es la trayectoria de cámaras
          const g = new THREE.BufferGeometry();
          g.setAttribute("position", new THREE.Float32BufferAttribute(pts.flat(), 3));
          const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: TRACK_COLORS[mode] || 0xffffff, depthTest: false }));
          line.renderOrder = 5;
          group.add(line);
        }
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
  if (live.marker && live.framed) live.marker.visible = e.target.checked;
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
  // cámara actual, suavizada: el modelo entrega una pose cada ~0.5 s y el marcador y la
  // vista se deslizan hacia ella en cada cuadro, en vez de saltar
  marker: null, tgtPos: new THREE.Vector3(), tgtQuat: new THREE.Quaternion(), hasPose: false,
  follow: true,
  // etapa 16: rango de puntos de cada frame (para re-registrarlo con `repose`), trayectorias de los
  // tres modos, keyframes de Stella y marcas de corrección
  ranges: {}, tracks: {}, kf: null, corr: null, nCorr: 0,
};
const TRACK_COLORS = { basic: 0xbbbbbb, stella: 0xff9933, hybrid: 0x33dd66 };
const TRACK_MAX = 20000;

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

  // marcador de la cámara actual: pirámide (mira a +Z, convención OpenCV) con un "techo"
  // para distinguir arriba de abajo. Tamaño unitario: se escala al encuadrar.
  const v = [0, 0, 0], a = [-0.6, -0.45, 1], b = [0.6, -0.45, 1], c = [0.6, 0.45, 1], d = [-0.6, 0.45, 1];
  const segs = [v, a, v, b, v, c, v, d, a, b, b, c, c, d, d, a, a, [0, -0.8, 1], [0, -0.8, 1], b];
  const mg = new THREE.BufferGeometry();
  mg.setAttribute("position", new THREE.Float32BufferAttribute(segs.flat(), 3));
  live.marker = new THREE.LineSegments(mg, new THREE.LineBasicMaterial({ color: 0xff3df0, depthTest: false }));
  live.marker.renderOrder = 10;
  live.marker.visible = false;
  live.group.add(live.marker);
  // etapa 16: trayectorias BASIC / STELLA / HYBRID (mismo mundo que la nube), keyframes y correcciones
  for (const m of Object.keys(TRACK_COLORS)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(TRACK_MAX * 3), 3));
    g.setDrawRange(0, 0);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: TRACK_COLORS[m], depthTest: false }));
    line.frustumCulled = false; line.renderOrder = 5;
    live.tracks[m] = { geo: g, line, n: 0 };
    live.group.add(line);
  }
  const mkPts = (n, color, size) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setDrawRange(0, 0);
    const p = new THREE.Points(g, new THREE.PointsMaterial({ color, size, sizeAttenuation: false, depthTest: false }));
    p.frustumCulled = false; p.renderOrder = 6;
    live.group.add(p);
    return { geo: g, pts: p, n: 0, max: n };
  };
  live.kf = mkPts(5000, 0x3399ff, 6);
  live.corr = mkPts(2000, 0xff2222, 12);
  scene.add(live.group);
}

// etapa 16: una posición nueva en la trayectoria de un modo
function liveTrackPush(mode, p) {
  const t = live.tracks[mode];
  if (!t || t.n >= TRACK_MAX || !p) return;
  t.geo.attributes.position.setXYZ(t.n, p[0], p[1], p[2]);
  t.n += 1;
  t.geo.attributes.position.needsUpdate = true;
  t.geo.setDrawRange(0, t.n);
}

// etapa 12/16: re-registrar los puntos de un frame con la corrección `delta` (4x4 fila a fila)
function liveRepose(fidx, d) {
  const r = live.ranges[fidx];
  if (!r) return;
  const [start, n] = r;
  const P = live.pos;
  for (let i = start; i < start + n; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    P[3 * i] = d[0] * x + d[1] * y + d[2] * z + d[3];
    P[3 * i + 1] = d[4] * x + d[5] * y + d[6] * z + d[7];
    P[3 * i + 2] = d[8] * x + d[9] * y + d[10] * z + d[11];
  }
  const pa = live.geo.attributes.position;
  pa.addUpdateRange(start * 3, n * 3); pa.needsUpdate = true;
}

function liveOnTracking(m) {
  const box = document.getElementById("trk-box");
  if (m.referencia || m.stella_status) box.style.display = "";
  if (m.referencia) {
    document.getElementById("trk-mode").textContent = m.referencia.modo.toUpperCase();
    const src = document.getElementById("trk-src");
    src.textContent = m.referencia.usado + (m.referencia.motivo ? ` (${m.referencia.motivo})` : "");
    src.style.color = m.referencia.usado === "STELLA" ? "#ff9933" : "#bbbbbb";
    document.getElementById("trk-rate").textContent = `${m.referencia.ritmo_stella_hz} poses/s`;
  }
  if (m.stella_status) document.getElementById("trk-stella").textContent = m.stella_status;
  document.getElementById("trk-stamp").textContent = (m.stamp ?? 0).toFixed(3) + " s";
  document.getElementById("trk-lat").textContent = m.latencia_s != null ? `${(m.latencia_s * 1000).toFixed(0)} ms` : "—";
  if (m.trayectorias) {
    liveEnsureScene();
    for (const k of Object.keys(m.trayectorias)) liveTrackPush(k, m.trayectorias[k]);
  }
}

function liveOnKeyframes(m) {
  liveEnsureScene();
  const g = live.kf.geo.attributes.position;
  const n = Math.min(m.positions.length, live.kf.max);
  for (let i = 0; i < n; i++) g.setXYZ(i, m.positions[i][0], m.positions[i][1], m.positions[i][2]);
  g.needsUpdate = true; live.kf.geo.setDrawRange(0, n);
}

function liveOnCorrection(m) {
  live.nCorr += 1;
  document.getElementById("trk-corr").textContent =
    `${live.nCorr} (último: ${m.keyframes_movidos} kf, ${m.frames_re_registrados} frames)`;
  if (live.hasPose && live.corr.n < live.corr.max) {
    const g = live.corr.geo.attributes.position;
    g.setXYZ(live.corr.n, live.tgtPos.x, live.tgtPos.y, live.tgtPos.z);
    live.corr.n += 1; g.needsUpdate = true; live.corr.geo.setDrawRange(0, live.corr.n);
  }
}

for (const [id, key] of [["trk-show-basic", "basic"], ["trk-show-stella", "stella"], ["trk-show-hybrid", "hybrid"]]) {
  document.getElementById(id).addEventListener("change", (e) => { if (live.tracks[key]) live.tracks[key].line.visible = e.target.checked; });
}
document.getElementById("trk-show-kf").addEventListener("change", (e) => { if (live.kf) live.kf.pts.visible = e.target.checked; });

function liveReset() {
  if (!live.group) return;
  live.count = 0; live.frames = 0; live.path = []; live.framed = false; live.sticky = false;
  live.framedDiag = 0; live.nReframe = 0;
  live.hasPose = false; live.marker.visible = false;
  live.follow = document.getElementById("live-follow").checked;
  live.down.set(0, 0, 0);
  live.geo.setDrawRange(0, 0);
  live.pathGeo.setDrawRange(0, 0);
  live.ranges = {}; live.nCorr = 0;
  for (const t of Object.values(live.tracks)) { t.n = 0; t.geo.setDrawRange(0, 0); }
  if (live.kf) { live.kf.geo.setDrawRange(0, 0); live.corr.n = 0; live.corr.geo.setDrawRange(0, 0); }
  document.getElementById("trk-corr").textContent = "0";
  document.getElementById("live-points").textContent = "0";
}

// Vista del video real que entra al modelo. Llega como mensaje binario con
// frame_idx = 0xFFFFFFFF (ningún frame real usa ese índice).
const PREVIEW_TAG = 0xFFFFFFFF;
let camViewUrl = null, camViewBusy = false, camViewPending = null;
function liveOnPreview(buf) {
  // Llega a ~15 fps. Si la imagen anterior todavía se está decodificando, se guarda
  // solo la más nueva: nunca se arma una cola (que es lo que hace que se vea atrasado).
  if (camViewBusy) { camViewPending = buf; return; }
  const view = document.getElementById("cam-view");
  view.style.display = view.classList.contains("full") ? "flex" : "block";
  const blob = new Blob([new Uint8Array(buf, 4)], { type: "image/jpeg" });
  const url = URL.createObjectURL(blob);
  const img = document.getElementById("cam-view-img");
  camViewBusy = true;
  img.onload = img.onerror = () => {
    if (camViewUrl) URL.revokeObjectURL(camViewUrl);
    camViewUrl = url; camViewBusy = false;
    if (camViewPending) { const b = camViewPending; camViewPending = null; liveOnPreview(b); }
  };
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

// Caja robusta de la nube en vivo (percentiles 2-98% de una muestra): unos pocos puntos lejanos o
// frames basura (cámara tapada, desenfoque) no deben decidir la escala de todo el mapa.
function liveRobustBox() {
  const n = live.count;
  if (n < 1000) return null;
  const step = Math.max(1, Math.floor(n / 40000));
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < n; i += step) { xs.push(live.pos[3 * i]); ys.push(live.pos[3 * i + 1]); zs.push(live.pos[3 * i + 2]); }
  const q = (a) => { a.sort((u, v) => u - v); return [a[Math.floor(a.length * 0.02)], a[Math.floor(a.length * 0.98)]]; };
  const [x0, x1] = q(xs), [y0, y1] = q(ys), [z0, z1] = q(zs);
  // la trayectoria entra completa: la cámara recorrió eso aunque la nube sea rala
  const box = new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));
  for (const p of live.path) box.expandByPoint(p);
  return box;
}

// Vuelve a normalizar el grupo en vivo con una caja nueva. No toca los controles (el tamaño de punto
// que haya elegido el usuario se respeta); si no se está siguiendo la cámara, reubica la vista.
function liveReframe(box, why, ratio) {
  // el tamaño de punto de three.js no escala con el grupo: se ajusta en la misma proporción para que
  // cada frame se vea igual de denso que antes del re-encuadre (dentro del rango del control)
  if (ratio) {
    const ps = document.getElementById("point-size");
    const v = Math.min(parseFloat(ps.max), Math.max(parseFloat(ps.min), parseFloat(ps.value) / ratio));
    ps.value = v;
    live.points.material.size = parseFloat(ps.value);
  }
  lastBox = box.clone();
  lastDown = live.down.clone().normalize();
  const nb = normalizeGroup(live.group, lastBox, lastDown);
  const diag = lastBox.getSize(new THREE.Vector3()).length();
  live.framedDiag = diag;
  live.marker.scale.setScalar(Math.max(diag * 0.04, 1e-4));
  if (!live.follow) {
    const c = nb.getCenter(new THREE.Vector3()), r = Math.max(nb.getSize(new THREE.Vector3()).length() * 0.5, 0.1);
    camera.position.copy(c).add(new THREE.Vector3(r, r * 0.6, r));
    orbit.target.copy(c); orbit.update();
  }
  if (why) {
    live.nReframe = (live.nReframe || 0) + 1;
    live.lastReframe = why;
    const el = document.getElementById("live-msg");
    if (!live.sticky) el.textContent = `vista re-encuadrada (${why})`;
  }
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
    live.ranges[frameIdx] = [live.count, take];
    live.count += take;
    live.geo.setDrawRange(0, live.count);
  }

  // c2w viene row-major: la traslación son los elementos 3, 7, 11
  // y la columna 1 (elementos 1, 5, 9) es el eje Y de la cámara = "abajo"
  live.down.add(new THREE.Vector3(c2w[1], c2w[5], c2w[9]));
  const p = new THREE.Vector3(c2w[3], c2w[7], c2w[11]);
  const rot = new THREE.Matrix4().set(c2w[0], c2w[1], c2w[2], 0, c2w[4], c2w[5], c2w[6], 0,
                                      c2w[8], c2w[9], c2w[10], 0, 0, 0, 0, 1);
  live.tgtPos.copy(p); live.tgtQuat.setFromRotationMatrix(rot);
  if (!live.hasPose) {               // la primera pose se toma directa, sin deslizar
    live.hasPose = true;
    live.marker.position.copy(p); live.marker.quaternion.copy(live.tgtQuat);
  }
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
    clearScene();                     // la sesión en vivo reemplaza lo que hubiera cargado
    lastBox = liveRobustBox() || (live.geo.computeBoundingBox(), live.geo.boundingBox.clone());
    lastDown = live.down.clone().normalize();
    frameScene(normalizeGroup(live.group, lastBox, lastDown));
    live.points.material.size = parseFloat(document.getElementById("point-size").value);
    // tamaño del marcador: 4% de la diagonal de lo que había al encuadrar
    const diag = lastBox.getSize(new THREE.Vector3()).length();
    live.framedDiag = diag;
    live.marker.scale.setScalar(Math.max(diag * 0.04, 1e-4));
    live.marker.visible = document.getElementById("show-cams").checked;
  } else if (live.framed && live.frames % 8 === 0) {
    // el mapa puede crecer mucho (deriva del streaming, recorrido largo) o resultar mucho más chico
    // que lo que se encuadró con los primeros frames: si la escala cambió más de 2x, re-encuadrar
    const box = liveRobustBox();
    if (box) {
      const d = box.getSize(new THREE.Vector3()).length();
      const r = d / Math.max(live.framedDiag || d, 1e-9);
      if (r > 2 || r < 0.5) liveReframe(box, r > 2 ? `el mapa creció ${r.toFixed(1)}x` : `el mapa es ${(1 / r).toFixed(1)}x más chico`, r);
    }
  }
}

// Se llama en cada cuadro: desliza el marcador hacia la última pose y, si "seguir la
// cámara" está activo, lleva la vista detrás y arriba de él. Con amortiguación
// exponencial, así el resultado no depende de los fps del navegador.
const _fw = new THREE.Vector3(), _wp = new THREE.Vector3(), _want = new THREE.Vector3();
function liveSmooth(dt) {
  if (!live.marker || !live.hasPose) return;
  const k = 1 - Math.exp(-dt * 5);
  live.marker.position.lerp(live.tgtPos, k);
  live.marker.quaternion.slerp(live.tgtQuat, k);
  if (!live.follow || !live.framed || !live.running || fps.active) return;
  live.marker.updateMatrixWorld();
  live.marker.getWorldPosition(_wp);
  live.marker.getWorldDirection(_fw);          // eje +Z del marcador = hacia donde mira
  _fw.y = 0;
  if (_fw.lengthSq() < 1e-6) _fw.set(0, 0, 1);
  _fw.normalize();
  const dist = CANON_DIAG * 0.22;
  _want.copy(_wp).addScaledVector(_fw, -dist).add(new THREE.Vector3(0, dist * 0.55, 0));
  const kc = 1 - Math.exp(-dt * 2.5);
  camera.position.lerp(_want, kc);
  orbit.target.lerp(_wp.addScaledVector(_fw, dist * 0.3), kc);
}
// arrastrar la vista toma el control: se deja de seguir hasta volver a marcar la casilla
orbit.addEventListener("start", () => {
  if (live.running && live.follow) {
    live.follow = false;
    document.getElementById("live-follow").checked = false;
  }
});
document.getElementById("live-follow").addEventListener("change", (e) => { live.follow = e.target.checked; });

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
      if (!live.sticky) document.getElementById("live-msg").textContent = (m.msg || "") +
        (live.nReframe ? ` · vista re-encuadrada ${live.nReframe}x (${live.lastReframe})` : "");
      document.getElementById("live-frames").textContent = m.frames ?? 0;
      document.getElementById("live-rate").textContent = m.fps ? m.fps.toFixed(2) : "—";
      document.getElementById("live-vram").textContent = m.vram_mb ? `${m.vram_mb} MB` : "—";
      const c = m.context;
      document.getElementById("live-ctx-row").style.display = c ? "" : "none";
      if (c) document.getElementById("live-ctx-stat").textContent =
        `${c.sent}/${c.read} enviados` + (c.synth ? `, ${c.synth} sintéticos` : "");
    } else if (m.type === "tracking") {
      liveOnTracking(m);
      const el = document.getElementById("live-still");
      if (el && m.movimiento_px !== undefined) {
        el.textContent = m.estatico ? `quieta (${m.movimiento_px ?? 0} px) — la pose no avanza`
                                    : (m.movimiento_px == null ? "—" : `en movimiento (${m.movimiento_px} px)`);
        el.style.color = m.estatico ? "#ffb347" : "";
      }
    } else if (m.type === "repose") {
      liveRepose(m.frame_idx, m.delta);
    } else if (m.type === "keyframes") {
      liveOnKeyframes(m);
    } else if (m.type === "correccion") {
      liveOnCorrection(m);
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
    device: 0,
    rotation: parseInt(document.getElementById("live-rot").value || "0"),
    path: document.getElementById("live-path").value.trim(),
    fps: parseFloat(document.getElementById("live-fps").value),
    max_frames: parseInt(document.getElementById("live-max").value) || 0,
    points_per_frame: parseInt(document.getElementById("live-pts").value) || 6000,
    conf_percentile: parseFloat(document.getElementById("live-conf").value),
    context: document.getElementById("live-ctx").checked,
    context_synth: document.getElementById("live-ctx").checked && document.getElementById("live-synth").checked,
    context_synth_strength: parseFloat(document.getElementById("live-synth-strength").value),
  };
  if (body.source === "webcam") {
    // la lista mezcla webcams locales y cámaras de teléfonos conectados por adb
    const opt = document.getElementById("live-device").selectedOptions[0];
    const d = opt && opt.dataset.dev ? JSON.parse(opt.dataset.dev) : { kind: "webcam", index: 0 };
    if (d.kind === "android") {
      Object.assign(body, { source: "android", serial: d.serial, camera_id: d.camera_id,
                            device_label: d.model || d.label,
                            cam_size: document.getElementById("live-camres").value });
    } else if (d.kind === "ssh") {
      // cámara de otro equipo por SSH: la resolución pedida es la de la cámara remota
      Object.assign(body, { source: "ssh", host: d.host, device: d.device, cam_tipo: d.tipo,
                            device_label: d.model || d.label,
                            cam_size: document.getElementById("live-camres").value });
    } else {
      body.device = d.index;
    }
  }
  if (body.context) body.fps = 0;      // con el analizador, el ritmo lo decide el movimiento
  body.ros2 = document.getElementById("live-ros2").checked;
  if (body.ros2) body.tracking_mode = document.getElementById("live-trkmode").value;
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
(async () => {
  try {
    const st = await (await fetch("/api/live/status")).json();
    document.getElementById("live-ros2").checked = !!st.ros2_default;
  } catch (e) { /* sin servidor */ }
})();
async function loadDevices() {
  const sel = document.getElementById("live-device");
  sel.innerHTML = "";
  const msg = document.createElement("option");
  msg.textContent = "buscando cámaras (webcams, teléfonos por adb y equipos por SSH)..."; msg.disabled = true;
  sel.appendChild(msg);
  try {
    const r = await fetch("/api/live/devices");
    const { devices = [], remote = [], ssh = [] } = await r.json();
    sel.innerHTML = "";
    const add = (group, list, empty) => {
      const g = document.createElement("optgroup"); g.label = group;
      const usable = list.filter((d) => d.usable);
      for (const d of [...usable, ...list.filter((x) => !x.usable)]) {
        const o = document.createElement("option");
        o.textContent = d.label + (d.usable || / — /.test(d.label) ? "" : " — no entrega imagen");
        o.disabled = !d.usable;
        o.dataset.dev = JSON.stringify(d);
        g.appendChild(o);
      }
      if (!list.length) {
        const o = document.createElement("option"); o.textContent = empty; o.disabled = true; g.appendChild(o);
      }
      sel.appendChild(g);
    };
    add("Cámaras remotas (celular por adb)", remote, "ningún teléfono conectado por adb");
    add("Cámaras de otros equipos (SSH: robot, vigia...)", ssh, "ningún equipo SSH con cámara alcanzable");
    add("Cámaras de este equipo", devices, "ninguna cámara de video");
    const first = [...sel.options].find((o) => !o.disabled);
    if (first) first.selected = true;
    liveDeviceChanged();
  } catch (e) {
    sel.innerHTML = "<option disabled>sin servidor</option>";
  }
}
function liveDeviceChanged() {
  const opt = document.getElementById("live-device").selectedOptions[0];
  const d = opt && opt.dataset.dev ? JSON.parse(opt.dataset.dev) : {};
  document.getElementById("live-camres").style.display = (d.kind === "android" || d.kind === "ssh") ? "" : "none";
  // la cámara del teléfono llega en la orientación del sensor (horizontal): sostenido en
  // vertical hay que girarla 90°, porque el modelo necesita la gravedad hacia abajo
  const rot = document.getElementById("live-rot");
  if (!rot.dataset.touched) rot.value = d.kind === "android" ? "90" : "0";
}
document.getElementById("live-rot").onchange = (e) => { e.target.dataset.touched = "1"; };
document.getElementById("live-device").onchange = liveDeviceChanged;
document.getElementById("live-device-refresh").onclick = loadDevices;
document.getElementById("live-source").onchange = (e) => {
  const v = e.target.value, cam = v === "webcam";
  const path = document.getElementById("live-path");
  path.style.display = cam ? "none" : "";
  path.placeholder = v === "url" ? "rtsp://... o http://.../video" : "ruta de la carpeta o del video";
  if (v === "url" && !/^(rtsp|https?):/.test(path.value)) path.value = "";
  document.getElementById("live-device-row").style.display = cam ? "flex" : "none";
  document.getElementById("live-rot-row").style.display = (cam || v === "url") ? "flex" : "none";
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
  pollGamepad(dt);

  if (fps.active) {
    applyFpsLook();   // orientar primero: fpsMove toma los ejes de la cámara ya orientada
    fpsMove(dt);
  } else {
    liveSmooth(dt);
    orbit.update();
  }
  if (fps.active) liveSmooth(dt);
  renderer.render(scene, camera);

  fpsAccum += dt; fpsFrames++;
  if (fpsAccum >= 0.5) {
    document.getElementById("fps-counter").textContent = (fpsFrames / fpsAccum).toFixed(0);
    fpsAccum = 0; fpsFrames = 0;
  }
}
animate();
// gancho de depuración para pruebas automáticas (p. ej. simular el mando en un navegador sin pantalla)
window.__visor = { camera, orbit, fps, readPad };

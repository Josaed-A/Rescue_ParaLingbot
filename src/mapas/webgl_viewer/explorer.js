// Explorador de pruebas: agrupa las pruebas por zona, categoría, carpeta o fecha,
// muestra sus mapas por tipo (un clic los carga), sus archivos, y permite editar
// título / zona / categorías / notas, guardar o descartar sesiones en vivo y
// construirles los mapas que les falten. Todo lo guarda el servidor en el info.json
// de cada prueba (ver src/mapas/webgl_viewer/catalog.py).

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const slug = (t) => String(t).toLowerCase().normalize("NFD").replace(/[^\w]+/g, "-");
const fmtSize = (b) => b > 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB` : b > 2 ** 20 ? `${(b / 2 ** 20).toFixed(1)} MB`
  : b > 1024 ? `${(b / 1024).toFixed(0)} KB` : `${b} B`;

const MAP_ORDER = ["nube", "alta", "cruda", "malla", "malla_f", "splat", "splat_f", "estructura", "glb"];
const MAP_SHORT = { nube: "nube", alta: "alta dens.", cruda: "cruda + tray.", malla: "malla", malla_f: "malla filtr.",
                    splat: "splat", splat_f: "splat filtr.", estructura: "estructura", glb: "glb viejo" };
const TASKS = [
  { id: "windowed", label: "Reprocesar en windowed (ventana 24)", d: "Vuelve a correr el modelo sobre frames/ en modo windowed: corrige la deriva del streaming. Recomendado para sesiones en vivo. ~5 min, GPU.", needs: "frames" },
  { id: "filtro", label: "Filtro geométrico + estructura", d: "Semántica (SegFormer), consistencia multivista, paredes/piso por planos y esquinas. No cambia la nube. ~2 min, GPU.", map: "estructura" },
  { id: "nube", label: "Nube densa", d: "Fusión por vóxel de todos los frames. ~1-3 min, RAM.", map: "nube" },
  { id: "alta", label: "Nube alta densidad", d: "Vóxel más fino. Archivos de cientos de MB. ~3-5 min.", map: "alta" },
  { id: "cruda", label: "Cruda + trayectoria", d: "Nube por frame y las cámaras del recorrido. <1 min.", map: "cruda" },
  { id: "malla", label: "Malla TSDF", d: "Superficie con color por fusión TSDF. ~3 min, CPU.", map: "malla" },
  { id: "splat", label: "Gaussian Splatting", d: "Reconstrucción fotorrealista (gsplat, 7000 iteraciones). ~5-8 min, GPU.", map: "splat" },
  { id: "malla_f", label: "Malla TSDF filtrada", d: "La malla TSDF con el filtro geométrico (requiere el filtro). ~3 min.", map: "malla_f" },
  { id: "splat_f", label: "Gaussian Splatting filtrado", d: "Splat con el filtro: sin personas, profundidad depurada y ajustada a planos. ~5-8 min, GPU.", map: "splat_f" },
  { id: "video", label: "Video del recorrido", d: "Render del recorrido con batch_demo (MP4). ~2 min, GPU.", map: null },
];

const ex = { data: null, filter: new Set(), loaded: null, open: new Set(), job: null };

function chip(c, extra = "") {
  return `<span class="chip c-${slug(c)} ${extra}" data-cat="${esc(c)}">${esc(c)}</span>`;
}

async function refresh(keepOpen = true) {
  const st = $("status");
  try {
    const r = await fetch("/api/explorer");
    ex.data = await r.json();
  } catch (e) {
    st.textContent = "no se pudo leer el catálogo: " + e;
    return;
  }
  if (!keepOpen) ex.open.clear();
  render();
  const n = ex.data.pruebas.length;
  if (!window.__viewer?.current?.()) st.textContent = `${n} pruebas en ${ex.data.root}`;
}

function matches(p) {
  const q = $("ex-search").value.trim().toLowerCase();
  if (q) {
    const hay = [p.titulo, p.zona, p.id, p.notas, ...(p.categorias || [])].join(" ").toLowerCase();
    if (!q.split(/\s+/).every((w) => hay.includes(w))) return false;
  }
  for (const c of ex.filter) if (!(p.categorias || []).includes(c)) return false;
  return true;
}

function groupsOf(p, mode) {
  if (mode === "zona") return [p.zona || "(sin zona)"];
  if (mode === "categoria") return p.categorias?.length ? p.categorias : ["(sin categoría)"];
  if (mode === "fecha") return [p.fecha || "(sin fecha)"];
  return [p.sitio];
}

function render() {
  const d = ex.data;
  if (!d) return;
  // filtro por categoría
  $("ex-filter").innerHTML = d.categorias.map((c) => chip(c, ex.filter.has(c) ? "on" : "")).join("");
  $("ex-filter").querySelectorAll(".chip").forEach((el) => el.onclick = () => {
    const c = el.dataset.cat;
    ex.filter.has(c) ? ex.filter.delete(c) : ex.filter.add(c);
    render();
  });
  const mode = $("ex-group").value;
  const groups = new Map();
  for (const p of d.pruebas.filter(matches)) {
    for (const g of groupsOf(p, mode)) {
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(p);
    }
  }
  const keys = [...groups.keys()].sort((a, b) => mode === "fecha" ? b.localeCompare(a) : a.localeCompare(b));
  const tree = $("ex-tree");
  if (!keys.length) { tree.innerHTML = `<div class="sub" style="padding:8px">ninguna prueba coincide</div>`; return; }
  tree.innerHTML = keys.map((g) => `
    <div class="ex-group ${ex.open.has("g:" + g) ? "collapsed" : ""}" data-g="${esc(g)}">
      <div class="gh"><span>${esc(g)}</span><span>${groups.get(g).length}</span></div>
      <div class="ex-items">${groups.get(g).map(card).join("")}</div>
    </div>`).join("");
  tree.querySelectorAll(".gh").forEach((h) => h.onclick = () => {
    const g = h.parentElement.dataset.g;
    ex.open.has("g:" + g) ? ex.open.delete("g:" + g) : ex.open.add("g:" + g);
    h.parentElement.classList.toggle("collapsed");
  });
  tree.querySelectorAll(".ex-maps button").forEach((b) => b.onclick = () => {
    const p = d.pruebas.find((x) => x.id === b.dataset.p);
    const m = p.maps.find((x) => x.file === b.dataset.f);
    ex.loaded = m.file;
    window.__viewer.loadMap(m);
    render();
  });
  tree.querySelectorAll("[data-act]").forEach((a) => a.onclick = () => {
    const p = d.pruebas.find((x) => x.id === a.dataset.p);
    if (a.dataset.act === "edit") openEditor(p);
    else if (a.dataset.act === "build") openBuilder(p);
    else if (a.dataset.act === "files") toggleFiles(p, a.closest(".ex-card"));
  });
  // reabrir los árboles de archivos que estaban abiertos
  tree.querySelectorAll(".ex-card").forEach((c) => {
    if (ex.open.has("f:" + c.dataset.p)) {
      const p = d.pruebas.find((x) => x.id === c.dataset.p);
      ex.open.delete("f:" + p.id);
      toggleFiles(p, c);
    }
  });
}

function card(p) {
  const maps = [...p.maps].sort((a, b) => MAP_ORDER.indexOf(a.type) - MAP_ORDER.indexOf(b.type));
  const mapBtns = maps.length
    ? maps.map((m) => `<button data-p="${esc(p.id)}" data-f="${esc(m.file)}" class="${m.file === ex.loaded ? "loaded" : ""}"
          title="${esc(m.file)} (${m.size_mb} MB)">${esc(MAP_SHORT[m.type] || m.type)}</button>`).join("")
    : `<span class="none">sin mapas todavía</span>`;
  const active = maps.some((m) => m.file === ex.loaded);
  return `<div class="ex-card ${active ? "active" : ""}" data-p="${esc(p.id)}">
    <div class="t">${esc(p.titulo || p.nombre)}</div>
    <div class="sub">${esc(p.id)}</div>
    ${p.zona ? `<div class="zona">📍 ${esc(p.zona)}</div>` : ""}
    <div class="chips">${(p.categorias || []).map((c) => chip(c)).join("")}</div>
    <div class="ex-maps">${mapBtns}</div>
    <div class="ex-actions">
      <a data-act="edit" data-p="${esc(p.id)}">${p.sin_guardar ? "💾 guardar sesión" : "✎ datos"}</a>
      <a data-act="build" data-p="${esc(p.id)}">⚙ construir mapas</a>
      <a data-act="files" data-p="${esc(p.id)}">📁 archivos</a>
    </div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Árbol de archivos (perezoso: una carpeta por pedido)
// ---------------------------------------------------------------------------
async function toggleFiles(p, cardEl) {
  const old = cardEl.querySelector(":scope > .ex-files");
  if (old) { old.remove(); ex.open.delete("f:" + p.id); return; }
  ex.open.add("f:" + p.id);
  const box = document.createElement("div");
  box.className = "ex-files";
  cardEl.appendChild(box);
  await fillDir(p, "", box);
}

async function fillDir(p, dir, box) {
  box.innerHTML = `<div class="sub">leyendo...</div>`;
  const r = await fetch(`/api/files?prueba=${encodeURIComponent(p.id)}&dir=${encodeURIComponent(dir)}`);
  const d = await r.json();
  if (!r.ok) { box.innerHTML = `<div class="sub">${esc(d.msg)}</div>`; return; }
  const mapFiles = new Map(p.maps.map((m) => [m.file, m]));
  box.innerHTML = d.entries.map((e) => e.dir
    ? `<div class="dir" data-path="${esc(e.path)}"><div class="row"><span class="n">▸ ${esc(e.name)}/</span><span class="s">${e.count} elem.</span></div><div class="kids"></div></div>`
    : `<div class="file"><div class="row"><span class="n">${mapFiles.has(e.data)
        ? `<a class="map" data-f="${esc(e.data)}" href="#" title="cargar en el visor">◆ ${esc(e.name)}</a>`
        : `<a href="/data/${esc(e.data)}" target="_blank" title="abrir en otra pestaña">${esc(e.name)}</a>`}</span>
        <span class="s">${fmtSize(e.size)}</span></div></div>`).join("")
    + (d.truncated ? `<div class="sub">... ${d.total - d.entries.length} más</div>` : "");
  box.querySelectorAll(":scope > .dir > .row .n").forEach((n) => n.onclick = async () => {
    const el = n.closest(".dir"); const kids = el.querySelector(":scope > .kids");
    if (kids.childElementCount) { kids.innerHTML = ""; n.textContent = n.textContent.replace("▾", "▸"); return; }
    n.textContent = n.textContent.replace("▸", "▾");
    await fillDir(p, el.dataset.path, kids);
  });
  box.querySelectorAll(":scope > .file a.map").forEach((a) => a.onclick = (ev) => {
    ev.preventDefault();
    const m = mapFiles.get(a.dataset.f);
    ex.loaded = m.file;
    window.__viewer.loadMap(m);
    render();
  });
}

// ---------------------------------------------------------------------------
// Modal: datos de la prueba (y guardar / descartar sesiones en vivo)
// ---------------------------------------------------------------------------
function modal(html) {
  $("modal-box").innerHTML = html;
  $("modal").classList.remove("hidden");
}
function closeModal() { $("modal").classList.add("hidden"); }
$("modal").addEventListener("mousedown", (e) => { if (e.target.id === "modal") closeModal(); });
window.addEventListener("keydown", (e) => { if (e.code === "Escape") closeModal(); });

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.ok === false) throw new Error(d.msg || `HTTP ${r.status}`);
  return d;
}

function openEditor(p) {
  const cats = new Set(p.categorias || []);
  const suggested = ex.data.categorias;
  const destDefault = "streaming/" + (p.id.includes("webcam") ? "webcam" : "sesiones");
  modal(`
    <h2>${p.sin_guardar ? "Guardar sesión en vivo" : "Datos de la prueba"}</h2>
    <div class="sub">${esc(p.id)}</div>
    <label>Título</label><input id="ed-titulo" type="text" value="${esc(p.titulo)}" placeholder="${esc(p.nombre)}">
    <label>Zona de la universidad</label>
    <input id="ed-zona" type="text" list="ed-zonas" value="${esc(p.zona)}" placeholder="ej. Fablab, escaleras del edificio X">
    <datalist id="ed-zonas">${ex.data.zonas.map((z) => `<option value="${esc(z)}">`).join("")}</datalist>
    <label>Categorías <span style="color:#7d8792">(clic en una sugerida para agregarla, × para quitarla)</span></label>
    <div id="ed-cats" class="chips"></div>
    <div style="display:flex;gap:6px;margin-top:6px">
      <input id="ed-newcat" type="text" list="ed-catlist" placeholder="nueva categoría" style="flex:1">
      <button id="ed-addcat">agregar</button>
    </div>
    <datalist id="ed-catlist">${suggested.map((c) => `<option value="${esc(c)}">`).join("")}</datalist>
    <div id="ed-sugg" class="chips"></div>
    <label>Notas</label><textarea id="ed-notas">${esc(p.notas)}</textarea>
    ${p.sin_guardar ? `
      <div class="section">
        <label>Carpeta donde guardarla (dentro de captures/; se crea prueba_N adentro)</label>
        <input id="ed-dest" type="text" value="${esc(destDefault)}">
        <div class="warn">La sesión se grabó en modo streaming, que deriva. Después de guardarla conviene
          "construir mapas" con la tarea <b>windowed</b>.</div>
      </div>` : ""}
    <div class="row">
      ${p.sin_guardar ? `<button id="ed-discard" class="danger">Descartar sesión</button>` : ""}
      <button id="ed-cancel">Cancelar</button>
      <button id="ed-save" class="primary">${p.sin_guardar ? "Guardar sesión" : "Guardar"}</button>
    </div>`);
  const drawCats = () => {
    $("ed-cats").innerHTML = [...cats].map((c) => `<span class="chip c-${slug(c)}" data-c="${esc(c)}">${esc(c)}<span class="x">×</span></span>`).join("")
      || `<span class="sub">ninguna</span>`;
    $("ed-cats").querySelectorAll(".chip").forEach((el) => el.onclick = () => { cats.delete(el.dataset.c); drawCats(); });
    $("ed-sugg").innerHTML = suggested.filter((c) => !cats.has(c)).map((c) => chip(c)).join("");
    $("ed-sugg").querySelectorAll(".chip").forEach((el) => el.onclick = () => { cats.add(el.dataset.cat); drawCats(); });
  };
  drawCats();
  const add = () => { const v = $("ed-newcat").value.trim(); if (v) { cats.add(v); $("ed-newcat").value = ""; drawCats(); } };
  $("ed-addcat").onclick = add;
  $("ed-newcat").onkeydown = (e) => { if (e.key === "Enter") add(); };
  $("ed-cancel").onclick = closeModal;
  $("ed-save").onclick = async () => {
    const body = { prueba: p.id, titulo: $("ed-titulo").value, zona: $("ed-zona").value,
                   categorias: [...cats], notas: $("ed-notas").value };
    try {
      if (p.sin_guardar) {
        const r = await post("/api/prueba/save", { ...body, destino: $("ed-dest").value });
        $("status").textContent = `sesión guardada en ${r.prueba}`;
      } else {
        await post("/api/prueba/meta", body);
        $("status").textContent = `datos guardados (${p.id})`;
      }
      closeModal();
      refresh();
    } catch (e) { alert("no se pudo guardar: " + e.message); }
  };
  if (p.sin_guardar) $("ed-discard").onclick = async () => {
    if (!confirm(`¿Borrar la sesión ${p.id} y todo lo que grabó? No se puede deshacer.`)) return;
    try { await post("/api/prueba/discard", { prueba: p.id }); closeModal(); refresh(); }
    catch (e) { alert("no se pudo descartar: " + e.message); }
  };
}

// ---------------------------------------------------------------------------
// Modal: construir mapas
// ---------------------------------------------------------------------------
function openBuilder(p) {
  const have = new Set(p.maps.map((m) => m.type));
  const isStream = (p.categorias || []).includes("streaming");
  const rows = TASKS.map((t) => {
    const disabled = (t.needs === "frames" && !p.tiene_frames) || (t.id !== "windowed" && !p.npz.length && !(p.tiene_frames));
    const done = t.map && have.has(t.map);
    const checked = !disabled && !done && (t.id !== "windowed" ? ["nube", "cruda", "malla", "splat"].includes(t.id) : isStream);
    return `<div class="task"><input type="checkbox" id="tk-${t.id}" ${checked ? "checked" : ""} ${disabled ? "disabled" : ""}>
      <div><label for="tk-${t.id}" style="margin:0;color:#e6e8eb;font-size:13px">${t.label}
        ${done ? `<span class="have">· ya existe</span>` : ""}</label><div class="d">${t.d}</div></div></div>`;
  }).join("");
  modal(`
    <h2>Construir mapas</h2>
    <div class="sub">${esc(p.titulo || p.nombre)} · ${esc(p.id)}</div>
    ${p.npz.length > 1 ? `<label>Predicciones</label><select id="bd-npz">${p.npz.map((n) => `<option>${esc(n)}</option>`).join("")}</select>`
      : p.npz.length ? `<div class="sub">predicciones: ${esc(p.npz[0])}</div>` : `<div class="warn">Esta prueba no tiene predicciones (.npz): sólo se puede reprocesar desde frames/.</div>`}
    ${rows}
    <div class="warn">Los mapas que ya existen no se rehacen. Las tareas corren una tras otra en segundo plano;
      mientras tanto no se puede iniciar el mapeo en vivo (comparten la GPU).</div>
    <div class="row"><button id="bd-cancel">Cancelar</button><button id="bd-go" class="primary">Construir</button></div>`);
  $("bd-cancel").onclick = closeModal;
  $("bd-go").onclick = async () => {
    const tasks = TASKS.map((t) => t.id).filter((id) => $("tk-" + id).checked);
    if (!tasks.length) { alert("elegí al menos una tarea"); return; }
    const npz = $("bd-npz") ? $("bd-npz").value : (p.npz[0] || null);
    try { await post("/api/jobs/start", { prueba: p.id, npz, tasks }); closeModal(); }
    catch (e) { alert("no se pudo iniciar: " + e.message); }
  };
}

// ---------------------------------------------------------------------------
// Progreso del trabajo (llega por el WebSocket del mapeo en vivo)
// ---------------------------------------------------------------------------
function renderJob(j) {
  const bar = $("job-bar");
  if (!j || (!j.running && j.rc === null)) { bar.classList.add("hidden"); return; }
  bar.classList.remove("hidden");
  const pct = j.total ? Math.round(100 * Object.keys(j.done || {}).length / j.total) : 0;
  const last = (j.log || []).slice(-1)[0] || "";
  bar.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center">
      <span>${j.running ? "⚙ construyendo" : (j.rc === 0 ? "✓ listo" : "⚠ terminado con fallas")}: ${esc(j.prueba || "")}</span>
      ${j.running ? `<button id="job-cancel">cancelar</button>` : `<button id="job-hide">ocultar</button>`}</div>
    <div class="bar"><div style="width:${pct}%"></div></div>
    <div>${(j.tasks || []).map((t) => `${j.done?.[t] === "ok" ? "✓" : j.done?.[t] === "falla" ? "✗" : t === j.current ? "▶" : "·"} ${t}`).join("  ")}</div>
    <div class="log" title="${esc(last)}">${esc(last)}</div>`;
  if ($("job-cancel")) $("job-cancel").onclick = () => post("/api/jobs/cancel", {});
  if ($("job-hide")) $("job-hide").onclick = () => bar.classList.add("hidden");
}
window.addEventListener("server-job", (e) => {
  const was = ex.job?.running;
  ex.job = e.detail;
  renderJob(ex.job);
  if (was && !ex.job.running) refresh();
  // un mapa nuevo terminado aparece en el árbol sin esperar al final
  else if (ex.job.running && ex.job.msg && /^ok /.test(ex.job.msg)) refresh();
});
window.addEventListener("server-saved", () => refresh());
window.addEventListener("map-loaded", (e) => { ex.loaded = e.detail.file; render(); });

$("ex-refresh").onclick = () => refresh();
$("ex-search").oninput = () => render();
$("ex-group").onchange = () => { ex.open.clear(); render(); };
const start = () => refresh(false);
if (window.__viewer) start(); else window.addEventListener("viewer-ready", start, { once: true });
window.__explorer = { refresh, openEditor, openBuilder };

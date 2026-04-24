/**
 * ApexVision-Core — TypeScript Gateway + Demo UI
 */

import { serve }  from "@hono/node-server";
import { Hono }   from "hono";
import { logger } from "hono/logger";
import axios from "axios";

const app  = new Hono();
const API  = process.env.APEX_API_URL ?? "http://localhost:8000";
const KEY  = process.env.APEX_API_KEY ?? "apexvision-master-dev-key";
const PORT = parseInt(process.env.TS_PORT ?? "3000");

app.use("*", logger());

app.get("/health", (c) =>
  c.json({ status: "ok", service: "ApexVision TS Gateway", upstream: API })
);

app.post("/analyze", async (c) => {
  try {
    const body = await c.req.json();
    const { data } = await axios.post(
      `${API}/api/v1/vision/analyze`,
      body,
      { headers: { "X-ApexVision-Key": KEY }, timeout: 120_000 }
    );
    return c.json(data);
  } catch (err: any) {
    const msg = err?.response?.data?.detail ?? err?.message ?? "Error";
    return c.json({ error: msg }, 500);
  }
});

app.get("/", (c) => c.html(buildHTML()));

serve({ fetch: app.fetch, port: PORT }, (info) => {
  console.log(`\n  ApexVision TS Gateway -> http://localhost:${info.port}`);
  console.log(`  Upstream API         -> ${API}`);
  console.log(`  API Docs             -> ${API}/docs\n`);
});

// ─── HTML ────────────────────────────────────────────────────────────────────

function buildHTML(): string {
  return `<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ApexVision-Core</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--surface:#1a1d2e;--border:#2a2d3e;
  --accent:#6366f1;--accent2:#8b5cf6;
  --green:#10b981;--yellow:#f59e0b;--red:#ef4444;--blue:#0ea5e9;--cyan:#06b6d4;--pink:#ec4899;
  --text:#e2e8f0;--muted:#64748b;--radius:12px;
}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:24px}
.header{display:flex;align-items:center;gap:12px;margin-bottom:28px}
.logo{width:40px;height:40px;background:linear-gradient(135deg,var(--accent),var(--accent2));border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px}
.header h1{font-size:22px;font-weight:700}
.header p{font-size:13px;color:var(--muted)}
.badge{margin-left:auto;background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:4px 14px;font-size:12px;color:var(--muted)}
.badge b{color:var(--green)}
.layout{display:grid;grid-template-columns:360px 1fr;gap:20px;max-width:1200px;margin:0 auto}
@media(max-width:800px){.layout{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;margin-bottom:16px}
.card:last-child{margin-bottom:0}
.card-title{font-size:12px;font-weight:700;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px}
.drop-zone{border:2px dashed var(--border);border-radius:var(--radius);padding:28px 16px;text-align:center;cursor:pointer;transition:all .2s;position:relative;overflow:hidden}
.drop-zone:hover,.drop-zone.drag{border-color:var(--accent);background:#6366f110}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop-icon{font-size:28px;margin-bottom:6px}
.drop-zone strong{display:block;font-size:14px;margin-bottom:4px}
.drop-zone span{font-size:12px;color:var(--muted)}
#preview-wrap{margin-top:12px;display:none}
#preview-img{width:100%;border-radius:8px;max-height:200px;object-fit:cover;border:1px solid var(--border)}
#preview-name{font-size:11px;color:var(--muted);margin-top:5px;text-align:center}
.tasks-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}
.task-btn{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px 4px;text-align:center;cursor:pointer;transition:all .15s;font-size:11px;color:var(--muted);user-select:none}
.task-btn .icon{font-size:18px;display:block;margin-bottom:3px}
.task-btn.on{border-color:var(--accent);background:#6366f118;color:var(--text)}
.conf-row{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.conf-row label{font-size:13px;color:var(--muted);white-space:nowrap}
input[type=range]{flex:1;accent-color:var(--accent)}
#conf-val{font-size:13px;font-weight:700;color:var(--accent);min-width:34px;text-align:right}
#analyze-btn{width:100%;padding:12px;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;border:none;border-radius:8px;font-size:14px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:opacity .2s}
#analyze-btn:hover{opacity:.9}
#analyze-btn:disabled{opacity:.45;cursor:not-allowed}
.spin{width:16px;height:16px;border:2px solid #ffffff40;border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}
.results-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;min-height:460px}
#results{display:flex;flex-direction:column;gap:12px}
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:360px;color:var(--muted);font-size:13px;gap:8px;text-align:center}
.empty .big{font-size:48px}
.rblock{background:var(--bg);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.rhead{padding:10px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--border);font-size:13px;font-weight:600}
.dot{width:8px;height:8px;border-radius:50%}
.rms{margin-left:auto;font-size:11px;font-weight:400;color:var(--muted);font-family:monospace}
.rbody{padding:12px 14px}
.timing{display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--surface);border:1px solid var(--border);border-radius:10px;font-size:12px;flex-wrap:wrap}
.tk{color:var(--muted)}
.tv{color:var(--text);font-family:monospace;font-weight:700}
.tsep{color:var(--border)}

/* Detection */
.box-count{font-size:26px;font-weight:700;color:var(--accent);line-height:1}
.box-sub{font-size:12px;color:var(--muted)}
.box-list{display:flex;flex-direction:column;gap:6px;margin-top:10px}
.box-item{display:flex;align-items:center;gap:8px;font-size:13px}
.blabel{flex:1;font-weight:500;text-transform:capitalize}
.bbar-wrap{width:80px;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.bbar{height:100%;border-radius:3px}
.bpct{font-size:11px;color:var(--muted);font-family:monospace;min-width:38px;text-align:right}

/* OCR */
.ocr-out{font-family:monospace;font-size:13px;line-height:1.7;color:var(--text);white-space:pre-wrap;word-break:break-word;max-height:200px;overflow-y:auto;padding:10px 12px;background:var(--surface);border:1px solid var(--border);border-radius:8px;margin-bottom:10px}
.ocr-copy{width:100%;padding:8px;background:var(--accent);color:#fff;border:none;border-radius:6px;font-size:12px;font-weight:600;cursor:pointer;transition:opacity .2s;margin-bottom:8px}
.ocr-copy:hover{opacity:.85}
.ocr-meta{display:flex;gap:10px;font-size:12px;color:var(--muted);flex-wrap:wrap}
.ocr-meta span{background:var(--surface);padding:2px 8px;border-radius:4px;border:1px solid var(--border)}
.ocr-blocks{margin-top:10px}
.block-row{display:flex;align-items:center;gap:8px;font-size:12px;padding:5px 8px;border-radius:6px;border:1px solid var(--border);margin-bottom:4px;background:var(--surface)}
.block-txt{flex:1;font-family:monospace}
.block-conf{font-size:11px;color:var(--muted);white-space:nowrap}
.ocr-blocks summary{font-size:12px;color:var(--muted);cursor:pointer;margin-top:4px;padding:4px 0}

/* Embedding */
.emb-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}
.emb-stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}
.emb-num{font-size:22px;font-weight:700;color:var(--accent)}
.emb-lbl{font-size:11px;color:var(--muted);margin-top:2px}
.emb-vec{grid-column:1/-1;font-family:monospace;font-size:11px;color:var(--muted);word-break:break-all;line-height:1.5;max-height:46px;overflow:hidden;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:8px}

/* Classification */
.clf-list{display:flex;flex-direction:column;gap:8px}
.clf-item{display:flex;align-items:center;gap:8px;font-size:13px}
.clf-rank{font-size:11px;color:var(--muted);font-family:monospace;min-width:16px}
.clf-lbl{flex:1;text-transform:capitalize}
.clf-bar-wrap{width:100px;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.clf-bar{height:100%;border-radius:3px;background:var(--blue)}

/* Face */
.face-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px}
.face-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px;font-size:12px;display:flex;flex-direction:column;gap:4px}
.fbadge{display:flex;gap:4px;align-items:center;font-size:11px;color:var(--muted)}
.fval{font-weight:600;color:var(--text)}

/* Error */
.err-box{background:#ef444420;border:1px solid #ef444450;border-radius:10px;padding:14px 16px;font-size:13px;color:#fca5a5}
.err-box strong{display:block;margin-bottom:4px;color:var(--red);font-size:14px}

::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:var(--surface)}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>

<div class="header">
  <div class="logo">&#128269;</div>
  <div>
    <h1>ApexVision-Core</h1>
    <p>Ultra Vision API Platform</p>
  </div>
  <div class="badge" id="apibadge">API: <b id="apistat">checking...</b></div>
</div>

<div class="layout">

  <div>
    <!-- Upload -->
    <div class="card">
      <div class="card-title">&#128247; Imagen</div>
      <div class="drop-zone" id="dropzone">
        <input type="file" id="fileinput" accept="image/*">
        <div class="drop-icon">&#128444;</div>
        <strong>Arrastra o click para subir</strong>
        <span>JPEG &middot; PNG &middot; WebP &middot; BMP</span>
      </div>
      <div id="preview-wrap">
        <img id="preview-img" src="" alt="preview">
        <div id="preview-name"></div>
      </div>
    </div>

    <!-- Tasks -->
    <div class="card">
      <div class="card-title">&#127919; Tasks</div>
      <div class="tasks-grid">
        <div class="task-btn on"  data-t="detect">  <span class="icon">&#128230;</span>detect</div>
        <div class="task-btn"     data-t="classify"><span class="icon">&#127991;</span>classify</div>
        <div class="task-btn"     data-t="ocr">     <span class="icon">&#128221;</span>ocr</div>
        <div class="task-btn"     data-t="face">    <span class="icon">&#128100;</span>face</div>
        <div class="task-btn"     data-t="embed">   <span class="icon">&#128290;</span>embed</div>
        <div class="task-btn"     data-t="depth">   <span class="icon">&#128208;</span>depth</div>
        <div class="task-btn"     data-t="segment"> <span class="icon">&#9986;</span>segment</div>
        <div class="task-btn" id="allbtn" style="background:#6366f118;border-color:#6366f1;color:#a5b4fc">
          <span class="icon">&#9889;</span>all
        </div>
      </div>
      <div class="conf-row">
        <label>Confianza</label>
        <input type="range" id="confrange" min="0.1" max="1.0" step="0.05" value="0.5">
        <span id="conf-val">0.50</span>
      </div>
    </div>

    <!-- Button -->
    <button id="analyze-btn" disabled>
      <div class="spin" id="spinner"></div>
      <span id="btntext">Selecciona una imagen</span>
    </button>
  </div>

  <!-- Results -->
  <div class="results-card">
    <div class="card-title">&#128202; Resultados</div>
    <div id="results">
      <div class="empty">
        <div class="big">&#128269;</div>
        <div>Sube una imagen y selecciona las tasks</div>
        <div style="color:#475569;font-size:12px">OCR extrae texto &bull; detect encuentra objetos &bull; embed genera vector</div>
      </div>
    </div>
  </div>

</div>

<script>
// ── API status ──────────────────────────────────────────────
function checkApi() {
  fetch("/health").then(function(r){ return r.json(); }).then(function(d){
    var el = document.getElementById("apistat");
    if(d.status === "ok"){ el.textContent = "online \u2705"; el.style.color = "#10b981"; }
    else { el.textContent = "degraded \u26A0\uFE0F"; el.style.color = "#f59e0b"; }
  }).catch(function(){
    var el = document.getElementById("apistat");
    el.textContent = "offline \u274C"; el.style.color = "#ef4444";
  });
}
checkApi();
setInterval(checkApi, 10000);

// ── File ───────────────────────────────────────────────────
var selectedFile = null;
var fileinput  = document.getElementById("fileinput");
var dropzone   = document.getElementById("dropzone");
var previewWrap= document.getElementById("preview-wrap");
var previewImg = document.getElementById("preview-img");
var previewName= document.getElementById("preview-name");
var analyzeBtn = document.getElementById("analyze-btn");
var btntext    = document.getElementById("btntext");

function handleFile(file) {
  if(!file || !file.type.startsWith("image/")) return;
  selectedFile = file;
  previewImg.src = URL.createObjectURL(file);
  previewName.textContent = file.name + " \u00B7 " + (file.size/1024).toFixed(1) + " KB";
  previewWrap.style.display = "block";
  analyzeBtn.disabled = false;
  btntext.textContent = "\uD83D\uDE80 Analizar imagen";
}

fileinput.addEventListener("change", function(e){ handleFile(e.target.files[0]); });
dropzone.addEventListener("dragover",  function(e){ e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", function(){ dropzone.classList.remove("drag"); });
dropzone.addEventListener("drop", function(e){
  e.preventDefault(); dropzone.classList.remove("drag");
  handleFile(e.dataTransfer.files[0]);
});

// ── Tasks ──────────────────────────────────────────────────
var TASKS = ["detect","classify","ocr","face","embed","depth","segment"];
var selectedTasks = new Set(["detect"]);

document.querySelectorAll(".task-btn[data-t]").forEach(function(btn){
  btn.addEventListener("click", function(){
    var t = btn.getAttribute("data-t");
    if(selectedTasks.has(t)){
      if(selectedTasks.size === 1) return;
      selectedTasks.delete(t); btn.classList.remove("on");
    } else {
      selectedTasks.add(t); btn.classList.add("on");
    }
  });
});

document.getElementById("allbtn").addEventListener("click", function(){
  var allOn = TASKS.every(function(t){ return selectedTasks.has(t); });
  if(allOn){
    TASKS.forEach(function(t){ selectedTasks.delete(t); });
    selectedTasks.add("detect");
    document.querySelectorAll(".task-btn[data-t]").forEach(function(b){
      b.classList.toggle("on", b.getAttribute("data-t") === "detect");
    });
  } else {
    TASKS.forEach(function(t){ selectedTasks.add(t); });
    document.querySelectorAll(".task-btn[data-t]").forEach(function(b){ b.classList.add("on"); });
  }
});

// ── Confidence ─────────────────────────────────────────────
var confrange = document.getElementById("confrange");
var confval   = document.getElementById("conf-val");
confrange.addEventListener("input", function(){
  confval.textContent = parseFloat(confrange.value).toFixed(2);
});

// ── Analyze ────────────────────────────────────────────────
analyzeBtn.addEventListener("click", function(){
  if(!selectedFile) return;
  var spinner = document.getElementById("spinner");
  analyzeBtn.disabled = true;
  spinner.style.display = "block";
  btntext.textContent = "Analizando...";

  fileToBase64(selectedFile).then(function(b64){
    return fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image:   { format: "base64", data: b64 },
        tasks:   Array.from(selectedTasks),
        options: { confidence_threshold: parseFloat(confrange.value) }
      })
    });
  }).then(function(r){ return r.json(); }).then(function(data){
    console.log('[ApexVision] Response:', JSON.stringify(data, null, 2));
    if(data.error) throw new Error(data.error);
    renderResults(data);
  }).catch(function(e){
    renderError(e.message || String(e));
  }).finally(function(){
    analyzeBtn.disabled = false;
    spinner.style.display = "none";
    btntext.textContent = "\uD83D\uDE80 Analizar imagen";
  });
});

function fileToBase64(file){
  return new Promise(function(resolve, reject){
    var r = new FileReader();
    r.onload  = function(){ resolve(r.result.split(",")[1]); };
    r.onerror = function(){ reject(new Error("Error leyendo archivo")); };
    r.readAsDataURL(file);
  });
}

// ── Render ─────────────────────────────────────────────────
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

function renderResults(data){
  var c = document.getElementById("results");
  c.innerHTML = "";

  // Timing
  var trow = document.createElement("div");
  trow.className = "timing";
  trow.innerHTML =
    "<span class='tk'>request_id:</span><span class='tv'>" + esc(data.request_id ? data.request_id.slice(0,8)+"..." : "-") + "</span>" +
    "<span class='tsep'>\u00B7</span>" +
    "<span class='tk'>imagen:</span><span class='tv'>" + esc(data.image_width) + "\u00D7" + esc(data.image_height) + "px</span>" +
    "<span class='tsep'>\u00B7</span>" +
    "<span class='tk'>total:</span><span class='tv'>" + esc(data.total_inference_ms ? data.total_inference_ms.toFixed(1) : "0") + "ms</span>" +
    "<span class='tsep'>\u00B7</span>" +
    "<span class='tk'>tasks:</span><span class='tv'>" + esc((data.tasks_ran || []).join(", ")) + "</span>";
  c.appendChild(trow);

  if(data.detection)     renderDetection(data.detection, c);
  if(data.classification) renderClassification(data.classification, c);
  if(data.ocr)           renderOCR(data.ocr, c);
  if(data.face)          renderFace(data.face, c);
  if(data.embedding)     renderEmbedding(data.embedding, c);
}

// ── Detection ──────────────────────────────────────────────
function renderDetection(det, c){
  var block = makeBlock("&#128230; Detecci\u00F3n \u2014 " + det.count + " objetos", det.inference_ms, "#6366f1");
  var body  = block.querySelector(".rbody");
  var html  = "";

  html += "<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:10px'>";
  html += "<span class='box-count'>" + esc(det.count) + "</span>";
  html += "<span class='box-sub'>objetos &middot; " + esc(det.model_used) + "</span>";
  html += "</div>";

  if(det.count === 0){
    html += "<p style='color:var(--muted);font-size:13px'>No se detectaron objetos con el threshold actual.</p>";
  } else {
    html += "<div class='box-list'>";
    var boxes = det.boxes.slice(0, 8);
    for(var i=0; i<boxes.length; i++){
      var b    = boxes[i];
      var pct  = (b.confidence * 100).toFixed(1);
      var color= b.confidence > 0.7 ? "#10b981" : b.confidence > 0.4 ? "#f59e0b" : "#ef4444";
      html += "<div class='box-item'>";
      html += "<span class='blabel'>" + esc(b.label) + "</span>";
      html += "<div class='bbar-wrap'><div class='bbar' style='width:" + pct + "%;background:" + color + "'></div></div>";
      html += "<span class='bpct'>" + pct + "%</span>";
      html += "</div>";
    }
    if(det.boxes.length > 8) html += "<p style='font-size:12px;color:var(--muted);margin-top:4px'>+ " + (det.boxes.length-8) + " m\u00E1s</p>";
    html += "</div>";
  }
  body.innerHTML = html;
  c.appendChild(block);
}

// ── Classification ─────────────────────────────────────────
function renderClassification(clf, c){
  var top   = clf.predictions && clf.predictions[0];
  var title = top
    ? "&#127991; Clasificaci\u00F3n \u2014 " + esc(top.label) + " (" + (top.confidence*100).toFixed(1) + "%)"
    : "&#127991; Clasificaci\u00F3n";
  var block = makeBlock(title, clf.inference_ms, "#0ea5e9");
  var body  = block.querySelector(".rbody");
  var html  = "<div class='clf-list'>";
  var preds = (clf.predictions || []).slice(0, 5);
  for(var i=0; i<preds.length; i++){
    var p   = preds[i];
    var pct = (p.confidence * 100).toFixed(1);
    html += "<div class='clf-item'>";
    html += "<span class='clf-rank'>#" + (i+1) + "</span>";
    html += "<span class='clf-lbl'>" + esc(p.label) + "</span>";
    html += "<div class='clf-bar-wrap'><div class='clf-bar' style='width:" + pct + "%'></div></div>";
    html += "<span class='bpct'>" + pct + "%</span>";
    html += "</div>";
  }
  html += "</div>";
  html += "<p style='font-size:11px;color:var(--muted);margin-top:8px'>Modelo: " + esc(clf.model_used) + "</p>";
  body.innerHTML = html;
  c.appendChild(block);
}

// ── OCR ────────────────────────────────────────────────────
function renderOCR(ocr, c){
  var title = ocr.text
    ? "&#128221; OCR \u2014 " + ocr.text.length + " caracteres detectados"
    : "&#128221; OCR \u2014 sin texto";
  var block = makeBlock(title, ocr.inference_ms, "#10b981");
  var body  = block.querySelector(".rbody");
  var html  = "";

  if(!ocr.text || ocr.text.trim() === ""){
    html += "<p style='color:var(--muted);font-size:13px'>No se detect\u00F3 texto en la imagen.</p>";
    html += "<p style='color:var(--muted);font-size:11px;margin-top:6px'>Tip: EasyOCR necesita estar instalado. Corre: <code style=\'background:var(--bg);padding:2px 6px;border-radius:4px\'>pip install easyocr</code></p>";
  } else {
    // Texto completo extraído
    html += "<div class='ocr-out' id='ocrout'>" + esc(ocr.text) + "</div>";

    // Botón copiar
    html += "<button class='ocr-copy' onclick='copyOCR()'>&#128203; Copiar texto</button>";

    // Meta info
    html += "<div class='ocr-meta'>";
    html += "<span>Idioma: " + esc(ocr.language_detected || "?") + "</span>";
    html += "<span>" + (ocr.blocks ? ocr.blocks.length : 0) + " bloques</span>";
    html += "<span>" + ocr.text.length + " caracteres</span>";
    html += "<span>" + ocr.text.split(" ").filter(function(w){ return w.length>0; }).length + " palabras</span>";
    html += "</div>";

    // Bloques individuales (collapsible)
    if(ocr.blocks && ocr.blocks.length > 0){
      html += "<div class='ocr-blocks'><details><summary>Ver " + ocr.blocks.length + " bloques individuales</summary>";
      html += "<div style='margin-top:8px;max-height:160px;overflow-y:auto'>";
      var blocks = ocr.blocks.slice(0, 30);
      for(var i=0; i<blocks.length; i++){
        var bl  = blocks[i];
        var pct = bl.confidence !== undefined ? (bl.confidence*100).toFixed(0) + "%" : "";
        html += "<div class='block-row'>";
        html += "<span class='block-txt'>" + esc(bl.text) + "</span>";
        html += "<span class='block-conf'>" + pct + "</span>";
        html += "</div>";
      }
      if(ocr.blocks.length > 30) html += "<p style='font-size:11px;color:var(--muted);margin-top:4px'>+ " + (ocr.blocks.length-30) + " m\u00E1s</p>";
      html += "</div></details></div>";
    }
  }
  body.innerHTML = html;
  c.appendChild(block);
}

// ── Face ───────────────────────────────────────────────────
function renderFace(face, c){
  var block = makeBlock("&#128100; Rostros \u2014 " + face.count + " detectados", face.inference_ms, "#f59e0b");
  var body  = block.querySelector(".rbody");
  var html  = "";

  if(face.count === 0){
    html += "<p style='color:var(--muted);font-size:13px'>No se detectaron rostros.</p>";
  } else {
    html += "<div class='face-grid'>";
    var faces = face.faces.slice(0,6);
    for(var i=0; i<faces.length; i++){
      var f = faces[i];
      var a = f.attributes || {};
      html += "<div class='face-card'>";
      html += "<div style='font-weight:700;color:var(--accent);font-size:13px'>Rostro #" + (i+1) + "</div>";
      if(a.age)     html += "<div class='fbadge'>&#127874; <span class='fval'>" + esc(a.age) + " a\u00F1os</span></div>";
      if(a.gender)  html += "<div class='fbadge'>&#128101; <span class='fval'>" + esc(a.gender) + "</span></div>";
      if(a.emotion) html += "<div class='fbadge'>&#128522; <span class='fval'>" + esc(a.emotion) + "</span></div>";
      if(f.bbox)    html += "<div class='fbadge'>conf: <span class='fval'>" + ((f.bbox.confidence||0)*100).toFixed(0) + "%</span></div>";
      html += "</div>";
    }
    html += "</div>";
  }
  body.innerHTML = html;
  c.appendChild(block);
}

// ── Embedding ──────────────────────────────────────────────
function renderEmbedding(emb, c){
  var block = makeBlock("&#128290; Embedding \u2014 " + emb.dimensions + "d", emb.inference_ms, "#8b5cf6");
  var body  = block.querySelector(".rbody");

  var norm    = 0;
  var preview = "";
  if(emb.embedding && emb.embedding.length > 0){
    for(var i=0; i<emb.embedding.length; i++) norm += emb.embedding[i] * emb.embedding[i];
    norm = Math.sqrt(norm);
    preview = "[" + emb.embedding.slice(0,16).map(function(v){ return v.toFixed(4); }).join(", ") + " ...]";
  }

  var html = "<div class='emb-grid'>";
  html += "<div class='emb-stat'><div class='emb-num'>" + esc(emb.dimensions) + "</div><div class='emb-lbl'>dimensiones</div></div>";
  html += "<div class='emb-stat'><div class='emb-num'>" + norm.toFixed(4) + "</div><div class='emb-lbl'>norma L2 (~1.0)</div></div>";
  html += "<div class='emb-vec'>" + esc(preview) + "</div>";
  html += "</div>";
  html += "<p style='font-size:11px;color:var(--muted);margin-top:8px'>Modelo: " + esc(emb.model_used) + "</p>";
  body.innerHTML = html;
  c.appendChild(block);
}

// ── Error ──────────────────────────────────────────────────
function renderError(msg){
  var c = document.getElementById("results");
  c.innerHTML = "";
  var d = document.createElement("div");
  d.className = "err-box";
  d.innerHTML = "<strong>&#10060; Error</strong>" + esc(msg);
  c.appendChild(d);
}

// ── Helper ─────────────────────────────────────────────────
function makeBlock(title, ms, color){
  var el = document.createElement("div");
  el.className = "rblock";
  var msHtml = ms != null ? "<span class='rms'>" + ms.toFixed(1) + "ms</span>" : "";
  el.innerHTML =
    "<div class='rhead'>" +
      "<div class='dot' style='background:" + color + "'></div>" +
      "<span>" + title + "</span>" +
      msHtml +
    "</div>" +
    "<div class='rbody'></div>";
  return el;
}

// ── Copy OCR ───────────────────────────────────────────────
function copyOCR(){
  var el = document.getElementById("ocrout");
  if(!el) return;
  navigator.clipboard.writeText(el.textContent).then(function(){
    var btn = document.querySelector(".ocr-copy");
    if(btn){ btn.textContent = "\u2705 Copiado!"; setTimeout(function(){ btn.textContent = "\uD83D\uDCCB Copiar texto"; }, 2000); }
  });
}
</script>
</body>
</html>`;
}
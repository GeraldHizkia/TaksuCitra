// ==========================================================
// TaksuCitra - Main Application Logic
// ==========================================================

let uploadedImageB64 = null;
let currentPanelKey = "channel-split";
let secondSampleValue = "sample3_edges.jpg";
let inpaintMaskCanvas = null, inpaintDrawCanvas = null, inpaintCtx = null;
let panoramaImages = [];
let watermarkedResultB64 = null;
let webcamStream = null, webcamRunning = false;

const panelTitleEl = document.getElementById("panel-title");
const panelDescEl = document.getElementById("panel-desc");
const controlsEl = document.getElementById("panel-controls");
const resultEl = document.getElementById("panel-result");
const imageSourceCard = document.getElementById("image-source-card");

// ============================================================
// HISTORY (localStorage) - Riwayat + Export PDF
// ============================================================
const HISTORY_KEY = "taksucitra_history_v1";

function getHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; }
}
function setHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
  updateHistoryBadge();
}
function addHistoryItem(item) {
  const items = getHistory();
  items.unshift({ ...item, id: Date.now(), time: new Date().toLocaleString("id-ID"), checked: true });
  if (items.length > 60) items.pop(); // batasi biar localStorage tidak penuh
  setHistory(items);
  renderHistoryList();
}
function updateHistoryBadge() {
  const count = getHistory().length;
  const badge = document.getElementById("history-count");
  if (count > 0) { badge.style.display = "flex"; badge.textContent = count; }
  else { badge.style.display = "none"; }
}

function renderHistoryList() {
  const items = getHistory();
  const listEl = document.getElementById("history-list");
  if (items.length === 0) {
    listEl.innerHTML = `<div class="history-empty">Belum ada riwayat.<br>Klik "Simpan ke Riwayat" setelah memproses gambar.</div>`;
    return;
  }
  listEl.innerHTML = items.map(it => `
    <div class="history-item">
      <input type="checkbox" data-hist-id="${it.id}" ${it.checked !== false ? "checked" : ""}>
      <img src="${it.after || it.before}">
      <div class="history-item-info">
        <div class="h-title">${it.title}</div>
        <div class="h-time">${it.time}</div>
      </div>
    </div>`).join("");

  listEl.querySelectorAll("input[type=checkbox]").forEach(cb => {
    cb.addEventListener("change", () => {
      const items2 = getHistory();
      const target = items2.find(x => x.id == cb.dataset.histId);
      if (target) target.checked = cb.checked;
      setHistory(items2);
    });
  });
}

function openHistoryDrawer() {
  document.getElementById("history-drawer").classList.add("open");
  document.getElementById("history-overlay").classList.add("open");
  renderHistoryList();
}
function closeHistoryDrawer() {
  document.getElementById("history-drawer").classList.remove("open");
  document.getElementById("history-overlay").classList.remove("open");
}

document.getElementById("btn-open-history").addEventListener("click", openHistoryDrawer);
document.getElementById("btn-close-history").addEventListener("click", closeHistoryDrawer);
document.getElementById("history-overlay").addEventListener("click", closeHistoryDrawer);
document.getElementById("btn-clear-history").addEventListener("click", () => {
  if (confirm("Hapus semua riwayat yang tersimpan?")) { setHistory([]); renderHistoryList(); }
});

document.getElementById("btn-export-pdf").addEventListener("click", async () => {
  const items = getHistory().filter(x => x.checked !== false);
  if (items.length === 0) { alert("Tidak ada item yang dicentang untuk dimasukkan ke laporan."); return; }

  const btn = document.getElementById("btn-export-pdf");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Membuat PDF...`;

  const payload = {
    student_info: {
      name: document.getElementById("student-name").value || "-",
      course: document.getElementById("student-course").value || "Image Processing Programming",
    },
    items: items.map(it => ({
      title: it.title, description: it.desc || "", before_b64: it.before, after_b64: it.after,
    })),
  };

  try {
    const res = await fetch("/api/report/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Gagal membuat laporan");
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "TaksuCitra_Laporan.pdf"; a.click();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert("Gagal export PDF: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "📄 Export Laporan PDF";
  }
});

// ============================================================
// UTILITAS UMUM
// ============================================================

function getSelectedSample() {
  const checked = document.querySelector('input[name="sample"]:checked');
  return checked ? checked.value : null;
}

function buildFieldsHTML(fields) {
  return fields.map(f => {
    const hintHTML = f.hint ? `<div class="field-hint">${f.hint}</div>` : "";
    if (f.type === "select") {
      return `<div class="control-field"><label>${f.label}</label>
        <select data-field="${f.name}">${f.options.map(([v, l]) => `<option value="${v}" ${v === f.value ? "selected" : ""}>${l}</option>`).join("")}</select>${hintHTML}</div>`;
    } else if (f.type === "range") {
      return `<div class="control-field"><label>${f.label} <span class="range-value" data-range-value="${f.name}">${f.value}</span></label>
        <input type="range" data-field="${f.name}" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.value}">${hintHTML}</div>`;
    } else if (f.type === "text") {
      return `<div class="control-field"><label>${f.label}</label>
        <input type="text" data-field="${f.name}" placeholder="${f.placeholder || ""}">${hintHTML}</div>`;
    } else if (f.type === "checkbox") {
      return `<div class="control-field control-field-checkbox"><label>
        <input type="checkbox" data-field="${f.name}" ${f.value ? "checked" : ""}> ${f.label}</label>${hintHTML}</div>`;
    } else if (f.type === "multicheck") {
      return `<div class="control-field"><label>${f.label}</label>
        <div class="multicheck-group">${f.options.map(([v, l]) => `
          <label class="multicheck-item"><input type="checkbox" data-multifield="${f.name}" value="${v}" ${f.defaultChecked && f.defaultChecked.includes(v) ? "checked" : ""}> ${l}</label>`).join("")}
        </div>${hintHTML}</div>`;
    }
    return "";
  }).join("");
}

function collectFieldValues() {
  const values = {};
  controlsEl.querySelectorAll("[data-field]").forEach(el => {
    values[el.dataset.field] = el.type === "checkbox" ? el.checked : el.value;
  });
  const multiGroups = {};
  controlsEl.querySelectorAll("[data-multifield]").forEach(el => {
    const name = el.dataset.multifield;
    if (!multiGroups[name]) multiGroups[name] = [];
    if (el.checked) multiGroups[name].push(el.value);
  });
  Object.assign(values, multiGroups);
  return values;
}

function bindRangeLiveUpdate() {
  controlsEl.querySelectorAll('input[type="range"]').forEach(inp => {
    inp.addEventListener("input", () => {
      const label = controlsEl.querySelector(`[data-range-value="${inp.dataset.field}"]`);
      if (label) label.textContent = inp.value;
    });
  });
}

function currentImagePreviewSrc() {
  if (uploadedImageB64) return uploadedImageB64;
  const sample = getSelectedSample();
  return sample ? `/static/samples/${sample}` : null;
}

// ============================================================
// RENDER PANEL UTAMA
// ============================================================

function renderPanel(key) {
  currentPanelKey = key;

  if (key === "home") return renderHomeDashboard();
  if (key === "ai-lab-home") return renderAiLabBento();

  const panel = PANELS[key];
  panelTitleEl.textContent = panel.title;
  panelDescEl.textContent = panel.desc;
  resultEl.style.display = "";

  // Tampilkan/sembunyikan image source card sesuai kebutuhan panel
  const needsStandardSource = !panel.customUI || panel.customUI === "mask-draw" || panel.customUI === "fe-concat" || panel.customUI === "morf-hmt";
  imageSourceCard.style.display = needsStandardSource ? "block" : "none";

  if (panel.customUI === "batch") return renderBatchMode();
  if (panel.customUI === "webcam") return renderWebcamMode();
  if (panel.customUI === "history-page") return renderHistoryPage();
  if (panel.customUI === "multi-upload") return renderMultiUploadPanel(panel);
  if (panel.customUI === "source-reference") return renderSourceReferencePanel(panel);
  if (panel.customUI === "watermark") return renderWatermarkPanel(panel);
  if (panel.customUI === "mask-draw") return renderMaskDrawPanel(panel);
  if (panel.customUI === "dual-upload-required") return renderDualUploadRequiredPanel(panel);
  if (panel.customUI === "dual-upload") return renderMLDualUploadPanel(panel);
  if (panel.customUI === "fe-concat") return renderConcatPanel(panel);
  if (panel.customUI === "image-database") return renderImageDatabasePanel(panel);
  if (panel.customUI === "audio-dtw") return renderAudioDTWPanel(panel);
  if (panel.customUI === "morf-hmt") return renderMorfHMTPanel(panel);

  if (panel.feEndpoint) return renderFEPanel(panel, key);

  renderStandardPanel(panel);
}

// ============================================================
// HOME DASHBOARD
// ============================================================

const AI_LAB_CARDS = [
  { key: "ai-super-res", icon: "high_quality", title: "Super Resolusi Gaib", desc: "Tingkatkan detail tekstur ukiran dan artefak digital hingga 4x lipat menggunakan jaringan neural restoratif.", grad: "linear-gradient(135deg, rgba(242,202,80,0.35), rgba(20,20,20,0.9))" },
  { key: "ai-face-blur", icon: "face_retouching_natural", title: "Deteksi Wajah Spiritual", desc: "Analisis ekspresi dan struktur wajah dengan tingkat presisi tinggi untuk restorasi potret klasik.", grad: "linear-gradient(135deg, rgba(255,192,164,0.3), rgba(20,20,20,0.9))" },
  { key: "ai-artistic", icon: "palette", title: "Transfer Gaya Kamasan", desc: "Terapkan palet dan goresan gaya lukisan klasik Bali ke dalam fotografi modern secara instan.", grad: "linear-gradient(135deg, rgba(255,181,157,0.3), rgba(20,20,20,0.9))" },
  { key: "ai-object-detection", icon: "category", title: "Deteksi Objek Nirwana", desc: "Kenali hingga 20 kelas objek dalam citra menggunakan model deteksi berbasis neural network.", grad: "linear-gradient(135deg, rgba(249,155,109,0.3), rgba(20,20,20,0.9))" },
  { key: "ai-remove-bg", icon: "layers_clear", title: "Penghapus Latar Gaib", desc: "Pisahkan subjek utama dari latar belakang secara otomatis dalam sekali proses.", grad: "linear-gradient(135deg, rgba(242,202,80,0.25), rgba(20,20,20,0.9))" },
];

function goHome() { navigateTo("home"); }

function renderHomeDashboard() {
  panelTitleEl.textContent = "Home";
  panelDescEl.textContent = "Ringkasan aktivitas platform kamu.";
  imageSourceCard.style.display = "none";
  resultEl.style.display = "none";

  const hist = getHistory().slice(0, 2);
  const recentHTML = hist.length ? hist.map(it => `
    <div class="ws-recent-item">
      <div class="ws-recent-thumb"><img src="${it.after || it.before}"></div>
      <div class="ws-recent-info">
        <h4>${it.title}</h4>
        <p>${it.time}</p>
      </div>
      <span class="ws-recent-tag">Tersimpan</span>
    </div>`).join("") : `
    <div class="ws-recent-item">
      <div class="ws-recent-thumb ws-recent-thumb-placeholder"><span class="material-symbols-outlined">image</span></div>
      <div class="ws-recent-info">
        <h4>Belum ada riwayat</h4>
        <p>Proses gambar pertamamu untuk mulai mengisi riwayat.</p>
      </div>
    </div>`;

  controlsEl.innerHTML = `
    <div class="dash-hero">
      <h2>Rahajeng Rauh</h2>
      <p>Selamat datang di dashboard TaksuCitra. Salurkan energi kreatifmu lewat pemrosesan citra tingkat lanjut. Workspace kamu sudah siap.</p>
      <button class="btn-process" id="dash-start-project">Mulai Proyek Baru</button>
    </div>

    <div class="dash-bento">
      <div class="dash-tile" id="dash-tile-webcam">
        <div class="dash-tile-top">
          <span class="material-symbols-outlined dash-tile-icon">videocam</span>
          <span class="dash-chip">Aktif</span>
        </div>
        <h3>Live Webcam</h3>
        <p>Pemrosesan citra real-time dan penerapan filter melalui feed webcam.</p>
      </div>
      <div class="dash-tile dash-tile-wide" id="dash-tile-ailab">
        <div class="dash-tile-top">
          <span class="material-symbols-outlined dash-tile-icon">psychology</span>
          <span class="dash-chip dash-chip-gold">Beta</span>
        </div>
        <h3>AI Lab Unggulan</h3>
        <p>Jelajahi model neural network mutakhir untuk transfer gaya dan upscaling.</p>
      </div>
    </div>

    <div class="dash-cols">
      <div class="dash-panel">
        <div class="dash-panel-head">
          <h3><span class="material-symbols-outlined">brush</span> Workspace Terakhir</h3>
          <button class="dash-link" id="dash-open-history">Lihat Semua</button>
        </div>
        <div class="ws-recent-list">${recentHTML}</div>
      </div>
      <div class="dash-panel">
        <h3><span class="material-symbols-outlined">monitoring</span> Statistik Platform</h3>
        <div class="dash-stats">
          <div class="dash-stat">
            <div class="dash-stat-label">Total Gambar Diproses</div>
            <div class="dash-stat-value">${getHistory().length}</div>
          </div>
          <div class="dash-stat">
            <div class="dash-stat-label">Fitur Tersedia</div>
            <div class="dash-stat-value">${Object.keys(PANELS).length}</div>
          </div>
          <div class="dash-stat">
            <div class="dash-stat-label">Sesi Ini</div>
            <div class="dash-progress"><div class="dash-progress-bar" style="width:75%"></div></div>
          </div>
        </div>
      </div>
    </div>`;

  document.getElementById("dash-start-project").addEventListener("click", () => navigateTo("channel-split"));
  document.getElementById("dash-tile-webcam").addEventListener("click", () => navigateTo("mode-webcam"));
  document.getElementById("dash-tile-ailab").addEventListener("click", () => navigateTo("ai-lab-home"));
  document.getElementById("dash-open-history").addEventListener("click", openHistoryDrawer);
}

// ============================================================
// AI LAB UNGGULAN — Bento Gallery
// ============================================================

function renderAiLabBento() {
  panelTitleEl.textContent = "Lab Kecerdasan Buatan";
  panelDescEl.textContent = "Eksplorasi kemampuan komputasional tingkat lanjut yang dipadukan dengan filosofi visual Taksu.";
  imageSourceCard.style.display = "none";
  resultEl.style.display = "none";

  controlsEl.innerHTML = `
    <div class="ai-bento-grid">
      ${AI_LAB_CARDS.map(c => `
        <div class="ai-bento-card" data-panel="${c.key}" style="background:${c.grad}">
          <div class="magic-aura"></div>
          <div class="ai-bento-icon"><span class="material-symbols-outlined">${c.icon}</span></div>
          <h3>${c.title}</h3>
          <p>${c.desc}</p>
          <span class="ai-bento-cta">Aktifkan Model <span class="material-symbols-outlined">arrow_forward</span></span>
        </div>`).join("")}
    </div>`;

  controlsEl.querySelectorAll(".ai-bento-card").forEach(card => {
    card.addEventListener("click", () => navigateTo(card.dataset.panel));
  });
}

let currentHMTKernel = [
  [-1, -1, -1],
  [-1,  1, -1],
  [-1, -1, -1]
];

function resetHMTKernel() {
  currentHMTKernel = [
    [-1, -1, -1],
    [-1,  1, -1],
    [-1, -1, -1]
  ];
  const grid = document.getElementById("hmt-kernel-grid");
  if (grid) {
    grid.querySelectorAll(".kernel-cell-btn").forEach(btn => {
      const r = parseInt(btn.dataset.row, 10);
      const c = parseInt(btn.dataset.col, 10);
      const v = currentHMTKernel[r][c];
      btn.dataset.val = v;
      btn.textContent = v === 1 ? "+1" : (v === -1 ? "-1" : "0");
    });
  }
}

function renderMorfHMTPanel(panel) {
  let explainHTML = "";
  if (panel.helpKey && window.renderFeatureHelpBox) {
    explainHTML = window.renderFeatureHelpBox(panel.helpKey);
  } else if (panel.explain) {
    explainHTML = `<div class="explain-box">${panel.explain}</div>`;
  }

  const kernelGridHTML = `
    <div class="kernel-3x3-container" id="hmt-kernel-container" style="display:none;">
      <label>Editor Kernel Kustom 3x3 (Klik kotak untuk mengubah status: +1 Hit / -1 Miss / 0 Bebas)</label>
      <div class="kernel-3x3-grid" id="hmt-kernel-grid">
        ${[0, 1, 2].map(r => [0, 1, 2].map(c => `
          <button type="button" class="kernel-cell-btn" data-row="${r}" data-col="${c}" data-val="${currentHMTKernel[r][c]}">
            ${currentHMTKernel[r][c] === 1 ? "+1" : (currentHMTKernel[r][c] === -1 ? "-1" : "0")}
          </button>
        `).join("")).join("")}
      </div>
      <div class="field-hint">+1 (Hijau): Wajib objek foreground &bull; -1 (Merah): Wajib latar background &bull; 0 (Abu-abu): Don't care</div>
    </div>`;

  controlsEl.innerHTML = `
    <h3>Parameter</h3>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    ${kernelGridHTML}
    ${explainHTML}
    <div class="control-actions-row">
      <button class="btn-process" id="btn-process">Proses</button>
      <button class="btn-reset-params" id="btn-reset-params" type="button" title="Kembalikan parameter ke default bawaan">
        <span class="material-symbols-outlined" style="font-size: 16px;">restart_alt</span> Reset Parameter
      </button>
    </div>`;

  bindRangeLiveUpdate();
  resultEl.innerHTML = `<div class="placeholder">Atur parameter di atas lalu klik <strong>Proses</strong> untuk melihat hasil.</div>`;

  const modeSelect = controlsEl.querySelector('[data-field="mode"]');
  const kernelContainer = document.getElementById("hmt-kernel-container");
  function checkHMTMode() {
    if (modeSelect && kernelContainer) {
      kernelContainer.style.display = modeSelect.value === "custom" ? "block" : "none";
    }
  }
  if (modeSelect) modeSelect.addEventListener("change", checkHMTMode);
  checkHMTMode();

  controlsEl.querySelectorAll(".kernel-cell-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const r = parseInt(btn.dataset.row, 10);
      const c = parseInt(btn.dataset.col, 10);
      let v = parseInt(btn.dataset.val, 10);
      if (v === 1) v = -1;
      else if (v === -1) v = 0;
      else v = 1;
      currentHMTKernel[r][c] = v;
      btn.dataset.val = v;
      btn.textContent = v === 1 ? "+1" : (v === -1 ? "-1" : "0");
    });
  });

  document.getElementById("btn-process").addEventListener("click", () => runStandardProcess(currentPanelKey));
  document.getElementById("btn-reset-params").addEventListener("click", () => resetPanelParams(panel));
}

function renderStandardPanel(panel) {
  let extraSourceHTML = "";
  if (panel.customImageSource === "dual-sample") {
    extraSourceHTML = `<div class="control-field" style="margin-top:14px"><label>Gambar 2</label>
      <select id="second-sample-select" data-field="__second_sample">
        ${window.__SAMPLES__.map(s => `<option value="${s}" ${s === secondSampleValue ? "selected" : ""}>${s}</option>`).join("")}
      </select></div>`;
  }

  let explainHTML = "";
  if (panel.helpKey && window.renderFeatureHelpBox) {
    explainHTML = window.renderFeatureHelpBox(panel.helpKey);
  } else if (panel.explain) {
    explainHTML = `<div class="explain-box">${panel.explain}</div>`;
  }

  controlsEl.innerHTML = `
    <h3>Parameter</h3>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    ${extraSourceHTML}
    ${explainHTML}
    <div class="control-actions-row">
      <button class="btn-process" id="btn-process">Proses</button>
      <button class="btn-reset-params" id="btn-reset-params" type="button" title="Kembalikan parameter ke default bawaan">
        <span class="material-symbols-outlined" style="font-size: 16px;">restart_alt</span> Reset Parameter
      </button>
    </div>`;

  bindRangeLiveUpdate();
  resultEl.innerHTML = `<div class="placeholder">Atur parameter di atas lalu klik <strong>Proses</strong> untuk melihat hasil.</div>`;
  document.getElementById("btn-process").addEventListener("click", () => runStandardProcess(currentPanelKey));
  document.getElementById("btn-reset-params").addEventListener("click", () => resetPanelParams(panel));
}

function resetPanelParams(panel) {
  if (!panel || !panel.fields) return;
  panel.fields.forEach(f => {
    const el = controlsEl.querySelector(`[data-field="${f.name}"]`);
    if (!el) return;
    if (el.type === "checkbox") {
      el.checked = f.value !== undefined ? !!f.value : false;
    } else {
      el.value = f.value !== undefined ? f.value : (f.default !== undefined ? f.default : "");
      if (el.type === "range") {
        const badge = controlsEl.querySelector(`[data-range-value="${f.name}"]`);
        if (badge) badge.textContent = el.value;
      }
    }
  });
  if (panel.customUI === "morf-hmt") {
    resetHMTKernel();
  }
}

async function runStandardProcess(key) {
  const panel = PANELS[key];
  const btn = document.getElementById("btn-process");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Memproses...`;

  let fieldValues = collectFieldValues();
  if (key === "morf-hit-or-miss" && fieldValues.mode === "custom") {
    fieldValues.custom_kernel = currentHMTKernel;
  }
  if (panel.parseFields) fieldValues = panel.parseFields(fieldValues);
  const payload = { ...fieldValues };

  if (uploadedImageB64) payload.image = uploadedImageB64;
  else payload.sample = getSelectedSample();

  if (panel.customImageSource === "dual-sample") {
    payload.sample1 = payload.sample;
    payload.sample2 = fieldValues.__second_sample || secondSampleValue;
    delete payload.sample;
    if (uploadedImageB64) { payload.image1 = uploadedImageB64; delete payload.image; }
  }

  try {
    const res = await fetch(panel.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    if (!res.ok) { const errData = await res.json().catch(() => ({})); throw new Error(errData.error || `Server error (${res.status})`); }
    const data = await res.json();
    resultEl.innerHTML = panel.render(data);
    (panel.sliderIds || []).forEach(initBeforeAfterSlider);
    bindSaveHistoryButton(panel, data);
  } catch (err) {
    resultEl.innerHTML = `<div class="error-box">Gagal memproses: ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Proses";
  }
}

function bindSaveHistoryButton(panel, data) {
  const btn = resultEl.querySelector("[data-save-panel]");
  if (!btn) return;
  btn.addEventListener("click", () => {
    addHistoryItem({
      title: panel.title, desc: panel.desc,
      before: data.original || data.source || data.image1 || null,
      after: data.result || data.transparent || null,
    });
    btn.textContent = "Tersimpan ke Riwayat ✓";
    btn.disabled = true;
  });
}

// ============================================================
// FEATURE EXTRACTION LAB: panel generik 1-citra (lewat /api/fe/run)
// ============================================================

function renderFEPanel(panel, key) {
  controlsEl.innerHTML = `
    <h3>Parameter</h3>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    <div class="explain-box">${panel.explain}</div>
    <button class="btn-process" id="btn-process">Proses</button>`;

  bindRangeLiveUpdate();
  resultEl.innerHTML = `<div class="placeholder">Atur parameter di atas lalu klik <strong>Proses</strong> untuk melihat proses bertahap ekstraksi fitur.</div>`;
  document.getElementById("btn-process").addEventListener("click", () => runFEProcess(key));
}

async function runFEProcess(key) {
  const panel = PANELS[key];
  const btn = document.getElementById("btn-process");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span> Memproses...`;

  let fieldValues = collectFieldValues();
  if (panel.parseFields) fieldValues = panel.parseFields(fieldValues);

  const payload = { panel: panel.fePanel, params: fieldValues };
  if (uploadedImageB64) payload.image = uploadedImageB64;
  else payload.sample = getSelectedSample();

  try {
    const res = await fetch(panel.feEndpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
    resultEl.innerHTML = renderFEStepGallery(data);
  } catch (err) {
    resultEl.innerHTML = `<div class="error-box">Gagal memproses: ${err.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Proses";
  }
}

// ============================================================
// FEATURE EXTRACTION LAB: Dual-upload WAJIB (Revisi 3)
// Matching Keypoint & Matching Engine -- tidak boleh pakai sample,
// kedua gambar harus di-upload user sendiri.
// ============================================================

function renderDualUploadRequiredPanel(panel) {
  controlsEl.innerHTML = `
    <h3>Upload Dua Gambar (Wajib)</h3>
    <p style="font-size:12.5px;color:var(--text-dim);margin-top:-6px">Kedua gambar wajib di-upload sendiri -- fitur ini tidak memakai sample.</p>
    <div class="dual-upload-wrap">
      <label class="dual-upload-slot" id="fe-dual-slot-1">
        <input type="file" id="fe-dual-upload-1" accept="image/*">
        <div class="du-label">Gambar 1</div>
        <div class="du-hint">Klik untuk upload</div>
      </label>
      <label class="dual-upload-slot" id="fe-dual-slot-2">
        <input type="file" id="fe-dual-upload-2" accept="image/*">
        <div class="du-label">Gambar 2</div>
        <div class="du-hint">Klik untuk upload</div>
      </label>
    </div>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    <div class="explain-box">${panel.explain}</div>
    <button class="btn-process" id="btn-process" style="margin-top:12px" disabled>Proses</button>`;

  bindRangeLiveUpdate();
  resultEl.innerHTML = `<div class="placeholder">Upload kedua gambar lalu klik Proses.</div>`;

  let img1B64 = null, img2B64 = null;
  const slot1 = document.getElementById("fe-dual-slot-1");
  const slot2 = document.getElementById("fe-dual-slot-2");
  const btn = document.getElementById("btn-process");

  function readAsB64(file, onDone) {
    const reader = new FileReader();
    reader.onload = (ev) => onDone(ev.target.result);
    reader.readAsDataURL(file);
  }

  function fillSlot(slotEl, b64) {
    slotEl.classList.add("filled");
    slotEl.querySelector(".du-hint").textContent = "Klik untuk ganti gambar";
    let img = slotEl.querySelector("img");
    if (!img) { img = document.createElement("img"); slotEl.appendChild(img); }
    img.src = b64;
  }

  document.getElementById("fe-dual-upload-1").addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    readAsB64(e.target.files[0], (b64) => { img1B64 = b64; fillSlot(slot1, b64); btn.disabled = !(img1B64 && img2B64); });
  });
  document.getElementById("fe-dual-upload-2").addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    readAsB64(e.target.files[0], (b64) => { img2B64 = b64; fillSlot(slot2, b64); btn.disabled = !(img1B64 && img2B64); });
  });

  btn.addEventListener("click", async () => {
    if (!img1B64 || !img2B64) { alert("Upload kedua gambar terlebih dahulu."); return; }
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Memproses...`;

    let fieldValues = collectFieldValues();
    if (panel.parseFields) fieldValues = panel.parseFields(fieldValues);
    const payload = { image1: img1B64, image2: img2B64, ...fieldValues };

    try {
      const res = await fetch(panel.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
      resultEl.innerHTML = renderFEStepGallery(data);
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">${err.message}</div>`;
    } finally {
      btn.disabled = false; btn.textContent = "Proses";
    }
  });
}

// ============================================================
// MATCHING LAB: Dual-Upload Flexible (Upload / Sample)
// ============================================================

function renderMLDualUploadPanel(panel) {
  controlsEl.innerHTML = `
    <h3>Pilih atau Upload Citra A & B</h3>
    <div class="dual-upload-wrap">
      <label class="dual-upload-slot" id="ml-dual-slot-1">
        <input type="file" id="ml-dual-upload-1" accept="image/*">
        <div class="du-label">Citra A / Referensi</div>
        <div class="du-hint">Klik untuk upload (atau pilih sample di bawah)</div>
      </label>
      <label class="dual-upload-slot" id="ml-dual-slot-2">
        <input type="file" id="ml-dual-upload-2" accept="image/*">
        <div class="du-label">Citra B / Target</div>
        <div class="du-hint">Klik untuk upload (atau pilih sample di bawah)</div>
      </label>
    </div>
    <div class="control-row">
      <div class="control-field">
        <label>Sample Citra A</label>
        <select id="ml-sample-select-1">
          <option value="">-- Gunakan Upload --</option>
          ${window.__SAMPLES__.map(s => `<option value="${s}">${s}</option>`).join("")}
        </select>
      </div>
      <div class="control-field">
        <label>Sample Citra B</label>
        <select id="ml-sample-select-2">
          <option value="">-- Gunakan Upload --</option>
          ${window.__SAMPLES__.map((s, i) => `<option value="${s}" ${i===1?"selected":""}>${s}</option>`).join("")}
        </select>
      </div>
    </div>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    <div class="explain-box">${panel.explain}</div>
    <button class="btn-process" id="btn-process" style="margin-top:12px">Jalankan Pipeline</button>`;

  bindRangeLiveUpdate();
  resultEl.innerHTML = `<div class="placeholder">Pilih citra A & B lalu klik "Jalankan Pipeline" untuk melihat langkah-langkah komputasi.</div>`;

  let img1B64 = null, img2B64 = null;
  const slot1 = document.getElementById("ml-dual-slot-1");
  const slot2 = document.getElementById("ml-dual-slot-2");
  const sample1 = document.getElementById("ml-sample-select-1");
  const sample2 = document.getElementById("ml-sample-select-2");
  const btn = document.getElementById("btn-process");

  function readAsB64(file, onDone) {
    const reader = new FileReader();
    reader.onload = (ev) => onDone(ev.target.result);
    reader.readAsDataURL(file);
  }

  function fillSlot(slotEl, b64) {
    slotEl.classList.add("filled");
    slotEl.querySelector(".du-hint").textContent = "Klik untuk ganti gambar";
    let img = slotEl.querySelector("img");
    if (!img) { img = document.createElement("img"); slotEl.appendChild(img); }
    img.src = b64;
  }

  document.getElementById("ml-dual-upload-1").addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    readAsB64(e.target.files[0], (b64) => { img1B64 = b64; fillSlot(slot1, b64); sample1.value = ""; });
  });

  document.getElementById("ml-dual-upload-2").addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    readAsB64(e.target.files[0], (b64) => { img2B64 = b64; fillSlot(slot2, b64); sample2.value = ""; });
  });

  sample1.addEventListener("change", () => {
    if (sample1.value) {
      img1B64 = null;
      slot1.classList.remove("filled");
      const img = slot1.querySelector("img");
      if (img) img.remove();
      slot1.querySelector(".du-hint").textContent = `Sample: ${sample1.value}`;
    }
  });

  sample2.addEventListener("change", () => {
    if (sample2.value) {
      img2B64 = null;
      slot2.classList.remove("filled");
      const img = slot2.querySelector("img");
      if (img) img.remove();
      slot2.querySelector(".du-hint").textContent = `Sample: ${sample2.value}`;
    }
  });

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Memproses Pipeline...`;

    let fieldValues = collectFieldValues();
    const payload = { params: fieldValues };

    if (img1B64) payload.image = img1B64;
    else if (sample1.value) payload.sample = sample1.value;
    else payload.sample = window.__SAMPLES__[0];

    if (img2B64) payload.image_b = img2B64;
    else if (sample2.value) payload.sample_b = sample2.value;
    else payload.sample_b = window.__SAMPLES__[1] || window.__SAMPLES__[0];

    try {
      const res = await fetch(panel.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      resultEl.innerHTML = panel.render ? panel.render(data) : renderMLGenericStepPipeline(data);
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">Gagal memproses: ${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Jalankan Pipeline";
    }
  });
}

// ============================================================
// §18: PENCOCOKAN CITRA 1-VS-DATABASE (1-TO-N)
// ============================================================

function renderImageDatabasePanel(panel) {
  let dbMode = "database"; // "database" (1-to-N) atau "1vs1"
  let queryB64 = null;
  let queryFilename = "";
  let customDatabaseImages = []; // Array of { filename: "...", image: "data:image/..." }
  let useSystemDb = true;

  let img1B64_1v1 = null, img2B64_1v1 = null;

  controlsEl.innerHTML = `
    <!-- Mode Switcher -->
    <div class="ml-mode-bar">
      <button class="ml-mode-btn active" id="btn-mode-db" type="button">
        <span class="material-symbols-outlined">inventory_2</span>
        Mode B: Pencocokan 1-vs-Database (1-to-N)
      </button>
      <button class="ml-mode-btn" id="btn-mode-1vs1" type="button">
        <span class="material-symbols-outlined">compare</span>
        Mode A: Pencocokan 1-vs-1 (Dua Gambar)
      </button>
    </div>

    <!-- Panel Mode Database (1-to-N) -->
    <div id="panel-mode-db-container">
      <div style="display: grid; grid-template-columns: 280px 1fr; gap: 16px; margin-bottom: 14px;">
        <!-- 1. Input Citra Uji (Query Image) -->
        <div class="ml-query-box">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; font-size: 13px; color: var(--accent); display: flex; align-items: center; gap: 6px;">
              <span class="material-symbols-outlined">search</span> 1. Citra Uji (Query)
            </h4>
            <button id="btn-clear-query" type="button" class="ml-btn-sec" style="padding: 2px 8px; font-size: 11px; display: none;">Hapus</button>
          </div>
          
          <div class="ml-query-slot" id="db-query-slot">
            <span class="material-symbols-outlined" style="font-size: 32px; color: var(--text-faint);">cloud_upload</span>
            <div class="du-hint">Klik atau seret 1 citra uji (JPG/PNG) ke sini</div>
            <input type="file" id="db-query-file" accept="image/*" style="display:none">
          </div>

          <div style="display: flex; flex-direction: column; gap: 4px;">
            <label style="font-size: 11px; color: var(--text-dim);">Atau Pilih Sampel Bawaan:</label>
            <select id="db-query-sample" style="font-size: 11.5px; padding: 6px 8px;">
              <option value="" disabled selected>-- Pilih Sampel Alternatif --</option>
              <option value="db_taksu_logo.jpg">db_taksu_logo.jpg (Logo Taksu Asli)</option>
              <option value="db_taksu_rot15.jpg">db_taksu_rot15.jpg (Logo Rotasi 15°)</option>
              <option value="db_taksu_dark.jpg">db_taksu_dark.jpg (Logo Variasi Gelap)</option>
              <option value="db_texture_grid.jpg">db_texture_grid.jpg (Pola Grid)</option>
              <option value="db_pattern_stars.jpg">db_pattern_stars.jpg (Pola Bintang)</option>
              <option value="sample1_shapes.jpg">sample1_shapes.jpg</option>
              <option value="sample2_face.jpg">sample2_face.jpg</option>
            </select>
          </div>
        </div>

        <!-- 2. Parameter & Konfigurasi Pencocokan -->
        <div style="display: flex; flex-direction: column; gap: 10px;">
          <div class="control-row">
            <div class="control-field">
              <label>Metode Representasi Fitur</label>
              <select id="db-param-method">
                <option value="orb" selected>ORB (Biner - Hamming + Lowe Ratio Test)</option>
                <option value="sift">SIFT (Float 128D - Euclidean L2 + FLANN)</option>
                <option value="deep_embedding">Deep Learned Embedding (Cosine Distance)</option>
                <option value="color_hist">Color Histogram (3D HSV Bhattacharyya)</option>
                <option value="hu_moments">Hu Moments Invariant (Bentuk Kontur)</option>
              </select>
            </div>
            <div class="control-field">
              <label>Jumlah Hasil (Top-K)</label>
              <select id="db-param-topk">
                <option value="3">Top-3 Hasil Teratas</option>
                <option value="5" selected>Top-5 Hasil Teratas</option>
                <option value="10">Top-10 Hasil Teratas</option>
              </select>
            </div>
          </div>

          <div class="control-row">
            <div class="control-field" style="flex: 2;">
              <label>Nilai Ambang Kemiripan Minimum (Threshold): <span id="db-thresh-val" style="color:var(--accent); font-weight:700;">50%</span></label>
              <input type="range" id="db-param-thresh" min="10" max="95" step="5" value="50">
            </div>
            <div class="control-field" style="flex: 1; display:flex; align-items:center; gap:8px; margin-top:16px;">
              <input type="checkbox" id="db-param-flann">
              <label for="db-param-flann" style="font-size: 11.5px; margin:0; cursor:pointer;">FLANN Matcher (SIFT)</label>
            </div>
          </div>

          <div class="explain-box" style="font-size: 12px; margin: 2px 0;">
            💡 <strong>Alur 1-to-N:</strong> Citra uji dicocokkan ke seluruh citra di database menggunakan metrik jarak yang sesuai. Pasangan yang memenuhi batas ambang ditandai <strong>MATCH</strong>.
          </div>
        </div>
      </div>

      <!-- 2. Kumpulan Database Target: User Upload Folder / File Sendiri -->
      <div class="ml-folder-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <h4 style="margin: 0; font-size: 13px; color: var(--accent); display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined">folder_special</span>
            2. Sumber Database Citra Target (Folder / Kumpulan File)
          </h4>
          <span id="db-badge-count" class="ml-badge-folder">
            <span class="material-symbols-outlined" style="font-size: 16px;">database</span>
            Database Bawaan Sistem (11 Citra)
          </span>
        </div>

        <div class="ml-folder-actions">
          <!-- Tombol Upload Folder -->
          <button class="ml-btn-gold" id="btn-pick-folder" type="button" style="padding: 8px 14px; font-size: 12px;">
            <span class="material-symbols-outlined" style="font-size: 18px;">create_new_folder</span>
            Pilih Folder Citra (a, b, c, ...)
          </button>
          <input type="file" id="db-folder-input" webkitdirectory directory multiple accept="image/*" style="display: none;">

          <!-- Tombol Upload Multiple Files -->
          <button class="ml-btn-sec" id="btn-pick-files" type="button" style="padding: 8px 14px; font-size: 12px;">
            <span class="material-symbols-outlined" style="font-size: 18px;">photo_library</span>
            Pilih Beberapa File Citra
          </button>
          <input type="file" id="db-files-input" multiple accept="image/*" style="display: none;">

          <!-- Tombol Reset / Gunakan Database Bawaan -->
          <button class="ml-btn-sec" id="btn-reset-db" type="button" style="padding: 8px 12px; font-size: 12px; display: none;">
            <span class="material-symbols-outlined" style="font-size: 16px;">restore</span>
            Kembalikan ke Database Sistem
          </button>
        </div>

        <!-- Preview Strip Gambar Database yang Diunggah Pengguna -->
        <div id="db-preview-container" style="display: none;">
          <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 4px;">Koleksi citra database yang dimuat:</div>
          <div class="ml-db-preview-strip" id="db-preview-strip"></div>
        </div>
      </div>

      <!-- Tombol Aksi Utama -->
      <div style="display: flex; gap: 10px; margin-top: 14px;">
        <button class="btn-process" id="btn-db-search" style="flex: 2;">
          <span class="material-symbols-outlined">manage_search</span> Cari di Database (1-to-N)
        </button>
      </div>
    </div>

    <!-- Panel Mode 1-vs-1 (Dua Gambar) -->
    <div id="panel-mode-1vs1-container" style="display: none;">
      <div class="ml-audio-dual-wrap">
        <!-- Slot Citra A -->
        <div class="ml-query-box">
          <h4 style="margin: 0; font-size: 13px; color: var(--accent);"><span class="material-symbols-outlined">image</span> Citra A (Masukan 1)</h4>
          <div class="ml-query-slot" id="slot-1v1-a">
            <span class="material-symbols-outlined" style="font-size: 32px; color: var(--text-faint);">upload</span>
            <div class="du-hint">Klik/tarik Citra A</div>
            <input type="file" id="file-1v1-a" accept="image/*" style="display:none">
          </div>
          <select id="sample-1v1-a" style="font-size: 11px;">
            <option value="db_taksu_logo.jpg" selected>Sampel: db_taksu_logo.jpg</option>
            <option value="sample1_shapes.jpg">Sampel: sample1_shapes.jpg</option>
            <option value="sample2_face.jpg">Sampel: sample2_face.jpg</option>
          </select>
        </div>

        <!-- Slot Citra B -->
        <div class="ml-query-box">
          <h4 style="margin: 0; font-size: 13px; color: var(--saffron);"><span class="material-symbols-outlined">image</span> Citra B (Masukan 2)</h4>
          <div class="ml-query-slot" id="slot-1v1-b">
            <span class="material-symbols-outlined" style="font-size: 32px; color: var(--text-faint);">upload</span>
            <div class="du-hint">Klik/tarik Citra B</div>
            <input type="file" id="file-1v1-b" accept="image/*" style="display:none">
          </div>
          <select id="sample-1v1-b" style="font-size: 11px;">
            <option value="db_taksu_rot15.jpg" selected>Sampel: db_taksu_rot15.jpg (Rotasi 15°)</option>
            <option value="db_taksu_dark.jpg">Sampel: db_taksu_dark.jpg</option>
            <option value="sample3_edges.jpg">Sampel: sample3_edges.jpg</option>
          </select>
        </div>
      </div>

      <div class="control-row" style="margin-top: 10px;">
        <div class="control-field">
          <label>Detektor & Descriptor</label>
          <select id="method-1v1">
            <option value="ORB" selected>ORB (Biner - Hamming)</option>
            <option value="SIFT">SIFT (Float - L2)</option>
          </select>
        </div>
      </div>
      <button class="btn-process" id="btn-process-1v1" style="margin-top: 10px;">
        <span class="material-symbols-outlined">compare</span> Jalankan Pencocokan 1-vs-1
      </button>
    </div>`;

  resultEl.innerHTML = `<div class="placeholder">Unggah citra uji dan folder database, lalu klik <strong>Cari di Database</strong>.</div>`;

  // Helper file reader
  function readFileAsB64(file, callback) {
    const reader = new FileReader();
    reader.onload = (ev) => callback(ev.target.result);
    reader.readAsDataURL(file);
  }

  // Tampilan Slot Citra Uji Rapi
  const qSlot = document.getElementById("db-query-slot");
  const qFile = document.getElementById("db-query-file");
  const qSample = document.getElementById("db-query-sample");
  const btnClearQuery = document.getElementById("btn-clear-query");

  function setQueryImage(b64, name) {
    queryB64 = b64;
    queryFilename = name;
    qSlot.classList.add("filled");
    qSlot.innerHTML = `
      <img src="${b64}" alt="${name}">
      <span class="du-overlay-hint">Klik untuk mengganti (${name})</span>
    `;
    btnClearQuery.style.display = "inline-block";
    qSample.value = "";
  }

  function clearQueryImage() {
    queryB64 = null;
    queryFilename = "";
    qSlot.classList.remove("filled");
    qSlot.innerHTML = `
      <span class="material-symbols-outlined" style="font-size: 32px; color: var(--text-faint);">cloud_upload</span>
      <div class="du-hint">Klik atau seret 1 citra uji (JPG/PNG) ke sini</div>
    `;
    btnClearQuery.style.display = "none";
  }

  qSlot.addEventListener("click", () => qFile.click());
  qFile.addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    const file = e.target.files[0];
    readFileAsB64(file, (b64) => setQueryImage(b64, file.name));
  });

  btnClearQuery.addEventListener("click", (e) => {
    e.stopPropagation();
    clearQueryImage();
  });

  qSample.addEventListener("change", () => {
    if (qSample.value) {
      clearQueryImage();
      qSlot.innerHTML = `
        <span class="material-symbols-outlined" style="font-size: 32px; color: var(--accent);">check_circle</span>
        <div class="du-hint">Sampel terpilih: <strong>${qSample.value}</strong></div>
      `;
    }
  });

  // Slider Threshold
  const threshSlider = document.getElementById("db-param-thresh");
  const threshValLabel = document.getElementById("db-thresh-val");
  threshSlider.addEventListener("input", () => {
    threshValLabel.textContent = `${threshSlider.value}%`;
  });

  // Mode Toggle Events
  const btnModeDb = document.getElementById("btn-mode-db");
  const btnMode1vs1 = document.getElementById("btn-mode-1vs1");
  const pModeDb = document.getElementById("panel-mode-db-container");
  const pMode1vs1 = document.getElementById("panel-mode-1vs1-container");

  btnModeDb.addEventListener("click", () => {
    dbMode = "database";
    btnModeDb.classList.add("active");
    btnMode1vs1.classList.remove("active");
    pModeDb.style.display = "block";
    pMode1vs1.style.display = "none";
  });

  btnMode1vs1.addEventListener("click", () => {
    dbMode = "1vs1";
    btnMode1vs1.classList.add("active");
    btnModeDb.classList.remove("active");
    pModeDb.style.display = "none";
    pMode1vs1.style.display = "block";
  });

  // Folder & Multiple Files Upload untuk Database Target
  const btnPickFolder = document.getElementById("btn-pick-folder");
  const btnPickFiles = document.getElementById("btn-pick-files");
  const folderInput = document.getElementById("db-folder-input");
  const filesInput = document.getElementById("db-files-input");
  const badgeDbCount = document.getElementById("db-badge-count");
  const btnResetDb = document.getElementById("btn-reset-db");
  const previewContainer = document.getElementById("db-preview-container");
  const previewStrip = document.getElementById("db-preview-strip");

  btnPickFolder.addEventListener("click", () => folderInput.click());
  btnPickFiles.addEventListener("click", () => filesInput.click());

  function processSelectedDatabaseFiles(files) {
    if (!files || files.length === 0) return;
    const imgFiles = Array.from(files).filter(f => f.type.startsWith("image/") || /\.(jpg|jpeg|png|webp|bmp)$/i.test(f.name));
    if (imgFiles.length === 0) {
      alert("Tidak ada file gambar yang valid dalam folder yang dipilih.");
      return;
    }

    customDatabaseImages = [];
    useSystemDb = false;
    previewStrip.innerHTML = "";
    previewContainer.style.display = "block";
    btnResetDb.style.display = "inline-flex";

    badgeDbCount.innerHTML = `
      <span class="material-symbols-outlined" style="font-size: 16px;">cloud_sync</span>
      Memproses ${imgFiles.length} citra...
    `;

    let loadedCount = 0;
    imgFiles.forEach((file) => {
      readFileAsB64(file, (b64) => {
        customDatabaseImages.push({ filename: file.name, image: b64 });
        
        // Buat mini thumbnail
        const thumbDiv = document.createElement("div");
        thumbDiv.className = "ml-db-preview-thumb";
        thumbDiv.title = file.name;
        thumbDiv.innerHTML = `<img src="${b64}"><div class="thumb-tag">${file.name}</div>`;
        previewStrip.appendChild(thumbDiv);

        loadedCount++;
        if (loadedCount === imgFiles.length) {
          badgeDbCount.innerHTML = `
            <span class="material-symbols-outlined" style="font-size: 16px; color:#64d2ff;">check_circle</span>
            ✓ ${customDatabaseImages.length} Citra Database Siap
          `;
        }
      });
    });
  }

  folderInput.addEventListener("change", (e) => processSelectedDatabaseFiles(e.target.files));
  filesInput.addEventListener("change", (e) => processSelectedDatabaseFiles(e.target.files));

  btnResetDb.addEventListener("click", () => {
    customDatabaseImages = [];
    useSystemDb = true;
    previewContainer.style.display = "none";
    previewStrip.innerHTML = "";
    btnResetDb.style.display = "none";
    badgeDbCount.innerHTML = `
      <span class="material-symbols-outlined" style="font-size: 16px;">database</span>
      Database Bawaan Sistem (11 Citra)
    `;
    folderInput.value = "";
    filesInput.value = "";
  });

  // Mode 1-vs-1 Slot handling
  const slot1v1A = document.getElementById("slot-1v1-a");
  const slot1v1B = document.getElementById("slot-1v1-b");
  const file1v1A = document.getElementById("file-1v1-a");
  const file1v1B = document.getElementById("file-1v1-b");
  const sample1v1A = document.getElementById("sample-1v1-a");
  const sample1v1B = document.getElementById("sample-1v1-b");

  slot1v1A.addEventListener("click", () => file1v1A.click());
  slot1v1B.addEventListener("click", () => file1v1B.click());

  function set1v1Slot(slotEl, b64, label) {
    slotEl.classList.add("filled");
    slotEl.innerHTML = `<img src="${b64}"><span class="du-overlay-hint">${label}</span>`;
  }

  file1v1A.addEventListener("change", (e) => {
    if (e.target.files[0]) readFileAsB64(e.target.files[0], (b64) => { img1B64_1v1 = b64; set1v1Slot(slot1v1A, b64, e.target.files[0].name); sample1v1A.value = ""; });
  });
  file1v1B.addEventListener("change", (e) => {
    if (e.target.files[0]) readFileAsB64(e.target.files[0], (b64) => { img2B64_1v1 = b64; set1v1Slot(slot1v1B, b64, e.target.files[0].name); sample1v1B.value = ""; });
  });

  // Tombol Cari di Database (1-to-N)
  document.getElementById("btn-db-search").addEventListener("click", async () => {
    const btn = document.getElementById("btn-db-search");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Mencocokkan 1-ke-Banyak...`;

    const payload = {
      params: {
        method: document.getElementById("db-param-method").value,
        threshold: parseFloat(threshSlider.value) / 100.0,
        top_k: parseInt(document.getElementById("db-param-topk").value),
        use_flann: document.getElementById("db-param-flann").checked,
        custom_database: (!useSystemDb && customDatabaseImages.length > 0) ? customDatabaseImages : null
      }
    };

    if (queryB64) {
      payload.image = queryB64;
    } else if (qSample.value) {
      payload.sample = qSample.value;
    } else {
      payload.sample = "db_taksu_logo.jpg";
    }

    try {
      const res = await fetch("/api/matching/database", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Gagal memproses pencocokan database.");

      // Render Galeri Khusus 1-ke-Banyak + Pipeline Lengkap
      let customHTML = `
        <div class="ml-container">
          <!-- Banner Keputusan -->
          <div class="ml-verdict-banner" style="background: ${data.result.metrics.matched_count > 0 ? "rgba(100,210,255,0.12)" : "rgba(255,100,100,0.12)"}; border-color: ${data.result.metrics.matched_count > 0 ? "#64d2ff" : "#ff7777"}; color: ${data.result.metrics.matched_count > 0 ? "#64d2ff" : "#ff9999"};">
            <span class="material-symbols-outlined">${data.result.metrics.matched_count > 0 ? "verified" : "error"}</span>
            <span>${data.summary}</span>
            <span style="margin-left: auto; font-family: var(--mono); font-size: 11px;">${data.timing_ms} ms</span>
          </div>

          <!-- Tampilan Antarmuka Dua Kolom: Citra Uji vs Galeri Hasil Teratas -->
          <div class="ml-db-layout">
            <!-- Kolom Kiri: Citra Uji -->
            <div class="ml-db-query-card">
              <h4><span class="material-symbols-outlined">target</span> Citra Uji (Query)</h4>
              <div class="ml-db-query-thumb">
                <img src="${data.query_thumb || (queryB64 || '')}">
              </div>
              <div style="font-size: 11.5px; color: var(--text-dim);">
                <div>Metode: <strong>${data.method.toUpperCase()}</strong></div>
                <div>Ambang Batas: <strong>${data.threshold}%</strong></div>
                <div>Basis Data: <strong>${!useSystemDb && customDatabaseImages.length > 0 ? "Folder Pengguna (" + customDatabaseImages.length + " citra)" : "Database Sistem (11 citra)"}</strong></div>
                <div>Lolos Ambang: <strong style="color:var(--accent);">${data.result.metrics.matched_count} dari ${data.all_results_count}</strong></div>
              </div>
            </div>

            <!-- Kolom Kanan: Galeri Top-K Hasil Database -->
            <div class="ml-db-results-container">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; font-size: 13px; color: var(--accent);">
                  Galeri Hasil Teratas (Top-${data.top_candidates.length} Hasil)
                </h4>
                <span style="font-size: 11px; color: var(--text-dim);">Urutan skor kemiripan tertinggi</span>
              </div>
              <div class="ml-db-grid">
                ${data.top_candidates.map((c, idx) => `
                  <div class="ml-db-card ${c.threshold_met ? "matched" : "non-matched"}">
                    <span class="ml-db-rank">#${idx + 1}</span>
                    <div class="ml-db-thumb-wrapper">
                      <img src="${c.thumb_b64}" alt="${c.filename}">
                    </div>
                    <div class="ml-db-name" title="${c.filename}">${c.filename}</div>
                    <div class="ml-db-meta">
                      <span>Kemiripan:</span>
                      <span class="ml-db-score-val">${c.score}%</span>
                    </div>
                    <div class="ml-db-meta">
                      <span>Jarak:</span>
                      <span>${c.distance}</span>
                    </div>
                    <div class="ml-badge-status ${c.threshold_met ? "match" : "nonmatch"}">
                      ${c.threshold_met ? "✓ MATCH" : "✗ NON-MATCH"}
                    </div>
                  </div>
                `).join("")}
              </div>
            </div>
          </div>

          <!-- Bagian Detail Stepper Pipeline Lengkap -->
          <div style="margin-top: 20px;">
            <h4 style="color: var(--accent); margin-bottom: 8px;">Tahapan Proses Ekstraksi & Pencocokan Sistem</h4>
            ${renderMLGenericStepPipeline(data)}
          </div>
        </div>`;

      resultEl.innerHTML = customHTML;
      if (typeof setupMLStepperEvents === "function") setupMLStepperEvents();

    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">Gagal memproses pencocokan 1-ke-Banyak: ${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span class="material-symbols-outlined">manage_search</span> Cari di Database (1-to-N)`;
    }
  });

  // Tombol Mode 1-vs-1
  document.getElementById("btn-process-1v1").addEventListener("click", async () => {
    const btn = document.getElementById("btn-process-1v1");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Menjalankan 1-vs-1...`;

    const payload = {
      params: {
        method: document.getElementById("method-1v1").value,
        ratio_threshold: 0.75,
        cross_check: true
      }
    };
    if (img1B64_1v1) payload.image = img1B64_1v1;
    else if (sample1v1A.value) payload.sample = sample1v1A.value;
    else payload.sample = "db_taksu_logo.jpg";

    if (img2B64_1v1) payload.image_b = img2B64_1v1;
    else if (sample1v1B.value) payload.sample_b = sample1v1B.value;
    else payload.sample_b = "db_taksu_rot15.jpg";

    try {
      const res = await fetch("/api/matching/local", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      resultEl.innerHTML = renderMLGenericStepPipeline(data);
      if (typeof setupMLStepperEvents === "function") setupMLStepperEvents();
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">Gagal memproses 1-vs-1: ${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span class="material-symbols-outlined">compare</span> Jalankan Pencocokan 1-vs-1`;
    }
  });
}


// ============================================================
// §19: PENCOCOKAN SUARA DYNAMIC TIME WARPING (DTW)
// ============================================================

function renderAudioDTWPanel(panel) {
  let audioMode = "database"; // "database" atau "1vs1"
  let audioObjective = "keyword_spotting"; // "keyword_spotting" atau "speaker_verification"

  // Mode B: Audio Uji & Kumpulan Database Audio
  let queryAudioB64 = null;
  let customDatabaseAudios = []; // Array of { filename: "...", audio: "data:audio/..." }
  let useSystemAudioDb = true;

  // Mode A: Audio 1-vs-1 (Audio A vs Audio B)
  let audioA_B64 = null, audioB_B64 = null;
  let audioA_Name = "Audio A", audioB_Name = "Audio B";

  // Recorder state
  let micRecording = false;
  let mediaRecorder = null;
  let audioChunks = [];
  let audioCtx = null;
  let analyserNode = null;
  let animId = null;
  let activeMicTarget = "query"; // "query", "1v1-a", "1v1-b"

  controlsEl.innerHTML = `
    <!-- Mode Switcher -->
    <div class="ml-mode-bar">
      <button class="ml-mode-btn active" id="btn-audio-mode-db" type="button">
        <span class="material-symbols-outlined">folder_shared</span>
        Mode B: Audio Uji vs Database Suara (1-to-N)
      </button>
      <button class="ml-mode-btn" id="btn-audio-mode-1vs1" type="button">
        <span class="material-symbols-outlined">graphic_eq</span>
        Mode A: 1-vs-1 (Audio A vs Audio B)
      </button>
    </div>

    <!-- Pilihan Objektif Aplikasi Suara -->
    <div style="display: flex; gap: 8px; margin-bottom: 12px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 8px; border: 1px solid var(--border);">
      <span style="font-size: 12px; color: var(--text-dim); display: flex; align-items: center; gap: 4px;">
        <span class="material-symbols-outlined" style="font-size: 16px;">tune</span> Objektif:
      </span>
      <button class="ml-mode-btn active" id="btn-obj-kw" type="button" style="padding: 4px 12px; font-size: 12px; flex: unset;">
        Pencarian Kata Kunci (Keyword Spotting)
      </button>
      <button class="ml-mode-btn" id="btn-obj-spk" type="button" style="padding: 4px 12px; font-size: 12px; flex: unset;">
        Verifikasi Penutur (Speaker Verification)
      </button>
    </div>

    <!-- ================= KONTEN MODE B (DATABASE 1-TO-N) ================= -->
    <div id="audio-panel-mode-db">
      <!-- 1. Input Audio Uji -->
      <div class="ml-audio-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <h4 style="margin: 0; font-size: 13px; color: var(--accent); display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined">mic</span> 1. Audio Uji (Query Audio)
          </h4>
          <span id="audio-rec-timer-query" style="font-family: var(--mono); font-size: 12px; color: #ff8888; display: none;">● REC 00:00</span>
        </div>

        <canvas class="ml-audio-canvas" id="canvas-mic-query"></canvas>

        <div class="ml-audio-rec-bar">
          <button class="ml-audio-ctrl-btn rec" id="btn-rec-query" type="button">
            <span class="material-symbols-outlined">mic</span> Rekam Mic Live
          </button>

          <label class="ml-audio-ctrl-btn" style="cursor: pointer; margin: 0;">
            <span class="material-symbols-outlined">upload_file</span> Unggah File Audio Uji (.wav/.mp3)
            <input type="file" id="audio-query-file" accept="audio/*" style="display: none;">
          </label>

          <div style="display: flex; align-items: center; gap: 6px; flex: 1;">
            <label style="font-size: 11px; white-space: nowrap;">Atau Sampel:</label>
            <select id="audio-query-sample" style="font-size: 11.5px; padding: 6px 8px; width: 100%;">
              <option value="kw_taksucitra_cepat.wav" selected>Sampel: TAKSUCITRA (Cepat - Uji DTW)</option>
              <option value="kw_taksucitra_normal.wav">Sampel: TAKSUCITRA (Normal)</option>
              <option value="kw_buka_cepat.wav">Sampel: BUKA (Cepat)</option>
              <option value="kw_tutup_normal.wav">Sampel: TUTUP (Normal)</option>
              <option value="spk_Budi_uji.wav">Sampel: Penutur Budi (Uji Biometrik)</option>
              <option value="spk_Siti_uji.wav">Sampel: Penutur Siti (Uji Biometrik)</option>
            </select>
          </div>
        </div>

        <audio id="player-query" class="ml-audio-player" controls src="/static/db_audio/kw_taksucitra_cepat.wav"></audio>
      </div>

      <!-- 2. Kumpulan Database Audio (Folder / File Upload Sendiri) -->
      <div class="ml-folder-box">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <h4 style="margin: 0; font-size: 13px; color: var(--accent); display: flex; align-items: center; gap: 6px;">
            <span class="material-symbols-outlined">folder_special</span>
            2. Sumber Database Audio Target (Folder / Kumpulan Rekaman)
          </h4>
          <span id="audio-db-badge-count" class="ml-badge-folder">
            <span class="material-symbols-outlined" style="font-size: 16px;">library_music</span>
            Database Audio Bawaan Sistem (26 Rekaman)
          </span>
        </div>

        <div class="ml-folder-actions">
          <button class="ml-btn-gold" id="btn-pick-audio-folder" type="button" style="padding: 8px 14px; font-size: 12px;">
            <span class="material-symbols-outlined" style="font-size: 18px;">create_new_folder</span>
            Pilih Folder Audio (a.wav, b.wav, ...)
          </button>
          <input type="file" id="audio-folder-input" webkitdirectory directory multiple accept="audio/*" style="display: none;">

          <button class="ml-btn-sec" id="btn-pick-audio-files" type="button" style="padding: 8px 14px; font-size: 12px;">
            <span class="material-symbols-outlined" style="font-size: 18px;">audio_file</span>
            Pilih Beberapa File Audio
          </button>
          <input type="file" id="audio-files-input" multiple accept="audio/*" style="display: none;">

          <button class="ml-btn-sec" id="btn-reset-audio-db" type="button" style="padding: 8px 12px; font-size: 12px; display: none;">
            <span class="material-symbols-outlined" style="font-size: 16px;">restore</span>
            Kembalikan ke Database Sistem
          </button>
        </div>

        <div id="audio-db-list-container" style="display: none;">
          <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 4px;">Daftar file audio database yang dimuat:</div>
          <div id="audio-db-list" style="display: flex; flex-direction: column; gap: 6px; max-height: 140px; overflow-y: auto; background: #0a0a0a; border: 1px solid var(--border); border-radius: 8px; padding: 8px;"></div>
        </div>
      </div>
    </div>

    <!-- ================= KONTEN MODE A (1-VS-1 AUDIO A VS AUDIO B) ================= -->
    <div id="audio-panel-mode-1vs1" style="display: none;">
      <div class="ml-audio-dual-wrap">
        <!-- Kotak Audio A (Audio Uji) -->
        <div class="ml-audio-slot-card" id="card-audio-a">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; font-size: 13px; color: var(--accent); display: flex; align-items: center; gap: 6px;">
              <span class="material-symbols-outlined">mic</span> Audio A (Audio Uji)
            </h4>
            <span id="audio-rec-timer-a" style="font-family: var(--mono); font-size: 11px; color: #ff8888; display: none;">● REC</span>
          </div>

          <canvas class="ml-audio-canvas" id="canvas-mic-a" style="height: 40px;"></canvas>

          <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <label class="ml-audio-ctrl-btn" style="flex: 1; justify-content: center; font-size: 11.5px; cursor: pointer; padding: 6px 10px;">
              <span class="material-symbols-outlined" style="font-size: 16px;">upload_file</span> Unggah Audio A
              <input type="file" id="file-audio-a" accept="audio/*" style="display: none;">
            </label>
            <button class="ml-audio-ctrl-btn rec" id="btn-rec-a" type="button" style="padding: 6px 10px; font-size: 11.5px;">
              <span class="material-symbols-outlined" style="font-size: 16px;">mic</span> Rekam Mic
            </button>
          </div>

          <audio id="player-audio-a" class="ml-audio-player" controls src="/static/db_audio/kw_taksucitra_normal.wav"></audio>
          <div id="label-audio-a" style="font-size: 11px; color: var(--text-dim); text-align: center;">Sampel: kw_taksucitra_normal.wav</div>
        </div>

        <!-- Kotak Audio B (Audio Target / Pembanding) -->
        <div class="ml-audio-slot-card" id="card-audio-b">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; font-size: 13px; color: var(--saffron); display: flex; align-items: center; gap: 6px;">
              <span class="material-symbols-outlined">audiotrack</span> Audio B (Audio Target)
            </h4>
            <span id="audio-rec-timer-b" style="font-family: var(--mono); font-size: 11px; color: #ff8888; display: none;">● REC</span>
          </div>

          <canvas class="ml-audio-canvas" id="canvas-mic-b" style="height: 40px;"></canvas>

          <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            <label class="ml-audio-ctrl-btn" style="flex: 1; justify-content: center; font-size: 11.5px; cursor: pointer; padding: 6px 10px;">
              <span class="material-symbols-outlined" style="font-size: 16px;">upload_file</span> Unggah Audio B
              <input type="file" id="file-audio-b" accept="audio/*" style="display: none;">
            </label>
            <button class="ml-audio-ctrl-btn rec" id="btn-rec-b" type="button" style="padding: 6px 10px; font-size: 11.5px;">
              <span class="material-symbols-outlined" style="font-size: 16px;">mic</span> Rekam Mic
            </button>
          </div>

          <audio id="player-audio-b" class="ml-audio-player" controls src="/static/db_audio/kw_taksucitra_cepat.wav"></audio>
          <div id="label-audio-b" style="font-size: 11px; color: var(--text-dim); text-align: center;">Sampel: kw_taksucitra_cepat.wav</div>
        </div>
      </div>
    </div>

    <!-- Parameter Kontrol DTW Suara -->
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-top: 12px;">
      <div class="control-field">
        <label>Batas Ambang Jarak DTW Maksimum (Threshold): <span id="dtw-thresh-label" style="color:var(--accent); font-weight:700;">22.0</span></label>
        <input type="range" id="dtw-thresh-slider" min="5.0" max="45.0" step="1.0" value="22.0">
        <div style="font-size: 11px; color: var(--text-faint); margin-top: 2px;">
          Jika jarak DTW $\le$ ambang $\to$ <strong>COCOK / MATCH</strong>; jika melebihi $\to$ <strong>TIDAK COCOK</strong>.
        </div>
      </div>
      <div class="control-field">
        <label>Jumlah Hasil (Top-K)</label>
        <select id="dtw-topk-select">
          <option value="3">Top-3 Hasil</option>
          <option value="5" selected>Top-5 Hasil</option>
          <option value="10">Top-10 Hasil</option>
        </select>
      </div>
    </div>

    <button class="btn-process" id="btn-run-audio-dtw" style="margin-top: 12px;">
      <span class="material-symbols-outlined">timeline</span> Jalankan Pencocokan Suara DTW
    </button>`;

  resultEl.innerHTML = `<div class="placeholder">Pilih atau rekam audio uji, lalu klik <strong>Jalankan Pencocokan Suara DTW</strong>.</div>`;

  // Visualizer Canvases Init
  function initEmptyCanvas(canvasId) {
    const cv = document.getElementById(canvasId);
    if (!cv) return;
    const ctx = cv.getContext("2d");
    cv.width = cv.clientWidth || 300;
    cv.height = cv.clientHeight || 45;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.strokeStyle = "#4d4635";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, cv.height / 2);
    ctx.lineTo(cv.width, cv.height / 2);
    ctx.stroke();
  }
  initEmptyCanvas("canvas-mic-query");
  initEmptyCanvas("canvas-mic-a");
  initEmptyCanvas("canvas-mic-b");

  // Mode Switcher Events
  const btnModeDb = document.getElementById("btn-audio-mode-db");
  const btnMode1vs1 = document.getElementById("btn-audio-mode-1vs1");
  const panelModeDb = document.getElementById("audio-panel-mode-db");
  const panelMode1vs1 = document.getElementById("audio-panel-mode-1vs1");

  btnModeDb.addEventListener("click", () => {
    audioMode = "database";
    btnModeDb.classList.add("active");
    btnMode1vs1.classList.remove("active");
    panelModeDb.style.display = "block";
    panelMode1vs1.style.display = "none";
  });

  btnMode1vs1.addEventListener("click", () => {
    audioMode = "1vs1";
    btnMode1vs1.classList.add("active");
    btnModeDb.classList.remove("active");
    panelModeDb.style.display = "none";
    panelMode1vs1.style.display = "block";
    initEmptyCanvas("canvas-mic-a");
    initEmptyCanvas("canvas-mic-b");
  });

  // Objective Switcher Events
  const btnObjKw = document.getElementById("btn-obj-kw");
  const btnObjSpk = document.getElementById("btn-obj-spk");
  btnObjKw.addEventListener("click", () => {
    audioObjective = "keyword_spotting";
    btnObjKw.classList.add("active");
    btnObjSpk.classList.remove("active");
  });
  btnObjSpk.addEventListener("click", () => {
    audioObjective = "speaker_verification";
    btnObjSpk.classList.add("active");
    btnObjKw.classList.remove("active");
  });

  // Threshold Slider
  const dtwSlider = document.getElementById("dtw-thresh-slider");
  const dtwThreshLabel = document.getElementById("dtw-thresh-label");
  dtwSlider.addEventListener("input", () => {
    dtwThreshLabel.textContent = parseFloat(dtwSlider.value).toFixed(1);
  });

  // Mode B: Audio Uji Upload / Sample
  const audioQueryFile = document.getElementById("audio-query-file");
  const audioQuerySample = document.getElementById("audio-query-sample");
  const playerQuery = document.getElementById("player-query");

  function readAudioFileAsB64(file, callback) {
    const reader = new FileReader();
    reader.onload = (ev) => callback(ev.target.result);
    reader.readAsDataURL(file);
  }

  audioQueryFile.addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    const file = e.target.files[0];
    readAudioFileAsB64(file, (b64) => {
      queryAudioB64 = b64;
      playerQuery.src = b64;
      audioQuerySample.value = "";
    });
  });

  audioQuerySample.addEventListener("change", () => {
    if (audioQuerySample.value) {
      queryAudioB64 = null;
      playerQuery.src = `/static/db_audio/${audioQuerySample.value}`;
    }
  });

  // Mode B: Folder / Kumpulan File Audio Database
  const btnPickAudioFolder = document.getElementById("btn-pick-audio-folder");
  const btnPickAudioFiles = document.getElementById("btn-pick-audio-files");
  const audioFolderInput = document.getElementById("audio-folder-input");
  const audioFilesInput = document.getElementById("audio-files-input");
  const badgeAudioDbCount = document.getElementById("audio-db-badge-count");
  const btnResetAudioDb = document.getElementById("btn-reset-audio-db");
  const audioDbListContainer = document.getElementById("audio-db-list-container");
  const audioDbList = document.getElementById("audio-db-list");

  btnPickAudioFolder.addEventListener("click", () => audioFolderInput.click());
  btnPickAudioFiles.addEventListener("click", () => audioFilesInput.click());

  function processSelectedAudioDatabaseFiles(files) {
    if (!files || files.length === 0) return;
    const audFiles = Array.from(files).filter(f => f.type.startsWith("audio/") || /\.(wav|mp3|ogg|flac|m4a)$/i.test(f.name));
    if (audFiles.length === 0) {
      alert("Tidak ada file audio yang valid dalam folder yang dipilih.");
      return;
    }

    customDatabaseAudios = [];
    useSystemAudioDb = false;
    audioDbList.innerHTML = "";
    audioDbListContainer.style.display = "block";
    btnResetAudioDb.style.display = "inline-flex";

    badgeAudioDbCount.innerHTML = `
      <span class="material-symbols-outlined" style="font-size: 16px;">cloud_sync</span>
      Memuat ${audFiles.length} audio...
    `;

    let loaded = 0;
    audFiles.forEach((file) => {
      readAudioFileAsB64(file, (b64) => {
        customDatabaseAudios.push({ filename: file.name, audio: b64 });
        
        // Buat item audio preview
        const row = document.createElement("div");
        row.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 11.5px;";
        row.innerHTML = `
          <span style="color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:180px;">🎵 ${file.name}</span>
          <audio controls src="${b64}" style="height:24px; max-width:200px;"></audio>
        `;
        audioDbList.appendChild(row);

        loaded++;
        if (loaded === audFiles.length) {
          badgeAudioDbCount.innerHTML = `
            <span class="material-symbols-outlined" style="font-size: 16px; color:#64d2ff;">check_circle</span>
            ✓ ${customDatabaseAudios.length} File Audio Database Siap
          `;
        }
      });
    });
  }

  audioFolderInput.addEventListener("change", (e) => processSelectedAudioDatabaseFiles(e.target.files));
  audioFilesInput.addEventListener("change", (e) => processSelectedAudioDatabaseFiles(e.target.files));

  btnResetAudioDb.addEventListener("click", () => {
    customDatabaseAudios = [];
    useSystemAudioDb = true;
    audioDbListContainer.style.display = "none";
    audioDbList.innerHTML = "";
    btnResetAudioDb.style.display = "none";
    badgeAudioDbCount.innerHTML = `
      <span class="material-symbols-outlined" style="font-size: 16px;">library_music</span>
      Database Audio Bawaan Sistem (26 Rekaman)
    `;
    audioFolderInput.value = "";
    audioFilesInput.value = "";
  });

  // Mode A: Upload Audio A & Audio B Mandiri oleh Pengguna
  const fileAudioA = document.getElementById("file-audio-a");
  const fileAudioB = document.getElementById("file-audio-b");
  const playerAudioA = document.getElementById("player-audio-a");
  const playerAudioB = document.getElementById("player-audio-b");
  const labelAudioA = document.getElementById("label-audio-a");
  const labelAudioB = document.getElementById("label-audio-b");
  const cardAudioA = document.getElementById("card-audio-a");
  const cardAudioB = document.getElementById("card-audio-b");

  fileAudioA.addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    const file = e.target.files[0];
    readAudioFileAsB64(file, (b64) => {
      audioA_B64 = b64;
      audioA_Name = file.name;
      playerAudioA.src = b64;
      labelAudioA.textContent = `File: ${file.name}`;
      cardAudioA.classList.add("filled");
    });
  });

  fileAudioB.addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    const file = e.target.files[0];
    readAudioFileAsB64(file, (b64) => {
      audioB_B64 = b64;
      audioB_Name = file.name;
      playerAudioB.src = b64;
      labelAudioB.textContent = `File: ${file.name}`;
      cardAudioB.classList.add("filled");
    });
  });

  // Shared Microphone Handler
  let recSecs = 0, recInterval = null;

  async function toggleRecording(target, btnEl, timerEl, canvasId, onFinish) {
    if (!micRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyserNode = audioCtx.createAnalyser();
        analyserNode.fftSize = 512;
        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyserNode);

        const canvasEl = document.getElementById(canvasId);
        const cvCtx = canvasEl.getContext("2d");
        const bufferLength = analyserNode.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        function drawWave() {
          animId = requestAnimationFrame(drawWave);
          analyserNode.getByteTimeDomainData(dataArray);
          cvCtx.fillStyle = "#090909";
          cvCtx.fillRect(0, 0, canvasEl.width, canvasEl.height);
          cvCtx.lineWidth = 2;
          cvCtx.strokeStyle = "#f2ca50";
          cvCtx.beginPath();
          const sliceWidth = canvasEl.width * 1.0 / bufferLength;
          let x = 0;
          for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * canvasEl.height / 2;
            if (i === 0) cvCtx.moveTo(x, y);
            else cvCtx.lineTo(x, y);
            x += sliceWidth;
          }
          cvCtx.lineTo(canvasEl.width, canvasEl.height / 2);
          cvCtx.stroke();
        }
        drawWave();

        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) audioChunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          const reader = new FileReader();
          reader.onloadend = () => {
            onFinish(reader.result);
          };
          reader.readAsDataURL(audioBlob);
          stream.getTracks().forEach(track => track.stop());
          if (audioCtx) { audioCtx.close(); audioCtx = null; }
          if (animId) cancelAnimationFrame(animId);
          initEmptyCanvas(canvasId);
        };

        mediaRecorder.start();
        micRecording = true;
        activeMicTarget = target;
        btnEl.classList.add("recording");
        btnEl.innerHTML = `<span class="material-symbols-outlined">stop</span> Stop`;
        timerEl.style.display = "inline";
        recSecs = 0;
        recInterval = setInterval(() => {
          recSecs++;
          const mm = String(Math.floor(recSecs / 60)).padStart(2, '0');
          const ss = String(recSecs % 60).padStart(2, '0');
          timerEl.textContent = `● REC ${mm}:${ss}`;
        }, 1000);

      } catch (err) {
        alert(`Gagal mengakses mikrofon: ${err.message}`);
      }
    } else {
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      micRecording = false;
      btnEl.classList.remove("recording");
      btnEl.innerHTML = `<span class="material-symbols-outlined">mic</span> Rekam Mic`;
      timerEl.style.display = "none";
      if (recInterval) clearInterval(recInterval);
    }
  }

  // Hook mic buttons
  document.getElementById("btn-rec-query").addEventListener("click", () => {
    toggleRecording(
      "query",
      document.getElementById("btn-rec-query"),
      document.getElementById("audio-rec-timer-query"),
      "canvas-mic-query",
      (b64) => {
        queryAudioB64 = b64;
        playerQuery.src = b64;
        audioQuerySample.value = "";
      }
    );
  });

  document.getElementById("btn-rec-a").addEventListener("click", () => {
    toggleRecording(
      "1v1-a",
      document.getElementById("btn-rec-a"),
      document.getElementById("audio-rec-timer-a"),
      "canvas-mic-a",
      (b64) => {
        audioA_B64 = b64;
        playerAudioA.src = b64;
        labelAudioA.textContent = "Rekaman Mikrofon Audio A";
        cardAudioA.classList.add("filled");
      }
    );
  });

  document.getElementById("btn-rec-b").addEventListener("click", () => {
    toggleRecording(
      "1v1-b",
      document.getElementById("btn-rec-b"),
      document.getElementById("audio-rec-timer-b"),
      "canvas-mic-b",
      (b64) => {
        audioB_B64 = b64;
        playerAudioB.src = b64;
        labelAudioB.textContent = "Rekaman Mikrofon Audio B";
        cardAudioB.classList.add("filled");
      }
    );
  });

  // Eksekusi Pencocokan DTW
  document.getElementById("btn-run-audio-dtw").addEventListener("click", async () => {
    const btn = document.getElementById("btn-run-audio-dtw");
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Menghitung Jalur DTW...`;

    try {
      if (audioMode === "database") {
        // Mode B: Audio Uji vs Database Suara
        const payload = {
          params: {
            objective: audioObjective,
            threshold: parseFloat(dtwSlider.value),
            top_k: parseInt(document.getElementById("dtw-topk-select").value),
            custom_database: (!useSystemAudioDb && customDatabaseAudios.length > 0) ? customDatabaseAudios : null
          }
        };

        if (queryAudioB64) {
          payload.audio = queryAudioB64;
        } else if (audioQuerySample.value) {
          payload.sample = audioQuerySample.value;
        } else {
          payload.sample = "kw_taksucitra_cepat.wav";
        }

        const res = await fetch("/api/matching/audio-db", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || "Gagal memproses audio database.");

        const isMatchTop = data.best_candidate && data.best_candidate.is_match;
        let customHTML = `
          <div class="ml-container">
            <!-- Banner Keputusan -->
            <div class="ml-verdict-banner" style="background: ${isMatchTop ? "rgba(100,210,255,0.14)" : "rgba(255,100,100,0.12)"}; border-color: ${isMatchTop ? "#64d2ff" : "#ff7777"}; color: ${isMatchTop ? "#64d2ff" : "#ff9999"};">
              <span class="material-symbols-outlined">${isMatchTop ? "verified" : "error"}</span>
              <span>${data.summary}</span>
              <span style="margin-left: auto; font-family: var(--mono); font-size: 11px;">${data.timing_ms} ms</span>
            </div>

            <!-- Galeri Hasil Suara Database Top-K -->
            <div style="margin-top: 14px; background: var(--panel-2); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <h4 style="margin: 0; font-size: 13.5px; color: var(--accent); display: flex; align-items: center; gap: 6px;">
                  <span class="material-symbols-outlined">queue_music</span>
                  Kandidat Suara Teratas (Top-${data.top_candidates.length} Hasil)
                </h4>
                <span style="font-size:11px; color:var(--text-dim);">
                  Basis Data: <strong>${!useSystemAudioDb && customDatabaseAudios.length > 0 ? "Folder Pengguna (" + customDatabaseAudios.length + " rekaman)" : "Database Sistem (26 rekaman)"}</strong>
                </span>
              </div>
              <div class="ml-db-grid">
                ${data.top_candidates.map((c, idx) => `
                  <div class="ml-db-card ${c.is_match ? "matched" : "non-matched"}">
                    <span class="ml-db-rank">#${idx + 1}</span>
                    <div style="font-size: 13px; font-weight: 700; color: var(--text); margin-top: 18px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;" title="${c.label}">${c.label}</div>
                    <div class="ml-db-meta">
                      <span>Jarak DTW:</span>
                      <span class="ml-db-score-val">${c.norm_dist}</span>
                    </div>
                    <div class="ml-db-meta">
                      <span>Confidence:</span>
                      <span>${c.confidence}%</span>
                    </div>
                    <audio class="ml-audio-player" controls src="${c.audio_url}" style="margin-top: 4px;"></audio>
                    <div class="ml-badge-status ${c.is_match ? "match" : "nonmatch"}" style="margin-top: 4px;">
                      ${c.is_match ? "✓ MATCH (Lolos Ambang)" : "✗ NON-MATCH"}
                    </div>
                  </div>
                `).join("")}
              </div>
            </div>

            <!-- Pipeline Visualisasi Langkah demi Langkah -->
            <div style="margin-top: 20px;">
              <h4 style="color: var(--accent); margin-bottom: 8px;">Tahapan Penyelarasan Temporal DTW & Spektrogram</h4>
              ${renderMLGenericStepPipeline(data)}
            </div>
          </div>`;

        resultEl.innerHTML = customHTML;
        if (typeof setupMLStepperEvents === "function") setupMLStepperEvents();

      } else {
        // Mode A: 1-vs-1 (Audio A vs Audio B) Mandiri oleh Pengguna
        const payload = {
          params: {
            threshold: parseFloat(dtwSlider.value)
          }
        };

        if (audioA_B64) payload.audio_a = audioA_B64;
        else payload.sample_a = "kw_taksucitra_normal.wav";

        if (audioB_B64) payload.audio_b = audioB_B64;
        else payload.sample_b = "kw_taksucitra_cepat.wav";

        const res = await fetch("/api/matching/audio-dtw", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok || !data.ok) throw new Error(data.error || "Gagal memproses DTW 1-vs-1.");

        resultEl.innerHTML = renderMLGenericStepPipeline(data);
        if (typeof setupMLStepperEvents === "function") setupMLStepperEvents();
      }
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">Gagal memproses audio DTW: ${err.message}</div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `<span class="material-symbols-outlined">timeline</span> Jalankan Pencocokan Suara DTW`;
    }
  });
}


// ============================================================
// FEATURE EXTRACTION LAB: Concat Playground
// ============================================================

function renderConcatPanel(panel) {
  controlsEl.innerHTML = `
    <h3>Pilih Fitur untuk Digabung</h3>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    <div class="explain-box">${panel.explain}</div>
    <button class="btn-process" id="btn-process" style="margin-top:12px">Gabungkan</button>`;

  resultEl.innerHTML = `<div class="placeholder">Pilih minimal 1 fitur lalu klik Gabungkan.</div>`;

  document.getElementById("btn-process").addEventListener("click", async () => {
    const btn = document.getElementById("btn-process");
    const values = collectFieldValues();
    const chosen = values.features || [];
    if (chosen.length === 0) { alert("Pilih minimal 1 fitur."); return; }

    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span> Memproses...`;

    const payload = {
      selections: chosen.map(k => ({ panel: k, params: {} })),
      normalize: !!values.normalize,
    };
    if (uploadedImageB64) payload.image = uploadedImageB64;
    else payload.sample = getSelectedSample();

    try {
      const res = await fetch(panel.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
      resultEl.innerHTML = renderFEStepGallery(data);
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">${err.message}</div>`;
    } finally {
      btn.disabled = false; btn.textContent = "Gabungkan";
    }
  });
}

// ============================================================
// PANEL KHUSUS: Source + Reference (histogram matching)
// ============================================================

function renderSourceReferencePanel(panel) {
  controlsEl.innerHTML = `
    <h3>Pilih Gambar Sumber & Referensi</h3>
    <div class="control-row">
      <div class="control-field"><label>Gambar Sumber</label>
        <select data-field="source_sample">${window.__SAMPLES__.map(s => `<option value="${s}">${s}</option>`).join("")}</select></div>
      <div class="control-field"><label>Gambar Referensi</label>
        <select data-field="reference_sample">${window.__SAMPLES__.map((s, i) => `<option value="${s}" ${i===1?"selected":""}>${s}</option>`).join("")}</select></div>
    </div>
    <div class="explain-box">${panel.explain}</div>
    <button class="btn-process" id="btn-process">Proses</button>`;

  resultEl.innerHTML = `<div class="placeholder">Pilih gambar sumber & referensi lalu klik Proses.</div>`;

  document.getElementById("btn-process").addEventListener("click", async () => {
    const btn = document.getElementById("btn-process");
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Memproses...`;
    const values = collectFieldValues();
    try {
      const res = await fetch(panel.endpoint, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_sample: values.source_sample, reference_sample: values.reference_sample }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal memproses");
      resultEl.innerHTML = panel.render(data);
      bindSaveHistoryButton(panel, { original: data.source, result: data.result });
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">Gagal memproses: ${err.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Proses"; }
  });
}

// ============================================================
// PANEL KHUSUS: Watermark (embed + extract)
// ============================================================

function renderWatermarkPanel(panel) {
  controlsEl.innerHTML = `
    <h3>Parameter</h3>
    <div class="control-row">${buildFieldsHTML(panel.fields)}</div>
    <div class="explain-box">${panel.explain}</div>
    <button class="btn-process" id="btn-embed">Sisipkan Pesan</button>
    <button class="btn-secondary" id="btn-extract" style="margin-left:8px">Ekstrak Pesan dari Hasil</button>`;

  resultEl.innerHTML = `<div class="placeholder">Tulis pesan lalu klik Sisipkan Pesan.</div>`;

  document.getElementById("btn-embed").addEventListener("click", async () => {
    const btn = document.getElementById("btn-embed");
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Memproses...`;
    const values = collectFieldValues();
    const payload = { text: values.text || "" };
    if (uploadedImageB64) payload.image = uploadedImageB64; else payload.sample = getSelectedSample();
    try {
      const res = await fetch(panel.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal memproses");
      watermarkedResultB64 = data.result;
      resultEl.innerHTML = panel.render(data);
      bindSaveHistoryButton(panel, data);
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">${err.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Sisipkan Pesan"; }
  });

  document.getElementById("btn-extract").addEventListener("click", async () => {
    if (!watermarkedResultB64) { alert("Sisipkan pesan dulu, baru bisa diekstrak."); return; }
    const btn = document.getElementById("btn-extract");
    btn.disabled = true; btn.textContent = "Mengekstrak...";
    try {
      const res = await fetch(panel.extractEndpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image: watermarkedResultB64 }) });
      const data = await res.json();
      resultEl.insertAdjacentHTML("beforeend", `<div class="explain-box" style="margin-top:14px">Pesan berhasil diekstrak: <b>"${data.extracted_text}"</b></div>`);
    } catch (err) {
      alert("Gagal ekstrak: " + err.message);
    } finally { btn.disabled = false; btn.textContent = "Ekstrak Pesan dari Hasil"; }
  });
}

// ============================================================
// PANEL KHUSUS: Mask Draw (Magic Eraser / Inpainting)
// ============================================================

function renderMaskDrawPanel(panel) {
  controlsEl.innerHTML = `
    <h3>Gambar Area yang Ingin Dihapus</h3>
    <p style="font-size:12.5px;color:var(--text-dim);margin-top:-6px">Klik dan seret di atas gambar untuk menandai objek yang ingin dihilangkan.</p>
    <div class="mask-canvas-wrap" style="margin:14px 0">
      <canvas id="inpaint-canvas" width="500" height="375"></canvas>
    </div>
    <div class="control-row">
      <div class="control-field"><label>Ukuran Kuas</label><input type="range" id="brush-size" min="5" max="60" value="20"></div>
      ${buildFieldsHTML(panel.fields)}
    </div>
    <button class="btn-secondary" id="btn-clear-mask">Bersihkan Coretan</button>
    <div class="explain-box" style="margin-top:12px">${panel.explain}</div>
    <button class="btn-process" id="btn-process" style="margin-top:12px">Hapus Objek</button>`;

  bindRangeLiveUpdate();
  resultEl.innerHTML = `<div class="placeholder">Tandai area lalu klik Hapus Objek.</div>`;

  const canvas = document.getElementById("inpaint-canvas");
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const ratio = img.height / img.width;
    canvas.width = 500; canvas.height = Math.round(500 * ratio);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  };
  img.src = currentImagePreviewSrc();

  const maskCanvas = document.createElement("canvas");
  let drawing = false;
  let brushSize = 20;
  document.getElementById("brush-size").addEventListener("input", (e) => brushSize = parseInt(e.target.value));

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return { x: (clientX - rect.left) * (canvas.width / rect.width), y: (clientY - rect.top) * (canvas.height / rect.height) };
  }

  function drawDot(x, y) {
    ctx.fillStyle = "rgba(255,71,71,0.55)";
    ctx.beginPath(); ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2); ctx.fill();
  }

  canvas.addEventListener("mousedown", (e) => { drawing = true; const p = getPos(e); drawDot(p.x, p.y); });
  canvas.addEventListener("mousemove", (e) => { if (drawing) { const p = getPos(e); drawDot(p.x, p.y); } });
  window.addEventListener("mouseup", () => drawing = false);
  canvas.addEventListener("touchstart", (e) => { drawing = true; const p = getPos(e); drawDot(p.x, p.y); e.preventDefault(); });
  canvas.addEventListener("touchmove", (e) => { if (drawing) { const p = getPos(e); drawDot(p.x, p.y); } e.preventDefault(); });
  window.addEventListener("touchend", () => drawing = false);

  document.getElementById("btn-clear-mask").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  });

  document.getElementById("btn-process").addEventListener("click", async () => {
    const btn = document.getElementById("btn-process");
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Menghapus objek...`;

    // Buat mask hitam-putih dari canvas: bandingkan piksel merah yang ditandai
    maskCanvas.width = canvas.width; maskCanvas.height = canvas.height;
    const maskCtx = maskCanvas.getContext("2d");
    const original = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const maskData = maskCtx.createImageData(canvas.width, canvas.height);
    for (let i = 0; i < original.data.length; i += 4) {
      const r = original.data[i], g = original.data[i + 1], b = original.data[i + 2];
      const isMarked = r > 150 && g < 120 && b < 120;
      const v = isMarked ? 255 : 0;
      maskData.data[i] = v; maskData.data[i+1] = v; maskData.data[i+2] = v; maskData.data[i+3] = 255;
    }
    maskCtx.putImageData(maskData, 0, 0);

    const cleanCanvas = document.createElement("canvas");
    cleanCanvas.width = canvas.width; cleanCanvas.height = canvas.height;
    cleanCanvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);

    const values = collectFieldValues();
    const payload = {
      image: cleanCanvas.toDataURL("image/png"),
      mask: maskCanvas.toDataURL("image/png"),
      method: values.method, radius: values.radius,
    };

    try {
      const res = await fetch(panel.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal memproses");
      resultEl.innerHTML = panel.render(data);
      (panel.sliderIds || []).forEach(initBeforeAfterSlider);
      bindSaveHistoryButton(panel, data);
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">${err.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Hapus Objek"; }
  });
}

// ============================================================
// PANEL KHUSUS: Multi-upload (Panorama)
// ============================================================

function renderMultiUploadPanel(panel) {
  panoramaImages = [];
  controlsEl.innerHTML = `
    <h3>Upload 2 atau Lebih Foto (Berurutan, Ada Area Tumpang Tindih)</h3>
    <input type="file" id="panorama-input" accept="image/*" multiple>
    <div class="batch-thumbs" id="panorama-thumbs"></div>
    <div class="explain-box" style="margin-top:12px">${panel.explain}</div>
    <button class="btn-process" id="btn-process" style="margin-top:12px" disabled>Gabungkan Panorama</button>`;

  resultEl.innerHTML = `<div class="placeholder">Upload minimal 2 foto untuk digabungkan.</div>`;

  document.getElementById("panorama-input").addEventListener("change", (e) => {
    const files = Array.from(e.target.files);
    panoramaImages = [];
    const thumbsEl = document.getElementById("panorama-thumbs");
    thumbsEl.innerHTML = "";
    let loaded = 0;
    files.forEach((file, idx) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        panoramaImages[idx] = ev.target.result;
        const img = document.createElement("img");
        img.src = ev.target.result;
        thumbsEl.appendChild(img);
        loaded++;
        if (loaded === files.length) {
          document.getElementById("btn-process").disabled = panoramaImages.length < 2;
        }
      };
      reader.readAsDataURL(file);
    });
  });

  document.getElementById("btn-process").addEventListener("click", async () => {
    const btn = document.getElementById("btn-process");
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Menggabungkan...`;
    try {
      const res = await fetch(panel.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ images: panoramaImages }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Gagal stitching");
      resultEl.innerHTML = panel.render(data);
      bindSaveHistoryButton(panel, { original: data.inputs[0], result: data.result });
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">${err.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Gabungkan Panorama"; }
  });
}

// ============================================================
// MODE KHUSUS: BATCH PROCESSING
// ============================================================

const BATCH_OPERATIONS = [
  ["grayscale", "Grayscale"], ["brightness", "Perubahan Kecerahan"], ["contrast", "Perubahan Kontras"],
  ["sharpen", "Sharpening"], ["equalize", "Histogram Equalization"], ["edge", "Deteksi Tepi"],
  ["stylize", "Artistic Stylization"], ["face_blur", "Anonimisasi Wajah"],
];

let batchImages = [];

function renderBatchMode() {
  controlsEl.innerHTML = `
    <h3>1. Upload Banyak Gambar</h3>
    <div class="batch-dropzone" id="batch-dropzone">
      <p>Klik atau drag-and-drop beberapa gambar ke sini</p>
      <input type="file" id="batch-input" accept="image/*" multiple style="display:none">
    </div>
    <div class="batch-thumbs" id="batch-thumbs"></div>

    <h3 style="margin-top:20px">2. Pilih Operasi</h3>
    <div class="control-row">
      <div class="control-field"><label>Operasi</label>
        <select id="batch-operation">${BATCH_OPERATIONS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></div>
    </div>
    <button class="btn-process" id="btn-batch-process" disabled>Proses Semua Gambar</button>`;

  resultEl.innerHTML = `<div class="placeholder">Upload gambar terlebih dahulu.</div>`;
  batchImages = [];

  const dropzone = document.getElementById("batch-dropzone");
  const input = document.getElementById("batch-input");
  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault(); dropzone.classList.remove("dragover");
    handleBatchFiles(Array.from(e.dataTransfer.files));
  });
  input.addEventListener("change", (e) => handleBatchFiles(Array.from(e.target.files)));

  function handleBatchFiles(files) {
    const thumbsEl = document.getElementById("batch-thumbs");
    let loaded = 0;
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        batchImages.push(ev.target.result);
        const img = document.createElement("img");
        img.src = ev.target.result;
        thumbsEl.appendChild(img);
        loaded++;
        document.getElementById("btn-batch-process").disabled = batchImages.length === 0;
      };
      reader.readAsDataURL(file);
    });
  }

  document.getElementById("btn-batch-process").addEventListener("click", async () => {
    const btn = document.getElementById("btn-batch-process");
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span> Memproses ${batchImages.length} gambar...`;
    const operation = document.getElementById("batch-operation").value;

    try {
      const res = await fetch("/api/batch/process", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: batchImages, operation, params: {} }),
      });
      const data = await res.json();
      const successResults = data.results.filter(r => r.success);

      resultEl.innerHTML = `
        <div class="tag-list"><span class="tag-chip">${successResults.length} / ${data.results.length} berhasil diproses</span></div>
        <div class="result-grid" style="margin-top:14px">
          ${data.results.map((r, i) => r.success
            ? `<div class="result-item"><img src="${r.result}"><div class="cap">Gambar ${i+1}</div></div>`
            : `<div class="result-item"><div class="error-box">Gagal: ${r.error}</div></div>`).join("")}
        </div>
        <button class="btn-process" id="btn-download-zip" style="margin-top:16px">Download Semua (ZIP)</button>`;

      document.getElementById("btn-download-zip").addEventListener("click", async () => {
        const zipRes = await fetch("/api/batch/download-zip", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ images: successResults.map(r => r.result) }),
        });
        const blob = await zipRes.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a"); a.href = url; a.download = "TaksuCitra_Batch.zip"; a.click();
        window.URL.revokeObjectURL(url);
      });
    } catch (err) {
      resultEl.innerHTML = `<div class="error-box">Gagal batch processing: ${err.message}</div>`;
    } finally { btn.disabled = false; btn.textContent = "Proses Semua Gambar"; }
  });
}

// ============================================================
// MODE KHUSUS: LIVE WEBCAM
// ============================================================

const WEBCAM_FILTERS = [
  ["none", "Normal"], ["grayscale", "Grayscale"], ["edge", "Deteksi Tepi"],
  ["sharpen", "Sharpening"], ["face_blur", "Anonimisasi Wajah"], ["stylize", "Artistic Stylization"], ["sketch", "Sketsa Pensil"],
];

function renderWebcamMode() {
  controlsEl.innerHTML = `
    <h3>Live Webcam Processing</h3>
    <div class="control-row">
      <div class="control-field"><label>Filter</label>
        <select id="webcam-filter">${WEBCAM_FILTERS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select></div>
    </div>
    <button class="btn-process" id="btn-webcam-start">Mulai Kamera</button>
    <button class="btn-secondary" id="btn-webcam-stop" style="margin-left:8px" disabled>Berhenti</button>
    <div class="explain-box" style="margin-top:12px">Frame diambil dari webcam browser lalu dikirim ke server untuk diproses dengan OpenCV secara berkala (bukan 30fps penuh, tapi cukup untuk demo real-time).</div>`;

  resultEl.innerHTML = `
    <div class="webcam-wrap">
      <div class="webcam-box"><video id="webcam-video" autoplay muted playsinline></video><div class="cap">Input Kamera</div></div>
      <div class="webcam-box"><canvas id="webcam-output" width="480" height="360"></canvas><div class="cap">Hasil Proses</div><div class="fps-indicator" id="webcam-fps"></div></div>
    </div>`;

  const video = document.getElementById("webcam-video");
  const outputCanvas = document.getElementById("webcam-output");
  const outputCtx = outputCanvas.getContext("2d");
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d");

  document.getElementById("btn-webcam-start").addEventListener("click", async () => {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
      video.srcObject = webcamStream;
      webcamRunning = true;
      document.getElementById("btn-webcam-start").disabled = true;
      document.getElementById("btn-webcam-stop").disabled = false;
      webcamLoop();
    } catch (err) {
      resultEl.insertAdjacentHTML("beforeend", `<div class="error-box" style="margin-top:12px">Tidak bisa mengakses webcam: ${err.message}</div>`);
    }
  });

  document.getElementById("btn-webcam-stop").addEventListener("click", () => {
    webcamRunning = false;
    if (webcamStream) webcamStream.getTracks().forEach(t => t.stop());
    document.getElementById("btn-webcam-start").disabled = false;
    document.getElementById("btn-webcam-stop").disabled = true;
  });

  async function webcamLoop() {
    if (!webcamRunning) return;
    const t0 = performance.now();
    captureCanvas.width = video.videoWidth || 480;
    captureCanvas.height = video.videoHeight || 360;
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const frameB64 = captureCanvas.toDataURL("image/jpeg", 0.7);
    const filter = document.getElementById("webcam-filter").value;

    try {
      const res = await fetch("/api/webcam/frame", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ frame: frameB64, filter }),
      });
      const data = await res.json();
      if (data.result) {
        const img = new Image();
        img.onload = () => {
          outputCanvas.width = img.width; outputCanvas.height = img.height;
          outputCtx.drawImage(img, 0, 0);
        };
        img.src = data.result;
      }
    } catch (e) { /* diamkan, lanjut frame berikutnya */ }

    const elapsed = performance.now() - t0;
    document.getElementById("webcam-fps").textContent = `Latensi: ${elapsed.toFixed(0)}ms per frame`;
    if (webcamRunning) setTimeout(webcamLoop, 200);
  }
}

// ============================================================
// MODE KHUSUS: HISTORY PAGE (versi inline, selain drawer)
// ============================================================

function renderHistoryPage() {
  controlsEl.innerHTML = `<h3>Kelola Riwayat</h3>
    <p style="font-size:13px;color:var(--text-dim)">Gunakan tombol 🕓 di kanan atas untuk membuka panel riwayat lengkap dan mengekspor ke PDF, atau klik tombol di bawah ini.</p>
    <button class="btn-process" id="btn-open-history-2">Buka Panel Riwayat</button>`;
  resultEl.innerHTML = `<div class="placeholder">Riwayat hasil proses kamu akan tersimpan otomatis setiap kali klik "Simpan ke Riwayat" pada fitur manapun.</div>`;
  document.getElementById("btn-open-history-2").addEventListener("click", openHistoryDrawer);
}

// ============================================================
// NAVIGASI SIDEBAR
// ============================================================

function navigateTo(key) {
  document.querySelectorAll(".nav-item, .nav-home").forEach(b => b.classList.remove("active"));
  const leafBtn = document.querySelector(`.nav-item[data-panel="${key}"]`);
  if (leafBtn) {
    leafBtn.classList.add("active");
    leafBtn.closest(".nav-group").classList.remove("collapsed");
  } else if (key === "home") {
    document.getElementById("nav-home-btn").classList.add("active");
  }
  renderPanel(key);
}

document.getElementById("nav-home-btn").addEventListener("click", () => navigateTo("home"));

document.querySelectorAll(".nav-item").forEach(btn => {
  btn.addEventListener("click", () => navigateTo(btn.dataset.panel));
});

document.querySelectorAll(".nav-group-title").forEach(title => {
  title.addEventListener("click", () => {
    title.closest(".nav-group").classList.toggle("collapsed");
    if (title.dataset.landing) navigateTo(title.dataset.landing);
  });
});

document.getElementById("sidebar-filter").addEventListener("input", (e) => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.style.display = btn.textContent.toLowerCase().includes(q) ? "block" : "none";
  });
  document.querySelectorAll(".nav-group").forEach(group => {
    const visible = Array.from(group.querySelectorAll(".nav-item")).some(b => b.style.display !== "none");
    group.style.display = visible ? "block" : "none";
    group.classList.toggle("collapsed", !q ? group.classList.contains("collapsed") : false);
  });
});

// ============================================================
// UPLOAD GAMBAR
// ============================================================

const uploadInput = document.getElementById("upload-input");
const uploadPreviewWrap = document.getElementById("upload-preview-wrap");
const uploadPreview = document.getElementById("upload-preview");
const clearUploadBtn = document.getElementById("clear-upload");

uploadInput.addEventListener("change", () => {
  const file = uploadInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadedImageB64 = e.target.result;
    uploadPreview.src = uploadedImageB64;
    uploadPreviewWrap.style.display = "flex";
  };
  reader.readAsDataURL(file);
});

clearUploadBtn.addEventListener("click", () => {
  uploadedImageB64 = null;
  uploadInput.value = "";
  uploadPreviewWrap.style.display = "none";
});

// ============================================================
// COMMAND PALETTE (Ctrl+K)
// ============================================================

const cmdkOverlay = document.getElementById("cmdk-overlay");
const cmdkInput = document.getElementById("cmdk-input");
const cmdkResults = document.getElementById("cmdk-results");
let cmdkSelectedIndex = 0;
let cmdkFilteredKeys = [];

function getAllPanelEntries() {
  return Object.keys(PANELS).map(key => ({ key, title: PANELS[key].title, group: findGroupForKey(key) }));
}

function findGroupForKey(key) {
  const btn = document.querySelector(`.nav-item[data-panel="${key}"]`);
  if (!btn) return "";
  const group = btn.closest(".nav-group");
  const label = group ? group.querySelector(".nav-group-label") : null;
  return label ? label.textContent.trim() : "";
}

function openCmdk() {
  cmdkOverlay.classList.add("open");
  cmdkInput.value = "";
  cmdkInput.focus();
  renderCmdkResults("");
}
function closeCmdk() { cmdkOverlay.classList.remove("open"); }

function renderCmdkResults(query) {
  const all = getAllPanelEntries();
  const filtered = query
    ? all.filter(e => e.title.toLowerCase().includes(query.toLowerCase()))
    : all;
  cmdkFilteredKeys = filtered.map(e => e.key);
  cmdkSelectedIndex = 0;

  cmdkResults.innerHTML = filtered.map((e, i) => `
    <div class="cmdk-item ${i === 0 ? "selected" : ""}" data-key="${e.key}" data-idx="${i}">
      <span>${e.title}</span><span class="cmdk-item-group">${e.group}</span>
    </div>`).join("") || `<div style="padding:16px;color:var(--text-faint);font-size:13px">Tidak ditemukan.</div>`;

  cmdkResults.querySelectorAll(".cmdk-item").forEach(item => {
    item.addEventListener("click", () => selectCmdkItem(item.dataset.key));
  });
}

function selectCmdkItem(key) {
  navigateTo(key);
  closeCmdk();
}

document.getElementById("cmdk-trigger").addEventListener("click", openCmdk);
cmdkOverlay.addEventListener("click", (e) => { if (e.target === cmdkOverlay) closeCmdk(); });
cmdkInput.addEventListener("input", (e) => renderCmdkResults(e.target.value));

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    cmdkOverlay.classList.contains("open") ? closeCmdk() : openCmdk();
  }
  if (cmdkOverlay.classList.contains("open")) {
    if (e.key === "Escape") closeCmdk();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      cmdkSelectedIndex = Math.min(cmdkSelectedIndex + 1, cmdkFilteredKeys.length - 1);
      updateCmdkSelection();
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      cmdkSelectedIndex = Math.max(cmdkSelectedIndex - 1, 0);
      updateCmdkSelection();
    }
    if (e.key === "Enter" && cmdkFilteredKeys[cmdkSelectedIndex]) {
      selectCmdkItem(cmdkFilteredKeys[cmdkSelectedIndex]);
    }
  }
});

function updateCmdkSelection() {
  cmdkResults.querySelectorAll(".cmdk-item").forEach((item, i) => {
    item.classList.toggle("selected", i === cmdkSelectedIndex);
  });
  const el = cmdkResults.querySelectorAll(".cmdk-item")[cmdkSelectedIndex];
  if (el) el.scrollIntoView({ block: "nearest" });
}

// ============================================================
// INIT
// ============================================================

window.__SAMPLES__ = Array.from(document.querySelectorAll('input[name="sample"]')).map(el => el.value);
updateHistoryBadge();
navigateTo("home");

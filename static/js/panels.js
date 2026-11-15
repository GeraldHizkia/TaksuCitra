// ==========================================================
// TaksuCitra - Konfigurasi Panel
// Setiap entry mendefinisikan: title, desc, endpoint, fields,
// explain, dan render(data) untuk menampilkan hasil.
// ==========================================================

function resultGrid(items) {
  return `<div class="result-grid">` +
    items.map(it => `<div class="result-item"><img src="${it.img}"><div class="cap">${it.cap}</div></div>`).join("") +
    `</div>`;
}

function beforeAfterSlider(beforeSrc, afterSrc, id) {
  return `
    <div class="ba-slider" id="${id}">
      <img src="${beforeSrc}" class="ba-before">
      <img src="${afterSrc}" class="ba-after">
      <div class="ba-handle"></div>
      <div class="ba-label ba-label-before">SEBELUM</div>
      <div class="ba-label ba-label-after">SESUDAH</div>
    </div>`;
}

function initBeforeAfterSlider(id) {
  const el = document.getElementById(id);
  if (!el) return;
  const afterImg = el.querySelector(".ba-after");
  const handle = el.querySelector(".ba-handle");

  function setPosition(clientX) {
    const rect = el.getBoundingClientRect();
    let pct = ((clientX - rect.left) / rect.width) * 100;
    pct = Math.max(0, Math.min(100, pct));
    afterImg.style.clipPath = `inset(0 0 0 ${pct}%)`;
    handle.style.left = `${pct}%`;
  }

  let dragging = false;
  el.addEventListener("mousedown", (e) => { dragging = true; setPosition(e.clientX); });
  window.addEventListener("mousemove", (e) => { if (dragging) setPosition(e.clientX); });
  window.addEventListener("mouseup", () => dragging = false);
  el.addEventListener("touchstart", (e) => { dragging = true; setPosition(e.touches[0].clientX); });
  el.addEventListener("touchmove", (e) => { if (dragging) setPosition(e.touches[0].clientX); });
  window.addEventListener("touchend", () => dragging = false);
}

function saveButtonHTML(panelKey) {
  return `<button class="btn-save-history" data-save-panel="${panelKey}">Simpan ke Riwayat</button>`;
}

function renderParamSummary(items) {
  if (!items || items.length === 0) return "";
  return `<div class="param-summary-bar">
    <div class="param-summary-title"><span class="material-symbols-outlined" style="font-size:16px;">tune</span> Parameter:</div>` +
    items.map(([label, val]) => `<span class="param-chip">${label}: <strong>${val}</strong></span>`).join("") +
    `</div>`;
}

function renderQualityMetrics(statsBefore, statsAfter) {
  if (!statsBefore || !statsAfter) return "";
  const bDiff = (statsAfter.brightness - statsBefore.brightness).toFixed(1);
  const cDiff = (statsAfter.contrast - statsBefore.contrast).toFixed(1);
  const hDiff = (statsAfter.entropy - statsBefore.entropy).toFixed(2);
  return `
    <div class="metric-cards-grid">
      <div class="metric-card">
        <div class="metric-card-label">Kecerahan (Mean)</div>
        <div class="metric-card-value">${statsAfter.brightness}</div>
        <div class="metric-card-sub">${statsBefore.brightness} &rarr; ${statsAfter.brightness} (${bDiff >= 0 ? '+' : ''}${bDiff})</div>
      </div>
      <div class="metric-card">
        <div class="metric-card-label">Kontras (RMS Std)</div>
        <div class="metric-card-value">${statsAfter.contrast}</div>
        <div class="metric-card-sub">${statsBefore.contrast} &rarr; ${statsAfter.contrast} (${cDiff >= 0 ? '+' : ''}${cDiff})</div>
      </div>
      <div class="metric-card">
        <div class="metric-card-label">Entropi Informasi</div>
        <div class="metric-card-value">${statsAfter.entropy}</div>
        <div class="metric-card-sub">${statsBefore.entropy} &rarr; ${statsAfter.entropy} (${hDiff >= 0 ? '+' : ''}${hDiff})</div>
      </div>
    </div>`;
}

function renderHistComparison(histBefore, histAfter) {
  if (!histBefore || !histAfter) return "";
  const maxB = Math.max(...histBefore, 1);
  const maxA = Math.max(...histAfter, 1);
  const barsB = histBefore.map(v => `<div class="hist-bar" style="height:${Math.max(2, (v / maxB) * 100)}%"></div>`).join("");
  const barsA = histAfter.map(v => `<div class="hist-bar" style="height:${Math.max(2, (v / maxA) * 100)}%"></div>`).join("");
  return `
    <div class="hist-comparison-wrap">
      <div class="hist-box">
        <h4>Distribusi Histogram Sebelum</h4>
        <div class="mini-hist-canvas-wrap">${barsB}</div>
      </div>
      <div class="hist-box">
        <h4>Distribusi Histogram Sesudah</h4>
        <div class="mini-hist-canvas-wrap">${barsA}</div>
      </div>
    </div>`;
}

function renderDenoiseEvalMetrics(metrics) {
  if (!metrics) return "";
  return `
    <div class="metric-cards-grid" style="margin-bottom:14px;">
      <div class="metric-card">
        <div class="metric-card-label">PSNR Noisy</div>
        <div class="metric-card-value">${metrics.psnr_noisy} dB</div>
        <div class="metric-card-sub">SSIM: ${metrics.ssim_noisy}</div>
      </div>
      <div class="metric-card">
        <div class="metric-card-label">PSNR Denoised</div>
        <div class="metric-card-value metric-card-positive">${metrics.psnr_denoised} dB</div>
        <div class="metric-card-sub">SSIM: ${metrics.ssim_denoised}</div>
      </div>
      <div class="metric-card">
        <div class="metric-card-label">Peningkatan PSNR</div>
        <div class="metric-card-value metric-card-positive">+${metrics.psnr_improvement} dB</div>
        <div class="metric-card-sub">${metrics.psnr_improvement > 0 ? "Kualitas Membaik" : "Stabil"}</div>
      </div>
    </div>`;
}

// ================= MATCHING LAB: Generic Step Pipeline Renderer =================
let currentMLData = null;
let currentMLStepIdx = 0;
let mlAutoplayInterval = null;

function renderMLGenericStepPipeline(data) {
  if (!data || !data.ok) {
    return `
      <div class="ml-card">
        <div class="ml-verdict-banner" style="background: rgba(255, 68, 68, 0.15); border-color: #ff4444; color: #ff8888;">
          <span class="material-symbols-outlined">error</span>
          <span>${data.error || "Gagal memproses pencocokan."}</span>
        </div>
        ${data.hint ? `<p style="font-size: 13px; color: var(--text-dim);">💡 Petunjuk: ${data.hint}</p>` : ""}
      </div>`;
  }

  currentMLData = data;
  currentMLStepIdx = 0;
  if (mlAutoplayInterval) { clearInterval(mlAutoplayInterval); mlAutoplayInterval = null; }

  const steps = data.steps || [];
  if (steps.length === 0) return `<div class="ml-card"><p>Tidak ada langkah yang dilaporkan.</p></div>`;

  let html = `<div class="ml-container">`;

  // Ringkasan Utama
  html += `
    <div class="ml-verdict-banner">
      <span class="material-symbols-outlined">verified</span>
      <span>${data.summary || "Proses pencocokan selesai."}</span>
      <span style="margin-left: auto; font-family: var(--mono); font-size: 11px; color: var(--text-dim);">${data.timing_ms || 0} ms</span>
    </div>`;

  // Warnings jika ada
  if (data.warnings && data.warnings.length > 0) {
    html += `
      <div style="background: rgba(255, 181, 157, 0.1); border: 1px solid var(--terracotta); border-radius: 8px; padding: 10px 14px; font-size: 12px; color: var(--terracotta);">
        ${data.warnings.map(w => `<div>⚠️ ${w}</div>`).join("")}
      </div>`;
  }

  // Stepper Tabs
  html += `<div class="ml-stepper" id="ml-stepper-tabs">`;
  steps.forEach((s, i) => {
    html += `
      <button class="ml-step-tab ${i === 0 ? "active" : ""}" data-ml-step="${i}" type="button">
        <span class="ml-step-num">${s.id || (i + 1)}</span>
        <span>${s.title}</span>
      </button>`;
  });
  html += `</div>`;

  // Stepper Controls (Play, Prev, Next, Grid Toggle, Download)
  html += `
    <div class="ml-stepper-ctrls">
      <button class="ml-btn-gold" id="ml-btn-prev" type="button"><span class="material-symbols-outlined">navigate_before</span> Prev</button>
      <button class="ml-btn-gold" id="ml-btn-next" type="button">Next <span class="material-symbols-outlined">navigate_next</span></button>
      <button class="ml-btn-sec" id="ml-btn-play" type="button"><span class="material-symbols-outlined">play_arrow</span> Autoplay</button>
      <button class="ml-btn-sec" id="ml-btn-toggle-grid" type="button"><span class="material-symbols-outlined">grid_view</span> Tampilkan Semua</button>
      <a class="ml-btn-sec" id="ml-btn-download" download="step.png" style="text-decoration:none;" href="#"><span class="material-symbols-outlined">download</span> Unduh PNG</a>
    </div>`;

  // Single Step Dynamic Container
  html += `<div id="ml-active-step-card" class="ml-card"></div>`;

  // Grid View Semua Langkah (Collapsible / Toggle)
  html += `<div id="ml-all-steps-grid" class="ml-grid-view" style="display: none;">`;
  steps.forEach((s, idx) => {
    html += `
      <div class="ml-grid-item">
        <div style="display: flex; justify-content: space-between; font-size: 11.5px; color: var(--accent); font-weight: 600;">
          <span>Langkah ${s.id || (idx+1)}</span>
          <span style="color: var(--text-dim);">${s.title}</span>
        </div>
        <img src="${s.image}" alt="${s.title}">
        <div class="ml-grid-cap">${s.explanation || ""}</div>
      </div>`;
  });
  html += `</div>`;

  html += `</div>`;
  setTimeout(() => initMLStepEvents(), 50);
  return html;
}

function updateMLActiveStepView() {
  if (!currentMLData || !currentMLData.steps) return;
  const step = currentMLData.steps[currentMLStepIdx];
  const container = document.getElementById("ml-active-step-card");
  if (!container || !step) return;

  // Update tabs active state
  document.querySelectorAll(".ml-step-tab").forEach((tab, idx) => {
    tab.classList.toggle("active", idx === currentMLStepIdx);
  });

  // Render konten kartu
  let inner = `
    <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 1px solid var(--border); padding-bottom: 10px;">
      <div>
        <h3 style="margin: 0; color: var(--accent); font-family: var(--font-display); font-size: 17px;">
          Langkah ${step.id || (currentMLStepIdx + 1)}: ${step.title}
        </h3>
        <p style="margin: 4px 0 0; color: var(--text-dim); font-size: 13px;">${step.explanation}</p>
      </div>
      <span class="ml-formula-chip">${step.formula || "Komputasi Matriks"}</span>
    </div>`;

  // Gambar Utama
  if (step.image) {
    inner += `
      <div class="ml-step-viewer">
        <img src="${step.image}" id="ml-current-img" alt="${step.title}">
      </div>`;
  }

  // Galeri Gambar Tambahan (misal scanline frames / sub-views)
  if (step.images && step.images.length > 0) {
    inner += `
      <div style="display: flex; gap: 8px; overflow-x: auto; padding: 6px 0;">
        ${step.images.map(im => `
          <div style="flex-shrink: 0; text-align: center;">
            <img src="${im.image}" style="height: 100px; border-radius: 6px; border: 1px solid var(--border);">
            <div style="font-size: 10px; color: var(--text-dim); margin-top: 2px;">${im.label}</div>
          </div>`).join("")}
      </div>`;
  }

  // Tabel Metrik jika tersedia
  if (step.table && step.table.rows && step.table.rows.length > 0) {
    inner += `
      <table class="ml-metrics-table">
        <thead>
          <tr>${step.table.headers.map(h => `<th>${h}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${step.table.rows.map(row => `<tr>${row.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>`;
  }

  container.innerHTML = inner;

  // Update download link
  const dlBtn = document.getElementById("ml-btn-download");
  if (dlBtn && step.image) {
    dlBtn.href = step.image;
    dlBtn.download = `step_${step.id || (currentMLStepIdx+1)}_${(step.title || "match").replace(/\s+/g, "_")}.png`;
  }
}

function initMLStepEvents() {
  updateMLActiveStepView();

  // Tab click
  document.querySelectorAll(".ml-step-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      currentMLStepIdx = parseInt(tab.dataset.mlStep);
      updateMLActiveStepView();
    });
  });

  // Prev / Next
  const btnPrev = document.getElementById("ml-btn-prev");
  const btnNext = document.getElementById("ml-btn-next");
  if (btnPrev) {
    btnPrev.onclick = () => {
      if (!currentMLData) return;
      currentMLStepIdx = Math.max(0, currentMLStepIdx - 1);
      updateMLActiveStepView();
    };
  }
  if (btnNext) {
    btnNext.onclick = () => {
      if (!currentMLData) return;
      currentMLStepIdx = Math.min(currentMLData.steps.length - 1, currentMLStepIdx + 1);
      updateMLActiveStepView();
    };
  }

  // Autoplay
  const btnPlay = document.getElementById("ml-btn-play");
  if (btnPlay) {
    btnPlay.onclick = () => {
      if (mlAutoplayInterval) {
        clearInterval(mlAutoplayInterval);
        mlAutoplayInterval = null;
        btnPlay.innerHTML = `<span class="material-symbols-outlined">play_arrow</span> Autoplay`;
      } else {
        btnPlay.innerHTML = `<span class="material-symbols-outlined">pause</span> Pause`;
        mlAutoplayInterval = setInterval(() => {
          if (!currentMLData || !currentMLData.steps) return;
          currentMLStepIdx = (currentMLStepIdx + 1) % currentMLData.steps.length;
          updateMLActiveStepView();
        }, 1200);
      }
    };
  }

  // Toggle Grid View
  const btnGrid = document.getElementById("ml-btn-toggle-grid");
  const gridView = document.getElementById("ml-all-steps-grid");
  const activeCard = document.getElementById("ml-active-step-card");
  if (btnGrid && gridView && activeCard) {
    btnGrid.onclick = () => {
      const isGrid = gridView.style.display !== "none";
      gridView.style.display = isGrid ? "none" : "grid";
      activeCard.style.display = isGrid ? "flex" : "none";
      btnGrid.innerHTML = isGrid
        ? `<span class="material-symbols-outlined">grid_view</span> Tampilkan Semua`
        : `<span class="material-symbols-outlined">view_carousel</span> Tampilan Stepper`;
    };
  }

  // Keyboard navigation
  window.onkeydown = (e) => {
    if (!currentMLData) return;
    if (e.key === "ArrowLeft") {
      currentMLStepIdx = Math.max(0, currentMLStepIdx - 1);
      updateMLActiveStepView();
    } else if (e.key === "ArrowRight") {
      currentMLStepIdx = Math.min(currentMLData.steps.length - 1, currentMLStepIdx + 1);
      updateMLActiveStepView();
    }
  };
}

// ================= FEATURE EXTRACTION LAB: helper & konstanta =================

// Dipakai render() semua panel FE Lab -- menampilkan proses bertahap
// (Revisi 1 dosen): tiap step punya judul, gambar PNG, dan caption.
function renderFEStepGallery(data) {
  const steps = data.steps || [];
  let html = `<div class="fe-steps">`;
  steps.forEach((s, i) => {
    html += `<div class="fe-step">
      <div class="fe-step-header">
        <span class="fe-step-badge">${i + 1}</span>
        <span class="fe-step-title">${s.title}</span>
      </div>
      <img src="${s.image}" alt="${s.title}">
      ${s.caption ? `<p class="fe-step-caption">${s.caption}</p>` : ""}
    </div>`;
    if (i < steps.length - 1) html += `<div class="fe-step-connector"></div>`;
  });
  html += `</div>`;

  // Matching Engine punya info.verdict + info.metrics -> tampilkan khusus (lebih "wow")
  if (data.info && data.info.verdict) {
    html += `<div class="fe-verdict">${data.info.verdict}</div>`;
    if (data.info.metrics) {
      html += `<table class="fe-metric-table">` + Object.entries(data.info.metrics).map(([k, v]) =>
        `<tr><td>${k}</td><td>${typeof v === "number" ? v.toFixed(4) : v}</td></tr>`).join("") + `</table>`;
    }
  } else if (data.info && Object.keys(data.info).length) {
    html += `<div class="fe-info-grid">` + Object.entries(data.info).map(([k, v]) => {
      const val = (typeof v === "object" && v !== null) ? JSON.stringify(v) : v;
      return `<span class="tag-chip">${k}: ${val}</span>`;
    }).join("") + `</div>`;
  }

  if (data.dim) html += `<span class="fe-dim-badge">Dimensi vector: ${data.dim}D</span>`;
  return html;
}

// REGISTRY function names di feature_extraction_lab.py -- dipakai dropdown
// "Fitur yang Diuji/Dibandingkan" pada Robustness Stress-Test & Matching Engine.
const FE_FEATURE_KEY_OPTIONS = [
  ["hu_moments_invariant", "Hu Moments"],
  ["intensity_statistics", "Statistik Intensitas"],
  ["shape_descriptors", "Shape Descriptors"],
  ["glcm_explorer", "GLCM Explorer"],
  ["gabor_single", "Gabor Single"],
  ["hog_feature", "HOG"],
  ["dct_feature", "DCT"],
  ["fft_radial_profile", "FFT Radial Profile"],
  ["wavelet_feature", "Wavelet"],
  ["lbp_explorer", "LBP Explorer (histogram)"],
  ["color_histogram_per_channel", "Histogram Warna (histogram)"],
  ["mb_lbp_features", "MB-LBP (histogram)"],
];

// Kunci panel FE Lab (harus sama persis dengan FE_PIPELINE_MAP di app.py) --
// dipakai checklist "Pilih Fitur" pada Concat Playground.
const FE_CONCAT_OPTIONS = [
  ["fe-intensity", "Statistik Intensitas"], ["fe-color-stats", "Statistik Warna"],
  ["fe-color-hist", "Histogram Warna"], ["fe-shape", "Shape Descriptors"],
  ["fe-moments", "Raw/Central Moments"], ["fe-hu", "Hu Moments"],
  ["fe-glcm", "GLCM Explorer"], ["fe-lbp", "LBP Explorer"], ["fe-ltp", "LTP"],
  ["fe-clbp", "CLBP"], ["fe-cslbp", "CS-LBP"], ["fe-mblbp", "MB-LBP"],
  ["fe-gabor-single", "Gabor Single"], ["fe-gabor-bank", "Gabor Bank"],
  ["fe-hog", "HOG"], ["fe-edge", "Fitur Tepi"], ["fe-projection", "Projection Profile"],
  ["fe-dct", "DCT"], ["fe-fft-radial", "FFT Radial"], ["fe-wavelet", "Wavelet"],
];

const PANELS = {

  // ================= BAB 2 - OPERASI PIKSEL =================
  "channel-split": {
    title: "Perubahan Warna (Channel Split)",
    desc: "Memisahkan citra menjadi kanal Red, Green, dan Blue secara terpisah.",
    endpoint: "/api/piksel/channel-split",
    fields: [],
    explain: "Setiap kanal warna (R/G/B) ditampilkan sendiri-sendiri dengan dua kanal lain diset ke nol, memakai <code>cv2.split()</code>.",
    render: (data) => resultGrid([
      { img: data.original, cap: "Citra Asli" },
      { img: data.red_channel, cap: "Red Channel" },
      { img: data.green_channel, cap: "Green Channel" },
      { img: data.blue_channel, cap: "Blue Channel" },
    ]) + saveButtonHTML("channel-split"),
  },

  "denoise": {
    title: "Pemadaman Noise",
    desc: "Menghilangkan noise pada citra: Median, Bilateral, Wavelet, Non-Local Means, Wiener (FFT), Low-pass.",
    endpoint: "/api/piksel/denoise",
    fields: [
      { type: "select", name: "method", label: "Metode Denoising", options: [
        ["median", "Median Filter"], ["bilateral", "Bilateral Filter"], ["nlm", "Non-Local Means"],
        ["wavelet", "Wavelet Denoising"], ["wiener", "Wiener Filter (FFT)"], ["lowpass", "Low-pass Filter (FFT)"],
      ]},
      { type: "range", name: "strength", label: "Kekuatan Filter", min: 3, max: 15, step: 2, value: 5 },
    ],
    explain: "Enam metode denoising sesuai Bab 2.1 buku, dari domain spasial hingga domain frekuensi (FFT).",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-denoise") + saveButtonHTML("denoise"),
    sliderIds: ["ba-denoise"],
  },

  "brightness": {
    title: "Perubahan Kecerahan",
    desc: "Meningkatkan/menurunkan intensitas piksel secara keseluruhan.",
    endpoint: "/api/piksel/brightness",
    fields: [{ type: "range", name: "factor", label: "Faktor Kecerahan", min: 0.2, max: 3, step: 0.1, value: 1.5 }],
    explain: "<code>cv2.convertScaleAbs(img, beta=...)</code> menambah nilai piksel sesuai faktor kecerahan.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-brightness") + saveButtonHTML("brightness"),
    sliderIds: ["ba-brightness"],
  },

  "contrast": {
    title: "Perubahan Kontras",
    desc: "Meningkatkan perbedaan antara area gelap dan terang.",
    endpoint: "/api/piksel/contrast",
    fields: [{ type: "range", name: "factor", label: "Faktor Kontras", min: 0.2, max: 3, step: 0.1, value: 1.5 }],
    explain: "<code>cv2.convertScaleAbs(img, alpha=contrast_factor)</code>.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-contrast") + saveButtonHTML("contrast"),
    sliderIds: ["ba-contrast"],
  },

  "sharpen": {
    title: "Filtrasi (Sharpening)",
    desc: "Menerapkan kernel konvolusi untuk mempertajam gambar.",
    endpoint: "/api/piksel/sharpen",
    fields: [{ type: "select", name: "variant", label: "Kernel", options: [["1", "Kernel Sharpening 1"], ["2", "Kernel Sharpening 2 (lebih kuat)"]] }],
    explain: "<code>cv2.filter2D</code> dengan kernel penguatan 3x3.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-sharpen") + saveButtonHTML("sharpen"),
    sliderIds: ["ba-sharpen"],
  },

  "rotate-scale": {
    title: "Rotasi & Skala",
    desc: "Merotasi citra dan mengubah ukurannya (scaling).",
    endpoint: "/api/piksel/rotate-scale",
    fields: [
      { type: "range", name: "angle", label: "Sudut Rotasi (derajat)", min: -180, max: 180, step: 5, value: 45 },
      { type: "range", name: "scale", label: "Faktor Skala", min: 0.1, max: 2, step: 0.1, value: 0.5 },
    ],
    explain: "<code>cv2.getRotationMatrix2D</code> + <code>warpAffine</code>, lalu <code>cv2.resize</code>.",
    render: (data) => resultGrid([
      { img: data.original, cap: "Asli" }, { img: data.rotated, cap: "Setelah Rotasi" }, { img: data.scaled, cap: "Setelah Rotasi + Skala" },
    ]) + saveButtonHTML("rotate-scale"),
  },

  "blend": {
    title: "Penggabungan Gambar",
    desc: "Menggabungkan dua citra menggunakan alpha blending.",
    endpoint: "/api/piksel/blend",
    fields: [{ type: "range", name: "alpha", label: "Alpha (transparansi citra 1)", min: 0, max: 1, step: 0.05, value: 0.5 }],
    customImageSource: "dual-sample",
    explain: "<code>cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)</code>.",
    render: (data) => resultGrid([{ img: data.image1, cap: "Gambar 1" }, { img: data.image2, cap: "Gambar 2" }, { img: data.result, cap: "Hasil Blending" }]) + saveButtonHTML("blend"),
  },

  // ================= BAB 2 - KONVERSI GAMBAR =================
  "colorspace": {
    title: "Konversi Ruang Warna",
    desc: "Mengonversi citra RGB ke ruang warna lain.",
    endpoint: "/api/konversi/colorspace",
    fields: [{ type: "select", name: "target", label: "Target Ruang Warna", options: [["gray", "Grayscale"], ["hsv", "HSV"], ["ycrcb", "YCrCb"], ["binary", "Biner (Black & White)"]] }],
    explain: "<code>cv2.cvtColor</code> dengan berbagai flag konversi ruang warna.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-colorspace") + saveButtonHTML("colorspace"),
    sliderIds: ["ba-colorspace"],
  },

  "resolution": {
    title: "Konversi Resolusi",
    desc: "Mengubah ukuran citra menjadi thumbnail.",
    endpoint: "/api/konversi/resolution",
    fields: [{ type: "select", name: "size", label: "Ukuran Thumbnail", options: [["32", "32 x 32"], ["64", "64 x 64"], ["128", "128 x 128"], ["256", "256 x 256"]] }],
    explain: "<code>img.thumbnail((size, size))</code> dari PIL.",
    render: (data) => `<div class="result-grid">
        <div class="result-item"><img src="${data.original}"><div class="cap">Asli (${data.original_dims})</div></div>
        <div class="result-item"><img src="${data.result}" style="image-rendering:pixelated"><div class="cap">Thumbnail (${data.result_dims})</div></div>
      </div>` + saveButtonHTML("resolution"),
  },

  "bitdepth": {
    title: "Konversi Bit Depth",
    desc: "Mengurangi jumlah level intensitas warna per piksel.",
    endpoint: "/api/konversi/bitdepth",
    fields: [{ type: "select", name: "bit_depth", label: "Bit Depth", options: [["1", "1-bit (2 level)"], ["2", "2-bit (4 level)"], ["4", "4-bit (16 level)"], ["6", "6-bit (64 level)"], ["8", "8-bit (256 level)"]] }],
    explain: "Kuantisasi ulang nilai piksel ke rentang bit depth baru.",
    render: (data) => `<div class="result-grid">
        <div class="result-item"><img src="${data.original}"><div class="cap">Asli (256 level)</div></div>
        <div class="result-item"><img src="${data.result}"><div class="cap">Hasil (${data.levels} level)</div></div>
      </div>` + saveButtonHTML("bitdepth"),
  },

  "format": {
    title: "Konversi Format Gambar",
    desc: "Membandingkan ukuran file pada format JPEG, PNG, dan BMP.",
    endpoint: "/api/konversi/format",
    fields: [],
    explain: "Perbedaan ukuran file akibat kompresi lossy vs lossless.",
    render: (data) => `<div class="result-grid"><div class="result-item"><img src="${data.original}"><div class="cap">Citra Sumber</div></div></div>
      <table class="info-table" style="margin-top:16px">${Object.entries(data.sizes).map(([f, s]) => `<tr><td>${f}</td><td>${(s/1024).toFixed(1)} KB</td></tr>`).join("")}</table>`,
  },

  // ================= BAB 2 - OPERASI HISTOGRAM =================
  "hist-analisis": {
    title: "Analisis Histogram", desc: "Menampilkan distribusi intensitas piksel.",
    endpoint: "/api/histogram/analisis", fields: [],
    explain: "<code>cv2.calcHist([img],[0],None,[256],[0,256])</code>.",
    render: (data) => resultGrid([{ img: data.original, cap: "Citra" }, { img: data.histogram, cap: "Histogram" }]),
  },
  "hist-equalization": {
    title: "Peningkatan Kontras (Equalization)", desc: "Meratakan histogram agar kontras meningkat.",
    endpoint: "/api/histogram/equalization", fields: [],
    explain: "<code>cv2.equalizeHist()</code> pada kanal luminansi (Y).",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-histeq") +
      resultGrid([{ img: data.hist_before, cap: "Histogram Sebelum" }, { img: data.hist_after, cap: "Histogram Sesudah" }]) + saveButtonHTML("hist-equalization"),
    sliderIds: ["ba-histeq"],
  },
  "hist-denoise": {
    title: "Pemadaman Noise (via Histogram)", desc: "Menghilangkan noise dan membandingkan histogram.",
    endpoint: "/api/histogram/denoise",
    fields: [{ type: "select", name: "method", label: "Metode", options: [["median", "Median Filter"], ["nlm", "Non-Local Means"]] }],
    explain: "Perbandingan histogram sebelum/sesudah denoising.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-histdenoise") +
      resultGrid([{ img: data.hist_before, cap: "Histogram Sebelum" }, { img: data.hist_after, cap: "Histogram Sesudah" }]),
    sliderIds: ["ba-histdenoise"],
  },
  "hist-edge": {
    title: "Deteksi Tepi", desc: "Mendeteksi tepi berdasarkan perubahan tajam intensitas.",
    endpoint: "/api/histogram/edge",
    fields: [{ type: "select", name: "method", label: "Metode", options: [["canny", "Canny Edge Detector"], ["sobel", "Sobel Operator"]] }],
    explain: "Deteksi tepi memanfaatkan gradien intensitas piksel.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-histedge") +
      resultGrid([{ img: data.hist_before, cap: "Histogram Asli" }, { img: data.hist_after, cap: "Histogram Tepi" }]),
    sliderIds: ["ba-histedge"],
  },
  "hist-segmentation": {
    title: "Segmentasi Gambar", desc: "Memisahkan objek dari latar belakang berdasarkan threshold.",
    endpoint: "/api/histogram/segmentation",
    fields: [{ type: "range", name: "threshold", label: "Nilai Threshold", min: 0, max: 255, step: 1, value: 127 }],
    explain: "<code>cv2.threshold(gray, T, 255, THRESH_BINARY)</code>.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-histseg") +
      resultGrid([{ img: data.hist_before, cap: "Histogram Asli" }, { img: data.hist_after, cap: "Histogram Hasil" }]),
    sliderIds: ["ba-histseg"],
  },
  "hist-intensity": {
    title: "Modifikasi Intensitas Piksel", desc: "Stretching atau shifting pada histogram intensitas.",
    endpoint: "/api/histogram/intensity",
    fields: [
      { type: "select", name: "mode", label: "Mode", options: [["stretch", "Histogram Stretching"], ["shift", "Histogram Shifting"]] },
      { type: "range", name: "shift", label: "Nilai Shift (mode Shifting)", min: -100, max: 100, step: 5, value: 30 },
    ],
    explain: "Modifikasi intensitas dengan menggeser/meregangkan histogram.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-histint") +
      resultGrid([{ img: data.hist_before, cap: "Histogram Sebelum" }, { img: data.hist_after, cap: "Histogram Sesudah" }]),
    sliderIds: ["ba-histint"],
  },

  // ================= MORFOLOGI MATEMATIKA =================
  "morf-dilasi": {
    title: "Dilasi Citra Biner",
    desc: "Membesarkan objek putih dan menyambung celah kecil pada citra biner.",
    endpoint: "/api/morfologi/dilasi",
    helpKey: "morf-dilasi",
    fields: [
      { type: "select", name: "se_shape", label: "Bentuk Structuring Element", options: [["ellipse", "Ellipse (Lingkaran/Halus)"], ["rect", "Rectangle (Persegi)"], ["cross", "Cross (Palang Salib)"]], value: "ellipse", hint: "Bentuk SE menentukan arah pembesaran objek" },
      { type: "range", name: "se_size", label: "Ukuran SE (Ganjil)", min: 3, max: 31, step: 2, value: 5, hint: "Ukuran SE harus ganjil (3–31). Makin besar makin kuat efeknya" },
      { type: "range", name: "iterations", label: "Jumlah Iterasi", min: 1, max: 10, step: 1, value: 1, hint: "Berapa kali operasi dilasi diulang berturut-turut" },
      { type: "select", name: "invert_mode", label: "Polaritas Biner", options: [["auto", "Otomatis (Objek = Area Minoritas)"], ["normal", "Normal (Tanpa Balik)"], ["invert", "Balikkan Polaritas (Invert)"]], value: "auto", hint: "Objek harus berwarna putih (foreground)" },
      { type: "checkbox", name: "use_manual", label: "Gunakan Implementasi Manual NumPy", value: false, hint: "Definisi A (+) B = {z | (B^)_z ∩ A != empty}" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["SE", `${data.se_shape} ${data.se_size}x${data.se_size}`],
        ["Iterasi", data.iterations],
        ["Polaritas", data.inverted ? "Dibalik (Inverted)" : "Normal"],
        ["Piksel Aktif", `${data.active_before} &rarr; ${data.active_after} (+${data.change_pct}%)`]
      ]);
      return summary +
        beforeAfterSlider(data.binary, data.result, "ba-morf-dilasi") +
        resultGrid([
          { img: data.original, cap: "Citra Asli" },
          { img: data.binary, cap: "Biner Praproses (Otsu)" },
          { img: data.result, cap: `Hasil Dilasi (${data.se_shape} ${data.se_size}px)` }
        ]) + saveButtonHTML("morf-dilasi");
    },
    sliderIds: ["ba-morf-dilasi"],
    parseFields: (v) => ({ ...v, se_size: parseInt(v.se_size, 10), iterations: parseInt(v.iterations, 10), use_manual: !!v.use_manual })
  },

  "morf-erosi": {
    title: "Erosi Citra Biner",
    desc: "Mengecilkan objek dan memutus sambungan tipis pada citra biner.",
    endpoint: "/api/morfologi/erosi",
    helpKey: "morf-erosi",
    fields: [
      { type: "select", name: "se_shape", label: "Bentuk Structuring Element", options: [["ellipse", "Ellipse (Lingkaran/Halus)"], ["rect", "Rectangle (Persegi)"], ["cross", "Cross (Palang Salib)"]], value: "ellipse", hint: "Bentuk SE menentukan pola pengikisan batas objek" },
      { type: "range", name: "se_size", label: "Ukuran SE (Ganjil)", min: 3, max: 31, step: 2, value: 5, hint: "Ukuran SE harus ganjil (3–31). Semakin besar makin agresif" },
      { type: "range", name: "iterations", label: "Jumlah Iterasi", min: 1, max: 10, step: 1, value: 1, hint: "Berapa kali erosi diulang berturut-turut" },
      { type: "select", name: "invert_mode", label: "Polaritas Biner", options: [["auto", "Otomatis (Objek = Area Minoritas)"], ["normal", "Normal (Tanpa Balik)"], ["invert", "Balikkan Polaritas (Invert)"]], value: "auto", hint: "Objek harus berwarna putih (foreground)" },
      { type: "checkbox", name: "use_manual", label: "Gunakan Implementasi Manual NumPy", value: false, hint: "Definisi A (-) B = {z | B_z subset A}" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["SE", `${data.se_shape} ${data.se_size}x${data.se_size}`],
        ["Iterasi", data.iterations],
        ["Polaritas", data.inverted ? "Dibalik (Inverted)" : "Normal"],
        ["Piksel Aktif", `${data.active_before} &rarr; ${data.active_after} (${data.change_pct}%)`]
      ]);
      return summary +
        beforeAfterSlider(data.binary, data.result, "ba-morf-erosi") +
        resultGrid([
          { img: data.original, cap: "Citra Asli" },
          { img: data.binary, cap: "Biner Praproses (Otsu)" },
          { img: data.result, cap: `Hasil Erosi (${data.se_shape} ${data.se_size}px)` }
        ]) + saveButtonHTML("morf-erosi");
    },
    sliderIds: ["ba-morf-erosi"],
    parseFields: (v) => ({ ...v, se_size: parseInt(v.se_size, 10), iterations: parseInt(v.iterations, 10), use_manual: !!v.use_manual })
  },

  "morf-opening": {
    title: "Opening (Buka)",
    desc: "Erosi lalu dilasi: membersihkan derau titik putih tanpa mengubah ukuran utama objek.",
    endpoint: "/api/morfologi/opening",
    helpKey: "morf-opening",
    fields: [
      { type: "select", name: "se_shape", label: "Bentuk Structuring Element", options: [["ellipse", "Ellipse (Lingkaran/Halus)"], ["rect", "Rectangle (Persegi)"], ["cross", "Cross (Palang Salib)"]], value: "ellipse", hint: "Pilih ellipse untuk membulatkan tonjolan secara alami" },
      { type: "range", name: "se_size", label: "Ukuran SE (Ganjil)", min: 3, max: 31, step: 2, value: 5, hint: "Pilih ukuran sedikit lebih besar daripada bintik derau putih" },
      { type: "select", name: "invert_mode", label: "Polaritas Biner", options: [["auto", "Otomatis (Objek = Area Minoritas)"], ["normal", "Normal (Tanpa Balik)"], ["invert", "Balikkan Polaritas (Invert)"]], value: "auto", hint: "Objek harus berwarna putih (foreground)" },
      { type: "checkbox", name: "use_manual", label: "Gunakan Implementasi Manual NumPy", value: false, hint: "A o B = (A (-) B) (+) B" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["SE", `${data.se_shape} ${data.se_size}x${data.se_size}`],
        ["Polaritas", data.inverted ? "Dibalik (Inverted)" : "Normal"],
        ["Noise Terbuang", `${data.removed_pixels} piksel`]
      ]);
      return summary +
        beforeAfterSlider(data.binary, data.result, "ba-morf-opening") +
        resultGrid([
          { img: data.original, cap: "Citra Asli" },
          { img: data.binary, cap: "Biner Praproses (Otsu)" },
          { img: data.result, cap: "Hasil Opening" },
          { img: data.diff, cap: `Noise yang Dibuang (${data.removed_pixels} px)` }
        ]) + saveButtonHTML("morf-opening");
    },
    sliderIds: ["ba-morf-opening"],
    parseFields: (v) => ({ ...v, se_size: parseInt(v.se_size, 10), use_manual: !!v.use_manual })
  },

  "morf-closing": {
    title: "Closing (Tutup)",
    desc: "Dilasi lalu erosi: menutup lubang hitam kecil dan menyambung celah tipis.",
    endpoint: "/api/morfologi/closing",
    helpKey: "morf-closing",
    fields: [
      { type: "select", name: "se_shape", label: "Bentuk Structuring Element", options: [["ellipse", "Ellipse (Lingkaran/Halus)"], ["rect", "Rectangle (Persegi)"], ["cross", "Cross (Palang Salib)"]], value: "ellipse", hint: "Pilih bentuk yang sesuai orientasi celah yang ingin ditutup" },
      { type: "range", name: "se_size", label: "Ukuran SE (Ganjil)", min: 3, max: 31, step: 2, value: 7, hint: "Harus lebih besar dari lebar celah atau retakan" },
      { type: "select", name: "invert_mode", label: "Polaritas Biner", options: [["auto", "Otomatis (Objek = Area Minoritas)"], ["normal", "Normal (Tanpa Balik)"], ["invert", "Balikkan Polaritas (Invert)"]], value: "auto", hint: "Objek harus berwarna putih (foreground)" },
      { type: "checkbox", name: "use_manual", label: "Gunakan Implementasi Manual NumPy", value: false, hint: "A • B = (A (+) B) (-) B" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["SE", `${data.se_shape} ${data.se_size}x${data.se_size}`],
        ["Polaritas", data.inverted ? "Dibalik (Inverted)" : "Normal"],
        ["Celah Terisi", `${data.filled_pixels} piksel`]
      ]);
      return summary +
        beforeAfterSlider(data.binary, data.result, "ba-morf-closing") +
        resultGrid([
          { img: data.original, cap: "Citra Asli" },
          { img: data.binary, cap: "Biner Praproses (Otsu)" },
          { img: data.result, cap: "Hasil Closing" },
          { img: data.filled, cap: `Celah yang Tertutup (${data.filled_pixels} px)` }
        ]) + saveButtonHTML("morf-closing");
    },
    sliderIds: ["ba-morf-closing"],
    parseFields: (v) => ({ ...v, se_size: parseInt(v.se_size, 10), use_manual: !!v.use_manual })
  },

  "morf-hit-or-miss": {
    title: "Hit-or-Miss Transform",
    desc: "Mendeteksi konfigurasi bentuk dan ukuran eksak (piksel terisolasi, titik ujung, persegi, kustom).",
    endpoint: "/api/morfologi/hit-or-miss",
    helpKey: "morf-hit-or-miss",
    customUI: "morf-hmt",
    fields: [
      { type: "select", name: "mode", label: "Mode Template / Target", options: [["isolated_pixel", "Piksel Terisolasi (Noise Garam 1px)"], ["endpoints", "Titik Ujung Garis (8 Template HMT)"], ["exact_square", "Persegi Tepat n x n"], ["custom", "Editor Kernel Kustom 3x3"]], value: "isolated_pixel", hint: "Pilih konfigurasi geometri biner yang ingin dicari persis" },
      { type: "range", name: "square_size", label: "Ukuran Persegi n (Mode Persegi)", min: 3, max: 41, step: 2, value: 21, hint: "Mencari kotak solid berukuran persis n x n piksel" },
      { type: "select", name: "invert_mode", label: "Polaritas Biner", options: [["auto", "Otomatis (Objek = Area Minoritas)"], ["normal", "Normal (Tanpa Balik)"], ["invert", "Balikkan Polaritas (Invert)"]], value: "auto", hint: "Objek harus berwarna putih (foreground)" },
      { type: "checkbox", name: "use_manual", label: "Gunakan Implementasi Manual NumPy", value: false, hint: "HMT(A) = (A (-) B_hit) ∩ (A^c (-) B_miss)" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["Mode", data.mode],
        ["Polaritas", data.inverted ? "Dibalik (Inverted)" : "Normal"],
        ["Kecocokan Ditemukan", `${data.match_count} lokasi`]
      ]);
      return summary +
        resultGrid([
          { img: data.binary, cap: "Biner Masukan" },
          { img: data.hitmiss, cap: `Peta Kecocokan HMT (${data.match_count} titik)` },
          { img: data.overlay, cap: "Sorotan Lokasi (Marker Emas)" }
        ]) + saveButtonHTML("morf-hit-or-miss");
    },
    parseFields: (v) => ({ ...v, square_size: parseInt(v.square_size, 10), use_manual: !!v.use_manual })
  },

  "morf-skeleton": {
    title: "Skeletonization & Topologi",
    desc: "Mereduksi objek menjadi kerangka 1 piksel dengan analisis titik ujung dan percabangan.",
    endpoint: "/api/morfologi/skeleton",
    helpKey: "morf-skeleton",
    fields: [
      { type: "select", name: "method", label: "Metode Skeleton", options: [["lee", "Lee (skimage - Ramping & Akurat)"], ["lantuejoul", "Lantuejoul (Morfologis Bersyarat)"], ["zhang_suen", "Zhang-Suen (Thinning Manual)"], ["medial_axis", "Medial Axis (+ Distance Transform)"]], value: "lee", hint: "Lantuejoul bersifat lossless; Lee menghasilkan garis 1-piksel terhubung" },
      { type: "range", name: "prune_len", label: "Pruning Spurs (Panjang Cabang Dibuang)", min: 0, max: 30, step: 1, value: 10, hint: "Pangkas cabang semu akibat tepi kasar (0 = tanpa pruning)" },
      { type: "checkbox", name: "show_endpoints", label: "Tandai Titik Ujung (Warna Merah)", value: true, hint: "Piksel skeleton dengan tepat 1 tetangga" },
      { type: "checkbox", name: "show_branches", label: "Tandai Titik Percabangan (Warna Biru)", value: true, hint: "Piksel persimpangan (>= 3 tetangga)" },
      { type: "select", name: "invert_mode", label: "Polaritas Biner", options: [["auto", "Otomatis (Objek = Area Minoritas)"], ["normal", "Normal (Tanpa Balik)"], ["invert", "Balikkan Polaritas (Invert)"]], value: "auto", hint: "Objek harus berwarna putih (foreground)" }
    ],
    render: (data) => {
      const summaryItems = [
        ["Metode", data.method],
        ["Pruning", `${data.prune_len} px`],
        ["Titik Ujung", `${data.endpoints_count} (Merah)`],
        ["Percabangan", `${data.branches_count} (Biru)`],
        ["Piksel Skeleton", `${data.skeleton_pixels} px`]
      ];
      if (data.recon_identical !== null) {
        summaryItems.push(["Rekonstruksi Lossless", data.recon_identical ? "Identik (100% Cocok)" : "Mendekati"]);
      }
      return renderParamSummary(summaryItems) +
        beforeAfterSlider(data.binary, data.skeleton, "ba-morf-skel") +
        resultGrid([
          { img: data.binary, cap: "Biner Masukan" },
          { img: data.skeleton, cap: `Skeleton 1-Piksel (${data.method})` },
          { img: data.overlay, cap: `Analisis Topologi (Ujung: ${data.endpoints_count}, Cabang: ${data.branches_count})` }
        ]) + saveButtonHTML("morf-skeleton");
    },
    sliderIds: ["ba-morf-skel"],
    parseFields: (v) => ({ ...v, prune_len: parseInt(v.prune_len, 10), show_endpoints: !!v.show_endpoints, show_branches: !!v.show_branches })
  },

  // ================= PENINGKATAN & ANALISIS CITRA =================
  "enh-brightness-contrast": {
    title: "Kecerahan & Kontras",
    desc: "Penyesuaian distribusi luminans citra: Linear, Gamma, Contrast Stretching, Equalization, dan CLAHE.",
    endpoint: "/api/enhancement/brightness-contrast",
    helpKey: "enh-brightness-contrast",
    fields: [
      { type: "select", name: "mode", label: "Metode Penyesuaian", options: [["linear", "Linear (Alpha Kontras & Beta Kecerahan)"], ["gamma", "Gamma Correction"], ["auto_gamma", "Gamma Otomatis (Target Luminans)"], ["contrast_stretch", "Contrast Stretching (Persentil)"], ["hist_equalize", "Histogram Equalization (Global)"], ["clahe", "CLAHE (Adaptif Kontras Terbatas)"]], value: "linear", hint: "Pilih algoritma koreksi kecerahan & kontras" },
      { type: "range", name: "alpha", label: "Alpha (Faktor Kontras Linear)", min: 0.2, max: 3.0, step: 0.1, value: 1.2, hint: "1.0 = normal, >1.0 kontras meningkat, <1.0 kontras menurun" },
      { type: "range", name: "beta", label: "Beta (Bias Kecerahan Linear)", min: -100, max: 100, step: 5, value: 10, hint: "Nilai penambah kecerahan (-100 s.d. +100)" },
      { type: "range", name: "gamma", label: "Gamma Value (Mode Gamma)", min: 0.1, max: 3.5, step: 0.1, value: 1.0, hint: "<1 mencerahkan area bayangan, >1 menggelapkan" },
      { type: "range", name: "target_mean", label: "Target Mean (Mode Auto-Gamma)", min: 0.1, max: 0.9, step: 0.05, value: 0.5, hint: "Rerata luminans yang dituju (0.5 = seimbang)" },
      { type: "range", name: "p_low", label: "Persentil Bawah p_low (Stretching)", min: 0, max: 15, step: 1, value: 1, hint: "Persentil bawah yang diregangkan ke 0" },
      { type: "range", name: "p_high", label: "Persentil Atas p_high (Stretching)", min: 85, max: 100, step: 1, value: 99, hint: "Persentil atas yang diregangkan ke 255" },
      { type: "range", name: "clip_limit", label: "Clip Limit (Mode CLAHE)", min: 1.0, max: 8.0, step: 0.5, value: 2.0, hint: "Batas penguatan kontras lokal per-tile" },
      { type: "range", name: "tile_size", label: "Ukuran Tile Grid (Mode CLAHE)", min: 4, max: 32, step: 2, value: 8, hint: "Ukuran pembagian grid lokal (8 = 8x8)" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["Mode", data.mode],
        ["Detail", data.extra_info.params || "-"]
      ]);
      const metricsHTML = renderQualityMetrics(data.stats_before, data.stats_after);
      const histHTML = renderHistComparison(data.hist_before, data.hist_after);
      return summary +
        metricsHTML +
        beforeAfterSlider(data.original, data.result, "ba-enh-bc") +
        histHTML +
        saveButtonHTML("enh-brightness-contrast");
    },
    sliderIds: ["ba-enh-bc"],
    parseFields: (v) => ({
      ...v,
      alpha: parseFloat(v.alpha),
      beta: parseFloat(v.beta),
      gamma: parseFloat(v.gamma),
      target_mean: parseFloat(v.target_mean),
      p_low: parseFloat(v.p_low),
      p_high: parseFloat(v.p_high),
      clip_limit: parseFloat(v.clip_limit),
      tile_size: parseInt(v.tile_size, 10)
    })
  },

  "enh-flip": {
    title: "Pencerminan (Flip / Mirror)",
    desc: "Membalik orientasi citra secara horizontal, vertikal, atau kedua sumbu (lossless).",
    endpoint: "/api/enhancement/flip",
    helpKey: "enh-flip",
    fields: [
      { type: "select", name: "mode", label: "Arah Pencerminan", options: [["h", "Horizontal (Kiri-Kanan)"], ["v", "Vertikal (Atas-Bawah)"], ["hv", "Horizontal + Vertikal (Rotasi 180°)"]], value: "h", hint: "Pencerminan berbasis slicing indeks matriks" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["Arah", data.mode_label],
        ["Dimensi", data.dimensions],
        ["Kualitas", "Lossless (Tanpa Interpolasi)"]
      ]);
      return summary +
        beforeAfterSlider(data.original, data.result, "ba-enh-flip") +
        saveButtonHTML("enh-flip");
    },
    sliderIds: ["ba-enh-flip"]
  },

  "enh-rotasi": {
    title: "Rotasi Citra",
    desc: "Merotasi citra dengan kelipatan 90° (lossless) atau sudut bebas dengan perluasan kanvas.",
    endpoint: "/api/enhancement/rotasi",
    helpKey: "enh-rotasi",
    fields: [
      { type: "select", name: "mode", label: "Mode Rotasi", options: [["free", "Sudut Bebas (-180° s.d. +180°)"], ["rot90", "Kelipatan 90° (Lossless)"]], value: "free", hint: "Sudut bebas memakai interpolasi bilinear; 90° murni permutasi indeks" },
      { type: "range", name: "angle", label: "Sudut Rotasi (derajat)", min: -180, max: 180, step: 1, value: 30, hint: "Sudut berlawanan arah jarum jam (positif)" },
      { type: "select", name: "times", label: "Kelipatan 90° (Mode 90°)", options: [["1", "90° Berlawanan Jarum Jam"], ["2", "180°"], ["3", "270° (90° Searah Jarum Jam)"]], value: "1", hint: "Rotasi 90 derajat bebas degradasi" },
      { type: "checkbox", name: "expand", label: "Perbesar Kanvas (Mencegah Sudut Terpotong)", value: true, hint: "Kanvas diperbesar otomatis sesuai bounding box baru" },
      { type: "range", name: "fill_color", label: "Warna Latar Isian Kosong", min: 0, max: 255, step: 5, value: 0, hint: "Warna area luar citra asli (0 = hitam, 255 = putih)" },
      { type: "checkbox", name: "use_manual", label: "Gunakan Rotasi Manual (Inverse Mapping Bilinear)", value: false, hint: "Implementasi manual inverse mapping vs OpenCV" }
    ],
    render: (data) => {
      const summaryItems = [
        ["Mode", data.mode === "rot90" ? "Kelipatan 90° (Lossless)" : `Bebas (${data.angle}°)`],
        ["Dimensi", `${data.orig_dims} &rarr; ${data.result_dims}`],
        ["Perbesar Kanvas", data.expand ? "Ya (Utuh)" : "Tidak (Terpotong)"]
      ];
      if (data.diff_vs_cv !== null) {
        summaryItems.push(["Selisih vs OpenCV", `${data.diff_vs_cv} (area dalam)`]);
      }
      return renderParamSummary(summaryItems) +
        beforeAfterSlider(data.original, data.result, "ba-enh-rotasi") +
        resultGrid([
          { img: data.original, cap: `Citra Asli (${data.orig_dims})` },
          { img: data.result, cap: `Hasil Rotasi (${data.result_dims})` }
        ]) + saveButtonHTML("enh-rotasi");
    },
    sliderIds: ["ba-enh-rotasi"],
    parseFields: (v) => ({ ...v, angle: parseFloat(v.angle), times: parseInt(v.times, 10), fill_color: parseInt(v.fill_color, 10), expand: !!v.expand, use_manual: !!v.use_manual })
  },

  "enh-edge-detection": {
    title: "Deteksi Tepi",
    desc: "Menemukan garis kontur objek dengan operator gradien (Sobel/Prewitt/Scharr), LoG, dan Canny.",
    endpoint: "/api/enhancement/edge-detection",
    helpKey: "enh-edge-detection",
    fields: [
      { type: "select", name: "method", label: "Metode Operator Tepi", options: [["canny_auto", "Auto-Canny OpenCV (Ambang Otomatis)"], ["canny_manual", "Canny Manual (NMS + Hysteresis Lengkap)"], ["sobel", "Sobel Filter (Turunan Pertama)"], ["prewitt", "Prewitt Filter"], ["scharr", "Scharr Filter (Gradien Presisi)"], ["log", "Laplacian of Gaussian (LoG - Zero Crossing)"]], value: "canny_auto", hint: "Sobel/Prewitt/Scharr cepat; Canny memberikan kontur 1-piksel bersih" },
      { type: "range", name: "sigma", label: "Sigma Gaussian Smoothing (LoG/Canny)", min: 0.5, max: 4.0, step: 0.1, value: 1.4, hint: "Makin besar sigma, noise makin ditekan tapi detail tepi berkurang" },
      { type: "range", name: "auto_sigma", label: "Sigma Ambang Auto-Canny", min: 0.1, max: 0.8, step: 0.05, value: 0.33, hint: "Faktor deviasi ambang dari nilai median intensitas (0.33 standar)" },
      { type: "text", name: "low_thresh", label: "Ambang Bawah (Canny Manual)", placeholder: "Kosongkan untuk otomatis (0.4 * high)", hint: "Batas bawah deteksi tepi lemah yang tersambung" },
      { type: "text", name: "high_thresh", label: "Ambang Atas (Canny Manual)", placeholder: "Kosongkan untuk persentil ke-92", hint: "Batas atas deteksi tepi kuat definitif" }
    ],
    render: (data) => {
      const summaryItems = [
        ["Metode", data.method],
        ["Piksel Tepi", `${data.edge_pixels} px (${data.edge_density_pct}%)`]
      ];
      if (data.agreement_f1 !== null) {
        summaryItems.push(["F1 Agreement vs OpenCV", `${data.agreement_f1}`]);
      }
      return renderParamSummary(summaryItems) +
        beforeAfterSlider(data.original, data.result, "ba-enh-edge") +
        resultGrid([
          { img: data.original, cap: "Citra Asli" },
          { img: data.result, cap: `Peta Kontur Tepi (${data.method})` }
        ]) + saveButtonHTML("enh-edge-detection");
    },
    sliderIds: ["ba-enh-edge"],
    parseFields: (v) => ({ ...v, sigma: parseFloat(v.sigma), auto_sigma: parseFloat(v.auto_sigma) })
  },

  "enh-noise-reduction": {
    title: "Pereduksian Derau (Denoise)",
    desc: "Meredam noise dengan filter Mean, Gaussian, Median, Adaptive Median, Bilateral, dan Non-Local Means.",
    endpoint: "/api/enhancement/noise-reduction",
    helpKey: "enh-noise-reduction",
    fields: [
      { type: "select", name: "method", label: "Metode Filter Denoise", options: [["median", "Median Filter (Salt & Pepper)"], ["adaptive_median", "Adaptive Median Filter (Salt & Pepper Ekstrem)"], ["nlm", "Non-Local Means (Gaussian / Tekstur Alami)"], ["bilateral", "Bilateral Filter (Tepi Tetap Tajam)"], ["gaussian", "Gaussian Blur"], ["mean", "Mean / Average Filter"]], value: "median", hint: "Pilih filter sesuai jenis gangguan derau pada citra" },
      { type: "range", name: "kernel_size", label: "Ukuran Kernel / Jendela (Ganjil)", min: 3, max: 15, step: 2, value: 5, hint: "Ukuran jendela pemfilteran spasial (harus ganjil)" },
      { type: "range", name: "sigma", label: "Sigma Spasial (Gaussian)", min: 0.5, max: 5.0, step: 0.5, value: 1.5, hint: "Standar deviasi blur Gaussian" },
      { type: "text", name: "h_param", label: "Parameter Kekuatan (h NLM / sigmaColor Bilateral)", placeholder: "Kosongkan untuk otomatis (Immerkaer estimate)", hint: "Nilai kekuatan peredaman; kosongkan untuk estimasi noise otomatis" },
      { type: "select", name: "add_test_noise", label: "Uji Coba: Tambah Noise Sintetis", options: [["none", "Citra Asli (Tanpa Noise Tambahan)"], ["salt_pepper", "Uji Salt & Pepper (Derau Impuls)"], ["gaussian", "Uji Gaussian (Noise Sensor Kamera)"], ["speckle", "Uji Speckle (Noise Multiplikatif)"], ["poisson", "Uji Poisson (Photon Noise)"]], value: "none", hint: "Beri noise sintetis untuk mengukur efektivitas filter (PSNR & SSIM)" },
      { type: "text", name: "noise_amount", label: "Tingkat Noise Uji (Opsional)", placeholder: "Default bawaan jenis noise (misal 0.05 untuk S&P)", hint: "Intensitas noise sintetis yang diinjeksikan" }
    ],
    render: (data) => {
      const summaryItems = [
        ["Filter", data.method],
        ["Estimasi Derau Sigma", `${data.estimated_noise_sigma}`]
      ];
      if (data.test_noise && data.test_noise !== "none") {
        summaryItems.push(["Jenis Uji", data.test_noise]);
      }
      const evalHTML = renderDenoiseEvalMetrics(data.metrics);
      const beforeImg = data.noisy || data.original;
      return renderParamSummary(summaryItems) +
        evalHTML +
        beforeAfterSlider(beforeImg, data.result, "ba-enh-denoise") +
        resultGrid([
          { img: data.original, cap: "Citra Asli" },
          ...(data.noisy ? [{ img: data.noisy, cap: `Citra Bernoise (${data.test_noise})` }] : []),
          { img: data.result, cap: `Hasil Denoise (${data.method})` }
        ]) + saveButtonHTML("enh-noise-reduction");
    },
    sliderIds: ["ba-enh-denoise"],
    parseFields: (v) => ({ ...v, kernel_size: parseInt(v.kernel_size, 10), sigma: parseFloat(v.sigma) })
  },

  "enh-pipeline": {
    title: "Pipeline Peningkatan Terpadu",
    desc: "Menjalankan rantai pemrosesan: Denoise -> CLAHE Kontras -> Deteksi Tepi Canny.",
    endpoint: "/api/enhancement/pipeline",
    helpKey: "enh-pipeline",
    fields: [
      { type: "select", name: "denoise_method", label: "Metode Denoise Tahap 1", options: [["nlm", "Non-Local Means (NLM)"], ["median", "Median Filter"]], value: "nlm", hint: "Tahap 1: Membersihkan noise awal sebelum kontras ditingkatkan" },
      { type: "range", name: "denoise_strength", label: "Kekuatan Denoise (h)", min: 3, max: 25, step: 1, value: 10, hint: "Kekuatan filter NLM pada tahap awal" },
      { type: "range", name: "clahe_clip", label: "CLAHE Clip Limit Tahap 2", min: 1.0, max: 5.0, step: 0.5, value: 2.5, hint: "Tahap 2: Peningkatan kontras lokal adaptif" },
      { type: "range", name: "edge_auto_sigma", label: "Sensitivitas Tepi Canny Tahap 3", min: 0.1, max: 0.6, step: 0.05, value: 0.33, hint: "Tahap 3: Deteksi garis kontur akhir" }
    ],
    render: (data) => {
      const summary = renderParamSummary([
        ["Alur Rantai", "1. Denoise &rarr; 2. CLAHE &rarr; 3. Canny Tepi"],
        ["Tepi Bersih", `${data.edge_count_clean} piksel`],
        ["Tepi Naive (Tanpa Denoise)", `${data.edge_count_naive} piksel (Kotor)`]
      ]);
      return summary +
        beforeAfterSlider(data.step2_clahe, data.step3_edges, "ba-enh-pipe") +
        resultGrid([
          { img: data.original, cap: "1. Input Asli" },
          { img: data.step1_denoised, cap: "2. Setelah Denoise" },
          { img: data.step2_clahe, cap: "3. Setelah CLAHE" },
          { img: data.step3_edges, cap: `4. Tepi Canny Bersih (${data.edge_count_clean} px)` },
          { img: data.naive_edges, cap: `Pembanding: Tepi Tanpa Denoise (${data.edge_count_naive} px - Kotor)` }
        ]) + saveButtonHTML("enh-pipeline");
    },
    sliderIds: ["ba-enh-pipe"],
    parseFields: (v) => ({ ...v, denoise_strength: parseFloat(v.denoise_strength), clahe_clip: parseFloat(v.clahe_clip), edge_auto_sigma: parseFloat(v.edge_auto_sigma) })
  },

  // ================= AI LAB =================
  "ai-face-blur": {
    title: "Deteksi & Anonimisasi Wajah",
    desc: "Mendeteksi wajah dengan Deep Learning (SSD ResNet10) lalu mem-blur otomatis untuk privasi.",
    endpoint: "/api/ai/face-blur",
    fields: [{ type: "select", name: "blur", label: "Mode", options: [["true", "Blur Wajah (anonimisasi)"], ["false", "Hanya Kotak Deteksi"]] }],
    explain: "Model <b>SSD dengan backbone ResNet-10 (Caffe)</b> — deep learning object detector, dijalankan lewat <code>cv2.dnn</code>.",
    render: (data) => `<div class="tag-list"><span class="tag-chip">${data.face_count} wajah terdeteksi</span></div>` +
      resultGrid([{ img: data.original, cap: "Asli" }, { img: data.detected, cap: "Terdeteksi" }, { img: data.result, cap: "Hasil" }]) + saveButtonHTML("ai-face-blur"),
    parseFields: (v) => ({ ...v, blur: v.blur === "true" }),
  },

  "ai-super-res": {
    title: "Super Resolution (AI Upscale)",
    desc: "Meningkatkan resolusi citra low-res menggunakan Convolutional Neural Network (ESPCN).",
    endpoint: "/api/ai/super-resolution",
    fields: [{ type: "select", name: "scale", label: "Faktor Upscale", options: [["2", "2x"], ["3", "3x"], ["4", "4x"]] }],
    explain: "<b>ESPCN (Efficient Sub-Pixel CNN)</b> — model deep learning untuk super-resolution, jauh lebih detail dibanding <code>cv2.resize</code> biasa.",
    render: (data) => `<div class="tag-list"><span class="tag-chip">${data.original_dims} to ${data.result_dims}</span></div>` +
      beforeAfterSlider(data.original, data.result, "ba-sr") + saveButtonHTML("ai-super-res"),
    sliderIds: ["ba-sr"],
  },

  "ai-object-detection": {
    title: "Deteksi Objek (20 Kelas)",
    desc: "Mendeteksi objek umum (orang, mobil, kucing, dll) menggunakan MobileNet-SSD.",
    endpoint: "/api/ai/object-detection",
    fields: [{ type: "range", name: "threshold", label: "Ambang Batas Confidence", min: 0.1, max: 0.9, step: 0.05, value: 0.4 }],
    explain: "<b>MobileNet-SSD</b> dilatih pada dataset Pascal VOC (20 kelas objek). Efisien untuk inference di CPU.",
    render: (data) => {
      const tags = data.objects.length
        ? data.objects.map(o => `<span class="tag-chip">${o.label} (${o.confidence}%)</span>`).join("")
        : `<span class="tag-chip">Tidak ada objek terdeteksi pada ambang batas ini</span>`;
      return `<div class="tag-list">${tags}</div>` + resultGrid([{ img: data.original, cap: "Asli" }, { img: data.result, cap: "Hasil Deteksi" }]) + saveButtonHTML("ai-object-detection");
    },
  },

  "ai-artistic": {
    title: "Artistic Rendering",
    desc: "Mengubah foto menjadi gaya artistik menggunakan algoritma non-photorealistic rendering.",
    endpoint: "/api/ai/artistic",
    fields: [{ type: "select", name: "style", label: "Gaya", options: [
      ["stylization", "Stylization (Cat Air)"], ["pencil_gray", "Sketsa Pensil (Hitam Putih)"],
      ["pencil_color", "Sketsa Pensil (Berwarna)"], ["detail_enhance", "Detail Enhancement"], ["edge_preserve", "Edge-Preserving Smooth"],
    ]}],
    explain: "Memakai modul <code>cv2.photo</code> (stylization, pencilSketch, detailEnhance) — algoritma edge-preserving filtering.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-artistic") + saveButtonHTML("ai-artistic"),
    sliderIds: ["ba-artistic"],
  },

  "ai-remove-bg": {
    title: "Penghapus Latar Belakang",
    desc: "Memisahkan objek utama dari latar belakang secara otomatis.",
    endpoint: "/api/ai/remove-background",
    fields: [],
    explain: "Algoritma <b>GrabCut</b> (iterative graph-cut segmentation) menghasilkan gambar transparan (PNG) siap pakai.",
    render: (data) => resultGrid([
      { img: data.original, cap: "Asli" }, { img: data.composited, cap: "Latar Putih" }, { img: data.transparent, cap: "PNG Transparan" },
    ]) + saveButtonHTML("ai-remove-bg"),
  },

  // ================= ADVANCED OPS =================
  "adv-inpaint": {
    title: "Magic Eraser (Inpainting)",
    desc: "Hapus objek dari gambar dengan menggambar area yang ingin dihilangkan.",
    endpoint: "/api/advanced/inpaint",
    customUI: "mask-draw",
    fields: [
      { type: "select", name: "method", label: "Algoritma", options: [["telea", "Telea (Fast Marching)"], ["ns", "Navier-Stokes"]] },
      { type: "range", name: "radius", label: "Radius Inpainting", min: 1, max: 20, step: 1, value: 7 },
    ],
    explain: "<code>cv2.inpaint()</code> merekonstruksi area yang ditandai menggunakan informasi tekstur di sekitarnya.",
    render: (data) => beforeAfterSlider(data.original, data.result, "ba-inpaint") + saveButtonHTML("adv-inpaint"),
    sliderIds: ["ba-inpaint"],
  },

  "adv-panorama": {
    title: "Panorama Stitching",
    desc: "Menggabungkan beberapa foto menjadi satu panorama lebar.",
    endpoint: "/api/advanced/panorama",
    customUI: "multi-upload",
    fields: [],
    explain: "<code>cv2.Stitcher</code> mendeteksi fitur antar gambar, mengestimasi homografi, lalu menyatukannya.",
    render: (data) => resultGrid(data.inputs.map((im, i) => ({ img: im, cap: `Input ${i+1}` }))) +
      `<div style="margin-top:16px"><h3 style="margin-bottom:8px">Hasil Panorama</h3><img src="${data.result}" style="width:100%;border-radius:10px;border:1px solid var(--border)"></div>` + saveButtonHTML("adv-panorama"),
  },

  "adv-morphology": {
    title: "Morphological Playground",
    desc: "Eksplorasi operasi morfologi pada citra biner: erosion, dilation, opening, closing, dll.",
    endpoint: "/api/advanced/morphology",
    fields: [
      { type: "select", name: "operation", label: "Operasi", options: [
        ["erode", "Erosion"], ["dilate", "Dilation"], ["opening", "Opening"], ["closing", "Closing"],
        ["gradient", "Morphological Gradient"], ["tophat", "Top Hat"], ["blackhat", "Black Hat"],
      ]},
      { type: "range", name: "kernel_size", label: "Ukuran Kernel", min: 1, max: 21, step: 2, value: 5 },
      { type: "range", name: "iterations", label: "Iterasi", min: 1, max: 10, step: 1, value: 1 },
    ],
    explain: "Operasi morfologi bekerja pada citra biner (hasil Otsu thresholding) menggunakan structuring element.",
    render: (data) => resultGrid([{ img: data.original, cap: "Asli" }, { img: data.binary, cap: "Biner (Otsu)" }, { img: data.result, cap: "Hasil Morfologi" }]) + saveButtonHTML("adv-morphology"),
  },

  "adv-frequency": {
    title: "Frequency Domain Explorer",
    desc: "Visualisasi spektrum FFT dan efek filter ideal low-pass/high-pass di domain frekuensi.",
    endpoint: "/api/advanced/frequency",
    fields: [
      { type: "select", name: "filter_type", label: "Jenis Filter", options: [["none", "Tanpa Filter"], ["lowpass", "Ideal Low-pass"], ["highpass", "Ideal High-pass"]] },
      { type: "range", name: "radius", label: "Radius Cutoff", min: 5, max: 150, step: 5, value: 40 },
    ],
    explain: "Transformasi Fourier 2D (<code>np.fft.fft2</code>) mengubah citra ke domain frekuensi. Filter ideal berupa mask lingkaran biner.",
    render: (data) => resultGrid([
      { img: data.original, cap: "Citra Asli" }, { img: data.spectrum, cap: "Spektrum FFT (Magnitude)" },
      { img: data.filtered_spectrum, cap: "Spektrum Setelah Filter" }, { img: data.result, cap: "Hasil Rekonstruksi (Inverse FFT)" },
    ]) + saveButtonHTML("adv-frequency"),
  },

  "adv-feature-matching": {
    title: "Feature Matching (ORB)",
    desc: "Mencocokkan titik fitur unik antara dua gambar — dasar dari image stitching & object recognition.",
    endpoint: "/api/advanced/feature-matching",
    customImageSource: "dual-sample",
    fields: [],
    explain: "<b>ORB (Oriented FAST and Rotated BRIEF)</b> mendeteksi keypoint, lalu dicocokkan dengan Brute-Force Hamming Matcher.",
    render: (data) => `<div class="tag-list">
        <span class="tag-chip">Keypoints Gambar 1: ${data.kp1_count}</span>
        <span class="tag-chip">Keypoints Gambar 2: ${data.kp2_count}</span>
        <span class="tag-chip">Total Match: ${data.match_count}</span>
      </div><img src="${data.result}" style="width:100%;border-radius:10px;border:1px solid var(--border);margin-top:12px">` + saveButtonHTML("adv-feature-matching"),
  },

  "adv-qr": {
    title: "QR / Barcode Scanner",
    desc: "Mendeteksi dan mendekode QR code dari gambar.",
    endpoint: "/api/advanced/qr-scan",
    fields: [],
    explain: "<code>cv2.QRCodeDetector().detectAndDecodeMulti()</code> — mendukung multi-QR dalam satu gambar.",
    render: (data) => {
      const tags = data.codes.length
        ? data.codes.map(c => `<span class="tag-chip">${c.detected ? "OK: " : "Gagal: "} ${c.text || "(tidak terbaca)"}</span>`).join("")
        : `<span class="tag-chip">Tidak ada QR code terdeteksi</span>`;
      return `<div class="tag-list">${tags}</div>` + resultGrid([{ img: data.original, cap: "Asli" }, { img: data.result, cap: "Hasil Deteksi" }]) + saveButtonHTML("adv-qr");
    },
  },

  "adv-histmatch": {
    title: "Histogram Matching",
    desc: "Menyesuaikan distribusi warna satu gambar agar menyerupai gambar referensi (histogram specification).",
    endpoint: "/api/advanced/histogram-matching",
    customUI: "source-reference",
    fields: [],
    explain: "Memetakan CDF (Cumulative Distribution Function) histogram sumber ke CDF histogram referensi per kanal warna.",
    render: (data) => resultGrid([{ img: data.source, cap: "Sumber" }, { img: data.reference, cap: "Referensi" }, { img: data.result, cap: "Hasil Matching" }]) +
      `<img src="${data.histogram}" style="width:100%;margin-top:14px;border-radius:10px">` + saveButtonHTML("adv-histmatch"),
  },

  "adv-watermark": {
    title: "Digital Watermark (LSB Steganography)",
    desc: "Menyembunyikan pesan rahasia di dalam citra tanpa mengubah tampilan visualnya.",
    endpoint: "/api/advanced/watermark-embed",
    extractEndpoint: "/api/advanced/watermark-extract",
    customUI: "watermark",
    fields: [{ type: "text", name: "text", label: "Pesan Rahasia", placeholder: "Tulis pesan yang ingin disembunyikan..." }],
    explain: "Teknik <b>Least Significant Bit (LSB) Steganography</b> — mengganti bit terakhir tiap nilai piksel dengan bit pesan, sehingga perubahan tidak terlihat mata manusia.",
    render: (data) => `<p style="font-size:13px;color:var(--text-dim);margin-bottom:12px">Gambar terlihat identik, tapi sudah menyimpan pesan rahasia di dalamnya.</p>` +
      resultGrid([{ img: data.original, cap: "Asli" }, { img: data.result, cap: "Sudah Disisipi Pesan (identik secara visual)" }]) + saveButtonHTML("adv-watermark"),
  },

  // ================= FEATURE EXTRACTION LAB =================
  // 3.1 Statistik & Warna
  "fe-intensity": {
    title: "Statistik Intensitas", desc: "Mean, std, min, max, entropy, dan histogram N-bin dari citra grayscale.",
    feEndpoint: "/api/fe/run", fePanel: "fe-intensity",
    fields: [{ type: "range", name: "n_bins", label: "Jumlah Bin", min: 4, max: 32, step: 1, value: 8 }],
    explain: "Statistik dasar + entropy Shannon dari histogram grayscale.",
  },
  "fe-color-stats": {
    title: "Statistik Warna Multi-Ruang", desc: "Mean & std per kanal pada ruang warna RGB/HSV/LAB.",
    feEndpoint: "/api/fe/run", fePanel: "fe-color-stats",
    fields: [{ type: "select", name: "space", label: "Ruang Warna", options: [["hsv", "HSV"], ["rgb", "RGB"], ["lab", "LAB"]], value: "hsv" }],
    explain: "Mean dan std tiap kanal warna pada ruang warna pilihan.",
  },
  "fe-color-hist": {
    title: "Histogram Warna per Kanal", desc: "Histogram tiap kanal warna digabung jadi satu vector.",
    feEndpoint: "/api/fe/run", fePanel: "fe-color-hist",
    fields: [
      { type: "select", name: "space", label: "Ruang Warna", options: [["hsv", "HSV"], ["rgb", "RGB"]], value: "hsv" },
      { type: "range", name: "n_bins", label: "Bin per Kanal", min: 4, max: 16, step: 1, value: 8 },
    ],
    explain: "Histogram per kanal (HSV/RGB) dikonkatenasi jadi satu vector.",
  },

  // 3.2 Bentuk & Moments
  "fe-shape": {
    title: "Shape Descriptors", desc: "Area, perimeter, aspect ratio, centroid, circularity dari kontur terbesar (Otsu).",
    feEndpoint: "/api/fe/run", fePanel: "fe-shape", fields: [],
    explain: "Citra dibinerisasi otomatis (Otsu) lalu diambil kontur terbesar.",
  },
  "fe-moments": {
    title: "Raw & Central Moments", desc: "m00, m10, m01, mu20, nu20 dari citra biner (Otsu).",
    feEndpoint: "/api/fe/run", fePanel: "fe-moments", fields: [],
    explain: "Momen mentah dan momen pusat dari citra biner.",
  },
  "fe-hu": {
    title: "Hu Moments ⭐", desc: "7 nilai Hu Moments invariant (tahan rotasi/skala/translasi).",
    feEndpoint: "/api/fe/run", fePanel: "fe-hu",
    fields: [{ type: "checkbox", name: "log_scale", label: "Skala Logaritmik", value: true }],
    explain: "Hu Moments adalah fitur bentuk klasik yang invariant terhadap rotasi, skala, dan translasi.",
  },

  // 3.3 Tekstur -- GLCM
  "fe-glcm": {
    title: "GLCM Explorer", desc: "Gray-Level Co-occurrence Matrix: contrast, homogeneity, energy, correlation, entropy, dsb.",
    feEndpoint: "/api/fe/run", fePanel: "fe-glcm",
    fields: [
      { type: "range", name: "distance", label: "Jarak (d)", min: 1, max: 8, step: 1, value: 1 },
      { type: "select", name: "angle_deg", label: "Sudut", options: [["0", "0°"], ["45", "45°"], ["90", "90°"], ["135", "135°"]], value: "0" },
      { type: "range", name: "levels", label: "Gray Levels", min: 4, max: 32, step: 1, value: 8 },
      { type: "select", name: "pooling", label: "Pooling", options: [["none", "Tanpa Pooling"], ["mean", "Mean"], ["mean_std", "Mean + Std"]], value: "none" },
    ],
    parseFields: (v) => ({
      distances: [Number(v.distance)], angles_deg: [Number(v.angle_deg)],
      levels: Number(v.levels), pooling: v.pooling === "none" ? null : v.pooling,
    }),
    explain: "GLCM mengukur hubungan spasial antar piksel pada jarak & sudut tertentu.",
  },
  "fe-glcm-block": {
    title: "GLCM Block/Overlap", desc: "GLCM dihitung per-blok (bisa overlap) lalu digabung.",
    feEndpoint: "/api/fe/run", fePanel: "fe-glcm-block",
    fields: [
      { type: "range", name: "block", label: "Ukuran Blok", min: 8, max: 64, step: 8, value: 32 },
      { type: "range", name: "stride", label: "Stride", min: 8, max: 64, step: 8, value: 32 },
      { type: "range", name: "levels", label: "Gray Levels", min: 4, max: 32, step: 1, value: 8 },
    ],
    parseFields: (v) => ({ block: Number(v.block), stride: Number(v.stride), levels: Number(v.levels) }),
    explain: "Stride lebih kecil dari Block berarti blok-blok saling overlap.",
  },
  "fe-glcm-color": {
    title: "GLCM Warna", desc: "GLCM dihitung terpisah per-kanal RGB.",
    feEndpoint: "/api/fe/run", fePanel: "fe-glcm-color",
    fields: [{ type: "range", name: "levels", label: "Gray Levels", min: 4, max: 32, step: 1, value: 8 }],
    explain: "GLCM per kanal R, G, B -- menangkap tekstur sekaligus informasi warna.",
  },

  // 3.3 Tekstur -- LBP
  "fe-lbp": {
    title: "LBP Explorer", desc: "Local Binary Pattern dasar: default/uniform/ror/nri_uniform/var.",
    feEndpoint: "/api/fe/run", fePanel: "fe-lbp",
    fields: [
      { type: "range", name: "P", label: "Jumlah Tetangga (P)", min: 4, max: 24, step: 1, value: 8 },
      { type: "range", name: "R", label: "Radius (R)", min: 1, max: 5, step: 1, value: 1 },
      { type: "select", name: "method", label: "Metode", options: [["uniform", "Uniform"], ["default", "Default"], ["ror", "Rotation-Invariant"], ["nri_uniform", "NRI-Uniform"], ["var", "Variance"]], value: "uniform" },
    ],
    parseFields: (v) => ({ P: Number(v.P), R: Number(v.R), method: v.method }),
    explain: "LBP membandingkan tiap piksel dengan tetangganya untuk membentuk kode biner lokal.",
  },
  "fe-lbp-block": {
    title: "LBP Block/Overlap", desc: "LBP dihitung per-blok lalu histogramnya digabung.",
    feEndpoint: "/api/fe/run", fePanel: "fe-lbp-block",
    fields: [
      { type: "range", name: "P", label: "P", min: 4, max: 16, step: 1, value: 8 },
      { type: "range", name: "R", label: "R", min: 1, max: 3, step: 1, value: 1 },
      { type: "select", name: "method", label: "Metode", options: [["uniform", "Uniform"], ["default", "Default"]], value: "uniform" },
      { type: "range", name: "block", label: "Ukuran Blok", min: 8, max: 64, step: 8, value: 16 },
      { type: "range", name: "stride", label: "Stride", min: 8, max: 64, step: 8, value: 16 },
    ],
    parseFields: (v) => ({ P: Number(v.P), R: Number(v.R), method: v.method, block: Number(v.block), stride: Number(v.stride) }),
    explain: "Histogram LBP dihitung per-blok untuk menangkap informasi spasial.",
  },
  "fe-lbp-multiscale": {
    title: "LBP Multi-scale", desc: "Gabungan histogram LBP dari 3 kombinasi (P,R) berbeda.",
    feEndpoint: "/api/fe/run", fePanel: "fe-lbp-multiscale",
    fields: [{ type: "select", name: "method", label: "Metode", options: [["uniform", "Uniform"], ["default", "Default"]], value: "uniform" }],
    explain: "Menggabungkan 3 skala LBP: (P=8,R=1), (P=16,R=2), (P=24,R=3).",
  },
  "fe-ltp": {
    title: "LTP", desc: "Local Ternary Pattern: pola dipecah upper/lower lalu histogram digabung.",
    feEndpoint: "/api/fe/run", fePanel: "fe-ltp",
    fields: [{ type: "range", name: "t", label: "Threshold (t)", min: 1, max: 20, step: 1, value: 5 }],
    parseFields: (v) => ({ t: Number(v.t) }),
    explain: "LTP lebih tahan noise dibanding LBP karena memakai threshold toleransi.",
  },
  "fe-clbp": {
    title: "CLBP", desc: "Completed LBP: komponen Sign (S), Magnitude (M), dan Center (C).",
    feEndpoint: "/api/fe/run", fePanel: "fe-clbp",
    fields: [
      { type: "range", name: "P", label: "P", min: 4, max: 16, step: 1, value: 8 },
      { type: "range", name: "R", label: "R", min: 1, max: 3, step: 1, value: 1 },
    ],
    parseFields: (v) => ({ P: Number(v.P), R: Number(v.R) }),
    explain: "CLBP menangkap informasi tanda DAN magnitude perbedaan piksel, bukan cuma tanda seperti LBP biasa.",
  },
  "fe-cslbp": {
    title: "CS-LBP", desc: "Center-Symmetric LBP: 4 pasang piksel berlawanan -> 16 kode.",
    feEndpoint: "/api/fe/run", fePanel: "fe-cslbp",
    fields: [
      { type: "range", name: "R", label: "Radius (R)", min: 1, max: 3, step: 1, value: 1 },
      { type: "range", name: "t", label: "Threshold (t)", min: 0, max: 20, step: 1, value: 0 },
    ],
    parseFields: (v) => ({ R: Number(v.R), t: Number(v.t) }),
    explain: "CS-LBP lebih ringkas (16 bin) dibanding LBP klasik (256 bin) karena hanya membandingkan pasangan piksel berlawanan.",
  },
  "fe-mblbp": {
    title: "MB-LBP", desc: "Multi-Block LBP: bandingkan rata-rata 8 blok sekitar dengan blok tengah.",
    feEndpoint: "/api/fe/run", fePanel: "fe-mblbp",
    fields: [{ type: "range", name: "block", label: "Ukuran Blok", min: 2, max: 6, step: 1, value: 3 }],
    parseFields: (v) => ({ block: Number(v.block) }),
    explain: "MB-LBP bekerja pada level blok, bukan piksel individual -- lebih tahan noise.",
  },

  // 3.3 Tekstur -- Gabor
  "fe-gabor-single": {
    title: "Gabor Single", desc: "Satu filter Gabor pada orientasi & wavelength tertentu.",
    feEndpoint: "/api/fe/run", fePanel: "fe-gabor-single",
    fields: [
      { type: "select", name: "theta_deg", label: "Orientasi", options: [["0", "0°"], ["45", "45°"], ["90", "90°"], ["135", "135°"]], value: "0" },
      { type: "range", name: "lam", label: "Wavelength (λ)", min: 5, max: 30, step: 1, value: 10 },
    ],
    parseFields: (v) => ({ theta_deg: Number(v.theta_deg), lam: Number(v.lam) }),
    explain: "Filter Gabor meniru respons sel sederhana korteks visual terhadap orientasi & frekuensi tertentu.",
  },
  "fe-gabor-bank": {
    title: "Gabor Bank", desc: "Filter bank 4 orientasi (0/45/90/135°) sekaligus.",
    feEndpoint: "/api/fe/run", fePanel: "fe-gabor-bank",
    fields: [{ type: "select", name: "stat", label: "Statistik", options: [["energy", "Energy saja"], ["mean_std_energy", "Mean + Std + Energy"]], value: "energy" }],
    explain: "Filter bank menangkap tekstur dari berbagai arah sekaligus.",
  },
  "fe-gabor-block": {
    title: "Gabor Block", desc: "Respons Gabor dihitung per-blok untuk tiap orientasi.",
    feEndpoint: "/api/fe/run", fePanel: "fe-gabor-block",
    fields: [{ type: "range", name: "block", label: "Ukuran Blok", min: 16, max: 64, step: 8, value: 32 }],
    parseFields: (v) => ({ block: Number(v.block) }),
    explain: "Statistik respons Gabor dihitung per-blok, bukan satu angka untuk seluruh citra.",
  },
  "fe-gabor-lbp": {
    title: "Gabor + LBP", desc: "Respons magnitude Gabor diubah jadi histogram LBP (fitur hybrid).",
    feEndpoint: "/api/fe/run", fePanel: "fe-gabor-lbp", fields: [],
    explain: "Kombinasi Gabor (menangkap frekuensi/orientasi) dengan LBP (menangkap pola lokal).",
  },

  // 3.4 Gradien, Tepi & Transform Domain
  "fe-hog": {
    title: "HOG", desc: "Histogram of Oriented Gradients -- fitur gradien klasik untuk deteksi objek/bentuk.",
    feEndpoint: "/api/fe/run", fePanel: "fe-hog",
    fields: [{ type: "range", name: "orientations", label: "Jumlah Orientasi", min: 4, max: 18, step: 1, value: 9 }],
    parseFields: (v) => ({ orientations: Number(v.orientations) }),
    explain: "HOG menghitung distribusi arah gradien pada cell-cell kecil citra.",
  },
  "fe-edge": {
    title: "Fitur Tepi", desc: "Edge density + histogram orientasi tepi (Canny + Sobel).",
    feEndpoint: "/api/fe/run", fePanel: "fe-edge",
    fields: [
      { type: "range", name: "canny_low", label: "Canny Low", min: 20, max: 150, step: 5, value: 80 },
      { type: "range", name: "canny_high", label: "Canny High", min: 50, max: 250, step: 5, value: 160 },
    ],
    parseFields: (v) => ({ canny_low: Number(v.canny_low), canny_high: Number(v.canny_high) }),
    explain: "Kerapatan tepi dan distribusi arah tepi (dibobot magnitude gradien).",
  },
  "fe-projection": {
    title: "Projection Profile", desc: "Proyeksi horizontal & vertikal dari citra biner.",
    feEndpoint: "/api/fe/run", fePanel: "fe-projection",
    fields: [{ type: "range", name: "thresh", label: "Threshold Biner", min: 50, max: 200, step: 5, value: 128 }],
    parseFields: (v) => ({ thresh: Number(v.thresh) }),
    explain: "Umum dipakai untuk segmentasi baris/karakter pada OCR.",
  },
  "fe-dct": {
    title: "DCT", desc: "Discrete Cosine Transform -- koefisien frekuensi rendah (mirip kompresi JPEG).",
    feEndpoint: "/api/fe/run", fePanel: "fe-dct",
    fields: [{ type: "range", name: "keep", label: "Blok Koefisien (k x k)", min: 2, max: 16, step: 1, value: 8 }],
    parseFields: (v) => ({ keep: Number(v.keep) }),
    explain: "Hanya koefisien frekuensi rendah (pojok kiri atas) yang diambil sebagai fitur.",
  },
  "fe-fft-radial": {
    title: "FFT Radial Profile", desc: "Spektrum FFT dirangkum jadi profil 1D berdasarkan jarak dari pusat frekuensi.",
    feEndpoint: "/api/fe/run", fePanel: "fe-fft-radial",
    fields: [{ type: "range", name: "n_radius_bins", label: "Jumlah Bin Radius", min: 8, max: 64, step: 4, value: 32 }],
    parseFields: (v) => ({ n_radius_bins: Number(v.n_radius_bins) }),
    explain: "Profil radial menangkap kandungan frekuensi citra secara rotation-invariant.",
  },
  "fe-wavelet": {
    title: "Wavelet", desc: "Dekomposisi wavelet Haar 1 level -> 4 subband (LL/LH/HL/HH).",
    feEndpoint: "/api/fe/run", fePanel: "fe-wavelet",
    fields: [{ type: "select", name: "stat", label: "Statistik", options: [["energy", "Energy saja"], ["mean_std_energy", "Mean + Std + Energy"]], value: "energy" }],
    explain: "Subband LL menyimpan approximation, LH/HL/HH menyimpan detail horizontal/vertikal/diagonal.",
  },

  // 3.5 Descriptor Lokal
  "fe-local-extract": {
    title: "Ekstrak Keypoint (ORB/SIFT)", desc: "Mendeteksi titik fitur lokal beserta descriptor-nya.",
    endpoint: "/api/fe/extract-keypoints",
    fields: [
      { type: "select", name: "method", label: "Metode", options: [["orb", "ORB (biner, cepat)"], ["sift", "SIFT (float, lebih presisi)"]], value: "orb" },
      { type: "range", name: "n_features", label: "Maks Keypoint", min: 100, max: 1000, step: 50, value: 500 },
    ],
    explain: "ORB menghasilkan descriptor biner 32-byte; SIFT menghasilkan descriptor float 128-D.",
    render: (data) => renderFEStepGallery(data),
  },
  "fe-local-match": {
    title: "Matching Keypoint", desc: "Mencocokkan keypoint antara dua gambar (wajib upload keduanya).",
    endpoint: "/api/fe/match-keypoints",
    customUI: "dual-upload-required",
    fields: [
      { type: "select", name: "method", label: "Metode", options: [["orb", "ORB (Hamming)"], ["sift", "SIFT (Ratio Test)"]], value: "orb" },
      { type: "range", name: "n_features", label: "Maks Keypoint", min: 100, max: 1000, step: 50, value: 500 },
      { type: "range", name: "ratio_thresh", label: "Ratio Threshold (khusus SIFT)", min: 0.5, max: 0.95, step: 0.05, value: 0.75 },
    ],
    explain: "Revisi dosen: kedua gambar wajib di-upload sendiri, tidak memakai sample.",
  },

  // 3.6 Robust & Deep Embedding
  "fe-resnet50": {
    title: "ResNet50 Embedding", desc: "Embedding 2048-D dari backbone ResNet50 pretrained (ImageNet).",
    feEndpoint: "/api/fe/run", fePanel: "fe-resnet50", fields: [],
    explain: "Butuh PyTorch + torchvision terpasang di server. Kalau belum ada, panel ini menampilkan status graceful, bukan error mentah.",
  },
  "fe-robustness": {
    title: "Robustness Stress-Test", desc: "Uji ketahanan fitur terhadap geser, rotasi, skala, dan perubahan brightness.",
    endpoint: "/api/fe/stress-test",
    fields: [{ type: "select", name: "feature_key", label: "Fitur yang Diuji", options: FE_FEATURE_KEY_OPTIONS, value: "hu_moments_invariant" }],
    explain: "Membandingkan vector fitur sebelum & sesudah 4 transformasi standar.",
    render: (data) => renderFEStepGallery(data),
  },

  // Lintas-Fitur
  "fe-matching-engine": {
    title: "Matching Engine", desc: "Membandingkan dua citra dengan fitur pilihan + metric otomatis (Euclidean/Cosine/Chi-Square/dll).",
    endpoint: "/api/fe/matching-engine",
    customUI: "dual-upload-required",
    fields: [
      { type: "select", name: "feature_key", label: "Fitur Pembanding", options: FE_FEATURE_KEY_OPTIONS, value: "hu_moments_invariant" },
      { type: "select", name: "feature_kind", label: "Jenis Fitur (untuk pilih metric)", options: [["continuous", "Continuous (Euclidean/Cosine)"], ["histogram", "Histogram (Chi-Square/Intersection)"], ["binary", "Binary (Hamming)"]], value: "continuous" },
    ],
    explain: "Revisi dosen: kedua gambar wajib di-upload sendiri, tidak memakai sample.",
  },
  "fe-concat": {
    title: "Concat Playground", desc: "Gabungkan beberapa fitur jadi satu vector besar (feature fusion).",
    endpoint: "/api/fe/concat",
    customUI: "fe-concat",
    fields: [
      { type: "multicheck", name: "features", label: "Pilih Fitur", options: FE_CONCAT_OPTIONS, defaultChecked: ["fe-hu", "fe-lbp", "fe-glcm"] },
      { type: "checkbox", name: "normalize", label: "Normalisasi (L2)", value: true },
    ],
    explain: "Vector dari tiap fitur terpilih dijalankan ulang di server lalu dikonkatenasi jadi satu vector besar.",
  },

  // ================= MATCHING LAB (Bumi & Mas) =================
  "ml-advisor": {
    title: "Matching Advisor (§16)",
    desc: "Pemandu pohon keputusan interaktif & analisis heuristik otomatis untuk memilih teknik pencocokan.",
    endpoint: "/api/matching/advisor",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "chosen_path", label: "Representasi Masukan", options: [["local", "Titik Fitur Lokal (Keypoints)"], ["template", "Template / Patch Spasial"], ["vector", "Vektor Global Numerik"], ["shape", "Bentuk / Kontur Biner"]], value: "local" },
      { type: "checkbox", name: "auto_analyze", label: "Analisis Citra Otomatis (Heuristik Resolusi & Keypoint)", value: false }
    ],
    explain: "Membantu memilih matcher terbaik berdasarkan representasi fitur, kondisi pencahayaan, dan invarian geometris.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-vector": {
    title: "Vector Distance Lab (§2)",
    desc: "Evaluasi jarak Euclidean, Manhattan, Minkowski, Chebyshev, Cosine, dan Mahalanobis.",
    endpoint: "/api/matching/vector",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "feature_type", label: "Ekstraktor Vektor", options: [["hu", "Hu Moments Invariant (7D)"], ["color_stat", "Statistik Warna BGR (6D)"], ["hog_compact", "HOG Ringkas Orientasi (12D)"]], value: "hu" },
      { type: "slider", name: "p_minkowski", label: "Parameter Minkowski (p)", min: 1, max: 6, step: 0.5, value: 3 },
      { type: "slider", name: "scale_k", label: "Uji Invariansi Skala (x k)", min: 1, max: 5, step: 0.5, value: 2 }
    ],
    explain: "Mendemonstrasikan mengapa Euclidean sensitif terhadap magnitudo skala sementara Cosine invarian.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-histogram": {
    title: "Histogram & Distribution Lab (§3)",
    desc: "Evaluasi Chi-square, Intersection, Bhattacharyya, Jensen-Shannon, dan Wasserstein/EMD.",
    endpoint: "/api/matching/histogram",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "color_space", label: "Ruang Warna", options: [["RGB", "RGB (Kanal R)"], ["HSV", "HSV (Hue)"], ["LAB", "LAB (Luminance)"]], value: "RGB" },
      { type: "slider", name: "n_bins", label: "Resolusi Kuantisasi Bin", min: 16, max: 128, step: 16, value: 32 },
      { type: "slider", name: "shift_bins", label: "Uji Geser Bin (Demo Sensitivitas EMD)", min: 1, max: 8, step: 1, value: 4 }
    ],
    explain: "Memperlihatkan keunggulan EMD: peka terhadap jarak geometris bin bertetangga, sedangkan Chi-Square mengabaikannya.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-binary": {
    title: "Binary Matching Lab (§4)",
    desc: "Pencocokan string biner berbasis Hamming Distance (POPCNT XOR) dan Jaccard Similarity.",
    endpoint: "/api/matching/binary",
    customUI: "dual-upload",
    fields: [
      { type: "checkbox", name: "use_material_preset", label: "Gunakan String Materi Formal (10110110 vs 10011110)", value: false }
    ],
    explain: "Descriptor biner (ORB/BRISK) dicocokkan bit-per-bit dengan operasi XOR berkecepatan tingkat instruksi CPU hardware.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-template": {
    title: "Template Matching Lab (§5)",
    desc: "Lokalisasi ROI spasial dengan SAD, SSD, NCC, dan ZNCC + analisis ketahanan gangguan.",
    endpoint: "/api/matching/template",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "method", label: "Metode Korelasi", options: [["ZNCC", "ZNCC (Zero-mean Normalized Cross-Correlation)"], ["SSD", "SSD (Sum of Squared Differences)"], ["NCC", "NCC (Normalized Cross-Correlation)"], ["SAD", "SAD (Sum of Absolute Differences)"]], value: "ZNCC" }
    ],
    explain: "Template matching cocok untuk translasi murni; ZNCC tahan variasi intensitas pencahayaan namun gagal terhadap rotasi.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-local": {
    title: "Local Feature Matching & Funnel (§6)",
    desc: "Detektor ORB, SIFT, AKAZE, BRISK + Lowe Ratio Test, Cross-Check, dan visualisasi corong reduksi.",
    endpoint: "/api/matching/local",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "method", label: "Detektor & Descriptor", options: [["ORB", "ORB (Biner - Hamming)"], ["SIFT", "SIFT (Float - L2)"], ["AKAZE", "AKAZE (Biner - Non-linear Scale)"], ["BRISK", "BRISK (Biner Multi-skala)"]], value: "ORB" },
      { type: "slider", name: "ratio_threshold", label: "Ambang Lowe Ratio (d1/d2)", min: 0.5, max: 0.9, step: 0.05, value: 0.75 },
      { type: "checkbox", name: "cross_check", label: "Aktifkan Cross-Check Simetris", value: true },
      { type: "checkbox", name: "use_flann", label: "Gunakan FLANN Matcher", value: false }
    ],
    explain: "Mengeliminasi korespondensi semu melalui rasio tetangga terdekat pertama terhadap kedua.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-geometric": {
    title: "RANSAC Geometric Verification & Homography (§7–8)",
    desc: "Verifikasi konsensus RANSAC, proyeksi bidang objek H 3x3, warp perspektif, dan penjahitan panorama.",
    endpoint: "/api/matching/geometric",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "mode", label: "Aplikasi Homografi", options: [["planar", "Deteksi Objek Planar"], ["panorama", "Penjahitan Citra Panorama"]], value: "planar" },
      { type: "slider", name: "reproj_threshold", label: "Ambang Reprojection Error (px)", min: 1.0, max: 10.0, step: 0.5, value: 3.0 }
    ],
    explain: "61 inlier geometris lebih bernilai daripada 78 descriptor yang hanya mirip secara fotometrik.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-shape": {
    title: "Shape & Contour Matching Lab (§9)",
    desc: "Pencocokan bentuk dengan Hausdorff, Chamfer DT, Shape Context log-polar, dan Fourier Descriptors.",
    endpoint: "/api/matching/shape",
    customUI: "dual-upload",
    fields: [
      { type: "checkbox", name: "invert", label: "Inversi Binerisasi Citra", value: false },
      { type: "slider", name: "k_fourier", label: "Jumlah Koefisien Harmonis Fourier (K)", min: 4, max: 32, step: 2, value: 12 },
      { type: "checkbox", name: "add_outlier", label: "Uji Outlier Ekstrem (Demonstrasi Kerentanan Hausdorff)", value: false }
    ],
    explain: "Metode kontur tidak mensyaratkan jumlah titik sampling yang sama antara himpunan A dan B.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-sequence": {
    title: "Sequence Matching — DTW (§10)",
    desc: "Dynamic Time Warping untuk penyelarasan deret waktu non-linier dan perbandingan vs Euclidean.",
    endpoint: "/api/matching/sequence",
    customUI: "dual-upload",
    fields: [
      { type: "checkbox", name: "use_material_preset", label: "Gunakan Deret Materi (A=[1,3,4,7,8], B=[1,2,3,3,4,6,7,8])", value: true }
    ],
    explain: "DTW memetakan sekuens dengan tempo dan durasi berbeda tanpa mendistorsi informasi struktural.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-assignment": {
    title: "Assignment & Optimal Transport (§11)",
    desc: "Hungarian 1-ke-1 vs Nearest Neighbor independen serta Sinkhorn Optimal Transport.",
    endpoint: "/api/matching/assignment",
    fields: [
      { type: "slider", name: "reg_entropy", label: "Regularisasi Entropi Sinkhorn", min: 0.02, max: 0.2, step: 0.02, value: 0.08 }
    ],
    explain: "Menghindari alokasi many-to-one yang serakah melalui optimasi bipartit global minimum.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-motion": {
    title: "Optical Flow Motion Lab (§12)",
    desc: "Lucas-Kanade sparse dengan verifikasi forward-backward error dan Farneback dense HSV.",
    endpoint: "/api/matching/motion",
    customUI: "dual-upload",
    fields: [],
    explain: "Mengestimasi vektor kecepatan piksel temporal dan memvalidasi keakuratan dengan motion compensation.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-deep": {
    title: "Deep Matching Lab (§13–14)",
    desc: "Siamese / Triplet Loss embedding (ResNet50) dan stub arsitektur modern (SuperPoint/LightGlue/LoFTR).",
    endpoint: "/api/matching/deep",
    customUI: "dual-upload",
    fields: [
      { type: "select", name: "mode", label: "Mode Demonstrasi", options: [["triplet", "Triplet Loss & Lingkaran Margin 2D"], ["modern_stub", "Kartu Arsitektur Transformer Modern"]], value: "triplet" },
      { type: "slider", name: "margin", label: "Margin Triplet Alpha", min: 0.1, max: 1.0, step: 0.1, value: 0.5 }
    ],
    explain: "Metode modern berbasis neural network tetap mengandalkan prinsip dasar korespondensi dan inlier konsistensi.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-guard": {
    title: "Compatibility Guard (§15)",
    desc: "Pendeteksi pelanggaran ruang fitur & eksperimen tabrakan pembuktian Euclidean(LBP, GLCM).",
    endpoint: "/api/matching/guard",
    fields: [
      { type: "checkbox", name: "collision_demo", label: "Jalankan Eksperimen Pembuktian 'Coba Tabrakan' LBP vs GLCM", value: true }
    ],
    explain: "Mencegah kesalahan fatal pencampuran representasi yang tidak sepadan (TIDAK BERMAKNA).",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-arena": {
    title: "Method Arena",
    desc: "Uji tanding satu pasangan citra ke berbagai matcher sekaligus (ORB, SIFT, AKAZE, BRISK, Histogram).",
    endpoint: "/api/matching/arena",
    customUI: "dual-upload",
    fields: [],
    explain: "Evaluasi multi-metode paralel untuk menentukan algoritma tercepat dan paling akurat.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-preset": {
    title: "Preset Kasus Nyata A–E (§17)",
    desc: "Jalankan skenario dunia nyata (Logo, Pengenalan Objek, Morfologi Daun, Panorama, DTW) dalam satu klik.",
    endpoint: "/api/matching/preset",
    fields: [
      { type: "select", name: "preset_key", label: "Pilihan Preset Skenario", options: [["A", "Kasus A: Logo di Foto (Template vs Local Feature)"], ["B", "Kasus B: Pengenalan Objek (Embedding Cosine)"], ["C", "Kasus C: Dua Bentuk Daun (Hausdorff & Chamfer)"], ["D", "Kasus D: Panorama (Stitching Dua Citra)"], ["E", "Kasus E: Signature / Sekuens (DTW)"]], value: "A" }
    ],
    explain: "Sampel prosedural otomatis dibangkitkan secara offline untuk demonstrasi instan.",
    render: (data) => renderMLGenericStepPipeline(data)
  },

  "ml-database": {
    title: "Pencocokan Citra 1-vs-Database (§18)",
    desc: "Pencocokan citra uji terhadap kumpulan database (1-to-N) dengan offline indexing, online query matching, thresholding, dan galeri Top-K.",
    endpoint: "/api/matching/database",
    customUI: "image-database",
    fields: [
      { type: "select", name: "method", label: "Representasi Fitur", options: [["orb", "ORB (Biner - Hamming + Lowe Ratio)"], ["sift", "SIFT (Float 128D - Euclidean L2 + FLANN)"], ["deep_embedding", "Deep Learned Embedding (Cosine Distance)"], ["color_hist", "Color Histogram (3D HSV Bhattacharyya)"], ["hu_moments", "Hu Moments Invariant (Bentuk Kontur)"]], value: "orb" },
      { type: "slider", name: "threshold", label: "Nilai Ambang / Threshold (%)", min: 10, max: 95, step: 5, value: 50 },
      { type: "select", name: "top_k", label: "Jumlah Hasil Teratas (Top-K)", options: [["3", "Top-3 Hasil"], ["5", "Top-5 Hasil"], ["10", "Top-10 Hasil"]], value: "5" },
      { type: "checkbox", name: "use_flann", label: "Gunakan FLANN Matcher (untuk SIFT)", value: false }
    ],
    explain: "Pra-ekstraksi representasi fitur secara luring (offline) memungkinkan pencarian real-time pada ribuan gambar tanpa komputasi ulang.",
    render: (data) => renderMLGenericStepPipeline(data)
  },
  "ml-audio-dtw": {
    title: "Pencocokan Suara DTW Lab (§19)",
    desc: "Dynamic Time Warping untuk audio: Verifikasi Penutur (Biometrik), Keyword Spotting, visualisasi Warping Path 2D, dan Live Recording.",
    endpoint: "/api/matching/audio-dtw",
    customUI: "audio-dtw",
    fields: [
      { type: "slider", name: "threshold", label: "Ambang Jarak DTW Maksimum", min: 5.0, max: 40.0, step: 1.0, value: 24.0 },
      { type: "select", name: "top_k", label: "Jumlah Hasil Teratas (Top-K)", options: [["3", "Top-3"], ["5", "Top-5"], ["10", "Top-10"]], value: "5" }
    ],
    explain: "DTW mengatasi perbedaan durasi dan tempo pengucapan dengan mencari jalur pelengkungan waktu berbiaya minimum.",
    render: (data) => renderMLGenericStepPipeline(data)
  },

  // ================= MODE KHUSUS =================
  "mode-batch": { title: "Batch Processing", desc: "Proses banyak gambar sekaligus dengan satu operasi.", customUI: "batch" },
  "mode-webcam": { title: "Live Webcam", desc: "Proses video real-time langsung dari webcam.", customUI: "webcam" },
  "mode-history": { title: "Riwayat & Laporan PDF", desc: "Lihat semua hasil yang tersimpan dan ekspor ke PDF.", customUI: "history-page" },
};

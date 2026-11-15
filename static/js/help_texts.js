// ==========================================================
// TaksuCitra - Kamus Teks Bantuan & Penjelasan Terpusat (i18n)
// Bahasa: Bahasa Indonesia
// Format: 3 kalimat per fitur: (a) apa, (b) kapan, (c) peringatan.
// Dilengkapi petunjuk ringkas (hints) untuk setiap parameter.
// ==========================================================

const FEATURE_HELP = {
  "morf-dilasi": {
    name: "Dilasi Citra Biner",
    shortExplain: "Membesarkan objek putih dan menyambung celah kecil. Cocok untuk menyambung garis putus. Hati-hati: objek yang berdekatan bisa menyatu.",
    what: "Membesarkan objek putih (foreground) dan menyambung celah kecil menggunakan elemen penstruktur (structuring element).",
    when: "Cocok untuk menyambung garis putus, menutup lubang kecil pada objek, atau menebalkan teks yang pudar.",
    caution: "Hati-hati: objek yang saling berdekatan bisa menyatu dan batas luar objek membesar.",
    hints: {
      se_shape: "Bentuk elemen: ellipse (lingkaran/halus), rect (persegi), cross (palang tipis).",
      se_size: "Ukuran SE harus bilangan ganjil (3–31). Makin besar ukuran, efek pembesaran makin kuat.",
      iterations: "Jumlah perulangan dilasi berturut-turut.",
      invert_mode: "Pilihan polaritas biner: 'Otomatis' menebak objek sebagai area minoritas.",
      blur: "Ukuran kernel blur sebelum Otsu thresholding.",
      use_manual: "Bandingkan implementasi manual NumPy vs OpenCV.",
    },
    defaults: { se_shape: "ellipse", se_size: 5, iterations: 1, invert_mode: "auto", blur: 5, use_manual: false }
  },

  "morf-erosi": {
    name: "Erosi Citra Biner",
    shortExplain: "Mengecilkan objek dan memutus sambungan tipis. Cocok untuk memisahkan objek yang menempel dan membuang noise kecil. Hati-hati: objek menyusut.",
    what: "Mengecilkan objek putih (foreground) dengan mengikis lapisan batas terluar sesuai bentuk elemen penstruktur.",
    when: "Cocok untuk memisahkan dua objek yang saling menempel secara tipis dan membuang titik noise putih kecil.",
    caution: "Hati-hati: seluruh objek akan menyusut, dan detail garis tipis bisa hilang sepenuhnya.",
    hints: {
      se_shape: "Bentuk elemen penstruktur (ellipse/rect/cross).",
      se_size: "Ukuran SE ganjil (3–31). Semakin besar ukuran, pengikisan semakin agresif.",
      iterations: "Jumlah pengulangan erosi.",
      invert_mode: "Pembalikan polaritas biner.",
      blur: "Filter praproses sebelum thresholding.",
      use_manual: "Gunakan algoritma manual definisi himpunan matematis.",
    },
    defaults: { se_shape: "ellipse", se_size: 5, iterations: 1, invert_mode: "auto", blur: 5, use_manual: false }
  },

  "morf-opening": {
    name: "Opening (Buka)",
    shortExplain: "Erosi lalu dilasi; membuang objek kecil dan tonjolan tanpa banyak mengubah ukuran objek utama. Cocok untuk noise titik putih. Hati-hati: sudut tajam ikut membulat.",
    what: "Melakukan erosi terlebih dahulu kemudian dilasi dengan structuring element yang sama.",
    when: "Sangat cocok untuk membersihkan derau titik-titik putih (salt noise) dan menghaluskan tonjolan kontur tanpa mengecilkan ukuran objek utama.",
    caution: "Hati-hati: sudut-sudut lancip pada objek akan ikut membulat.",
    hints: {
      se_shape: "Bentuk elemen (ellipse dianjurkan untuk kontur alami).",
      se_size: "Ukuran SE ganjil. Pilih ukuran sedikit lebih besar daripada bintik noise.",
      invert_mode: "Deteksi polaritas objek.",
      blur: "Blur awal untuk menstabilkan thresholding.",
      use_manual: "Implementasi manual opening (erode -> dilate).",
    },
    defaults: { se_shape: "ellipse", se_size: 5, invert_mode: "auto", blur: 5, use_manual: false }
  },

  "morf-closing": {
    name: "Closing (Tutup)",
    shortExplain: "Dilasi lalu erosi; menutup lubang dan celah kecil. Cocok untuk menyambung retakan/garis putus. Hati-hati: objek berdekatan bisa tergabung.",
    what: "Melakukan dilasi terlebih dahulu kemudian diikuti oleh erosi menggunakan structuring element yang sama.",
    when: "Cocok untuk menutup lubang hitam kecil di dalam objek (pepper noise), menyambung retakan, dan mengisi celah tipis.",
    caution: "Hati-hati: dua objek terpisah yang letaknya berdekatan dapat menyatu menjadi satu komponen.",
    hints: {
      se_shape: "Bentuk structuring element.",
      se_size: "Ukuran SE ganjil. Harus cukup besar untuk menjangkau lebar celah yang ingin ditutup.",
      invert_mode: "Deteksi objek otomatis atau manual.",
      blur: "Tingkat penghalusan awal.",
      use_manual: "Implementasi manual closing (dilate -> erode).",
    },
    defaults: { se_shape: "ellipse", se_size: 7, invert_mode: "auto", blur: 5, use_manual: false }
  },

  "morf-hit-or-miss": {
    name: "Hit-or-Miss Transform",
    shortExplain: "Mendeteksi pola bentuk dan ukuran yang tepat (hit = harus objek, miss = harus latar, don't care = bebas). Cocok untuk piksel terisolasi, titik ujung/persimpangan, dan template matching biner. Hati-hati: sangat sensitif noise, bersihkan citra dulu.",
    what: "Mencari kecocokan pola biner eksak: elemen 'hit' wajib berada di foreground, elemen 'miss' wajib di background, dan 'don't care' diabaikan.",
    when: "Cocok untuk menemukan piksel terisolasi, mendeteksi titik ujung garis (endpoints), dan mencari bentuk geometris tertentu berukuran pas.",
    caution: "Hati-hati: sangat sensitif terhadap noise; bahkan satu piksel yang meleset akan menggagalkan deteksi.",
    hints: {
      mode: "Preset pencocokan pola: piksel terisolasi, titik ujung skeleton, persegi tepat n x n, atau custom kernel 3x3.",
      square_size: "Sisi persegi (n x n) yang dicari secara tepat jika memilih preset persegi.",
      invert_mode: "Penentuan latar vs objek.",
      use_manual: "Bandingkan komputasi manual (A(-)B_hit AND A^c(-)B_miss) vs OpenCV.",
    },
    defaults: { mode: "isolated_pixel", square_size: 21, invert_mode: "auto", blur: 5, use_manual: false }
  },

  "morf-skeleton": {
    name: "Skeletonization & Topologi",
    shortExplain: "Mereduksi objek menjadi garis setebal 1 piksel yang mewakili kerangkanya. Cocok untuk pembuluh darah, tulisan tangan, jaringan jalan/retakan. Hati-hati: tepi kasar menghasilkan cabang palsu, gunakan pruning.",
    what: "Mereduksi objek biner hingga menjadi kerangka median setebal 1 piksel dengan mempertahankan konektivitas dan topologi aslinya.",
    when: "Sangat bermanfaat untuk analisis struktural pembuluh darah retina, pola sidik jari, pengenalan karakter tulisan tangan, dan jaringan jalan.",
    caution: "Hati-hati: batas tepi yang kasar memicu cabang palsu (spurs); gunakan parameter pruning untuk memangkasnya.",
    hints: {
      method: "Metode skeleton: Lantuejoul (morfologis lossless), Zhang-Suen (manual), Lee (skimage), atau Medial Axis.",
      prune_len: "Panjang cabang pendek (spurs) yang dipangkas (0–30 piksel).",
      show_endpoints: "Tandai titik ujung cabang dengan warna merah pada overlay.",
      show_branches: "Tandai titik persimpangan/cabang dengan warna biru pada overlay.",
      invert_mode: "Pengaturan polaritas citra.",
    },
    defaults: { method: "lee", prune_len: 10, show_endpoints: true, show_branches: true, invert_mode: "auto", blur: 5 }
  },

  "enh-brightness-contrast": {
    name: "Kecerahan & Kontras",
    shortExplain: "Brightness mengatur terang-gelap, contrast mengatur selisih gelap-terang. Linear untuk koreksi cepat, gamma untuk foto gelap/terlalu terang, CLAHE untuk citra medis atau pencahayaan tidak merata. Hati-hati: nilai ekstrem membuat area terang/gelap kehilangan detail (clipping) dan noise ikut menguat.",
    what: "Memodifikasi distribusi luminans dan rentang dinamis piksel citra secara global maupun adaptif.",
    when: "Linear untuk koreksi cepat, Gamma untuk memperbaiki area bayangan tanpa membakar area terang, dan CLAHE untuk citra dengan pencahayaan tidak seragam.",
    caution: "Hati-hati: penguatan kontras yang terlalu tinggi menimbulkan efek clipping (detail hilang jadi putih/hitam polos) dan memperparah noise latar.",
    hints: {
      mode: "Pilihan algoritma: Linear (alpha & beta), Gamma, Gamma Otomatis, Contrast Stretching, Histogram Equalization, atau CLAHE.",
      alpha: "Faktor kontras linear (1.0 = normal, >1.0 meningkatkan kontras).",
      beta: "Nilai bias kecerahan linear (-100 s.d. +100).",
      gamma: "Nilai kurva gamma (<1 mencerahkan bayangan, >1 menggelapkan).",
      target_mean: "Rerata kecerahan target untuk Gamma Otomatis (0.1–0.9).",
      p_low: "Batas persentil bawah untuk contrast stretching (default 1%).",
      p_high: "Batas persentil atas untuk contrast stretching (default 99%).",
      clip_limit: "Batas penguatan kontras lokal pada CLAHE agar noise terkendali.",
      tile_size: "Ukuran grid ubin lokal untuk CLAHE (mis. 8 = grid 8x8).",
    },
    defaults: { mode: "linear", alpha: 1.2, beta: 10, gamma: 1.0, target_mean: 0.5, p_low: 1, p_high: 99, clip_limit: 2.0, tile_size: 8 }
  },

  "enh-flip": {
    name: "Pencerminan (Flip / Mirror)",
    shortExplain: "Membalik citra seperti cermin (horizontal/vertikal) tanpa kehilangan kualitas. Cocok untuk augmentasi data. Hati-hati: teks, angka, dan penanda kiri/kanan (mis. X-ray) bisa berubah makna.",
    what: "Membalik susunan indeks koordinat matriks piksel secara horizontal, vertikal, atau kedua sumbu sekaligus secara lossless.",
    when: "Sangat baik untuk augmentasi data citra machine learning dan memperbaiki orientasi foto hasil tangkapan kamera depan.",
    caution: "Hati-hati: teks terbaca terbalik dan penanda orientasi penting (misal sisi kiri/kanan pada foto medis/X-ray) dapat tertukar.",
    hints: {
      mode: "Arah pencerminan: Horizontal (kiri-kanan), Vertikal (atas-bawah), atau Keduanya (rotasi 180°).",
    },
    defaults: { mode: "h" }
  },

  "enh-rotasi": {
    name: "Rotasi Citra",
    shortExplain: "Memutar citra; 90 derajat bersifat lossless, sudut bebas memakai interpolasi. Cocok untuk meluruskan dokumen dan augmentasi. Hati-hati: tanpa 'perbesar kanvas' sudut citra terpotong.",
    what: "Memutar citra mengelilingi titik pusat dengan pemetaan koordinat berlawanan arah jarum jam.",
    when: "Rotasi 90° cocok untuk orientasi potret/lanskap, sedangkan sudut bebas berguna meluruskan dokumen miring (deskewing).",
    caution: "Hati-hati: jika opsi 'Perbesar Kanvas' dinonaktifkan, keempat sudut citra akan terpotong keluar bingkai.",
    hints: {
      mode: "Pilihan mode: Sudut Bebas (-180° s.d. +180°) atau Kelipatan 90° (lossless).",
      angle: "Besaran derajat rotasi (nilai positif = berlawanan jarum jam).",
      times: "Berapa kali kelipatan 90° (1 = 90°, 2 = 180°, 3 = 270°).",
      expand: "Perbesar dimensi kanvas output agar tidak ada sudut yang terpotong.",
      fill_color: "Intensitas warna latar belakang untuk area kosong di luar citra asli (0 = hitam, 255 = putih).",
      use_manual: "Gunakan algoritma manual inverse mapping + interpolasi bilinear.",
    },
    defaults: { mode: "free", angle: 30, times: 1, expand: true, fill_color: 0, use_manual: false }
  },

  "enh-edge-detection": {
    name: "Deteksi Tepi",
    shortExplain: "Menemukan batas objek dari perubahan intensitas. Sobel/Prewitt/Scharr cepat dan menunjukkan kekuatan gradien, Canny memberi garis tipis yang bersih. Hati-hati: noise memunculkan tepi palsu, kurangi noise dulu atau naikkan sigma.",
    what: "Mendeteksi diskontinuitas intensitas lokal yang menandai batas fisik, perubahan material, atau tepian objek dalam citra.",
    when: "Sobel/Prewitt/Scharr untuk peta magnitudo gradien cepat, Laplacian of Gaussian untuk zero-crossing, dan Canny untuk kontur biner ramping 1-piksel.",
    caution: "Hati-hati: derau acak memicu banyak deteksi tepi palsu; terapkan denoising sebelumnya atau tingkatkan nilai sigma.",
    hints: {
      method: "Operator gradien (Sobel, Prewitt, Scharr), LoG, Canny Manual (NMS + Hysteresis), atau Auto-Canny OpenCV.",
      sigma: "Standar deviasi Gaussian smoothing untuk meredam noise sebelum deteksi tepi.",
      low_thresh: "Ambang batas bawah hysteresis pada Canny (tepi lemah yang terhubung).",
      high_thresh: "Ambang batas atas hysteresis pada Canny (tepi kuat pasti dipertahankan).",
      auto_sigma: "Faktor penentu ambang otomatis berbasis median pada Auto-Canny (default 0.33).",
    },
    defaults: { method: "canny_auto", sigma: 1.4, low_thresh: "", high_thresh: "", auto_sigma: 0.33 }
  },

  "enh-noise-reduction": {
    name: "Pereduksian Derau (Noise Reduction)",
    shortExplain: "Mengurangi bintik/derau. Median/adaptive median untuk salt-and-pepper, NLM atau bilateral untuk noise Gaussian dengan tepi terjaga, Gaussian/mean untuk noise ringan. Hati-hati: filter yang terlalu kuat mengaburkan detail.",
    what: "Meredam fluktuasi intensitas acak yang tidak diinginkan dengan metode filtering domain spasial maupun non-lokal.",
    when: "Median & Adaptive Median untuk salt-and-pepper; Bilateral & Non-Local Means (NLM) untuk Gaussian dengan ketajaman tepi yang tetap terjaga.",
    caution: "Hati-hati: ukuran filter yang terlalu besar akan mengikis tekstur halus dan mengaburkan batas objek penting.",
    hints: {
      method: "Jenis filter: Mean, Gaussian, Median, Adaptive Median, Bilateral, atau Non-Local Means (NLM).",
      kernel_size: "Ukuran kernel jendela spasial (harus ganjil, 3–15).",
      sigma: "Standar deviasi untuk filter Gaussian.",
      h_param: "Parameter kekuatan filter (sigmaColor pada bilateral, h pada NLM). Kosongkan untuk estimasi noise otomatis.",
      add_test_noise: "Generator derau uji coba (Gaussian, Salt & Pepper, Speckle, Poisson) untuk menguji performa filter.",
      noise_amount: "Tingkat intensitas noise sintetis yang ditambahkan pada mode uji.",
    },
    defaults: { method: "median", kernel_size: 5, sigma: 1.5, h_param: "", add_test_noise: "none", noise_amount: "" }
  },

  "enh-pipeline": {
    name: "Pipeline Peningkatan Terpadu",
    shortExplain: "Rangkaian pemrosesan citra multi-tahap (denoise -> kontras -> deteksi tepi). Cocok untuk alur otomasi komprehensif. Hati-hati: kesalahan di tahap awal akan merambat ke tahap berikutnya.",
    what: "Menjalankan rangkaian bertahap terpadu: peredaman derau (Denoise) -> perbaikan kontras adaptif (CLAHE) -> deteksi kontur biner (Canny).",
    when: "Ideal untuk menangani citra beresolusi rendah, gelap, atau sarat noise sebelum diekstraksi fiturnya atau dikenali oleh AI.",
    caution: "Hati-hati: parameter yang keliru di tahap denoise dapat menghilangkan struktur halus yang diperlukan pada tahap deteksi tepi.",
    hints: {
      denoise_method: "Metode denoise awal: Non-Local Means (NLM) atau Median.",
      denoise_strength: "Kekuatan filter denoise.",
      clahe_clip: "Batas kontras CLAHE pada tahap kedua.",
      edge_auto_sigma: "Sensitivitas ambang deteksi tepi Canny pada tahap akhir.",
    },
    defaults: { denoise_method: "nlm", denoise_strength: 10, clahe_clip: 2.5, edge_auto_sigma: 0.33 }
  }
};

/**
 * Render elemen bantuan ramah pengguna untuk diletakkan di UI.
 * @param {string} featureKey
 * @returns {string} HTML markup
 */
function renderFeatureHelpBox(featureKey) {
  const f = FEATURE_HELP[featureKey];
  if (!f) return "";
  return `
    <div class="explain-box explain-box-enhanced">
      <div class="explain-header">
        <span class="material-symbols-outlined" style="font-size: 17px; color: var(--accent); vertical-align: middle;">info</span>
        <strong>Bantuan & Panduan Fitur</strong>
      </div>
      <div class="explain-body">
        <p><strong>Fungsi:</strong> ${f.what}</p>
        <p><strong>Kapan Dipakai:</strong> ${f.when}</p>
        <p class="explain-caution"><span class="material-symbols-outlined" style="font-size: 15px; color: var(--danger); vertical-align: middle;">warning</span> <strong>Peringatan:</strong> ${f.caution}</p>
      </div>
    </div>`;
}

// Ekspor ke window agar dapat diakses oleh panels.js dan main.js
if (typeof window !== "undefined") {
  window.FEATURE_HELP = FEATURE_HELP;
  window.renderFeatureHelpBox = renderFeatureHelpBox;
}

"""
processing/match_vector.py
==========================
Implementasi §2: Vector Distance Lab, §3: Histogram Matching Lab, §4: Binary Matching Lab.
Menyediakan pipeline bertahap (minimal 4 langkah berurutan dengan visualisasi berbeda).
Dilengkapi kompatibilitas formula materi:
- Euclidean, Manhattan, Minkowski (p-norm), Chebyshev, Cosine, Mahalanobis (shrinkage/regularized)
- Chi-square, Intersection, Bhattacharyya, Jensen-Shannon, Wasserstein / EMD
- Hamming, Jaccard
"""

import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance

from processing.match_viz import (
    DEFAULTS, create_step, bgr_to_b64, fig_to_b64, style_dark_axes, draw_side_by_side
)
from processing.feature_extraction_lab import (
    euclidean_distance, cosine_similarity, hamming_distance,
    chi_square_distance, histogram_intersection, hu_moments_invariant
)


# =============================================================================
# §2 VECTOR DISTANCE LAB
# =============================================================================

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def minkowski_distance(a: np.ndarray, b: np.ndarray, p: float = 3.0) -> float:
    if p <= 0:
        p = 1.0
    return float(np.sum(np.abs(a - b) ** p) ** (1.0 / p))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def extract_feature_vector(img_bgr: np.ndarray, feature_type: str = "hu") -> tuple[np.ndarray, list[str]]:
    """Mengekstraksi representasi vektor dari citra (Hu Moments, HOG ringkas, atau statistik warna)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if feature_type == "hu":
        hu = hu_moments_invariant(gray, log_scale=True)
        return hu["vector"], [f"h_{i+1}" for i in range(len(hu["vector"]))]
    elif feature_type == "color_stat":
        means, stds = cv2.meanStdDev(img_bgr)
        vec = np.concatenate([means.flatten(), stds.flatten()]).astype(np.float32)
        labels = ["B_mean", "G_mean", "R_mean", "B_std", "G_std", "R_std"]
        return vec, labels
    elif feature_type == "hog_compact":
        resized = cv2.resize(gray, (64, 64))
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(ang, bins=12, range=(0, 360), weights=mag)
        hist = hist / (np.linalg.norm(hist) + 1e-7)
        return hist.astype(np.float32), [f"bin_{i*30}deg" for i in range(12)]
    else:
        # Default Hu
        hu = hu_moments_invariant(gray, log_scale=True)
        return hu["vector"], [f"h_{i+1}" for i in range(len(hu["vector"]))]


def compute_vector_lab(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Eksekusi Pipeline §2 Vector Distance Lab.
    Langkah:
    1. Input Citra A & B
    2. Ekstraksi Vektor A vs B (Bar Chart)
    3. Selisih per komponen |a_i - b_i| (Bar Chart kontribusi terbesar)
    4. Evaluasi Multi-Metrik Jarak
    5. Analisis Invariansi Magnitude (Demo Euclidean vs Cosine)
    """
    t0 = time.time()
    warnings = []
    
    feat_type = params.get("feature_type", "hu")
    p_minkowski = float(params.get("p_minkowski", 3.0))
    scale_k = float(params.get("scale_k", 2.0))
    custom_a = params.get("custom_vector_a", None)
    custom_b = params.get("custom_vector_b", None)
    
    # 1. Tentukan Vektor
    if custom_a is not None and custom_b is not None and len(custom_a) > 0 and len(custom_a) == len(custom_b):
        vec_a = np.array(custom_a, dtype=np.float32)
        vec_b = np.array(custom_b, dtype=np.float32)
        labels = [f"dim_{i+1}" for i in range(len(vec_a))]
        raw_vis = np.full((180, 500, 3), 20, dtype=np.uint8)
        cv2.putText(raw_vis, f"Vektor Kustom: {len(vec_a)} Dimensi", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (242, 202, 80), 2, cv2.LINE_AA)
        step1_img = bgr_to_b64(raw_vis)
    else:
        if img1 is None or img2 is None:
            return {"ok": False, "error": "Dua citra masukan atau pasangan vektor kustom dibutuhkan.", "hint": "Unggah citra A dan B atau isi vektor manual."}
        vec_a, labels = extract_feature_vector(img1, feat_type)
        vec_b, _ = extract_feature_vector(img2, feat_type)
        step1_img = bgr_to_b64(draw_side_by_side(img1, img2, "Citra A", "Citra B"))
    
    # 2. Plot Vektor A vs B
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax, title="Perbandingan Nilai Fitur (Vektor A vs Vektor B)", xlabel="Komponen Fitur", ylabel="Magnitudo")
    x = np.arange(len(vec_a))
    w = 0.38
    ax.bar(x - w/2, vec_a, width=w, label="Vektor A", color=DEFAULTS["gold"], edgecolor=DEFAULTS["border"])
    ax.bar(x + w/2, vec_b, width=w, label="Vektor B", color=DEFAULTS["saffron"], edgecolor=DEFAULTS["border"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    ax.grid(axis="y", color="#2a2a2a", linestyle="--", alpha=0.5)
    step2_img = fig_to_b64(fig)
    
    # 3. Selisih per komponen
    diff = np.abs(vec_a - vec_b)
    fig_diff, ax_diff = plt.subplots(figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_diff, title="Selisih Mutlak per Komponen |a_i - b_i|", xlabel="Komponen", ylabel="Selisih")
    bars = ax_diff.bar(x, diff, color=DEFAULTS["terracotta"], edgecolor=DEFAULTS["border"], width=0.5)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax_diff.grid(axis="y", color="#2a2a2a", linestyle="--", alpha=0.5)
    max_idx = int(np.argmax(diff))
    bars[max_idx].set_color("#ff4444")
    ax_diff.annotate(f"Maks: {diff[max_idx]:.3f}", xy=(max_idx, diff[max_idx]),
                     xytext=(0, 5), textcoords="offset points", ha='center',
                     color=DEFAULTS["text_color"], fontsize=8, fontweight="bold")
    step3_img = fig_to_b64(fig_diff)
    
    # 4. Hitung Jarak-Jarak
    d_euc = euclidean_distance(vec_a, vec_b)
    d_man = manhattan_distance(vec_a, vec_b)
    d_mink = minkowski_distance(vec_a, vec_b, p=p_minkowski)
    d_cheb = chebyshev_distance(vec_a, vec_b)
    sim_cos = cosine_similarity(vec_a, vec_b)
    
    # Mahalanobis Guard
    # Catatan: Bila hanya 1 pasang tanpa galeri, tolak dengan regularisasi / penjelasan
    d_mahal = None
    mahal_status = "Ditolak: Mahalanobis mensyaratkan kovarians populasi (n >= 2*d sampel). Digunakan regularisasi S = I."
    warnings.append("Jarak Mahalanobis dievaluasi dengan regularisasi identitas karena tidak ada galeri sampel multi-vektor.")
    cov = np.eye(len(vec_a))
    d_mahal = float(mahalanobis(vec_a, vec_b, np.linalg.inv(cov)))
    
    metrics_table = {
        "headers": ["Metrik", "Formula / Definisi", "Nilai Numerik", "Keterangan Invariansi"],
        "rows": [
            ["Euclidean (L2)", "sqrt(sum((a_i - b_i)^2))", f"{d_euc:.4f}", "Sensitif terhadap skala magnitudo"],
            ["Manhattan (L1)", "sum(|a_i - b_i|)", f"{d_man:.4f}", "Robus terhadap outlier ekstrem"],
            [f"Minkowski (p={p_minkowski})", f"(sum(|a_i - b_i|^{p_minkowski}))^(1/{p_minkowski})", f"{d_mink:.4f}", "Generalisasi L1 dan L2"],
            ["Chebyshev (L_inf)", "max(|a_i - b_i|)", f"{d_cheb:.4f}", "Didominasi oleh koordinat selisih terbesar"],
            ["Cosine Similarity", "(a . b) / (||a|| ||b||)", f"{sim_cos:.4f}", "Invarian terhadap perbesaran magnitudo k*A"],
            ["Mahalanobis", "sqrt((a-b)^T S^-1 (a-b))", f"{d_mahal:.4f}", mahal_status]
        ]
    }
    
    fig_tbl, ax_tbl = plt.subplots(figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_tbl, title="Perbandingan Metrik Jarak Ternormalisasi")
    norm_metrics = [d_euc, d_man, d_mink, d_cheb]
    names = ["Euclidean", "Manhattan", f"Minkowski(p={p_minkowski})", "Chebyshev"]
    max_val = max(norm_metrics) if max(norm_metrics) > 0 else 1.0
    bar_vals = [v / max_val for v in norm_metrics]
    ax_tbl.barh(names, bar_vals, color=[DEFAULTS["gold"], DEFAULTS["saffron"], DEFAULTS["terracotta"], "#64d2ff"])
    ax_tbl.set_xlim(0, 1.15)
    for i, v in enumerate(bar_vals):
        ax_tbl.text(v + 0.02, i, f"{norm_metrics[i]:.3f}", va='center', color=DEFAULTS["text_color"], fontsize=8)
    step4_img = fig_to_b64(fig_tbl)
    
    # 5. Eksperimen Skala Magnitudo (Demo Euclidean vs Cosine)
    vec_a_scaled = vec_a * scale_k
    euc_scaled = euclidean_distance(vec_a_scaled, vec_b)
    cos_scaled = cosine_similarity(vec_a_scaled, vec_b)
    
    fig_scale, ax_scale = plt.subplots(figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_scale, title=f"Uji Skala Magnitudo (A x {scale_k}): Cosine vs Euclidean")
    bars_demo = ax_scale.bar(["Euclidean Asli", f"Euclidean (x{scale_k})", "Cosine Asli", f"Cosine (x{scale_k})"],
                 [d_euc, euc_scaled, sim_cos, cos_scaled],
                 color=[DEFAULTS["gold"], DEFAULTS["terracotta"], DEFAULTS["gold"], DEFAULTS["gold"]])
    for b in bars_demo:
        h = b.get_height()
        ax_scale.text(b.get_x() + b.get_width()/2, h + 0.02, f"{h:.3f}", ha='center', color=DEFAULTS["text_color"], fontsize=8)
    step5_img = fig_to_b64(fig_scale)
    
    # Contoh manual 3 komponen materi: A=[1,2,3], B=[2,4,4] -> 2.449
    sub_a = np.array([1.0, 2.0, 3.0])
    sub_b = np.array([2.0, 4.0, 4.0])
    sample_calc = f"Verifikasi Materi: A=[1,2,3], B=[2,4,4] -> d = sqrt((2-1)^2 + (4-2)^2 + (4-3)^2) = sqrt(1 + 4 + 1) = sqrt(6) ≈ 2.449"
    
    steps = [
        create_step(1, "Input Citra & Pemilihan Representasi",
                    f"Menerima pasangan masukan dan mengekstraksi vektor berdimensi {len(vec_a)} menggunakan ekstraktor {feat_type}.",
                    formula="x ∈ R^D",
                    image_b64=step1_img,
                    metrics={"dimensi": len(vec_a), "fitur": feat_type}),
        create_step(2, "Ekstraksi & Penjajaran Komponen Vektor",
                    "Representasi komponen fitur dijajarkan berdampingan untuk mengamati profil nilai masing-masing entitas.",
                    formula="A = [a_1, a_2, ..., a_D], B = [b_1, b_2, ..., b_D]",
                    image_b64=step2_img,
                    metrics={"mean_a": float(np.mean(vec_a)), "mean_b": float(np.mean(vec_b))}),
        create_step(3, "Dekomposisi Selisih per Dimensi",
                    "Menghitung selisih absolut untuk mendeteksi dimensi mana yang memberikan kontribusi perbedaan paling dominan.",
                    formula="Δ_i = |a_i - b_i|",
                    image_b64=step3_img,
                    metrics={"komponen_pembeda_maks": labels[max_idx], "selisih_maks": float(diff[max_idx])}),
        create_step(4, "Kalkulasi Matriks Jarak Geometris",
                    "Menghitung berbagai metrik jarak L1, L2, L_p, Chebyshev, Cosine, dan Mahalanobis.",
                    formula="d_L2 = ||a - b||_2, d_L1 = ||a - b||_1",
                    image_b64=step4_img,
                    table=metrics_table,
                    metrics={"euclidean": d_euc, "manhattan": d_man, "cosine": sim_cos, "minkowski": d_mink}),
        create_step(5, "Analisis Perilaku Skala & Verifikasi",
                    f"{sample_calc}. Demonstrasi menunjukkan bahwa Cosine invarian terhadap penskalaan vektor x{scale_k}, sedangkan Euclidean berubah sebanding dengan magnitudo.",
                    formula="cos(k*A, B) = cos(A, B)",
                    image_b64=step5_img,
                    metrics={"cosine_original": sim_cos, "cosine_scaled": cos_scaled, "euclidean_scaled": euc_scaled})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "vector_matching",
        "summary": f"Evaluasi jarak vektor {len(vec_a)}D: Euclidean={d_euc:.4f}, Cosine={sim_cos:.4f}, Manhattan={d_man:.4f}.",
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {"euclidean": d_euc, "manhattan": d_man, "cosine": sim_cos, "chebyshev": d_cheb},
            "table": metrics_table
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }


# =============================================================================
# §3 HISTOGRAM & DISTRIBUTION LAB
# =============================================================================

def bhattacharyya_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """BC = sum(sqrt(p * q)), d = -ln(BC)."""
    p = h1 / (np.sum(h1) + 1e-12)
    q = h2 / (np.sum(h2) + 1e-12)
    bc = np.sum(np.sqrt(np.clip(p * q, 0, None)))
    bc = np.clip(bc, 1e-12, 1.0)
    return float(-np.log(bc))


def jensen_shannon_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """JS divergence d = sqrt(0.5 * KL(p||m) + 0.5 * KL(q||m))."""
    p = h1 / (np.sum(h1) + 1e-12)
    q = h2 / (np.sum(h2) + 1e-12)
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / (b[mask] + 1e-12)))
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(0.0, js)))


def compute_histogram_lab(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Eksekusi Pipeline §3 Histogram & Distribution Lab.
    Langkah:
    1. Input Citra & Histogram per Kanal
    2. Normalisasi Distribusi Probabilitas
    3. Analisis Area Overlap (Intersection)
    4. Evaluasi Wasserstein Distance / EMD (Perpindahan Massa CDF)
    5. Demonstrasi Sensitivitas Bin Geser (Poin Wow: Chi-Sq vs EMD)
    6. Tabel Komprehensif Seluruh Metrik Distribusi
    """
    t0 = time.time()
    warnings = []
    
    if img1 is None or img2 is None:
        return {"ok": False, "error": "Dua citra diperlukan untuk komparasi distribusi.", "hint": "Unggah citra A dan B."}
        
    space = params.get("color_space", "RGB").upper()
    n_bins = int(params.get("n_bins", 32))
    shift_demo = int(params.get("shift_bins", 4))
    
    # 1. Konversi Ruang Warna
    if space == "HSV":
        c1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        c2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    elif space == "LAB":
        c1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        c2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    else:
        c1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        c2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
    # Hitung histogram kanal pertama (L/V/R) untuk demonstrasi visual utama
    h1, _ = np.histogram(c1[:, :, 0], bins=n_bins, range=(0, 256))
    h2, _ = np.histogram(c2[:, :, 0], bins=n_bins, range=(0, 256))
    
    # 1. Visual Input & Raw Hist
    fig1, ax1 = plt.subplots(figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax1, title=f"Histogram Mentah Kanal 0 ({space})", xlabel="Indeks Bin", ylabel="Frekuensi Piksel")
    bins_x = np.arange(n_bins)
    ax1.plot(bins_x, h1, label="Citra A", color=DEFAULTS["gold"], lw=2)
    ax1.plot(bins_x, h2, label="Citra B", color=DEFAULTS["saffron"], lw=2)
    ax1.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step1_img = fig_to_b64(fig1)
    
    # 2. Normalisasi Probabilitas
    p1 = h1.astype(np.float64) / (np.sum(h1) + 1e-12)
    p2 = h2.astype(np.float64) / (np.sum(h2) + 1e-12)
    
    fig2, ax2 = plt.subplots(figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax2, title="Distribusi Probabilitas Ternormalisasi P(x)", xlabel="Bin", ylabel="Probabilitas")
    ax2.fill_between(bins_x, p1, color=DEFAULTS["gold"], alpha=0.4, label="P(A)")
    ax2.plot(bins_x, p1, color=DEFAULTS["gold"], lw=1.5)
    ax2.fill_between(bins_x, p2, color=DEFAULTS["saffron"], alpha=0.4, label="P(B)")
    ax2.plot(bins_x, p2, color=DEFAULTS["saffron"], lw=1.5)
    ax2.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step2_img = fig_to_b64(fig2)
    
    # 3. Histogram Intersection (Area Overlap diwarnai Gold)
    overlap = np.minimum(p1, p2)
    intersect_val = float(np.sum(overlap))
    fig3, ax3 = plt.subplots(figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax3, title=f"Area Overlap Distribusi (Intersection = {intersect_val:.4f})", xlabel="Bin", ylabel="Probabilitas")
    ax3.plot(bins_x, p1, color=DEFAULTS["text_dim"], lw=1, linestyle="--", label="P(A)")
    ax3.plot(bins_x, p2, color=DEFAULTS["text_dim"], lw=1, linestyle=":", label="P(B)")
    ax3.fill_between(bins_x, overlap, color=DEFAULTS["gold"], alpha=0.85, label=f"Irisan ({intersect_val*100:.1f}%)")
    ax3.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step3_img = fig_to_b64(fig3)
    
    # 4. Visual EMD / Wasserstein: CDF & Luas Arsir
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)
    emd_val = float(wasserstein_distance(bins_x, bins_x, u_weights=p1, v_weights=p2))
    
    fig4, ax4 = plt.subplots(figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax4, title=f"Visualisasi EMD (Wasserstein = {emd_val:.4f}): Area Antara CDF", xlabel="Bin", ylabel="Kumulatif F(x)")
    ax4.plot(bins_x, cdf1, color=DEFAULTS["gold"], lw=2, label="CDF A")
    ax4.plot(bins_x, cdf2, color=DEFAULTS["saffron"], lw=2, label="CDF B")
    ax4.fill_between(bins_x, cdf1, cdf2, color=DEFAULTS["terracotta"], alpha=0.5, label="Biaya Kerja Transport (EMD)")
    ax4.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step4_img = fig_to_b64(fig4)
    
    # 5. Poin Wow Materi: Bin Shift Demo (Chi-Sq vs EMD)
    # Buat distribusi sintetis satu puncak tajam, lalu geser 1 bin dan 4 bin
    spike_0 = np.zeros(n_bins, dtype=np.float64)
    spike_0[n_bins // 2] = 1.0
    spike_near = np.zeros(n_bins, dtype=np.float64)
    spike_near[min(n_bins - 1, n_bins // 2 + 1)] = 1.0
    spike_far = np.zeros(n_bins, dtype=np.float64)
    spike_far[min(n_bins - 1, n_bins // 2 + shift_demo)] = 1.0
    
    chi_near = chi_square_distance(spike_0, spike_near)
    chi_far = chi_square_distance(spike_0, spike_far)
    emd_near = float(wasserstein_distance(bins_x, bins_x, u_weights=spike_0, v_weights=spike_near))
    emd_far = float(wasserstein_distance(bins_x, bins_x, u_weights=spike_0, v_weights=spike_far))
    
    fig5, ax5 = plt.subplots(figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax5, title="Bukti Formal §3: Ketidakpekaan Chi-Square vs Kepekaan Metrik Ruang EMD")
    x_pos = np.arange(2)
    ax5.bar(x_pos - 0.2, [chi_near, chi_far], width=0.35, label="Chi-Square (Konstan)", color=DEFAULTS["terracotta"])
    ax5.bar(x_pos + 0.2, [emd_near, emd_far], width=0.35, label="EMD (Proporsional Jarak Bin)", color=DEFAULTS["gold"])
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(["Geser 1 Bin", f"Geser {shift_demo} Bin"])
    ax5.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step5_img = fig_to_b64(fig5)
    
    # 6. Hitung Semua Metrik
    d_chi = chi_square_distance(p1, p2)
    d_bhat = bhattacharyya_distance(p1, p2)
    d_js = jensen_shannon_distance(p1, p2)
    
    table_metrics = {
        "headers": ["Metrik Distribusi", "Sifat / Karakteristik", "Nilai", "Skala Kesamaan"],
        "rows": [
            ["Histogram Intersection", "Non-metrik (Similarity)", f"{intersect_val:.4f}", "1.0 = Identik, 0.0 = Saling Lepas"],
            ["Chi-Square Distance", "Sensitif terhadap bin yang sama persis", f"{d_chi:.4f}", "0.0 = Identik, >0 = Beda"],
            ["Bhattacharyya", "Jarak geometris probabilitas", f"{d_bhat:.4f}", "0.0 = Identik, inf = Berbeda total"],
            ["Jensen-Shannon", "Metrik simetris turunan KL-divergence", f"{d_js:.4f}", "[0, 1] rentang terikat"],
            ["Wasserstein / EMD", "Memperhitungkan geometri jarak antar-bin", f"{emd_val:.4f}", "Semakin kecil semakin murah transport"]
        ]
    }
    
    steps = [
        create_step(1, f"Histogram Per-Kanal ({space})",
                    f"Mengekstraksi distribusi frekuensi intensitas pada kanal utama dengan resolusi {n_bins} bin kuantisasi.",
                    formula="H(b) = sum_{p} [I(p) in bin_b]",
                    image_b64=step1_img,
                    metrics={"bins": n_bins, "ruang_warna": space}),
        create_step(2, "Normalisasi Distribusi Massa Probabilitas",
                    "Mengubah frekuensi absolut piksel menjadi fungsi massa probabilitas diskrit dengan jumlah total = 1.",
                    formula="P(b) = H(b) / sum(H)",
                    image_b64=step2_img,
                    metrics={"sum_p1": float(np.sum(p1)), "sum_p2": float(np.sum(p2))}),
        create_step(3, "Kalkulasi Histogram Intersection",
                    f"Menghitung volume irisan bersama antara kedua distribusi. Diperoleh overlap sebesar {intersect_val*100:.1f}%.",
                    formula="K(P, Q) = sum min(P(b), Q(b))",
                    image_b64=step3_img,
                    metrics={"intersection": intersect_val}),
        create_step(4, "Kalkulasi Wasserstein Distance (EMD 1D)",
                    "Menghitung luas area mutlak di antara kedua Cumulative Distribution Function (CDF) sebagai biaya perpindahan massa.",
                    formula="EMD(P, Q) = integral |F_P(x) - F_Q(x)| dx",
                    image_b64=step4_img,
                    metrics={"emd_wasserstein": emd_val}),
        create_step(5, "Demonstrasi Sensitivitas Bin Bertetangga",
                    f"Memperlihatkan keunggulan EMD: saat puncak bergeser dari 1 ke {shift_demo} bin, Chi-Square tetap konstan ({chi_near:.2f}), sedangkan EMD meningkat dari {emd_near:.2f} ke {emd_far:.2f}.",
                    formula="d_EMD ~ |mu_1 - mu_2|",
                    image_b64=step5_img,
                    table=table_metrics,
                    metrics={"chi_square": d_chi, "bhattacharyya": d_bhat, "jensen_shannon": d_js, "emd": emd_val})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "histogram_matching",
        "summary": f"Komparasi histogram {space} ({n_bins} bin): Overlap={intersect_val:.4f}, EMD={emd_val:.4f}, Chi-Square={d_chi:.4f}.",
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {"intersection": intersect_val, "emd": emd_val, "chi_square": d_chi, "jensen_shannon": d_js},
            "table": table_metrics
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }


# =============================================================================
# §4 BINARY MATCHING LAB
# =============================================================================

def jaccard_binary_similarity(b1: np.ndarray, b2: np.ndarray) -> float:
    """Jaccard index pada himpunan bit-1: |A ∩ B| / |A ∪ B|."""
    intersection = np.count_nonzero((b1 == 1) & (b2 == 1))
    union = np.count_nonzero((b1 == 1) | (b2 == 1))
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_binary_lab(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Eksekusi Pipeline §4 Binary Matching Lab.
    Langkah:
    1. Ekstraksi Descriptor Biner (ORB / BRISK)
    2. Visualisasi Grid Bit 8x32 dengan Penandaan Bit Berbeda
    3. Kalkulasi Hamming & Jaccard + Contoh Materi Formal
    4. Distribusi Jarak Hamming Seluruh Pasangan Pasangan Descriptor
    """
    t0 = time.time()
    warnings = []
    
    preset_example = params.get("use_material_preset", False)
    
    if preset_example:
        # Contoh persis materi: 10110110 vs 10011110
        bits_a = np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
        bits_b = np.array([1, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        n_features = 1
    else:
        if img1 is None or img2 is None:
            return {"ok": False, "error": "Citra masukan dibutuhkan untuk mengekstrak descriptor biner.", "hint": "Unggah citra atau pilih preset materi."}
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return {"ok": False, "error": "Tidak ditemukan keypoint biner yang memadai pada salah satu citra.", "hint": "Gunakan citra dengan tekstur yang lebih jelas."}
            
        bits_a = np.unpackbits(des1[0])[:256]
        bits_b = np.unpackbits(des2[0])[:256]
        n_features = min(len(des1), len(des2))
    
    # 1. Visual Input
    if preset_example:
        raw_vis = np.full((160, 520, 3), 20, dtype=np.uint8)
        cv2.putText(raw_vis, "Preset Materi: 10110110 vs 10011110", (24, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (242, 202, 80), 2, cv2.LINE_AA)
        step1_img = bgr_to_b64(raw_vis)
    else:
        step1_img = bgr_to_b64(draw_side_by_side(img1, img2, f"Keypoints A: {len(kp1)}", f"Keypoints B: {len(kp2)}"))
        
    # 2. Visualisasi Grid Bit (misal 8 baris x 32 kolom = 256 bit)
    n_show = len(bits_a)
    cols = 32 if n_show >= 32 else len(bits_a)
    rows = int(np.ceil(n_show / cols))
    
    fig2, (ax_ba, ax_bb, ax_xor) = plt.subplots(3, 1, figsize=(8, 4.2), facecolor=DEFAULTS["bg_color"])
    for ax, title in [(ax_ba, "Descriptor A (Bitstream)"), (ax_bb, "Descriptor B (Bitstream)"), (ax_xor, "Perbedaan Bit (XOR: Merah = Berbeda)")]:
        style_dark_axes(ax, title=title)
        ax.set_xticks([])
        ax.set_yticks([])
        
    grid_a = np.zeros((rows, cols), dtype=np.float32)
    grid_b = np.zeros((rows, cols), dtype=np.float32)
    grid_xor = np.zeros((rows, cols), dtype=np.float32)
    
    for i in range(n_show):
        r, c = i // cols, i % cols
        grid_a[r, c] = bits_a[i]
        grid_b[r, c] = bits_b[i]
        grid_xor[r, c] = 1.0 if bits_a[i] != bits_b[i] else 0.0
        
    ax_ba.imshow(grid_a, cmap="gold" if "gold" in plt.colormaps() else "YlOrBr", aspect="auto")
    ax_bb.imshow(grid_b, cmap="saffron" if "saffron" in plt.colormaps() else "Oranges", aspect="auto")
    ax_xor.imshow(grid_xor, cmap="Reds", aspect="auto")
    step2_img = fig_to_b64(fig2)
    
    # 3. Hitung Hamming dan Jaccard
    ham_dist = float(np.count_nonzero(bits_a != bits_b))
    jac_sim = jaccard_binary_similarity(bits_a, bits_b)
    
    fig3, ax3 = plt.subplots(figsize=(8, 2.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax3, title="Evaluasi Metrik Jarak & Kesamaan Biner")
    ax3.barh(["Jaccard Similarity", "Hamming Ternormalisasi"], [jac_sim, ham_dist / float(len(bits_a))],
             color=[DEFAULTS["gold"], DEFAULTS["terracotta"]], height=0.45)
    ax3.set_xlim(0, 1.1)
    ax3.text(jac_sim + 0.02, 0, f"{jac_sim:.4f}", va='center', color=DEFAULTS["text_color"], fontsize=8)
    ax3.text((ham_dist / float(len(bits_a))) + 0.02, 1, f"{ham_dist:.0f}/{len(bits_a)} ({ham_dist/len(bits_a):.2%})",
             va='center', color=DEFAULTS["text_color"], fontsize=8)
    step3_img = fig_to_b64(fig3)
    
    # 4. Distribusi Jarak Hamming jika mengevaluasi citra penuh
    if not preset_example and des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        dists = [m.distance for m in matches]
        fig4, ax4 = plt.subplots(figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
        style_dark_axes(ax4, title="Distribusi Jarak Hamming Seluruh Pasangan Terbaik", xlabel="Jarak Hamming (bit)", ylabel="Frekuensi")
        ax4.hist(dists, bins=25, color=DEFAULTS["gold"], edgecolor=DEFAULTS["border"])
        ax4.axvline(np.mean(dists), color=DEFAULTS["terracotta"], linestyle="--", label=f"Rata-rata: {np.mean(dists):.1f}")
        ax4.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
        step4_img = fig_to_b64(fig4)
    else:
        # Tampilkan dummy plot perbandingan string materi
        fig4, ax4 = plt.subplots(figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
        style_dark_axes(ax4, title="Verifikasi Contoh Materi: 10110110 vs 10011110")
        diff_indices = [i for i in range(len(bits_a)) if bits_a[i] != bits_b[i]]
        ax4.text(0.1, 0.6, f"Bit A: {''.join(map(str, bits_a))}\nBit B: {''.join(map(str, bits_b))}",
                 color=DEFAULTS["gold"], fontsize=12, family="monospace")
        ax4.text(0.1, 0.2, f"Bit Berbeda di Indeks: {diff_indices}\nHamming Distance = {ham_dist} bit",
                 color=DEFAULTS["terracotta"], fontsize=11, family="monospace", fontweight="bold")
        ax4.set_axis_off()
        step4_img = fig_to_b64(fig4)
        
    table_bin = {
        "headers": ["Operasi", "Rumus Formal", "Hasil", "Keterangan"],
        "rows": [
            ["Hamming Distance", "sum(a_i XOR b_i)", f"{ham_dist:.0f} bit", f"Dari total {len(bits_a)} bit string"],
            ["Jaccard Index", "|A ∩ B| / |A ∪ B|", f"{jac_sim:.4f}", "Dihitung pada himpunan posisi bit bernilai 1"],
            ["Kesesuaian Hardware", "POPCNT instruction", "O(1) CPU word", "Sangat cepat untuk perangkat edge"]
        ]
    }
    
    steps = [
        create_step(1, "Akuisisi Descriptor Biner",
                    f"Menerima pasangan masukan dan mengonversi representasi byte ke dalam bitstream {len(bits_a)} bit.",
                    formula="b ∈ {0, 1}^B",
                    image_b64=step1_img,
                    metrics={"total_bit": len(bits_a)}),
        create_step(2, "Visualisasi Peta Bit & Operasi XOR",
                    "Memetakan bitstream ke dalam kisi 2D dan menyorot posisi ketidaksesuaian menggunakan gerbang logika XOR.",
                    formula="XOR(a_i, b_i) = a_i ⊕ b_i",
                    image_b64=step2_img,
                    metrics={"bit_berbeda": int(ham_dist)}),
        create_step(3, "Kalkulasi Jarak Hamming & Jaccard",
                    f"Menghitung jumlah bit berbeda. Dihasilkan Hamming = {ham_dist:.0f} bit dan Jaccard = {jac_sim:.4f}.",
                    formula="d_H(a, b) = ||a ⊕ b||_1",
                    image_b64=step3_img,
                    table=table_bin,
                    metrics={"hamming": ham_dist, "jaccard": jac_sim}),
        create_step(4, "Analisis Distribusi Populasi Match",
                    "Menampilkan sebaran jarak Hamming untuk mengevaluasi ambang batas separabilitas antara inlier dan outlier.",
                    formula="P(d_H | match) vs P(d_H | non-match)",
                    image_b64=step4_img,
                    metrics={"populasi": n_features})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "binary_matching",
        "summary": f"Pencocokan biner ({len(bits_a)} bit): Hamming={ham_dist:.0f} bit, Jaccard={jac_sim:.4f}.",
        "steps": steps,
        "result": {
            "image": step2_img,
            "metrics": {"hamming": ham_dist, "jaccard": jac_sim, "total_bit": len(bits_a)},
            "table": table_bin
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }

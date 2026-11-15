"""
processing/match_shape.py
=========================
Implementasi §9: Shape & Contour Matching Lab.
Metode:
- Hausdorff Distance (berarah dan simetris, visualisasi garis ke worst-point)
- Chamfer Distance (Distance Transform Heatmap pada kontur B + titik A)
- Shape Context (Histogram Log-Polar 5x12 per titik + korespondensi Hungarian)
- Fourier Descriptors (Normalisasi translasi, rotasi, skala, titik awal + rekonstruksi K koefisien)
- Hu Moments Invariant (Reused, log-transform)
- Uji Ketahanan Outlier Ekstrem (Hausdorff melonjak vs Chamfer stabil)
"""

import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from processing.match_viz import (
    DEFAULTS, create_step, bgr_to_b64, fig_to_b64, style_dark_axes,
    draw_side_by_side, create_heatmap_bgr
)
from processing.feature_extraction_lab import hu_moments_invariant


def extract_largest_contour(img: np.ndarray, invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mengekstraksi kontur terbesar setelah thresholding Otsu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    if invert:
        gray = 255 - gray
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32), bw
    c = max(contours, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float32)
    return pts, bw


def sample_contour_points(pts: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """Melakukan subsampling deterministik seragam pada kurva kontur."""
    if len(pts) <= n_samples:
        return pts
    idx = np.linspace(0, len(pts) - 1, n_samples, dtype=int)
    return pts[idx]


def compute_hausdorff_directed(pts_a: np.ndarray, pts_b: np.ndarray) -> tuple[float, int, int]:
    """h(A, B) = max_{a in A} min_{b in B} ||a - b||."""
    D = cdist(pts_a, pts_b, metric='euclidean') # shape (Na, Nb)
    min_dist_per_a = np.min(D, axis=1)
    worst_a_idx = int(np.argmax(min_dist_per_a))
    closest_b_idx = int(np.argmin(D[worst_a_idx]))
    h_ab = float(min_dist_per_a[worst_a_idx])
    return h_ab, worst_a_idx, closest_b_idx


def compute_chamfer_distance(pts_a: np.ndarray, bw_b: np.ndarray) -> tuple[float, np.ndarray]:
    """Chamfer distance menggunakan distance transform citra B."""
    # Kontur B sebagai 0, latar belakang sebagai jarak
    contour_b_img = np.ones_like(bw_b, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(bw_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contour_b_img, contours, -1, 0, 1)
    dt = cv2.distanceTransform(contour_b_img, cv2.DIST_L2, 5)
    
    # Ambil nilai DT di koordinat titik A
    h_img, w_img = bw_b.shape
    dists = []
    for pt in pts_a:
        x, y = int(np.clip(pt[0], 0, w_img - 1)), int(np.clip(pt[1], 0, h_img - 1))
        dists.append(dt[y, x])
    mean_chamfer = float(np.mean(dists)) if dists else 0.0
    return mean_chamfer, dt


def compute_shape_contexts(pts: np.ndarray, n_r: int = 5, n_theta: int = 12, r_inner: float = 0.125, r_outer: float = 2.0) -> np.ndarray:
    """Menghitung histogram log-polar Shape Context per titik."""
    N = len(pts)
    if N == 0:
        return np.zeros((0, n_r * n_theta), dtype=np.float32)
    # Normalisasi skala rata-rata jarak antar titik
    D = cdist(pts, pts)
    mean_dist = np.mean(D) + 1e-7
    D_norm = D / mean_dist
    
    # Sudut theta
    dx = pts[:, 0, None] - pts[None, :, 0]
    dy = pts[:, 1, None] - pts[None, :, 1]
    theta = np.arctan2(dy, dx) # [-pi, pi]
    theta = np.mod(theta + 2*np.pi, 2*np.pi) # [0, 2pi]
    
    r_bins = np.logspace(np.log10(r_inner), np.log10(r_outer), n_r + 1)
    theta_bins = np.linspace(0, 2*np.pi, n_theta + 1)
    
    sc = np.zeros((N, n_r * n_theta), dtype=np.float32)
    for i in range(N):
        r_i = D_norm[i]
        th_i = theta[i]
        # Buang titik diri sendiri
        mask = (r_i > 0)
        h, _, _ = np.histogram2d(r_i[mask], th_i[mask], bins=[r_bins, theta_bins])
        sc[i] = h.flatten() / (np.sum(h) + 1e-7)
    return sc


def compute_fourier_descriptors(pts: np.ndarray, n_coeffs: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Menghitung Fourier Descriptors yang invarian translasi, rotasi, skala, titik awal."""
    z = pts[:, 0] + 1j * pts[:, 1]
    Z = np.fft.fft(z)
    
    # 1. Invarian Translasi: Z[0] = 0
    Z_inv = Z.copy()
    Z_inv[0] = 0.0
    
    # 2. Invarian Skala: bagi dengan |Z[1]|
    scale = np.abs(Z_inv[1]) + 1e-7
    Z_norm = Z_inv / scale
    
    # Ambil magnitudo untuk invarian rotasi & fase titik awal
    mags = np.abs(Z_norm)
    feature_vec = np.concatenate([mags[1:n_coeffs//2 + 1], mags[-n_coeffs//2:]]).astype(np.float32)
    
    # Rekonstruksi K koefisien
    Z_recon = np.zeros_like(Z)
    Z_recon[0] = Z[0] # kembalikan pusat untuk rekonstruksi posisi
    k_half = min(len(Z) // 2, n_coeffs // 2)
    Z_recon[1:k_half+1] = Z[1:k_half+1]
    Z_recon[-k_half:] = Z[-k_half:]
    z_rec = np.fft.ifft(Z_recon)
    pts_rec = np.column_stack([np.real(z_rec), np.imag(z_rec)]).astype(np.float32)
    return feature_vec, pts_rec


def compute_shape_lab(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Eksekusi Pipeline §9 Shape & Contour Matching Lab.
    Langkah:
    1. Input Citra Bentuk & Binerisasi Otsu
    2. Ekstraksi Kontur & Subsampling Titik (Jumlah Titik A vs B berbeda)
    3. Visualisasi Hausdorff Distance (Garis Terpanjang ke Titik Terburuk)
    4. Chamfer Matching (Distance Transform Heatmap Citra B + Titik A)
    5. Shape Context Log-Polar & Hungarian Assignment
    6. Rekonstruksi Kontur Fourier Descriptors (Slider K)
    7. Uji Ketahanan Outlier Ekstrem (Hausdorff vs Chamfer) + Tabel Rangkuman
    """
    t0 = time.time()
    warnings = []
    
    if img1 is None or img2 is None:
        return {"ok": False, "error": "Dua citra bentuk masukan diperlukan.", "hint": "Unggah citra bentuk daun, logo, atau siluet."}
        
    invert = params.get("invert", False)
    k_fourier = int(params.get("k_fourier", 12))
    add_outlier = params.get("add_outlier", False)
    
    # 1. Binerisasi Otsu & Kontur
    pts_a, bw_a = extract_largest_contour(img1, invert=invert)
    pts_b, bw_b = extract_largest_contour(img2, invert=invert)
    
    if len(pts_a) < 10 or len(pts_b) < 10:
        return {
            "ok": False,
            "error": "Kontur bentuk tidak terdeteksi secara jelas pada salah satu citra.",
            "hint": "Pastikan latar belakang memiliki kontras yang cukup atau aktifkan tombol Invert."
        }
        
    step1_img = bgr_to_b64(draw_side_by_side(bw_a, bw_b, "Binerisasi Citra A", "Binerisasi Citra B"))
    
    # 2. Subsampling Titik (Tegaskan jumlah titik tidak harus sama: misal 80 vs 140)
    pts_a_sub = sample_contour_points(pts_a, n_samples=80)
    pts_b_sub = sample_contour_points(pts_b, n_samples=140)
    
    if add_outlier:
        # Tambah 1 titik outlier ekstrem jauh di luar bentuk
        outlier_pt = np.array([[10.0, 10.0]], dtype=np.float32)
        pts_a_sub = np.vstack([pts_a_sub, outlier_pt])
        warnings.append("Titik outlier ekstrem disisipkan ke Himpunan A untuk demonstrasi lonjakan jarak Hausdorff.")
        
    fig2, ax2 = plt.subplots(figsize=(8, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax2, title=f"Sampling Titik Kontur: Set A ({len(pts_a_sub)} Titik) vs Set B ({len(pts_b_sub)} Titik)")
    ax2.scatter(pts_a_sub[:, 0], -pts_a_sub[:, 1], color=DEFAULTS["gold"], s=20, label="Titik Kontur A")
    ax2.scatter(pts_b_sub[:, 0], -pts_b_sub[:, 1], color=DEFAULTS["saffron"], s=15, alpha=0.6, label="Titik Kontur B")
    ax2.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    ax2.set_aspect("equal")
    step2_img = fig_to_b64(fig2)
    
    # 3. Hausdorff Distance (Berarah & Simetris)
    h_ab, w_a, c_b = compute_hausdorff_directed(pts_a_sub, pts_b_sub)
    h_ba, w_b, c_a = compute_hausdorff_directed(pts_b_sub, pts_a_sub)
    h_sym = max(h_ab, h_ba)
    
    fig3, ax3 = plt.subplots(figsize=(8, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax3, title=f"Hausdorff Distance Simetris = {h_sym:.2f} px (Worst Pair Garis Merah)")
    ax3.scatter(pts_a_sub[:, 0], -pts_a_sub[:, 1], color=DEFAULTS["gold"], s=15, alpha=0.5, label="Set A")
    ax3.scatter(pts_b_sub[:, 0], -pts_b_sub[:, 1], color=DEFAULTS["saffron"], s=15, alpha=0.5, label="Set B")
    
    # Sorot pasangan worst-case
    worst_p1 = pts_a_sub[w_a] if h_ab >= h_ba else pts_b_sub[w_b]
    worst_p2 = pts_b_sub[c_b] if h_ab >= h_ba else pts_a_sub[c_a]
    ax3.plot([worst_p1[0], worst_p2[0]], [-worst_p1[1], -worst_p2[1]], color="#ff4444", lw=2.5, linestyle="--")
    ax3.scatter([worst_p1[0]], [-worst_p1[1]], color="#ff4444", s=80, edgecolors="white", label="Titik Terburuk (Worst Point)")
    ax3.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    ax3.set_aspect("equal")
    step3_img = fig_to_b64(fig3)
    
    # 4. Chamfer Distance (Distance Transform Heatmap B + Titik A)
    chamfer_dist, dt_b = compute_chamfer_distance(pts_a_sub, bw_b)
    dt_heatmap = create_heatmap_bgr(dt_b, cv2.COLORMAP_VIRIDIS)
    
    # Overlay titik A di atas heatmap DT
    dt_vis = dt_heatmap.copy()
    for pt in pts_a_sub:
        cv2.circle(dt_vis, (int(pt[0]), int(pt[1])), 2, (255, 255, 255), -1)
    step4_img = bgr_to_b64(dt_vis)
    
    # 5. Shape Context Log-Polar & Hungarian Assignment
    sc_a = compute_shape_contexts(pts_a_sub[:40]) # Ambil subset 40 titik untuk komputasi cepat
    sc_b = compute_shape_contexts(pts_b_sub[:40])
    
    # Matriks Biaya Chi-Square antar descriptor shape context
    cost_matrix = cdist(sc_a, sc_b, lambda u, v: 0.5 * np.sum((u - v)**2 / (u + v + 1e-10)))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    sc_total_cost = float(cost_matrix[row_ind, col_ind].sum())
    
    # Visual Log-Polar Histogram Titik Pertama
    fig5, (ax_sc1, ax_sc2) = plt.subplots(1, 2, figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_sc1, title="Histogram Log-Polar Titik A_0 (5 Radius x 12 Sudut)")
    im_sc1 = ax_sc1.imshow(sc_a[0].reshape(5, 12), cmap="YlOrBr", aspect="auto")
    style_dark_axes(ax_sc2, title="Histogram Log-Polar Titik B_0 Terdekat")
    im_sc2 = ax_sc2.imshow(sc_b[col_ind[0]].reshape(5, 12), cmap="YlOrBr", aspect="auto")
    step5_img = fig_to_b64(fig5)
    
    # 6. Fourier Descriptors & Rekonstruksi K Koefisien
    fd_a, pts_rec_a = compute_fourier_descriptors(pts_a, n_coeffs=k_fourier)
    fd_b, pts_rec_b = compute_fourier_descriptors(pts_b, n_coeffs=k_fourier)
    
    fig6, ax6 = plt.subplots(figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax6, title=f"Rekonstruksi Kontur Fourier ({k_fourier} Koefisien Harmonis Terpilih)")
    ax6.plot(pts_a[:, 0], -pts_a[:, 1], color="#555555", lw=1, linestyle=":", label="Kontur A Asli")
    ax6.plot(pts_rec_a[:, 0], -pts_rec_a[:, 1], color=DEFAULTS["gold"], lw=2, label=f"Rekonstruksi A (K={k_fourier})")
    ax6.plot(pts_rec_b[:, 0], -pts_rec_b[:, 1], color=DEFAULTS["saffron"], lw=2, label=f"Rekonstruksi B (K={k_fourier})")
    ax6.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    ax6.set_aspect("equal")
    step6_img = fig_to_b64(fig6)
    
    # 7. Hu Moments (Reuse)
    hu_a = hu_moments_invariant(bw_a)["vector"]
    hu_b = hu_moments_invariant(bw_b)["vector"]
    d_hu = float(np.linalg.norm(hu_a - hu_b))
    
    shape_table = {
        "headers": ["Metode Pencocokan Bentuk", "Nilai Jarak / Skor", "Apakah Titik Harus Sama?", "Sensitivitas Outlier"],
        "rows": [
            ["Hausdorff Distance (Simetris)", f"{h_sym:.2f} px", "TIDAK (Set A != Set B)", "SANGAT TINGGI (Didominasi titik terburuk)"],
            ["Chamfer Distance (Mean DT)", f"{chamfer_dist:.2f} px", "TIDAK (Menggunakan Distance Map)", "MODERAT (Rata-rata jarak titik)"],
            ["Shape Context + Hungarian", f"{sc_total_cost:.2f} (Biaya)", "TIDAK (Bisa dummy-padded)", "RENDAH (Distribusi log-polar relatif)"],
            ["Fourier Descriptors (L2)", f"{float(np.linalg.norm(fd_a - fd_b)):.4f}", "TIDAK (Terekspansi ke K harmonis)", "RENDAH (Frekuensi tinggi dibuang)"],
            ["Hu Invariant Moments", f"{d_hu:.4f}", "TIDAK (Berdasarkan integral momen)", "RENDAH terhadap noise spasial"]
        ]
    }
    
    steps = [
        create_step(1, "Binerisasi Otsu & Segmentasi Objek",
                    "Memisahkan siluet bentuk dari latar belakang menggunakan pengambang otomatis Otsu.",
                    formula="T = \\arg\\max \\sigma_B^2(T)",
                    image_b64=step1_img),
        create_step(2, "Subsampling Himpunan Titik Kontur",
                    f"Mengekstraksi {len(pts_a_sub)} titik pada Kontur A dan {len(pts_b_sub)} titik pada Kontur B. Konsep materi menegaskan: algoritma set matching tidak menuntut jumlah titik yang identik.",
                    formula="A = \\{a_1, ..., a_{N_A}\\}, \\quad B = \\{b_1, ..., b_{N_B}\\}",
                    image_b64=step2_img,
                    metrics={"titik_a": len(pts_a_sub), "titik_b": len(pts_b_sub)}),
        create_step(3, "Kalkulasi Hausdorff Distance & Worst-Point",
                    f"Menghitung jarak minimax. Pasangan titik terburuk (worst-case error) berjarak {h_sym:.2f} piksel.",
                    formula="H(A, B) = \\max \\{ \\sup_{a} \\inf_{b} d(a,b), \\sup_{b} \\inf_{a} d(b,a) \\}",
                    image_b64=step3_img,
                    metrics={"hausdorff_simetris": h_sym, "h_AB": h_ab, "h_BA": h_ba}),
        create_step(4, "Chamfer Distance via Distance Transform",
                    f"Memetakan seluruh titik A ke medan peta jarak kontur B. Rata-rata jarak perpindahan kontur adalah {chamfer_dist:.2f} piksel.",
                    formula="d_{\\text{Chamfer}}(A, B) = \\frac{1}{|A|} \\sum_{a \\in A} \\min_{b \\in B} \\|a - b\\|",
                    image_b64=step4_img,
                    metrics={"chamfer_mean": chamfer_dist}),
        create_step(5, "Shape Context Log-Polar & Hungarian Korespondensi",
                    "Menghitung distribusi spasial relatif tetangga dalam kisi log-polar untuk mendeskripsikan bentuk lokal setiap titik kontur.",
                    formula="h_i(k) = \\#\\{q \\ne p_i : (q - p_i) \\in \\text{bin}(k)\\}",
                    image_b64=step5_img,
                    metrics={"biaya_hungarian": sc_total_cost}),
        create_step(6, "Spektrum Fourier Descriptors & Rekonstruksi",
                    f"Transformasi Fourier pada kurva parametrik z(t) = x(t) + i y(t). Rekonstruksi menggunakan {k_fourier} koefisien pertama mempertahankan morfologi global.",
                    formula="z_k = \\sum_{n=0}^{N-1} (x_n + i y_n) e^{-i 2\\pi k n / N}",
                    image_b64=step6_img,
                    table=shape_table,
                    metrics={"k_koefisien": k_fourier, "jarak_hu": d_hu})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "shape_matching",
        "summary": f"Pencocokan Kontur: Hausdorff={h_sym:.2f} px, Chamfer={chamfer_dist:.2f} px, Fourier(K={k_fourier}) L2={float(np.linalg.norm(fd_a - fd_b)):.4f}.",
        "steps": steps,
        "result": {
            "image": step3_img,
            "metrics": {"hausdorff": h_sym, "chamfer": chamfer_dist, "shape_context_cost": sc_total_cost, "hu_dist": d_hu},
            "table": shape_table
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }

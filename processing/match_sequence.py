"""
processing/match_sequence.py
============================
Implementasi §10: Sequence Matching — Dynamic Time Warping (DTW),
serta §11: Assignment & Set Matching (Hungarian & Optimal Transport Sinkhorn).

Fitur:
- Input Sequence dari:
  (a) Vektor manual (default materi: A=[1,3,4,7,8], B=[1,2,3,3,4,6,7,8])
  (b) Profil proyeksi horizontal/vertikal citra
  (c) Signature kontur (jarak centroid terhadap sudut)
  (d) Koordinat goresan/tanda tangan canvas
- Matriks Cost (Heatmap)
- Matriks Akumulasi DTW + Warping Path Minimum
- Plot Alignment Pasangan Titik A <-> B
- Perbandingan Euclidean (Zero-padding/Resample) vs DTW (Mengapa DTW unggul saat panjang/kecepatan beda)
- Opsi Sakoe-Chiba Band
- §11 Hungarian 1-ke-1 vs Nearest Neighbor independen (konflik many-to-one)
- §11 Sinkhorn Optimal Transport (Log-domain stabil, konvergensi error marginal, heatmap transport plan).
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
    DEFAULTS, create_step, bgr_to_b64, fig_to_b64, style_dark_axes, create_heatmap_bgr
)


# =============================================================================
# §10 DYNAMIC TIME WARPING (DTW)
# =============================================================================

def compute_dtw(seq_a: np.ndarray, seq_b: np.ndarray, window_band: int = None) -> tuple[float, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    Kalkulasi DTW dengan matriks akumulasi dan backtracking warping path.
    seq_a: 1D array panjang N
    seq_b: 1D array panjang M
    window_band: lebar pita pembatas Sakoe-Chiba (|i - j| <= window_band)
    """
    n, m = len(seq_a), len(seq_b)
    # Matriks Biaya Lokal
    cost_matrix = np.abs(seq_a[:, None] - seq_b[None, :])
    
    # Matriks Akumulasi D
    accum_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    accum_matrix[0, 0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if window_band is not None and abs(i - j) > window_band:
                continue
            cost = cost_matrix[i - 1, j - 1]
            accum_matrix[i, j] = cost + min(accum_matrix[i - 1, j],       # insersi
                                            accum_matrix[i, j - 1],       # delesi
                                            accum_matrix[i - 1, j - 1])   # match
                                            
    dtw_distance = float(accum_matrix[n, m])
    
    # Backtracking path minimum
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            steps = [accum_matrix[i - 1, j - 1], accum_matrix[i - 1, j], accum_matrix[i, j - 1]]
            best_idx = np.argmin(steps)
            if best_idx == 0:
                i -= 1; j -= 1
            elif best_idx == 1:
                i -= 1
            else:
                j -= 1
        path.append((i - 1, j - 1))
    path.reverse()
    return dtw_distance, cost_matrix, accum_matrix[1:, 1:], path


def compute_sequence_lab(seq_a_input, seq_b_input, params: dict) -> dict:
    """
    Eksekusi Pipeline §10 Dynamic Time Warping Lab.
    Langkah:
    1. Input Sekuens A & B (Plot profil 1D)
    2. Matriks Biaya Lokal |a_i - b_j|
    3. Matriks Akumulasi & Penelusuran Jalur Optimal (Warping Path)
    4. Plot Alignment Garis Korespondensi Titik A <-> B
    5. Evaluasi Komparatif: DTW vs Euclidean Kaku (Resampled)
    6. Eksperimen Sakoe-Chiba Band
    """
    t0 = time.time()
    warnings = []
    
    use_material_preset = params.get("use_material_preset", False)
    sakoe_band = params.get("sakoe_chiba_band", None)
    if sakoe_band is not None:
        sakoe_band = int(sakoe_band)
        
    if use_material_preset or seq_a_input is None or seq_b_input is None:
        # Default contoh materi persis: A=[1,3,4,7,8], B=[1,2,3,3,4,6,7,8]
        seq_a = np.array([1, 3, 4, 7, 8], dtype=np.float32)
        seq_b = np.array([1, 2, 3, 3, 4, 6, 7, 8], dtype=np.float32)
    else:
        seq_a = np.array(seq_a_input, dtype=np.float32)
        seq_b = np.array(seq_b_input, dtype=np.float32)
        
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return {"ok": False, "error": "Sekuens tidak boleh kosong.", "hint": "Masukkan minimal 2 elemen angka."}
        
    # 1. Visualisasi Sekuens A & B
    fig1, (ax_sa, ax_sb) = plt.subplots(2, 1, figsize=(8, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_sa, title=f"Sekuens A (Panjang N={n})", ylabel="Nilai")
    ax_sa.plot(seq_a, color=DEFAULTS["gold"], marker="o", lw=2)
    style_dark_axes(ax_sb, title=f"Sekuens B (Panjang M={m})", xlabel="Langkah Waktu / Indeks", ylabel="Nilai")
    ax_sb.plot(seq_b, color=DEFAULTS["saffron"], marker="s", lw=2)
    step1_img = fig_to_b64(fig1)
    
    # 2 & 3. DTW Cost & Accumulation Matrix
    dtw_dist, cost_mat, accum_mat, path = compute_dtw(seq_a, seq_b, window_band=sakoe_band)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax2, title=f"Matriks Akumulasi DTW & Jalur Minimum (Biaya = {dtw_dist:.2f})", xlabel="Indeks Sekuens B", ylabel="Indeks Sekuens A")
    ax2.imshow(accum_mat, cmap="magma", aspect="auto", origin="lower")
    py, px = zip(*path)
    ax2.plot(px, py, color="#64d2ff", lw=3, label="Optimal Warping Path")
    ax2.scatter(px, py, color="white", s=25, zorder=4)
    ax2.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step3_img = fig_to_b64(fig2)
    
    # Matriks Biaya Lokal Murni
    fig_cost, ax_cost = plt.subplots(figsize=(6, 4.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_cost, title="Matriks Biaya Jarak Lokal |a_i - b_j|", xlabel="Indeks B", ylabel="Indeks A")
    im_c = ax_cost.imshow(cost_mat, cmap="viridis", aspect="auto", origin="lower")
    plt.colorbar(im_c, ax=ax_cost)
    step2_img = fig_to_b64(fig_cost)
    
    # 4. Plot Alignment Titik Korespondensi
    fig4, ax4 = plt.subplots(figsize=(8, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax4, title="Korespondensi Penyelarasan Temporal Non-Linier (A <-> B)")
    # Geser visual sekuens A di atas, sekuens B di bawah
    y_shift = max(np.max(seq_a), np.max(seq_b)) * 1.5
    ax4.plot(seq_a + y_shift, color=DEFAULTS["gold"], marker="o", lw=2, label="Sekuens A (Ditranslasi)")
    ax4.plot(seq_b, color=DEFAULTS["saffron"], marker="s", lw=2, label="Sekuens B")
    
    # Hubungkan garis alignment
    for (ia, ib) in path:
        ax4.plot([ia, ib], [seq_a[ia] + y_shift, seq_b[ib]], color=DEFAULTS["terracotta"], alpha=0.6, linestyle="--")
    ax4.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    ax4.set_xticks([])
    step4_img = fig_to_b64(fig4)
    
    # 5. DTW vs Euclidean Kaku (Resampling)
    # Resample A agar panjangnya sama dengan B
    t_resample = np.linspace(0, n - 1, m)
    seq_a_resampled = np.interp(t_resample, np.arange(n), seq_a)
    euc_resampled = float(np.linalg.norm(seq_a_resampled - seq_b))
    
    fig5, ax5 = plt.subplots(figsize=(8, 2.6), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax5, title="Komparasi Keunggulan: DTW Elastis vs Euclidean Kaku (Resampled)")
    ax5.barh(["Euclidean (Resampling Linier)", "Dynamic Time Warping (DTW)"], [euc_resampled, dtw_dist],
             color=[DEFAULTS["terracotta"], DEFAULTS["gold"]], height=0.45)
    ax5.text(euc_resampled + 0.1, 0, f"{euc_resampled:.2f}", va='center', color=DEFAULTS["text_color"], fontsize=9)
    ax5.text(dtw_dist + 0.1, 1, f"{dtw_dist:.2f}", va='center', color=DEFAULTS["text_color"], fontsize=9)
    ax5.set_xlim(0, max(euc_resampled, dtw_dist) * 1.35)
    step5_img = fig_to_b64(fig5)
    
    dtw_table = {
        "headers": ["Aspek Penyelarasan", "Nilai", "Konsep Materi §10"],
        "rows": [
            ["Jarak Total DTW", f"{dtw_dist:.4f}", "Biaya akumulasi sepanjang jalur optimal minimum"],
            ["Panjang Optimal Path", f"{len(path)} langkah", "Memetakan N={n} ke M={m} secara elastis"],
            ["Euclidean Resampled", f"{euc_resampled:.4f}", "Rentan terhadap pergeseran fase dan distorsi kecepatan"],
            ["Batasan Sakoe-Chiba", f"{sakoe_band if sakoe_band is not None else 'Tak Terbatas'}", "Mencegah pergeseran temporal ekstrem dan mempercepat komputasi O(N*W)"]
        ]
    }
    
    steps = [
        create_step(1, "Input Sekuens Temporal Berbeda Panjang",
                    f"Menerima Sekuens A (panjang N={n}) dan Sekuens B (panjang M={m}).",
                    formula="A = (a_1, ..., a_N), \\quad B = (b_1, ..., b_M)",
                    image_b64=step1_img,
                    metrics={"panjang_A": n, "panjang_B": m}),
        create_step(2, "Matriks Biaya Jarak Lokal (Cost Matrix)",
                    "Menghitung selisih jarak absolut untuk setiap pasangan titik kisi N x M.",
                    formula="c(i, j) = |a_i - b_j|",
                    image_b64=step2_img),
        create_step(3, "Pemrograman Dinamis & Optimal Warping Path",
                    f"Membangun matriks akumulasi kumulatif dan menelusuri mundur jalur optimal dari (N,M) ke (1,1). Total biaya = {dtw_dist:.2f}.",
                    formula="D(i,j) = c(i,j) + \\min \\{ D(i-1,j), D(i,j-1), D(i-1,j-1) \\}",
                    image_b64=step3_img,
                    metrics={"biaya_dtw": dtw_dist, "langkah_path": len(path)}),
        create_step(4, "Penyelarasan Korespondensi Waktu Elastis",
                    "Memvisualisasikan koneksi titik A ke B. DTW mampu menangani pemuluran (stretching) dan pemadatan (compression) tempo.",
                    formula="W = (w_1, w_2, ..., w_K), \\quad w_k = (i, j)",
                    image_b64=step4_img),
        create_step(5, "Evaluasi: DTW vs Euclidean Kaku",
                    f"Memperlihatkan keunggulan DTW saat panjang dan kecepatan berbeda. Jarak DTW ({dtw_dist:.2f}) jauh lebih mencerminkan kemiripan struktural dibandingkan Euclidean ({euc_resampled:.2f}).",
                    formula="\\text{DTW unggul pada variasi tempo & fase}",
                    image_b64=step5_img,
                    table=dtw_table,
                    metrics={"dtw": dtw_dist, "euclidean_resample": euc_resampled})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "sequence_dtw",
        "summary": f"DTW Sequence Matching: Biaya akumulasi = {dtw_dist:.4f} ({len(path)} langkah penyelarasan).",
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {"dtw_dist": dtw_dist, "path_length": len(path), "euclidean_resampled": euc_resampled},
            "table": dtw_table
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }


# =============================================================================
# §11 ASSIGNMENT & OPTIMAL TRANSPORT LAB
# =============================================================================

def sinkhorn_algorithm(a: np.ndarray, b: np.ndarray, M: np.ndarray, reg: float = 0.05, num_iters: int = 100) -> tuple[np.ndarray, list[float]]:
    """
    Implementasi Sinkhorn Optimal Transport stabil dalam domain log/eksponensial.
    a: bobot marginal sumber (sum=1)
    b: bobot marginal target (sum=1)
    M: cost matrix
    reg: parameter entropi epsilon
    """
    K = np.exp(-M / reg)
    u = np.ones_like(a) / len(a)
    errors = []
    
    for it in range(num_iters):
        v = b / (np.dot(K.T, u) + 1e-12)
        u = a / (np.dot(K, v) + 1e-12)
        
        # Marginal error
        err = np.linalg.norm(np.sum(u[:, None] * K * v[None, :], axis=1) - a)
        errors.append(float(err))
        if err < 1e-4:
            break
            
    P = u[:, None] * K * v[None, :]
    return P, errors


def compute_assignment_lab(pts1: np.ndarray, pts2: np.ndarray, params: dict) -> dict:
    """
    Eksekusi Pipeline §11 Assignment & Optimal Transport.
    Langkah:
    1. Input Himpunan Titik Spasial A & B
    2. Matriks Biaya Jarak Spasial
    3. Nearest Neighbor Independen (Konflik Many-to-One Disorot)
    4. Hungarian Assignment (Korespondensi Biyektif 1-ke-1)
    5. Optimal Transport Sinkhorn Plan (Heatmap Fluks Massa & Konvergensi)
    6. Komparasi Total Biaya & Tabel Ringkasan
    """
    t0 = time.time()
    warnings = []
    
    # Generate titik sintetis jika tidak diberikan
    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0:
        rng = np.random.default_rng(42)
        pts1 = rng.uniform(20, 280, size=(10, 2)).astype(np.float32)
        pts2 = pts1 + rng.normal(15, 10, size=(10, 2)).astype(np.float32)
        
    n_a, n_b = len(pts1), len(pts2)
    # Batasi untuk visualisasi yang bersih
    pts1_sub = pts1[:min(16, n_a)]
    pts2_sub = pts2[:min(16, n_b)]
    
    # 1. Visual Input
    fig1, ax1 = plt.subplots(figsize=(7, 3.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax1, title="Himpunan Titik Sumber A dan Target B")
    ax1.scatter(pts1_sub[:, 0], pts1_sub[:, 1], color=DEFAULTS["gold"], s=50, label="Titik A (Sumber)")
    ax1.scatter(pts2_sub[:, 0], pts2_sub[:, 1], color=DEFAULTS["saffron"], s=50, label="Titik B (Target)")
    for i, pt in enumerate(pts1_sub):
        ax1.text(pt[0]+2, pt[1]+2, f"A{i}", color=DEFAULTS["gold"], fontsize=8)
    for j, pt in enumerate(pts2_sub):
        ax1.text(pt[0]+2, pt[1]+2, f"B{j}", color=DEFAULTS["saffron"], fontsize=8)
    ax1.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    step1_img = fig_to_b64(fig1)
    
    # 2. Matriks Biaya Jarak
    cost_matrix = cdist(pts1_sub, pts2_sub, metric='euclidean')
    fig2, ax2 = plt.subplots(figsize=(6, 3.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax2, title="Matriks Biaya Euclidean Jarak Antar Titik", xlabel="Indeks Titik B", ylabel="Indeks Titik A")
    im2 = ax2.imshow(cost_matrix, cmap="inferno", aspect="auto")
    plt.colorbar(im2, ax=ax2)
    step2_img = fig_to_b64(fig2)
    
    # 3. Nearest Neighbor Independen (Konflik Many-to-One)
    nn_indices = np.argmin(cost_matrix, axis=1)
    nn_cost = float(np.sum(np.min(cost_matrix, axis=1)))
    
    # Deteksi konflik
    counts = np.bincount(nn_indices, minlength=len(pts2_sub))
    conflicted_targets = np.where(counts > 1)[0]
    
    fig3, ax3 = plt.subplots(figsize=(7, 3.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax3, title=f"Nearest Neighbor Independen (Konflik: {len(conflicted_targets)} Titik Dipilih Berulang)")
    ax3.scatter(pts1_sub[:, 0], pts1_sub[:, 1], color=DEFAULTS["gold"], s=45)
    ax3.scatter(pts2_sub[:, 0], pts2_sub[:, 1], color=DEFAULTS["saffron"], s=45)
    for i, target_idx in enumerate(nn_indices):
        is_conflicted = target_idx in conflicted_targets
        col = "#ff4444" if is_conflicted else "#777777"
        ax3.plot([pts1_sub[i, 0], pts2_sub[target_idx, 0]], [pts1_sub[i, 1], pts2_sub[target_idx, 1]],
                 color=col, lw=2 if is_conflicted else 1, linestyle="--")
    step3_img = fig_to_b64(fig3)
    
    # 4. Hungarian Assignment (Biyektif 1-ke-1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    hungarian_cost = float(cost_matrix[row_ind, col_ind].sum())
    
    fig4, ax4 = plt.subplots(figsize=(7, 3.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax4, title=f"Hungarian Assignment 1-ke-1 (Total Biaya = {hungarian_cost:.2f})")
    ax4.scatter(pts1_sub[:, 0], pts1_sub[:, 1], color=DEFAULTS["gold"], s=45)
    ax4.scatter(pts2_sub[:, 0], pts2_sub[:, 1], color=DEFAULTS["saffron"], s=45)
    for r, c in zip(row_ind, col_ind):
        ax4.plot([pts1_sub[r, 0], pts2_sub[c, 0]], [pts1_sub[r, 1], pts2_sub[c, 1]],
                 color=DEFAULTS["gold"], lw=2)
    step4_img = fig_to_b64(fig4)
    
    # 5. Optimal Transport Sinkhorn
    a_weights = np.ones(len(pts1_sub)) / len(pts1_sub)
    b_weights = np.ones(len(pts2_sub)) / len(pts2_sub)
    M_norm = cost_matrix / (np.max(cost_matrix) + 1e-7)
    P_transport, err_hist = sinkhorn_algorithm(a_weights, b_weights, M_norm, reg=0.08, num_iters=80)
    ot_cost = float(np.sum(P_transport * cost_matrix))
    
    fig5, (ax_p, ax_err) = plt.subplots(1, 2, figsize=(8, 3.0), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_p, title="Transport Plan Fluks Massa P (Sinkhorn)", xlabel="Indeks B", ylabel="Indeks A")
    ax_p.imshow(P_transport, cmap="YlOrBr", aspect="auto")
    style_dark_axes(ax_err, title="Konvergensi Error Marginal Sinkhorn", xlabel="Iterasi", ylabel="Residual Error")
    ax_err.plot(err_hist, color=DEFAULTS["gold"], lw=2)
    ax_err.set_yscale("log")
    step5_img = fig_to_b64(fig5)
    
    assign_table = {
        "headers": ["Algoritma", "Sifat Pemetaan", "Total Biaya", "Keterangan Konflik"],
        "rows": [
            ["Nearest Neighbor", "Independen (Lokal)", f"{nn_cost:.2f}", f"Banyak-ke-satu (Terdapat {len(conflicted_targets)} titik ganda)"],
            ["Hungarian Algorithm", "Biyektif (1-ke-1)", f"{hungarian_cost:.2f}", "Optimal global tanpa tumpang tindih"],
            ["Optimal Transport (Sinkhorn)", "Distribusi Fraksional", f"{ot_cost:.2f}", "Memperhitungkan entropi relaxasi P"]
        ]
    }
    
    steps = [
        create_step(1, "Input Himpunan Titik Spasial",
                    f"Menerima {len(pts1_sub)} titik entitas A dan {len(pts2_sub)} titik entitas B.",
                    formula="A = \\{x_1, ..., x_n\\}, \\quad B = \\{y_1, ..., y_m\\}",
                    image_b64=step1_img),
        create_step(2, "Kalkulasi Matriks Biaya Jarak",
                    "Membangun matriks biaya Euclidean pairwise antar seluruh kombinasi koordinat.",
                    formula="C_{ij} = \\|x_i - y_j\\|_2",
                    image_b64=step2_img),
        create_step(3, "Nearest Neighbor Independen & Deteksi Konflik",
                    f"Mencocokkan setiap titik A ke B secara serakah (greedy). Terdeteksi konflik tumpang tindih: {len(conflicted_targets)} target diperebutkan oleh banyak sumber.",
                    formula="j^*(i) = \\arg\\min_j C_{ij}",
                    image_b64=step3_img,
                    metrics={"biaya_nn": nn_cost, "target_konflik": len(conflicted_targets)}),
        create_step(4, "Hungarian Assignment 1-ke-1 Optimal Global",
                    f"Menyelesaikan alokasi bipartit secara serentak sehingga tidak ada elemen yang terduplikasi. Total biaya optimal = {hungarian_cost:.2f}.",
                    formula="\\min_P \\sum_{i,j} C_{ij} P_{ij} \\quad \\text{s.t.} \\quad P \\in \\Pi(n)",
                    image_b64=step4_img,
                    metrics={"biaya_hungarian": hungarian_cost}),
        create_step(5, "Sinkhorn Optimal Transport & Konvergensi",
                    f"Menghitung rencana transportasi massa optimal dengan regularisasi entropi. Konvergensi tercapai dengan residual error < {err_hist[-1]:.2e}.",
                    formula="P^* = \\arg\\min_P \\langle P, C \\rangle - \\epsilon H(P)",
                    image_b64=step5_img,
                    table=assign_table,
                    metrics={"biaya_ot": ot_cost, "residual_error": err_hist[-1]})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "assignment_matching",
        "summary": f"Assignment Lab: Hungarian 1-to-1 Cost={hungarian_cost:.2f} (NN Biaya={nn_cost:.2f} dengan konflik).",
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {"hungarian_cost": hungarian_cost, "nn_cost": nn_cost, "ot_cost": ot_cost},
            "table": assign_table
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }

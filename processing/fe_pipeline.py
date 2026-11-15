"""
processing/fe_pipeline.py
==========================
Lapisan "cerita visual" di atas feature_extraction_lab.py.

MASALAH YANG DIPERBAIKI (revisi dari dosen):
  Revisi 1: Setiap panel HARUS menunjukkan proses bertahap (citra masuk ->
            representasi antara -> representasi antara lagi (kalau ada) ->
            vector akhir), bukan langsung lompat ke satu bar chart hasil.
  Revisi 2: "Citra Asli" grayscale JANGAN dipaksakan tampil di semua panel.
            Fitur tekstur/bentuk/gradien -> citra asli GRAYSCALE (memang
            input ekstraksinya). Fitur warna -> citra asli BERWARNA. Fitur
            matching/robustness -> TIDAK ada step "citra asli" generik sama
            sekali, karena yang relevan adalah citra-citra yang dibandingkan/
            ditransformasi itu sendiri.
  Revisi 3: Matching (ORB/SIFT) & Matching Engine TIDAK pakai sample default
            untuk gambar kedua -- keduanya wajib upload dari user (ditegakkan
            di app.py, modul ini hanya menerima dua citra apa adanya).

KONTRAK OUTPUT (app.py tinggal jsonify apa adanya -- tidak ada array numpy
mentah yang bocor ke JSON, semua sudah dikonversi/dibuang di sini):
  {
    "steps": [ {"title": str, "caption": str, "image": bytes_png}, ... ],
    "vector": list[float] | None,
    "labels": list[str] | None,
    "dim": int | None,
    "info": dict,
  }
"""

import cv2
import numpy as np
import matplotlib
from skimage.feature import local_binary_pattern

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import feature_extraction_lab as fel


# =============================================================================
# HELPER DASAR
# =============================================================================

def _png(fig):
    return fel._fig_to_png_bytes(fig)


def step_image(img, title, cmap=None, caption=""):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    if img.ndim == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(img, cmap=cmap or "gray")
    ax.set_title(title, fontsize=10, color="#f0f1f5")
    ax.axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return {"title": title, "caption": caption, "image": _png(fig)}


def step_images_row(images_titles, suptitle="", caption=""):
    n = len(images_titles)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (img, t) in zip(axes, images_titles):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(t, fontsize=9, color="#f0f1f5")
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color="#f0f1f5")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return {"title": suptitle or "Perbandingan", "caption": caption, "image": _png(fig)}


def step_png(png_bytes, title, caption=""):
    return {"title": title, "caption": caption, "image": png_bytes}


def step_vector_bar(vector, labels, title, caption="", max_bars=40):
    png = fel.visualize_vector(np.asarray(vector), labels, title, max_bars=max_bars)
    return {"title": title, "caption": caption, "image": png}


def _clean_info(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = _clean_info(v)
        else:
            out[k] = v
    return out


def finalize(steps, vector=None, labels=None, info=None):
    return {
        "steps": steps,
        "vector": (np.asarray(vector).astype(float).tolist() if vector is not None else None),
        "labels": labels,
        "dim": (len(vector) if vector is not None else None),
        "info": _clean_info(info or {}),
    }


def _draw_grid_overlay(img_gray, block, stride):
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    H, W = img_gray.shape
    for y in range(0, H - block + 1, stride):
        cv2.line(overlay, (0, y), (W, y), (77, 208, 225), 1)
    for x in range(0, W - block + 1, stride):
        cv2.line(overlay, (x, 0), (x, H), (77, 208, 225), 1)
    return overlay


# =============================================================================
# 3.1 — STATISTIK & WARNA
# =============================================================================

def pipeline_intensity_statistics(img_gray, img_bgr, n_bins=8):
    r = fel.intensity_statistics(img_gray, n_bins=n_bins)
    hist = np.array(r["representasi_antara"]["histogram_bins"])
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(len(hist)), hist, color="#4f7cff")
    ax.set_title(f"Histogram Intensitas ({n_bins} bin)", fontsize=10)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Jumlah Piksel")
    fel._style_dark_axes(ax)
    fig.tight_layout()
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)", caption="Input untuk statistik intensitas."),
        step_png(_png(fig), "2. Histogram Intensitas", caption="Representasi antara: distribusi nilai piksel."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir",
                         caption="[mean, std, min, max, entropy] + histogram N-bin."),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_color_statistics_multispace(img_gray, img_bgr, space="hsv"):
    r = fel.color_statistics_multispace(img_bgr, space=space)
    conv = cv2.cvtColor(img_bgr, {"rgb": cv2.COLOR_BGR2RGB, "hsv": cv2.COLOR_BGR2HSV,
                                    "lab": cv2.COLOR_BGR2LAB}[space.lower()])
    ch_names = {"rgb": ["R", "G", "B"], "hsv": ["H", "S", "V"], "lab": ["L", "a", "b"]}[space.lower()]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for i, (ax, name) in enumerate(zip(axes, ch_names)):
        ax.imshow(conv[:, :, i], cmap="gray")
        ax.set_title(f"Kanal {name}", fontsize=9, color="#f0f1f5")
        ax.axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    steps = [
        step_image(img_bgr, "1. Citra Asli (Warna)", caption="Fitur ini berbasis warna, bukan grayscale."),
        step_png(_png(fig), f"2. Konversi ke {space.upper()}",
                 caption=f"Citra dipecah jadi 3 kanal {space.upper()} secara terpisah."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir", caption="Mean & std tiap kanal."),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_color_histogram_per_channel(img_gray, img_bgr, space="hsv", n_bins=8):
    r = fel.color_histogram_per_channel(img_bgr, space=space, n_bins=n_bins)
    per_ch = r["representasi_antara"]["per_channel_hist"]
    ch_names = {"rgb": ["R", "G", "B"], "hsv": ["H", "S", "V"]}[space.lower()]
    colors = {"rgb": ["#ff6b6b", "#5bd17a", "#4f7cff"], "hsv": ["#f2ca50", "#ff9f6b", "#4dd0e1"]}[space.lower()]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))
    for ax, h, name, c in zip(axes, per_ch, ch_names, colors):
        ax.bar(range(len(h)), h, color=c)
        ax.set_title(f"Histogram {name}", fontsize=9, color="#f0f1f5")
        fel._style_dark_axes(ax)
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    steps = [
        step_image(img_bgr, "1. Citra Asli (Warna)"),
        step_png(_png(fig), "2. Histogram per Kanal", caption=f"Ruang warna {space.upper()}, {n_bins} bin/kanal."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (concat)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


# =============================================================================
# 3.2 — BENTUK & MOMENTS
# =============================================================================

def _draw_shape_overlay(img_gray, bw):
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(overlay, [c], -1, (0, 255, 120), 2)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (77, 208, 225), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            cv2.circle(overlay, (cx, cy), 5, (255, 100, 100), -1)
    return overlay


def pipeline_shape_descriptors(img_gray, img_bgr):
    bw = fel._binarize_otsu(img_gray)
    r = fel.shape_descriptors(img_gray, bw)
    overlay = _draw_shape_overlay(img_gray, bw)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_image(bw, "2. Citra Biner (Otsu)", caption="Segmentasi awal -- wajib sebelum fitur bentuk."),
        step_image(overlay, "3. Kontur + Bounding Box + Centroid",
                   caption="Hijau=kontur, biru=bbox, merah=centroid."),
        step_vector_bar(r["vector"], r["labels"], "4. Vector Akhir",
                         caption="[area, perimeter, aspect_ratio, cx, cy, circularity]"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_raw_central_moments(img_gray, img_bgr):
    bw = fel._binarize_otsu(img_gray)
    r = fel.raw_central_moments(img_gray, bw)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_image(bw, "2. Citra Biner (Otsu)"),
        step_vector_bar(r["vector"], r["labels"], "3. Tabel Moments (m00,m10,m01,mu20,nu20)",
                         caption="m10/m01 peka posisi; mu20/nu20 relatif terhadap centroid."),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_hu_moments_invariant(img_gray, img_bgr, log_scale=True):
    bw = fel._binarize_otsu(img_gray)
    r = fel.hu_moments_invariant(img_gray, bw, log_scale=log_scale)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_image(bw, "2. Citra Biner (Otsu)"),
        step_vector_bar(r["vector"], r["labels"], "3. Hu Moments (7D, log-scale)",
                         caption="Stabil terhadap translasi/rotasi/skala -- coba juga Robustness Stress-Test."),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


# =============================================================================
# 3.3 — GLCM
# =============================================================================

def pipeline_glcm_explorer(img_gray, img_bgr, distances=(1,), angles_deg=(0,), levels=8,
                            symmetric=True, pooling=None):
    q = fel._quantize(img_gray, levels)
    r = fel.glcm_explorer(img_gray, distances=distances, angles_deg=angles_deg, levels=levels,
                           symmetric=symmetric, pooling=pooling)
    heatmap_png = fel.visualize_glcm_heatmap(img_gray, distance=distances[0], angle_deg=angles_deg[0],
                                              levels=levels)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_image(q, f"2. Citra Terkuantisasi ({levels} level)",
                   caption="GLCM butuh level abu-abu terbatas, bukan 256 penuh."),
        step_png(heatmap_png, f"3. Matriks GLCM (d={distances[0]}, {angles_deg[0]}°)",
                 caption="Representasi antara -- BUKAN feature akhir."),
        step_vector_bar(r["vector"], r["labels"], "4. Vector Akhir (Haralick features)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_glcm_block(img_gray, img_bgr, block=32, stride=None, levels=8, distances=(1,),
                         angles_deg=(0, 90)):
    r = fel.glcm_block(img_gray, block=block, stride=stride, levels=levels,
                        distances=distances, angles_deg=angles_deg)
    stride_eff = stride or block
    grid_img = _draw_grid_overlay(img_gray, block, stride_eff)
    F = np.array(r["vector"]).reshape(r["representasi_antara"]["n_blok"], -1)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(F, cmap="Blues", aspect="auto")
    ax.set_title("Matriks Fitur per Blok", fontsize=10)
    ax.set_xlabel("Statistik GLCM")
    ax.set_ylabel("Nomor Blok")
    fel._style_dark_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    steps = [
        step_image(grid_img, f"1. Pembagian Blok ({block}px, stride={stride_eff})",
                   caption="Overlap" if stride_eff < block else "Non-overlap"),
        step_png(_png(fig), "2. Heatmap Fitur per Blok",
                 caption=f"{r['representasi_antara']['n_blok']} blok."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (flatten)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_glcm_color(img_gray, img_bgr, levels=8):
    r = fel.glcm_color(img_bgr, levels=levels)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for i, (ax, name) in enumerate(zip(axes, ["R", "G", "B"])):
        ax.imshow(rgb[:, :, i], cmap="gray")
        ax.set_title(f"Kanal {name}", fontsize=9, color="#f0f1f5")
        ax.axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    steps = [
        step_image(img_bgr, "1. Citra Asli (Warna)"),
        step_png(_png(fig), "2. Pemisahan Kanal R/G/B", caption="GLCM dihitung terpisah tiap kanal."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (3 kanal x 3 statistik)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


# =============================================================================
# 3.3 — LBP
# =============================================================================

def pipeline_lbp_explorer(img_gray, img_bgr, P=8, R=1, method="uniform"):
    r = fel.lbp_explorer(img_gray, P=P, R=R, method=method)
    combo_png = fel.visualize_lbp_map(img_gray, P=P, R=R, method=method)
    steps = [
        step_png(combo_png, "1-2. Citra Asli -> LBP Map -> Histogram",
                 caption="Tiga tahap sekaligus: representasi antara (map) sampai vector (histogram)."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (histogram dinormalisasi)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_lbp_block(img_gray, img_bgr, P=8, R=1, method="uniform", block=16, stride=None):
    r = fel.lbp_block(img_gray, P=P, R=R, method=method, block=block, stride=stride)
    stride_eff = stride or block
    grid_img = _draw_grid_overlay(img_gray, block, stride_eff)
    n_blok, n_bins = r["representasi_antara"]["feature_matrix_shape"]
    F = np.array(r["vector"]).reshape(n_blok, n_bins)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(F, cmap="viridis", aspect="auto")
    ax.set_title(f"Histogram LBP per Blok ({n_blok}x{n_bins})", fontsize=10)
    ax.set_xlabel("Bin LBP")
    ax.set_ylabel("Nomor Blok")
    fel._style_dark_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    steps = [
        step_image(grid_img, f"1. Pembagian Blok ({block}px, stride={stride_eff})"),
        step_png(_png(fig), "2. Heatmap Histogram per Blok", caption="Tiap baris = histogram LBP 1 blok."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (flatten)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_lbp_multiscale(img_gray, img_bgr, configs=((8, 1), (16, 2), (24, 3)), method="uniform"):
    r = fel.lbp_multiscale(img_gray, configs=configs, method=method)
    imgs = [(local_binary_pattern(img_gray, P=P, R=R, method=method), f"P={P},R={R}") for P, R in configs]
    fig, axes = plt.subplots(1, len(imgs), figsize=(3.2 * len(imgs), 3.4))
    if len(imgs) == 1:
        axes = [axes]
    for ax, (m, t) in zip(axes, imgs):
        ax.imshow(m, cmap="nipy_spectral")
        ax.set_title(t, fontsize=9, color="#f0f1f5")
        ax.axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_png(_png(fig), "2. LBP Map per Skala (P,R)", caption="Radius berbeda menangkap skala tekstur berbeda."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (concat semua skala)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_ltp_features(img_gray, img_bgr, t=5, P=8, R=1):
    r = fel.ltp_features(img_gray, t=t, P=P, R=R)
    upper, lower = r["representasi_antara"]["upper_map"], r["representasi_antara"]["lower_map"]
    combo = step_images_row([(img_gray, "Citra Asli"), (upper, "Upper Pattern"), (lower, "Lower Pattern")],
                             suptitle="LTP: Threshold Ganda", caption=f"threshold t={t}")
    steps = [combo, step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (hist upper + lower)", max_bars=30)]
    info = {k: v for k, v in r["representasi_antara"].items() if k not in ("upper_map", "lower_map")}
    return finalize(steps, r["vector"], r["labels"], info)


def pipeline_clbp_features(img_gray, img_bgr, P=8, R=1):
    r = fel.clbp_features(img_gray, P=P, R=R)
    sign_map, mag_map = r["representasi_antara"]["sign_map"], r["representasi_antara"]["magnitude_map"]
    combo = step_images_row([(img_gray, "Citra Asli"), (sign_map, "Sign (S)"), (mag_map, "Magnitude (M)")],
                             suptitle="CLBP: Sign + Magnitude + Center")
    steps = [combo, step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (hist S + hist M + proporsi C)",
                                     max_bars=30)]
    info = {k: v for k, v in r["representasi_antara"].items() if k not in ("sign_map", "magnitude_map")}
    return finalize(steps, r["vector"], r["labels"], info)


def pipeline_cs_lbp_features(img_gray, img_bgr, R=1, t=0):
    r = fel.cs_lbp_features(img_gray, R=R, t=t)
    code_map = r["representasi_antara"]["code_map"]
    steps = [
        step_images_row([(img_gray, "Citra Asli"), (code_map, "Code Map (4 pasangan berlawanan)")],
                         suptitle="CS-LBP"),
        step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (16D)"),
    ]
    info = {k: v for k, v in r["representasi_antara"].items() if k != "code_map"}
    return finalize(steps, r["vector"], r["labels"], info)


def pipeline_mb_lbp_features(img_gray, img_bgr, block=3):
    r = fel.mb_lbp_features(img_gray, block=block)
    grid_img = _draw_grid_overlay(img_gray, block * 3, block * 3)
    steps = [
        step_image(grid_img, f"1. Pembagian Region 3x3-blok (blok={block}px)",
                   caption="Tiap region 3x3 blok dibandingkan mean-nya (bukan piksel tunggal)."),
        step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (histogram 256 kode)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


# =============================================================================
# 3.3 — GABOR
# =============================================================================

def pipeline_gabor_single(img_gray, img_bgr, theta_deg=0, lam=10.0, sigma=4.0, gamma=0.5):
    r = fel.gabor_single(img_gray, theta_deg=theta_deg, lam=lam, sigma=sigma, gamma=gamma)
    kernel = fel._gabor_kernel(theta_deg, lam, sigma, gamma)
    resp = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
    steps = [
        step_images_row([(img_gray, "Citra Asli"), (kernel, f"Kernel {theta_deg}°"), (resp, "Respons Filter")],
                         suptitle="Gabor: Kernel -> Respons"),
        step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (mean_abs, std, energy)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_gabor_bank(img_gray, img_bgr, thetas_deg=(0, 45, 90, 135), lambdas=(10.0,), stat="energy"):
    r = fel.gabor_bank(img_gray, thetas_deg=thetas_deg, lambdas=lambdas, stat=stat)
    grid_png = fel.visualize_gabor_grid(img_gray, thetas_deg=thetas_deg, lam=lambdas[0])
    heatmap = np.array(r["representasi_antara"]["energy_heatmap"])
    fig, ax = plt.subplots(figsize=(4.6, max(2.4, 0.6 * len(lambdas))))
    im = ax.imshow(heatmap, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(thetas_deg)))
    ax.set_xticklabels([f"{t}°" for t in thetas_deg])
    ax.set_yticks(range(len(lambdas)))
    ax.set_yticklabels([f"λ={l}" for l in lambdas])
    ax.set_title("Energi per Orientasi x Skala", fontsize=10)
    fel._style_dark_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    steps = [
        step_png(grid_png, "1. Kernel & Respons per Orientasi"),
        step_png(_png(fig), "2. Heatmap Energi (Filter Bank)", caption="Setiap kotak = 1 kombinasi orientasi-skala."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"],
                     {k: v for k, v in r["representasi_antara"].items() if k != "energy_heatmap"})


def pipeline_gabor_block(img_gray, img_bgr, thetas_deg=(0, 45, 90, 135), lam=10.0, block=32, stride=None):
    r = fel.gabor_block(img_gray, thetas_deg=thetas_deg, lam=lam, block=block, stride=stride)
    grid_png = fel.visualize_gabor_grid(img_gray, thetas_deg=thetas_deg, lam=lam)
    stride_eff = stride or block
    grid_overlay = _draw_grid_overlay(img_gray, block, stride_eff)
    steps = [
        step_png(grid_png, "1. Kernel & Respons per Orientasi"),
        step_image(grid_overlay, f"2. Pembagian Blok ({block}px, stride={stride_eff})"),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (per blok x filter)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_gabor_lbp_hybrid(img_gray, img_bgr, thetas_deg=(0, 45, 90, 135), lam=10.0):
    r = fel.gabor_lbp_hybrid(img_gray, thetas_deg=thetas_deg, lam=lam)
    grid_png = fel.visualize_gabor_grid(img_gray, thetas_deg=thetas_deg, lam=lam)
    steps = [
        step_png(grid_png, "1. Kernel & Respons Gabor"),
        step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (LBP histogram per respons Gabor)",
                         caption="Respons Gabor -> normalisasi -> LBP -> histogram.", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


# =============================================================================
# 3.4 — GRADIEN, TEPI & TRANSFORM DOMAIN
# =============================================================================

def pipeline_hog_feature(img_gray, img_bgr, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    r = fel.hog_feature(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                         cells_per_block=cells_per_block)
    hog_png = fel.visualize_hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                 cells_per_block=cells_per_block)
    steps = [
        step_png(hog_png, "1. Citra Asli -> Visualisasi HOG",
                 caption="Deretan histogram orientasi gradien lokal per sel."),
        step_vector_bar(r["vector"], r["labels"], "2. Vector Akhir (ratusan D)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_edge_based_feature(img_gray, img_bgr, canny_low=80, canny_high=160, n_bins=9, mag_thresh=20.0):
    r = fel.edge_based_feature(img_gray, canny_low=canny_low, canny_high=canny_high,
                                n_bins=n_bins, mag_thresh=mag_thresh)
    edges = cv2.Canny(img_gray, canny_low, canny_high)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_image(edges, "2. Edge Map (Canny)", caption="Representasi antara -- belum jadi vector."),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (density + histogram orientasi)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_projection_profile(img_gray, img_bgr, target_size=64, thresh=128):
    r = fel.projection_profile(img_gray, target_size=target_size, thresh=thresh)
    h_profile = np.array(r["representasi_antara"]["h_profile"])
    v_profile = np.array(r["representasi_antara"]["v_profile"])
    resized = cv2.resize(img_gray, (target_size, target_size))
    bw = (resized < thresh).astype(np.uint8) * 255
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    axes[0].plot(h_profile, range(len(h_profile)), color="#4dd0e1")
    axes[0].invert_yaxis()
    axes[0].set_title("Horizontal Projection", fontsize=9)
    fel._style_dark_axes(axes[0])
    axes[1].plot(v_profile, color="#f2ca50")
    axes[1].set_title("Vertical Projection", fontsize=9)
    fel._style_dark_axes(axes[1])
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    steps = [
        step_image(bw, "1. Citra Biner", caption=f"Diresize ke {target_size}x{target_size} dulu."),
        step_png(_png(fig), "2. Horizontal & Vertical Projection"),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (gabungan H+V)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], {"target_size": target_size, "thresh": thresh})


def pipeline_dct_feature(img_gray, img_bgr, target_size=64, keep=8):
    r = fel.dct_feature(img_gray, target_size=target_size, keep=keep)
    heatmap_block = np.array(r["representasi_antara"]["heatmap_block"])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap_block, cmap="magma")
    ax.set_title(f"Koefisien DCT {keep}x{keep} (frekuensi rendah)", fontsize=10)
    fel._style_dark_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    steps = [
        step_image(cv2.resize(img_gray, (target_size, target_size)), f"1. Citra {target_size}x{target_size}"),
        step_png(_png(fig), "2. Heatmap Koefisien DCT"),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (flatten)", max_bars=30),
    ]
    return finalize(steps, r["vector"], r["labels"], {"target_size": target_size, "keep": keep})


def pipeline_fft_radial_profile(img_gray, img_bgr, n_radius_bins=32):
    r = fel.fft_radial_profile(img_gray, n_radius_bins=n_radius_bins)
    combo_png = fel.visualize_fft_radial(img_gray, n_radius_bins=n_radius_bins)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_png(combo_png, "2. Spektrum FFT -> Radial Profile"),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (radial profile)", max_bars=32),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_wavelet_feature(img_gray, img_bgr, wavelet="haar", stat="energy"):
    r = fel.wavelet_feature(img_gray, wavelet=wavelet, stat=stat)
    combo_png = fel.visualize_wavelet_subbands(img_gray, wavelet=wavelet)
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_png(combo_png, "2. Subband Wavelet (LL/LH/HL/HH)"),
        step_vector_bar(r["vector"], r["labels"], "3. Vector Akhir (energi/statistik per subband)"),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


# =============================================================================
# 3.5 — DESCRIPTOR LOKAL (ORB / SIFT)
# =============================================================================

def pipeline_extract_keypoints(img_gray, img_bgr, method="orb", n_features=500):
    r = fel.extract_orb(img_gray, n_features=n_features) if method == "orb" else fel.extract_sift(img_gray)
    drawn = fel.draw_keypoints(img_gray, r["keypoints"])
    steps = [
        step_image(img_gray, "1. Citra Asli (Grayscale)"),
        step_image(drawn, f"2. Keypoint Terdeteksi ({method.upper()})",
                   caption=f"{r['n_keypoint']} keypoint, tiap titik = 1 descriptor "
                            f"{r['descriptor_size_bytes']} byte ({r['descriptor_type']})."),
    ]
    info = {"n_keypoint": r["n_keypoint"], "descriptor_size_bytes": r["descriptor_size_bytes"],
            "descriptor_type": r["descriptor_type"], "method": method}
    return finalize(steps, vector=None, labels=None, info=info)


def pipeline_match_keypoints(img1_gray, img2_gray, method="orb", n_features=500, ratio_thresh=0.75):
    """Revisi 3: img1 & img2 WAJIB dua-duanya dari user (app.py yang enforce
    tidak ada default sample di route ini)."""
    if method == "orb":
        r1 = fel.extract_orb(img1_gray, n_features=n_features)
        r2 = fel.extract_orb(img2_gray, n_features=n_features)
        m = fel.match_orb(r1["descriptors"], r2["descriptors"])
    else:
        r1, r2 = fel.extract_sift(img1_gray), fel.extract_sift(img2_gray)
        m = fel.match_sift(r1["descriptors"], r2["descriptors"], ratio_thresh=ratio_thresh)

    kp1_drawn = fel.draw_keypoints(img1_gray, r1["keypoints"])
    kp2_drawn = fel.draw_keypoints(img2_gray, r2["keypoints"])
    matched = fel.draw_matches(img1_gray, r1["keypoints"], img2_gray, r2["keypoints"], m["matches"])

    steps = [
        step_images_row([(img1_gray, "Gambar 1"), (img2_gray, "Gambar 2")], suptitle="1. Dua Citra Input (dari user)"),
        step_images_row([(kp1_drawn, f"{r1['n_keypoint']} keypoint"), (kp2_drawn, f"{r2['n_keypoint']} keypoint")],
                         suptitle=f"2. Keypoint Terdeteksi ({method.upper()})"),
        step_image(matched, f"3. Hasil Matching ({m['n_match']} pasangan)",
                   caption="Garis menghubungkan keypoint yang dianggap sama."),
    ]
    info = {"n_keypoint_1": r1["n_keypoint"], "n_keypoint_2": r2["n_keypoint"],
            "n_match": m["n_match"], "method": method,
            "best_distance": m.get("best_distance"), "distances_sample": m["distances"][:10]}
    return finalize(steps, vector=None, labels=None, info=info)


# =============================================================================
# 3.6 — ROBUST & DEEP EMBEDDING
# =============================================================================

def pipeline_resnet50_embedding(img_gray, img_bgr):
    r = fel.resnet50_embedding(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if r["status"] != "ok":
        return finalize([step_image(img_bgr, "Citra Asli (Warna)")],
                         info={"status": r["status"], "reason": r["reason"]})
    steps = [
        step_image(img_bgr, "1. Citra Asli (Warna)"),
        step_vector_bar(r["vector"], r["labels"], "2. Embedding ResNet50 (2048D)",
                         caption="Global Average Pooling -- tidak interpretable per-dimensi seperti HOG/LBP.",
                         max_bars=40),
    ]
    return finalize(steps, r["vector"], r["labels"], r["representasi_antara"])


def pipeline_robustness_stress_test(img, feature_key, feature_kwargs=None):
    feature_fn = fel.REGISTRY.get(feature_key)
    if feature_fn is None:
        raise ValueError(f"Fitur '{feature_key}' tidak ada di REGISTRY untuk stress-test")
    st = fel.run_stress_test(img, feature_fn, feature_kwargs or {})
    versions = fel.apply_transformations(img)
    gallery = step_images_row([(im, name) for name, im in versions.items()],
                               suptitle="1. Empat Transformasi Diterapkan")
    changes = {k: v["stability_vs_asli"]["relative_l2_change"]
               for k, v in st["per_transformasi"].items() if v["stability_vs_asli"]}
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(list(changes.keys()), list(changes.values()), color="#ff6b6b")
    ax.set_title("Perubahan Relatif Vector vs Asli", fontsize=10)
    ax.set_ylabel("Relative L2 Change")
    fel._style_dark_axes(ax)
    fig.tight_layout()
    steps = [gallery, step_png(_png(fig), "2. Skor Stabilitas per Transformasi", caption=st["kesimpulan"])]
    return finalize(steps, vector=None, labels=st["labels"],
                     info={"kesimpulan": st["kesimpulan"], "feature_key": feature_key})


# =============================================================================
# MATCHING ENGINE (lintas-fitur, generik) -- Revisi 3: dua citra WAJIB dari user
# =============================================================================

def pipeline_matching_engine(img1_gray, img1_bgr, img2_gray, img2_bgr, feature_key, feature_kind,
                              feature_kwargs=None):
    feature_kwargs = feature_kwargs or {}
    feature_fn = fel.REGISTRY.get(feature_key)
    if feature_fn is None:
        raise ValueError(f"Fitur '{feature_key}' tidak ada di REGISTRY")

    needs_color = feature_key in ("color_statistics_multispace", "color_histogram_per_channel", "glcm_color")
    img1 = img1_bgr if needs_color else img1_gray
    img2 = img2_bgr if needs_color else img2_gray

    r1 = feature_fn(img1, **feature_kwargs)
    r2 = feature_fn(img2, **feature_kwargs)
    v1, v2, labels = np.asarray(r1["vector"]), np.asarray(r2["vector"]), r1["labels"]

    result = fel.compute_all_relevant(v1, v2, feature_kind)

    fig, ax = plt.subplots(figsize=(max(6, min(len(v1), 30) * 0.25), 3.2))
    n_show = min(len(v1), 30)
    x = np.arange(n_show)
    w = 0.38
    ax.bar(x - w / 2, v1[:n_show], width=w, label="Gambar 1", color="#4f7cff")
    ax.bar(x + w / 2, v2[:n_show], width=w, label="Gambar 2", color="#f2ca50")
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:n_show], rotation=60, ha="right", fontsize=6.5)
    ax.legend(facecolor="#1a1d29", labelcolor="#f0f1f5", fontsize=8)
    ax.set_title("Vector Gambar 1 vs Gambar 2" + (f" (30 dari {len(v1)}D)" if len(v1) > 30 else ""), fontsize=10)
    fel._style_dark_axes(ax)
    fig.tight_layout()

    steps = [
        step_images_row([(img1_bgr, "Gambar 1 (dari user)"), (img2_bgr, "Gambar 2 (dari user)")],
                         suptitle="1. Dua Citra yang Dibandingkan"),
        step_png(_png(fig), "2. Vector Berdampingan"),
    ]
    info = {"feature_key": feature_key, "feature_kind": feature_kind,
            "metrics": result["metrics"], "primary_metric": result["primary_metric"],
            "verdict": result["verdict"], "dim": len(v1)}
    return finalize(steps, vector=None, labels=labels, info=info)


# =============================================================================
# FEATURE CONCATENATION PLAYGROUND
# =============================================================================

def pipeline_concat_playground(feature_results: dict, normalize: bool = False):
    cleaned = {k: {"vector": np.asarray(v["vector"]), "labels": v.get("labels", [])}
               for k, v in feature_results.items()}
    r = fel.concatenate_features(cleaned, normalize=normalize)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    names = [b["nama_fitur"] for b in r["breakdown"]]
    dims = [b["dimensi"] for b in r["breakdown"]]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(names)))
    ax.barh(names, dims, color=colors)
    for i, d in enumerate(dims):
        ax.text(d, i, f" {d}D", va="center", fontsize=8, color="#f0f1f5")
    ax.set_title("Breakdown Dimensi per Komponen", fontsize=10)
    fel._style_dark_axes(ax)
    fig.tight_layout()
    steps = [
        step_png(_png(fig), "1. Kontribusi Dimensi Tiap Fitur"),
        step_vector_bar(r["vector"], None, f"2. Vector Gabungan ({r['dim_total']}D)"
                         + (" -- dinormalisasi (L2)" if normalize else ""), max_bars=40),
    ]
    return finalize(steps, r["vector"], None, {"breakdown": r["breakdown"], "dim_total": r["dim_total"]})

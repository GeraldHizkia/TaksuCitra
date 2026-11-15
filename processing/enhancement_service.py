"""
processing/enhancement_service.py
=================================
Service layer untuk fitur Peningkatan & Analisis Citra TaksuCitra,
mengadaptasi logika dan fungsi dari processing/image_enhancement_toolkit.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from processing import image_enhancement_toolkit as iet


def compute_histogram_bins(img: np.ndarray, bins: int = 64) -> list[int]:
    """Hitung histogram luminans (grayscale) dalam jumlah bin tertentu untuk frontend."""
    gray = iet.to_gray(img)
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    return [int(x[0]) for x in hist]


def process_brightness_contrast(
    img_bgr: np.ndarray,
    mode: str = "linear",
    alpha: float = 1.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    target_mean: float = 0.5,
    p_low: float = 1.0,
    p_high: float = 99.0,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> dict:
    """
    Penyesuaian Kecerahan & Kontras:
    mode: 'linear', 'gamma', 'auto_gamma', 'contrast_stretch', 'hist_equalize', 'clahe'.
    """
    stats_before = iet.quality_stats(img_bgr)
    hist_before = compute_histogram_bins(img_bgr, bins=64)
    extra_info = {}

    if mode == "linear":
        out = iet.linear_adjust(img_bgr, alpha=float(alpha), beta=float(beta))
        extra_info["params"] = f"alpha={alpha}, beta={beta}"

    elif mode == "gamma":
        g_val = max(0.05, min(10.0, float(gamma)))
        out = iet.gamma_correct(img_bgr, gamma=g_val)
        extra_info["params"] = f"gamma={g_val:.2f}"

    elif mode == "auto_gamma":
        tgt = max(0.1, min(0.9, float(target_mean)))
        out, selected_gamma = iet.auto_gamma(img_bgr, target=tgt)
        extra_info["selected_gamma"] = round(selected_gamma, 3)
        extra_info["params"] = f"target={tgt}, gamma terpilih={selected_gamma:.2f}"

    elif mode == "contrast_stretch":
        plo = max(0.0, min(40.0, float(p_low)))
        phi = max(60.0, min(100.0, float(p_high)))
        out = iet.contrast_stretch(img_bgr, p_low=plo, p_high=phi)
        extra_info["params"] = f"p_low={plo}%, p_high={phi}%"

    elif mode == "hist_equalize":
        out = iet.hist_equalize(img_bgr)
        extra_info["params"] = "Equalisasi Global (Kanal Luminans Y)"

    elif mode == "clahe":
        cl = max(0.5, min(20.0, float(clip_limit)))
        ts = max(2, min(32, int(tile_size)))
        out = iet.clahe(img_bgr, clip=cl, tiles=ts)
        extra_info["params"] = f"clip={cl}, tile={ts}x{ts}"

    else:
        raise ValueError(f"Mode brightness/contrast '{mode}' tidak valid")

    stats_after = iet.quality_stats(out)
    hist_after = compute_histogram_bins(out, bins=64)

    return {
        "original": img_bgr,
        "result": out,
        "mode": mode,
        "extra_info": extra_info,
        "stats_before": {
            "brightness": round(stats_before["brightness"], 1),
            "contrast": round(stats_before["contrast"], 1),
            "entropy": round(stats_before["entropy"], 2),
        },
        "stats_after": {
            "brightness": round(stats_after["brightness"], 1),
            "contrast": round(stats_after["contrast"], 1),
            "entropy": round(stats_after["entropy"], 2),
        },
        "hist_before": hist_before,
        "hist_after": hist_after,
    }


def process_flip(img_bgr: np.ndarray, mode: str = "h") -> dict:
    """Pencerminan (flip) citra: 'h', 'v', atau 'hv'."""
    mode = str(mode).lower()
    if mode not in ("h", "v", "hv"):
        mode = "h"
    out = iet.flip(img_bgr, mode=mode)
    labels = {
        "h": "Horizontal (Kiri-Kanan)",
        "v": "Vertikal (Atas-Bawah)",
        "hv": "Horizontal + Vertikal (Rotasi 180°)",
    }
    return {
        "original": img_bgr,
        "result": out,
        "mode": mode,
        "mode_label": labels.get(mode, mode),
        "dimensions": f"{out.shape[1]} x {out.shape[0]}",
    }


def process_rotation(
    img_bgr: np.ndarray,
    mode: str = "free",
    angle: float = 45.0,
    times: int = 1,
    expand: bool = True,
    fill_color: int = 0,
    use_manual: bool = False,
) -> dict:
    """
    Rotasi citra:
    mode: 'rot90' (kelipatan 90° lossless) atau 'free' (sudut bebas).
    """
    diff_vs_cv = None
    if mode == "rot90":
        times = int(times) % 4
        out = iet.rotate90(img_bgr, times=times)
        actual_angle = times * 90
        is_lossless = True
    else:  # 'free'
        angle = float(angle)
        actual_angle = angle
        is_lossless = False
        fill_val = int(fill_color)

        if use_manual:
            out = iet.rotate_manual(img_bgr, angle=angle, expand=expand, fill=fill_val)
            # Evaluasi perbedaan vs OpenCV untuk verifikasi matematis
            ref_cv = iet.rotate_cv(img_bgr, angle=angle, expand=expand)
            if out.shape == ref_cv.shape:
                diff_vs_cv = float(np.abs(out.astype(int) - ref_cv.astype(int)).mean())
        else:
            out = iet.rotate_cv(img_bgr, angle=angle, expand=expand)

    return {
        "original": img_bgr,
        "result": out,
        "mode": mode,
        "angle": actual_angle,
        "expand": expand,
        "is_lossless": is_lossless,
        "orig_dims": f"{img_bgr.shape[1]} x {img_bgr.shape[0]}",
        "result_dims": f"{out.shape[1]} x {out.shape[0]}",
        "diff_vs_cv": round(diff_vs_cv, 3) if diff_vs_cv is not None else None,
    }


def process_edge_detection(
    img_bgr: np.ndarray,
    method: str = "canny_auto",
    sigma: float = 1.4,
    low_thresh: float | None = None,
    high_thresh: float | None = None,
    auto_sigma: float = 0.33,
) -> dict:
    """
    Deteksi Tepi:
    method: 'sobel', 'prewitt', 'scharr', 'log', 'canny_manual', 'canny_auto'.
    """
    gray = iet.to_gray(img_bgr)
    agreement_f1 = None

    if method in ("sobel", "prewitt", "scharr"):
        out = iet.edge_operator(gray, op=method)
    elif method == "log":
        out = iet.laplacian_of_gaussian(gray, sigma=float(sigma))
    elif method == "canny_manual":
        low_t = float(low_thresh) if low_thresh is not None and str(low_thresh).strip() != "" else None
        high_t = float(high_thresh) if high_thresh is not None and str(high_thresh).strip() != "" else None
        out = iet.canny_manual(gray, sigma=float(sigma), low=low_t, high=high_t)
        # Bandingkan F1 score terhadap OpenCV auto-canny
        ref_auto = iet.canny_auto(gray, sigma=float(auto_sigma))
        agreement_f1 = iet.edge_agreement(out, ref_auto, tol=2)
    elif method == "canny_auto":
        out = iet.canny_auto(gray, sigma=float(auto_sigma))
    else:
        raise ValueError(f"Metode deteksi tepi '{method}' tidak valid")

    edge_pixels = int((out > 0).sum())
    total_pixels = out.size
    edge_density = round((edge_pixels / total_pixels) * 100, 2)

    return {
        "original": img_bgr,
        "result": out,
        "method": method,
        "edge_pixels": edge_pixels,
        "edge_density_pct": edge_density,
        "agreement_f1": round(agreement_f1, 2) if agreement_f1 is not None else None,
    }


def process_noise_reduction(
    img_bgr: np.ndarray,
    method: str = "median",
    kernel_size: int = 5,
    sigma: float = 1.5,
    h_param: float | None = None,
    add_test_noise: str = "none",
    noise_amount: float | None = None,
    noise_seed: int = 42,
) -> dict:
    """
    Pereduksian Derau (Noise Reduction) & Generator Noise Pengujian.
    filter: 'mean', 'gaussian', 'median', 'adaptive_median', 'bilateral', 'nlm'.
    """
    rng = np.random.default_rng(int(noise_seed))
    has_test_noise = add_test_noise not in ("none", "", None)

    # 1. Terapkan noise sintetis jika diminta untuk pengujian
    if has_test_noise:
        noisy_img = iet.add_noise(img_bgr, add_test_noise, rng, amount=noise_amount)
    else:
        noisy_img = img_bgr.copy()

    # 2. Estimasi sigma noise
    est_sigma = iet.estimate_noise_sigma(noisy_img)

    # 3. Konfigurasi parameter per filter
    kw = {}
    k_val = max(3, int(kernel_size))
    if k_val % 2 == 0:
        k_val += 1

    if method == "mean":
        kw["k"] = k_val
    elif method == "gaussian":
        kw["sigma"] = float(sigma) if sigma else 1.5
    elif method == "median":
        kw["k"] = k_val
    elif method == "adaptive_median":
        kw["max_size"] = k_val
    elif method == "bilateral":
        kw["d"] = k_val
        kw["sc"] = float(h_param) if h_param else float(np.clip(1.5 * est_sigma, 25, 100))
        kw["ss"] = k_val
    elif method == "nlm":
        kw["h"] = float(h_param) if h_param else float(np.clip(0.9 * est_sigma, 3, 40))

    # 4. Jalankan denoise
    denoised = iet.denoise(noisy_img, method=method, **kw)

    # 5. Hitung metrik PSNR dan SSIM jika ada citra referensi asli
    metrics = None
    if has_test_noise:
        psnr_noisy, ssim_noisy = iet.evaluate(img_bgr, noisy_img)
        psnr_clean, ssim_clean = iet.evaluate(img_bgr, denoised)
        metrics = {
            "psnr_noisy": round(float(psnr_noisy), 2),
            "ssim_noisy": round(float(ssim_noisy), 3),
            "psnr_denoised": round(float(psnr_clean), 2),
            "ssim_denoised": round(float(ssim_clean), 3),
            "psnr_improvement": round(float(psnr_clean - psnr_noisy), 2),
        }

    return {
        "original": img_bgr,
        "noisy": noisy_img if has_test_noise else None,
        "result": denoised,
        "method": method,
        "test_noise": add_test_noise,
        "estimated_noise_sigma": round(est_sigma, 2),
        "metrics": metrics,
    }


def process_enhancement_pipeline(
    img_bgr: np.ndarray,
    denoise_method: str = "nlm",
    denoise_strength: float = 10.0,
    clahe_clip: float = 2.5,
    edge_auto_sigma: float = 0.33,
) -> dict:
    """
    Pipeline multi-tahap: Input -> Denoise -> CLAHE -> Deteksi Tepi (Canny).
    Juga menyediakan pembanding (CLAHE tanpa Denoise) untuk melihat efek noise.
    """
    gray = iet.to_gray(img_bgr)

    # Tahap 1: Denoise
    kw = {"h": float(denoise_strength)} if denoise_method == "nlm" else {"k": 5}
    step1_denoised = iet.denoise(img_bgr, denoise_method, **kw)

    # Tahap 2: CLAHE
    step2_clahe = iet.clahe(step1_denoised, clip=float(clahe_clip))

    # Tahap 3: Deteksi Tepi
    step3_edges = iet.canny_auto(iet.to_gray(step2_clahe), sigma=float(edge_auto_sigma))

    # Pembanding tanpa denoise (noise ikut teramplifikasi)
    naive_clahe = iet.clahe(img_bgr, clip=float(clahe_clip))
    naive_edges = iet.canny_auto(iet.to_gray(naive_clahe), sigma=float(edge_auto_sigma))

    return {
        "original": img_bgr,
        "step1_denoised": step1_denoised,
        "step2_clahe": step2_clahe,
        "step3_edges": step3_edges,
        "naive_edges": naive_edges,
        "edge_count_clean": int((step3_edges > 0).sum()),
        "edge_count_naive": int((naive_edges > 0).sum()),
    }

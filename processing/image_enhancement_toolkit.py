#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_enhancement_toolkit.py
============================
Toolkit Computer Vision (non-morfologi):

  1. Brightness & Contrast : linear (g = a*f + b), gamma, contrast stretching,
                             histogram equalization, CLAHE (+ metrik kualitas)
  2. Pencerminan & Rotasi  : flip H/V, rot90 lossless, rotasi sudut bebas
                             (implementasi manual + interpolasi bilinear)
  3. Deteksi Tepi          : Sobel / Prewitt / Scharr / LoG, Canny manual
                             (Gaussian -> gradien -> NMS -> hysteresis) + auto-Canny
  4. Noise Reduction       : mean, Gaussian, median, adaptive median, bilateral,
                             Non-Local Means (+ PSNR / SSIM)
  5. Pipeline lengkap      : denoise -> CLAHE -> edge, dan augmentasi geometri

Jalankan:
    python image_enhancement_toolkit.py                      # demo bawaan
    python image_enhancement_toolkit.py --input foto.jpg
    python image_enhancement_toolkit.py --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# 0. UTILITAS
# ---------------------------------------------------------------------------
def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def load_image(path: str | None, color: bool = True) -> np.ndarray:
    """Baca citra (BGR). Tanpa path -> foto bawaan scikit-image."""
    if path:
        img = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")
        return img
    if color:
        return cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    return skdata.camera()


def show_rgb(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im


def entropy(gray: np.ndarray) -> float:
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def quality_stats(img: np.ndarray) -> dict:
    """Brightness = rerata luminans, Contrast = simpangan baku (RMS contrast)."""
    g = to_gray(img)
    return {"brightness": float(g.mean()), "contrast": float(g.std()),
            "entropy": entropy(g)}


def grid(items, path: Path, cols: int, title: str, hists: bool = False):
    """items: [(judul, citra)]. hists=True -> baris kedua berisi histogram."""
    n = len(items)
    rows = int(np.ceil(n / cols)) * (2 if hists else 1)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.1 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (name, im) in enumerate(items):
        r, c = divmod(i, cols)
        ax = axes[r * (2 if hists else 1), c]
        ax.imshow(show_rgb(im), cmap="gray", vmin=0, vmax=255)
        ax.set_title(name, fontsize=8.5)
        if hists:
            hx = axes[r * 2 + 1, c]
            hx.axis("on")
            hx.hist(to_gray(im).ravel(), bins=64, range=(0, 255), color="0.35")
            hx.set_yticks([])
            hx.set_xlim(0, 255)
            hx.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)
    print(f"  tersimpan: {path}")


# ---------------------------------------------------------------------------
# 1. BRIGHTNESS & CONTRAST
# ---------------------------------------------------------------------------
def linear_adjust(img: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """g(x,y) = alpha * f(x,y) + beta.  alpha -> kontras, beta -> kecerahan.
    (Tidak memakai convertScaleAbs: fungsi itu mengambil nilai mutlak.)"""
    out = alpha * img.astype(np.float32) + beta
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    """g = 255 * (f/255)^gamma.  gamma<1 mencerahkan, gamma>1 menggelapkan.
    Non-linear: memperbaiki area gelap tanpa 'membakar' area terang."""
    lut = (np.arange(256) / 255.0) ** gamma * 255.0
    return cv2.LUT(img, np.clip(lut, 0, 255).astype(np.uint8))


def auto_gamma(img: np.ndarray, target: float = 0.5) -> tuple[np.ndarray, float]:
    """Pilih gamma agar rerata luminans -> target (0..1)."""
    m = np.clip(to_gray(img).mean() / 255.0, 1e-3, 0.999)
    gamma = float(np.clip(np.log(target) / np.log(m), 0.3, 3.0))
    return gamma_correct(img, gamma), gamma


def contrast_stretch(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Rentangkan [percentile_low, percentile_high] ke [0,255]; tahan outlier."""
    lo, hi = np.percentile(to_gray(img), [p_low, p_high])
    if hi - lo < 1:
        return img.copy()
    out = (img.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def hist_equalize(img: np.ndarray) -> np.ndarray:
    """Equalisasi histogram global (pada kanal luminans Y untuk citra warna)."""
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE: equalisasi adaptif per-tile dengan batas kontras (clip limit)
    sehingga noise tidak ikut diperkuat berlebihan. Warna: kanal L pada LAB."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    if img.ndim == 2:
        return op.apply(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# 2. PENCERMINAN & ROTASI
# ---------------------------------------------------------------------------
def flip(img: np.ndarray, mode: str = "h") -> np.ndarray:
    """h: F(x,y) = I(x, w-1-y) ; v: F(x,y) = I(h-1-x, y) ; hv: keduanya.
    Murni pengindeksan -> lossless."""
    if mode == "h":
        return img[:, ::-1].copy()
    if mode == "v":
        return img[::-1].copy()
    if mode == "hv":
        return img[::-1, ::-1].copy()
    raise ValueError("mode harus 'h', 'v', atau 'hv'")


def rotate90(img: np.ndarray, times: int = 1) -> np.ndarray:
    """Rotasi kelipatan 90 derajat (berlawanan jarum jam), lossless."""
    return np.ascontiguousarray(np.rot90(img, times))


def rotate_manual(img: np.ndarray, angle: float, expand: bool = False,
                  fill: int = 0) -> np.ndarray:
    """Rotasi sudut bebas dengan *inverse mapping* + interpolasi bilinear.
    angle > 0 = berlawanan arah jarum jam (sama dengan OpenCV).
    Setiap piksel output dicari asalnya di citra sumber:
        x_src =  cos*x' - sin*y' + cx ;  y_src = sin*x' + cos*y' + cy
    expand=True -> kanvas diperbesar agar tidak ada bagian yang terpotong."""
    src = img if img.ndim == 3 else img[..., None]
    h, w, _ = src.shape
    t = np.deg2rad(angle)
    c, s = np.cos(t), np.sin(t)
    if expand:
        nw = int(np.ceil(abs(w * c) + abs(h * s)))
        nh = int(np.ceil(abs(w * s) + abs(h * c)))
    else:
        nw, nh = w, h
    cx, cy = (w - 1) / 2, (h - 1) / 2
    ncx, ncy = (nw - 1) / 2, (nh - 1) / 2
    ys, xs = np.mgrid[0:nh, 0:nw].astype(np.float32)
    xd, yd = xs - ncx, ys - ncy
    xsrc = c * xd - s * yd + cx
    ysrc = s * xd + c * yd + cy

    x0, y0 = np.floor(xsrc).astype(np.int32), np.floor(ysrc).astype(np.int32)
    fx, fy = (xsrc - x0)[..., None], (ysrc - y0)[..., None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    f = src.astype(np.float32)
    out = (f[y0c, x0c] * (1 - fx) * (1 - fy) + f[y0c, x1c] * fx * (1 - fy)
           + f[y1c, x0c] * (1 - fx) * fy + f[y1c, x1c] * fx * fy)
    valid = (xsrc >= 0) & (xsrc <= w - 1) & (ysrc >= 0) & (ysrc <= h - 1)
    out[~valid] = fill
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out[..., 0] if img.ndim == 2 else out


def rotate_cv(img: np.ndarray, angle: float, expand: bool = False,
              interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Versi OpenCV (warpAffine), dipakai sebagai pembanding & untuk kecepatan."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    nw, nh = w, h
    if expand:
        c, s = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(np.ceil(h * s + w * c)), int(np.ceil(h * c + w * s))
        m[0, 2] += (nw - 1) / 2 - cx
        m[1, 2] += (nh - 1) / 2 - cy
    return cv2.warpAffine(img, m, (nw, nh), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def random_augment(img: np.ndarray, rng: np.random.Generator,
                   max_angle: float = 15.0, p_flip: float = 0.5) -> np.ndarray:
    """Augmentasi geometri untuk data latih deep learning."""
    out = img
    if rng.random() < p_flip:
        out = flip(out, "h")
    return rotate_cv(out, float(rng.uniform(-max_angle, max_angle)))


# ---------------------------------------------------------------------------
# 3. DETEKSI TEPI
# ---------------------------------------------------------------------------
KERNELS = {
    "sobel": (np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),
              np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)),
    "prewitt": (np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32),
                np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)),
    "scharr": (np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], np.float32),
               np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], np.float32)),
}


def gradient(gray: np.ndarray, op: str = "sobel"):
    """Turunan pertama lewat konvolusi: magnitudo |G| = sqrt(Gx^2 + Gy^2),
    arah theta = atan2(Gy, Gx)."""
    kx, ky = KERNELS[op]
    g = gray.astype(np.float32)
    gx = cv2.filter2D(g, cv2.CV_32F, kx, borderType=cv2.BORDER_REFLECT_101)
    gy = cv2.filter2D(g, cv2.CV_32F, ky, borderType=cv2.BORDER_REFLECT_101)
    return np.hypot(gx, gy), np.arctan2(gy, gx), gx, gy


def edge_operator(gray: np.ndarray, op: str = "sobel") -> np.ndarray:
    mag = gradient(gray, op)[0]
    return np.clip(mag * 255.0 / max(mag.max(), 1e-6), 0, 255).astype(np.uint8)


def laplacian_of_gaussian(gray: np.ndarray, sigma: float = 1.6) -> np.ndarray:
    """LoG (turunan kedua): tepi = zero-crossing. Dikembalikan sebagai peta
    respons |LoG| (lebih terang = tepi lebih kuat)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3))
    return np.clip(lap * 255.0 / max(lap.max(), 1e-6), 0, 255).astype(np.uint8)


def non_max_suppression(mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Pertahankan hanya maksimum lokal searah gradien -> tepi setipis 1 piksel.
    Arah dikuantisasi ke 0/45/90/135 derajat (koordinat citra, y ke bawah)."""
    h, w = mag.shape
    q = np.round((np.rad2deg(theta) % 180) / 45).astype(np.int32) % 4
    p = np.pad(mag, 1)

    def sh(dy, dx):
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    n1 = np.select([q == 0, q == 1, q == 2, q == 3],
                   [sh(0, 1), sh(1, 1), sh(1, 0), sh(1, -1)])
    n2 = np.select([q == 0, q == 1, q == 2, q == 3],
                   [sh(0, -1), sh(-1, -1), sh(-1, 0), sh(-1, 1)])
    return (mag >= n1) & (mag >= n2)


def hysteresis(mag: np.ndarray, cand: np.ndarray, low: float, high: float) -> np.ndarray:
    """Tepi lemah (>= low) dipertahankan HANYA jika terhubung (8-konektivitas)
    ke tepi kuat (>= high)."""
    strong = cand & (mag >= high)
    weak = cand & (mag >= low)
    _, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def canny_manual(gray: np.ndarray, sigma: float = 1.4, low: float | None = None,
                 high: float | None = None, high_pct: float = 92.0) -> np.ndarray:
    """Canny dari nol: Gaussian -> Sobel -> NMS -> double threshold -> hysteresis.
    Jika low/high kosong: high = persentil magnitudo (hanya piksel hasil NMS)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    mag, theta, _, _ = gradient(g, "sobel")
    nms = non_max_suppression(mag, theta)
    if high is None:
        high = float(np.percentile(mag[nms], high_pct))
    if low is None:
        low = 0.4 * high
    return (hysteresis(mag, nms, low, high).astype(np.uint8)) * 255


def canny_auto(gray: np.ndarray, sigma: float = 0.33, blur: int = 5) -> np.ndarray:
    """Auto-Canny OpenCV: ambang dari median intensitas (tanpa tuning manual)."""
    g = cv2.GaussianBlur(gray, (blur, blur), 0)
    v = float(np.median(g))
    return cv2.Canny(g, int(max(0, (1 - sigma) * v)), int(min(255, (1 + sigma) * v)))


def edge_agreement(a: np.ndarray, b: np.ndarray, tol: int = 1) -> float:
    """F1 antara dua peta tepi dengan toleransi 'tol' piksel."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    a, b = a > 0, b > 0
    if not a.any() or not b.any():
        return 0.0
    prec = (a & (cv2.dilate(b.astype(np.uint8), k) > 0)).sum() / a.sum()
    rec = (b & (cv2.dilate(a.astype(np.uint8), k) > 0)).sum() / b.sum()
    return float(2 * prec * rec / max(prec + rec, 1e-9))


# ---------------------------------------------------------------------------
# 4. NOISE & NOISE REDUCTION
# ---------------------------------------------------------------------------
def add_noise(img: np.ndarray, kind: str, rng: np.random.Generator,
              amount: float | None = None) -> np.ndarray:
    f = img.astype(np.float32)
    if kind == "gaussian":
        out = f + rng.normal(0, amount or 25.0, f.shape)
    elif kind == "salt_pepper":
        p = amount or 0.05
        out = img.copy()
        r = rng.random(img.shape[:2])
        out[r < p / 2] = 0
        out[r > 1 - p / 2] = 255
        return out
    elif kind == "speckle":
        out = f + f * rng.normal(0, amount or 0.25, f.shape)
    elif kind == "poisson":
        peak = amount or 30.0
        out = rng.poisson(f / 255.0 * peak) / peak * 255.0
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_median(img: np.ndarray, max_size: int = 7) -> np.ndarray:
    """Adaptive median filter (vektorisasi): jendela membesar 3,5,7 sampai median
    bukan impuls. Piksel yang BUKAN impuls dipertahankan -> detail lebih awet
    daripada median biasa. Cocok untuk salt-and-pepper kepadatan tinggi."""
    if img.ndim == 3:
        return np.dstack([adaptive_median(img[..., c], max_size) for c in range(3)])
    cur = img.astype(np.int16)
    out = cur.copy()
    done = np.zeros(img.shape, bool)
    for s in range(3, max_size + 1, 2):
        zmed = cv2.medianBlur(img, s).astype(np.int16)
        se = np.ones((s, s), np.uint8)
        zmin = cv2.erode(img, se).astype(np.int16)
        zmax = cv2.dilate(img, se).astype(np.int16)
        level_b = ~done & (zmed > zmin) & (zmed < zmax)
        keep = level_b & (cur > zmin) & (cur < zmax)
        out[keep] = cur[keep]
        repl = level_b & ~keep
        out[repl] = zmed[repl]
        done |= level_b
        if s == max_size:
            rest = ~done
            out[rest] = zmed[rest]
    return out.astype(np.uint8)


def estimate_noise_sigma(img: np.ndarray) -> float:
    """Estimasi sigma noise Gaussian (metode Immerkaer): respons Laplacian
    'beda-beda' menghilangkan struktur citra, sisanya ~ noise."""
    g = to_gray(img).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6.0)


def denoise(img: np.ndarray, method: str, **kw) -> np.ndarray:
    """h (NLM) dan sigmaColor (bilateral) diatur otomatis dari estimasi sigma
    noise bila tidak diberikan -> tidak perlu tuning manual per citra."""
    if method == "mean":
        k = kw.get("k", 5)
        return cv2.blur(img, (k, k))
    if method == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), kw.get("sigma", 1.5))
    if method == "median":
        return cv2.medianBlur(img, kw.get("k", 5))
    if method == "adaptive_median":
        return adaptive_median(img, kw.get("max_size", 7))
    if method == "bilateral":
        sc = kw.get("sc") or float(np.clip(1.5 * estimate_noise_sigma(img), 25, 100))
        return cv2.bilateralFilter(img, kw.get("d", 7), sc, kw.get("ss", 7))
    if method == "nlm":
        h = kw.get("h") or float(np.clip(0.9 * estimate_noise_sigma(img), 3, 40))
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    raise ValueError(method)


FILTERS = ["mean", "gaussian", "median", "adaptive_median", "bilateral", "nlm"]


def evaluate(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    ch = -1 if ref.ndim == 3 else None
    return (psnr(ref, test, data_range=255),
            ssim(ref, test, data_range=255, channel_axis=ch))


# ---------------------------------------------------------------------------
# 5. DEMO PER FITUR
# ---------------------------------------------------------------------------
def demo_brightness_contrast(img, out: Path):
    dark = linear_adjust(img, 0.35, 25)                    # simulasi foto gelap
    gam, g = auto_gamma(dark)
    methods = [
        ("Input gelap & kontras rendah", dark),
        ("Linear a=2.2, b=10", linear_adjust(dark, 2.2, 10)),
        (f"Gamma otomatis (g={g:.2f})", gam),
        ("Contrast stretching 1-99%", contrast_stretch(dark)),
        ("Histogram equalization", hist_equalize(dark)),
        ("CLAHE (clip 2.0, 8x8)", clahe(dark)),
    ]
    items = []
    print(f"\n  {'metode':<32}{'brightness':>11}{'contrast':>10}{'entropy':>9}")
    for name, im in methods:
        s = quality_stats(im)
        print(f"  {name:<32}{s['brightness']:>11.1f}{s['contrast']:>10.1f}{s['entropy']:>9.2f}")
        items.append((f"{name}\nB={s['brightness']:.0f}  C={s['contrast']:.0f}  H={s['entropy']:.2f}", im))
    grid(items, out / "1_brightness_contrast.png", 6,
         "Brightness & Contrast (B=rerata, C=simpangan baku, H=entropi)", hists=True)
    return dark


def demo_geometry(img, out: Path):
    rng = np.random.default_rng(3)
    man = rotate_manual(img, 30, expand=True)
    cvr = rotate_cv(img, 30, expand=True)
    diff = float(np.abs(man.astype(int) - cvr.astype(int)).mean())
    items = [
        ("Asli", img), ("Flip horizontal", flip(img, "h")),
        ("Flip vertikal", flip(img, "v")), ("Flip H+V (= rot 180)", flip(img, "hv")),
        ("Rot 90 (lossless)", rotate90(img)),
        ("Rotasi 30 (kanvas tetap, sudut terpotong)", rotate_manual(img, 30)),
        (f"Rotasi 30 expand (manual)\nselisih vs OpenCV = {diff:.2f}", man),
        ("Augmentasi acak (flip + rot ±15)", random_augment(img, rng)),
    ]
    grid(items, out / "2_flip_rotasi.png", 4, "Pencerminan & Rotasi")
    return diff


def demo_edges(gray, out: Path):
    rng = np.random.default_rng(5)
    noisy = add_noise(gray, "gaussian", rng, 20)
    cm, cc = canny_manual(gray), canny_auto(gray)
    f1 = edge_agreement(cm, cc, tol=2)
    items = [
        ("Citra grayscale", gray),
        ("Sobel", edge_operator(gray, "sobel")),
        ("Prewitt", edge_operator(gray, "prewitt")),
        ("Scharr", edge_operator(gray, "scharr")),
        ("Laplacian of Gaussian", laplacian_of_gaussian(gray)),
        ("Canny manual (NMS + hysteresis)", cm),
        (f"Auto-Canny OpenCV (F1 vs manual={f1:.2f})", cc),
        ("Sobel pada citra bernoise σ=20", edge_operator(noisy, "sobel")),
        ("Canny manual pada citra bernoise", canny_manual(noisy, sigma=2.0)),
    ]
    grid(items, out / "3_deteksi_tepi.png", 3, "Deteksi tepi")
    return f1


def demo_noise(gray, out: Path):
    rng = np.random.default_rng(11)
    kinds = [("gaussian", "Gaussian σ=25"), ("salt_pepper", "Salt&Pepper 5%"),
             ("speckle", "Speckle 0.25")]
    crop = (slice(40, 240), slice(150, 350))
    items, table = [], {}
    for kind, label in kinds:
        noisy = add_noise(gray, kind, rng)
        p0, s0 = evaluate(gray, noisy)
        items.append((f"{label}\nPSNR {p0:.1f} SSIM {s0:.2f}", noisy[crop]))
        table[label] = {"noisy": (p0, s0)}
        for m in FILTERS:
            res = denoise(noisy, m)
            p, s = evaluate(gray, res)
            table[label][m] = (p, s)
            items.append((f"{m}\nPSNR {p:.1f} SSIM {s:.2f}", res[crop]))
    grid(items, out / "4_noise_reduction.png", len(FILTERS) + 1,
         "Noise reduction: 3 jenis noise x 6 filter")
    return table


def pipeline(img, out: Path):
    """Foto gelap + bernoise -> denoise -> CLAHE -> tepi (Canny)."""
    rng = np.random.default_rng(21)
    low = add_noise(linear_adjust(img, 0.45, 15), "gaussian", rng, 12)
    den = denoise(low, "nlm", h=10)
    enh = clahe(den, clip=2.5)
    edges = canny_auto(to_gray(enh))
    naive = canny_auto(to_gray(clahe(low, clip=2.5)))   # tanpa denoise
    grid([
        ("1. Input gelap + noise", low),
        ("2. Denoise (NLM)", den),
        ("3. CLAHE", enh),
        ("4. Tepi (Canny)", edges),
        ("Pembanding: CLAHE tanpa denoise -> tepi kotor", naive),
    ], out / "5_pipeline.png", 5, "Pipeline: denoise -> kontras -> tepi")
    return int((edges > 0).sum()), int((naive > 0).sum())


# ---------------------------------------------------------------------------
# 6. SELF-TEST
# ---------------------------------------------------------------------------
def selftest():
    gray = load_image(None, color=False)
    color = load_image(None, color=True)
    # flip == OpenCV
    for mode, code in (("h", 1), ("v", 0), ("hv", -1)):
        assert np.array_equal(flip(color, mode), cv2.flip(color, code)), mode
    print("  [OK] flip == cv2.flip (h, v, hv)")
    # rotasi 90 == cv2
    assert np.array_equal(rotate90(color), cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE))
    print("  [OK] rot90 == cv2.rotate")
    # rotasi manual ~ OpenCV (area dalam)
    for ang, exp in ((30, False), (-17, True), (90, False)):
        a, b = rotate_manual(color, ang, exp), rotate_cv(color, ang, exp)
        assert a.shape == b.shape, (a.shape, b.shape)
        m = slice(20, -20)
        d = np.abs(a[m, m].astype(int) - b[m, m].astype(int)).mean()
        assert d < 1.0, d
        print(f"  [OK] rotasi manual vs OpenCV sudut {ang:>4}, expand={exp}: selisih {d:.3f}")
    # Sobel manual == cv2.Sobel
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    assert np.allclose(gradient(gray, "sobel")[2], gx, atol=1e-3)
    print("  [OK] gradien Sobel manual == cv2.Sobel")
    # Canny manual cukup mirip OpenCV
    f1 = edge_agreement(canny_manual(gray), canny_auto(gray), tol=2)
    assert f1 > 0.5, f1
    print(f"  [OK] Canny manual vs auto-Canny OpenCV, F1 = {f1:.2f}")
    # adaptive median memperbaiki S&P
    rng = np.random.default_rng(0)
    noisy = add_noise(gray, "salt_pepper", rng, 0.2)
    assert evaluate(gray, adaptive_median(noisy))[0] > evaluate(gray, noisy)[0] + 8
    print("  [OK] adaptive median menaikkan PSNR pada S&P 20%")
    # gamma monoton & CLAHE menaikkan kontras citra gelap
    dark = linear_adjust(gray, 0.35, 25)
    assert quality_stats(clahe(dark))["contrast"] > quality_stats(dark)["contrast"]
    print("  [OK] CLAHE menaikkan kontras citra gelap")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Toolkit enhancement & edge & noise")
    ap.add_argument("--input", help="path citra (default: foto bawaan scikit-image)")
    ap.add_argument("--output", default="hasil_enhancement")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("Self-test:")
        selftest()
        return

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    img = load_image(args.input, color=True)
    gray = to_gray(img) if args.input else load_image(None, color=False)

    print("1) Brightness & contrast")
    demo_brightness_contrast(img, out)
    print("2) Pencerminan & rotasi")
    diff = demo_geometry(img, out)
    print(f"   selisih rotasi manual vs OpenCV: {diff:.3f}")
    print("3) Deteksi tepi")
    f1 = demo_edges(gray, out)
    print(f"   kesesuaian Canny manual vs OpenCV (F1, toleransi 2 px): {f1:.2f}")
    print("4) Noise reduction")
    table = demo_noise(gray, out)
    print(f"\n   {'noise / filter':<20}" + "".join(f"{m:>17}" for m in ["noisy"] + FILTERS))
    for label, row in table.items():
        print(f"   {label:<20}" + "".join(f"{row[m][0]:>9.1f}/{row[m][1]:<6.2f} " for m in ["noisy"] + FILTERS))
        best = max(FILTERS, key=lambda m: row[m][0])
        print(f"   -> terbaik (PSNR): {best}")
    print("\n5) Pipeline")
    e1, e2 = pipeline(img, out)
    print(f"   piksel tepi dengan denoise: {e1}  | tanpa denoise: {e2}")


if __name__ == "__main__":
    main()

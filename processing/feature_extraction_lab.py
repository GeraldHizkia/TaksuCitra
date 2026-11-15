"""
processing/feature_extraction_lab.py
=====================================
Feature Extraction Lab (Sesi 3) — satu file, banyak sub-fitur, persis pola
advanced_ops.py / ai_lab.py yang sudah ada di project ini.

Sub-bagian di dalam file ini (dipetakan ke sub-header sidebar UI):
  3.1 Statistik & Warna
  3.2 Bentuk & Moments
  3.3 Tekstur Klasik (GLCM / LBP / Gabor)
  3.4 Gradien, Tepi & Transform Domain
  3.5 Descriptor Lokal (ORB / SIFT)
  3.6 Robust & Deep Embedding
  + Matching Engine (lintas-fitur)
  + Feature Concatenation Playground

Semua dim di bawah dihitung otomatis saat runtime (tergantung parameter user
seperti n_bins, P/R, block/stride) — jangan di-hardcode di frontend.

Kontrak return seragam untuk semua fungsi ekstraksi fitur (kecuali ORB/SIFT
dan matching, yang secara desain punya bentuk return berbeda):
  {
    "representasi_antara": dict,   # data/gambar perantara (untuk UI)
    "vector": np.ndarray,          # feature vector akhir (1D)
    "labels": list[str],           # nama tiap komponen vector
    "dim": int,                    # panjang vector (jangan di-hardcode di UI)
  }
"""

import io

import cv2
import matplotlib
import numpy as np
import pywt
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor="#12141c")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _style_dark_axes(ax):
    ax.set_facecolor("#12141c")
    ax.tick_params(colors="#c8cad6", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3a3f52")
    ax.title.set_color("#f0f1f5")
    ax.xaxis.label.set_color("#c8cad6")
    ax.yaxis.label.set_color("#c8cad6")


# =============================================================================
# GENERIC HELPERS — dipakai lintas-fitur, bukan fitur tersendiri
# =============================================================================

def overlapping_block(feature_fn, img: np.ndarray, block: int = 32, stride: int | None = None,
                       **feature_kwargs) -> dict:
    """Strategi spatial generik -- konsep eksplisit modul GLCM: 'Overlapping
    block dapat memakai feature apa saja'. Membungkus fungsi fitur APAPUN jadi
    versi block/overlapping-nya tanpa bikin *_block() manual per fitur."""
    stride = stride or block
    H, W = img.shape[:2]
    feats, n_blok = [], 0
    for y in range(0, H - block + 1, stride):
        for x in range(0, W - block + 1, stride):
            patch = img[y:y + block, x:x + block]
            r = feature_fn(patch, **feature_kwargs)
            feats.append(r["vector"])
            n_blok += 1
    if n_blok == 0:
        raise ValueError("Citra terlalu kecil untuk ukuran block ini")
    base_labels = feature_fn(img[:block, :block], **feature_kwargs)["labels"]
    F = np.array(feats)
    vector = F.flatten().astype(np.float32)
    labels = [f"blok{i}_{lb}" for i in range(n_blok) for lb in base_labels]
    return {
        "representasi_antara": {"feature_matrix_shape": list(F.shape), "n_blok": n_blok,
                                 "overlap": stride < block, "block": block, "stride": stride},
        "vector": vector, "labels": labels, "dim": len(vector),
    }


def visualize_vector(vector: np.ndarray, labels: list[str] | None = None,
                      title: str = "Feature Vector", max_bars: int = 40) -> bytes:
    """WOW-FACTOR UNIVERSAL: bar chart dari vector APAPUN -- membuat 27+ fitur
    langsung tampil visual tanpa plot khusus satu-satu."""
    vector = np.asarray(vector).flatten()
    n = len(vector)
    truncated = n > max_bars
    show = vector[:max_bars] if truncated else vector
    show_labels = (labels[:max_bars] if labels else [str(i) for i in range(len(show))])
    fig, ax = plt.subplots(figsize=(max(6, len(show) * 0.22), 3.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(show)))
    ax.bar(range(len(show)), show, color=colors)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels(show_labels, rotation=60, ha="right", fontsize=6.5)
    suffix = f" (menampilkan {max_bars} dari {n} dimensi)" if truncated else f" ({n}D)"
    ax.set_title(title + suffix, fontsize=10)
    _style_dark_axes(ax)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# =============================================================================
# 3.1 & 3.2 — STATISTIK, WARNA, BENTUK, MOMENTS
# =============================================================================

# ---------------------------------------------------------------------------
# 3.1 — STATISTIK & WARNA
# ---------------------------------------------------------------------------

def intensity_statistics(img_gray: np.ndarray, n_bins: int = 8) -> dict:
    """Mean, std, min, max, histogram N-bin, entropy dari citra grayscale.
    Dimensi vector = 4 (mean,std,min,max) + n_bins."""
    img = img_gray.astype(np.float32)
    mean, std = float(img.mean()), float(img.std())
    mn, mx = float(img.min()), float(img.max())

    hist, edges = np.histogram(img, bins=n_bins, range=(0, 255))
    hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-12)

    # entropy dari histogram (bukan dari nilai piksel mentah)
    p = hist_norm[hist_norm > 0]
    entropy = float(-np.sum(p * np.log2(p)))

    vector = np.r_[mean, std, mn, mx, entropy, hist_norm]
    labels = ["mean", "std", "min", "max", "entropy"] + [f"h{i}" for i in range(n_bins)]

    return {
        "representasi_antara": {"histogram_bins": hist.tolist(), "bin_edges": edges.tolist()},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def color_statistics_multispace(img_bgr: np.ndarray, space: str = "hsv") -> dict:
    """Mean + std per kanal pada ruang warna pilihan: rgb / hsv / lab.
    Dimensi vector = 6 (mean1,mean2,mean3,std1,std2,std3)."""
    space = space.lower()
    if space == "rgb":
        conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ch_names = ["R", "G", "B"]
    elif space == "hsv":
        conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ch_names = ["H", "S", "V"]
    elif space == "lab":
        conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        ch_names = ["L", "a", "b"]
    else:
        raise ValueError("space harus salah satu dari: rgb, hsv, lab")

    flat = conv.reshape(-1, 3).astype(np.float32)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)

    vector = np.r_[means, stds]
    labels = [f"mean_{c}" for c in ch_names] + [f"std_{c}" for c in ch_names]

    return {
        "representasi_antara": {"space": space, "converted_shape": conv.shape},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def color_histogram_per_channel(img_bgr: np.ndarray, space: str = "hsv", n_bins: int = 8) -> dict:
    """Histogram tiap kanal (concat), dimensi = 3 * n_bins."""
    space = space.lower()
    if space == "rgb":
        conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ranges = [(0, 256), (0, 256), (0, 256)]
        ch_names = ["R", "G", "B"]
    elif space == "hsv":
        conv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ranges = [(0, 180), (0, 256), (0, 256)]
        ch_names = ["H", "S", "V"]
    else:
        raise ValueError("space harus rgb atau hsv")

    feats, labels, hist_maps = [], [], []
    for i, name in enumerate(ch_names):
        h = cv2.calcHist([conv], [i], None, [n_bins], list(ranges[i])).flatten()
        h = h / (h.sum() + 1e-12)
        feats.append(h)
        hist_maps.append(h.tolist())
        labels += [f"{name}_h{b}" for b in range(n_bins)]

    vector = np.concatenate(feats)
    return {
        "representasi_antara": {"space": space, "per_channel_hist": hist_maps},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


# ---------------------------------------------------------------------------
# 3.2 — BENTUK & MOMENTS
# ---------------------------------------------------------------------------

def _binarize_otsu(img_gray: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def shape_descriptors(img_gray: np.ndarray, bw: np.ndarray | None = None) -> dict:
    """Area, perimeter, bounding box, aspect ratio, centroid, circularity.
    Dimensi vector = 6. Butuh citra biner (Otsu default kalau tidak dikasih)."""
    if bw is None:
        bw = _binarize_otsu(img_gray)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Tidak ada kontur ditemukan — citra biner kosong / perlu invert")

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = w / h if h != 0 else 0.0
    circularity = (4 * np.pi * area / (peri ** 2)) if peri != 0 else 0.0

    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] != 0 else 0.0
    cy = M["m01"] / M["m00"] if M["m00"] != 0 else 0.0

    vector = np.array([area, peri, aspect_ratio, cx, cy, circularity], dtype=np.float32)
    labels = ["area", "perimeter", "aspect_ratio", "centroid_x", "centroid_y", "circularity"]

    return {
        "representasi_antara": {"bounding_box": [int(x), int(y), int(w), int(h)], "binary_used": True},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def raw_central_moments(img_gray: np.ndarray, bw: np.ndarray | None = None) -> dict:
    """Tabel m00,m10,m01,mu20,nu20 (representasi antara, bukan vector akhir yang
    umum dipakai untuk matching -- tapi tetap disediakan sebagai vector 5D
    untuk UI 'tabel highlight mana yang berubah/tidak'."""
    if bw is None:
        bw = _binarize_otsu(img_gray)
    M = cv2.moments(bw)
    vector = np.array([M["m00"], M["m10"], M["m01"], M["mu20"], M["nu20"]], dtype=np.float64)
    labels = ["m00", "m10", "m01", "mu20", "nu20"]
    return {
        "representasi_antara": {"raw_moments_dict": {k: float(v) for k, v in M.items()}},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def hu_moments_invariant(img_gray: np.ndarray, bw: np.ndarray | None = None, log_scale: bool = True) -> dict:
    """7 nilai Hu moments (opsional log-scale). Dimensi vector = 7.
    Fitur andalan untuk Robustness Stress-Test (lihat processing/robustness.py)."""
    if bw is None:
        bw = _binarize_otsu(img_gray)
    hu = cv2.HuMoments(cv2.moments(bw)).flatten()
    if log_scale:
        hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    labels = [f"Hu{i+1}" for i in range(7)]
    return {
        "representasi_antara": {"log_scale": log_scale},
        "vector": hu.astype(np.float32),
        "labels": labels,
        "dim": len(hu),
    }


# ---------------------------------------------------------------------------
# Registry kecil supaya panel UI bisa memanggil fungsi lewat nama string


# =============================================================================
# 3.3 — TEKSTUR KLASIK: GLCM / LBP / GABOR
# =============================================================================

# ---------------------------------------------------------------------------
# PANEL A — GLCM
# ---------------------------------------------------------------------------

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]


def _quantize(img_gray: np.ndarray, levels: int) -> np.ndarray:
    return (img_gray.astype(np.int32) * int(levels) // 256).clip(0, int(levels) - 1).astype(np.uint8)


def _glcm_entropy(g: np.ndarray) -> np.ndarray:
    """Entropy manual (skimage tidak menyediakan) -- kelompok 'Ketidakpastian'
    pada tabel Haralick features di modul."""
    eps = 1e-12
    p = g.astype(np.float64)
    return -np.sum(p * np.log2(p + eps), axis=(0, 1))


def glcm_explorer(
    img_gray: np.ndarray,
    distances: list[int] = (1,),
    angles_deg: list[int] = (0,),
    levels: int = 8,
    symmetric: bool = True,
    pooling: str | None = None,  # None | "mean" | "mean_std"
    include_entropy: bool = True,
) -> dict:
    """GLCM dengan parameter bebas: multi-distance & multi-angle.
    Dimensi vector (tanpa pooling) = len(props) * len(distances) * len(angles).
    Dengan pooling="mean" -> dimensi = len(props).
    Dengan pooling="mean_std" -> dimensi = len(props) * 2.
    """
    q = _quantize(img_gray, levels)
    angles_rad = [np.deg2rad(a) for a in angles_deg]
    g = graycomatrix(q, distances=list(distances), angles=angles_rad,
                      levels=levels, symmetric=symmetric, normed=True)

    props = GLCM_PROPS + (["entropy"] if include_entropy else [])
    raw = {p: graycoprops(g, p) for p in GLCM_PROPS}
    if include_entropy:
        raw["entropy"] = _glcm_entropy(g)

    if pooling is None:
        vector, labels = [], []
        for p in props:
            for di, d in enumerate(distances):
                for ai, a in enumerate(angles_deg):
                    vector.append(raw[p][di, ai])
                    labels.append(f"{p}_d{d}_a{a}")
    elif pooling == "mean":
        vector, labels = [], []
        for p in props:
            vector.append(float(np.mean(raw[p])))
            labels.append(f"{p}_mean")
    elif pooling == "mean_std":
        vector, labels = [], []
        for p in props:
            vector.append(float(np.mean(raw[p])))
            vector.append(float(np.std(raw[p])))
            labels += [f"{p}_mean", f"{p}_std"]
    else:
        raise ValueError("pooling harus None, 'mean', atau 'mean_std'")

    vector = np.array(vector, dtype=np.float32)
    return {
        "representasi_antara": {
            "glcm_matrix_shape": list(g.shape), "levels": levels,
            "glcm_matrix_first_pair": g[:, :, 0, 0].tolist(),
        },
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def visualize_glcm_heatmap(img_gray: np.ndarray, distance: int = 1, angle_deg: int = 0,
                            levels: int = 8) -> bytes:
    """Heatmap matriks GLCM -- persis figur modul 'GLCM (arah 0°, jarak 1)'."""
    q = _quantize(img_gray, levels)
    g = graycomatrix(q, distances=[distance], angles=[np.deg2rad(angle_deg)],
                      levels=levels, symmetric=True, normed=True)[:, :, 0, 0]
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(g, cmap="Blues")
    ax.set_title(f"GLCM (d={distance}, {angle_deg}°)", fontsize=10)
    ax.set_xlabel("gray j")
    ax.set_ylabel("gray i")
    for i in range(levels):
        for j in range(levels):
            ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center",
                     color="white" if g[i, j] < g.max() / 2 else "black", fontsize=7)
    _style_dark_axes(ax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color="#c8cad6")
    plt.setp(cbar.ax.get_yticklabels(), color="#c8cad6")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def glcm_block(img_gray: np.ndarray, block: int = 32, stride: int | None = None,
               levels: int = 8, distances=(1,), angles_deg=(0, 90)) -> dict:
    """Local/Overlapping GLCM. stride=None -> non-overlap (stride=block).
    Dimensi = n_blok * len(props)."""
    stride = stride or block
    H, W = img_gray.shape
    feats = []
    n_blok = 0
    for y in range(0, H - block + 1, stride):
        for x in range(0, W - block + 1, stride):
            patch = img_gray[y:y + block, x:x + block]
            r = glcm_explorer(patch, distances=distances, angles_deg=angles_deg,
                               levels=levels, pooling="mean")
            feats.append(r["vector"])
            n_blok += 1
    F = np.array(feats)  # (n_blok, len(props))
    vector = F.flatten()
    _glcm_props_with_entropy = GLCM_PROPS + ["entropy"]
    labels = [f"blok{i}_{p}" for i in range(n_blok) for p in _glcm_props_with_entropy]
    return {
        "representasi_antara": {"feature_matrix_shape": F.shape, "n_blok": n_blok,
                                 "overlap": stride < block},
        "vector": vector.astype(np.float32),
        "labels": labels,
        "dim": len(vector),
    }


def glcm_color(img_bgr: np.ndarray, levels: int = 8) -> dict:
    """GLCM per-kanal RGB. Dimensi = 3 kanal * 3 statistik = 9."""
    levels = int(levels)
    feats, labels = [], []
    props = ["contrast", "homogeneity", "energy"]
    for ch_idx, ch_name in enumerate(["R", "G", "B"]):
        channel = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[:, :, ch_idx]
        q = _quantize(channel, levels)
        g = graycomatrix(q, [1], [0, np.pi / 2], levels=levels, symmetric=True, normed=True)
        for p in props:
            feats.append(float(graycoprops(g, p).mean()))
            labels.append(f"{ch_name}_{p}")
    vector = np.array(feats, dtype=np.float32)
    return {"representasi_antara": {"levels": levels}, "vector": vector, "labels": labels, "dim": len(vector)}


# ---------------------------------------------------------------------------
# PANEL B — LBP
# ---------------------------------------------------------------------------

def _lbp_n_bins(P: int, method: str) -> int:
    if method == "default":
        return 2 ** P
    if method == "uniform":
        return P + 2
    if method == "ror":
        return 2 ** P  # kode kanonik tetap dalam rentang [0, 2^P)
    if method == "nri_uniform":
        return P * (P - 1) + 3
    if method == "var":
        return 32  # var kontinu -> dibinning manual jadi 32 bin
    raise ValueError(f"method LBP tidak dikenal: {method}")


def lbp_explorer(img_gray: np.ndarray, P: int = 8, R: int = 1, method: str = "uniform") -> dict:
    """LBP dasar: default/uniform/ror/nri_uniform/var.
    Dimensi vector = P+2 (uniform) / 2^P (default,ror) / P(P-1)+3 (nri_uniform) / 32 (var)."""
    lbp_map = local_binary_pattern(img_gray, P=P, R=R, method=method)
    n_bins = _lbp_n_bins(P, method)

    if method == "var":
        # var bernilai kontinu (termasuk nan di area homogen) -> clip & bin manual
        vals = lbp_map[~np.isnan(lbp_map)]
        hist, _ = np.histogram(vals, bins=n_bins)
    else:
        hist, _ = np.histogram(lbp_map.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-12)
    labels = [f"h{i}" for i in range(n_bins)]

    return {
        "representasi_antara": {"lbp_map_shape": lbp_map.shape, "method": method, "P": P, "R": R},
        "vector": hist,
        "labels": labels,
        "dim": len(hist),
    }


def lbp_block(img_gray: np.ndarray, P: int = 8, R: int = 1, method: str = "uniform",
              block: int = 16, stride: int | None = None) -> dict:
    """Block/Overlapping LBP. Dimensi = n_blok * n_bins."""
    stride = stride or block
    lbp_map = local_binary_pattern(img_gray, P=P, R=R, method=method)
    n_bins = _lbp_n_bins(P, method)
    H, W = lbp_map.shape
    feats = []
    for y in range(0, H - block + 1, stride):
        for x in range(0, W - block + 1, stride):
            patch = lbp_map[y:y + block, x:x + block]
            h, _ = np.histogram(patch.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
            h = h.astype(np.float32) / (h.sum() + 1e-12)
            feats.append(h)
    F = np.array(feats)
    vector = F.flatten()
    return {
        "representasi_antara": {"feature_matrix_shape": F.shape, "overlap": stride < block},
        "vector": vector,
        "labels": [f"blok{i}_h{j}" for i in range(F.shape[0]) for j in range(n_bins)],
        "dim": len(vector),
    }


def lbp_multiscale(img_gray: np.ndarray, configs: list[tuple[int, int]] = ((8, 1), (16, 2), (24, 3)),
                    method: str = "uniform") -> dict:
    """Gabung histogram beberapa (P,R). Dimensi = sum(n_bins tiap config)."""
    all_feat, labels = [], []
    for P, R in configs:
        r = lbp_explorer(img_gray, P=P, R=R, method=method)
        all_feat.append(r["vector"])
        labels += [f"P{P}R{R}_{lb}" for lb in r["labels"]]
    vector = np.concatenate(all_feat)
    return {
        "representasi_antara": {"configs": list(configs)},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


# --- Varian lanjutan (accordion "Varian Lanjutan" di UI) ---

def ltp_features(img_gray: np.ndarray, t: int = 5, P: int = 8, R: int = 1) -> dict:
    """Local Ternary Pattern: dipecah upper/lower pattern lalu histogram digabung.
    Dimensi = 2 * (P+2) memakai skema uniform-like binning sederhana."""
    img = img_gray.astype(np.int16)
    H, W = img.shape
    upper = np.zeros((H, W), dtype=np.uint8)
    lower = np.zeros((H, W), dtype=np.uint8)

    # implementasi sederhana P=8,R=1 (8 tetangga grid 3x3, tanpa interpolasi)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    padded = np.pad(img, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        diff = neighbor - center
        upper |= ((diff > t).astype(np.uint8) << bit)
        lower |= ((diff < -t).astype(np.uint8) << bit)

    n_bins = 256
    hu, _ = np.histogram(upper.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hl, _ = np.histogram(lower.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hu = hu.astype(np.float32) / (hu.sum() + 1e-12)
    hl = hl.astype(np.float32) / (hl.sum() + 1e-12)

    vector = np.concatenate([hu, hl])
    labels = [f"upper_h{i}" for i in range(n_bins)] + [f"lower_h{i}" for i in range(n_bins)]
    return {
        "representasi_antara": {"threshold_t": t, "upper_map_shape": upper.shape,
                                 "upper_map": upper, "lower_map": lower},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def clbp_features(img_gray: np.ndarray, P: int = 8, R: int = 1) -> dict:
    """CLBP: sign (S) + magnitude (M) + center (C).
    Dimensi = n_bins(S) + n_bins(M) + 1(C-binary -> direpresentasikan sbg proporsi)."""
    lbp_s = local_binary_pattern(img_gray, P=P, R=R, method="uniform")
    n_bins_s = P + 2
    hist_s, _ = np.histogram(lbp_s.ravel(), bins=np.arange(0, n_bins_s + 1), range=(0, n_bins_s))
    hist_s = hist_s.astype(np.float32) / (hist_s.sum() + 1e-12)

    # magnitude: |neighbor - center| dirata-ratakan lalu threshold ke mean magnitude -> histogram biner
    padded = np.pad(img_gray.astype(np.int16), 1, mode="edge")
    center = padded[1:-1, 1:-1]
    mags = []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for dy, dx in offsets:
        neighbor = padded[1 + dy:1 + dy + img_gray.shape[0], 1 + dx:1 + dx + img_gray.shape[1]]
        mags.append(np.abs(neighbor - center))
    mag_stack = np.mean(mags, axis=0)
    mag_thresh = mag_stack.mean()
    hist_m, _ = np.histogram(mag_stack.ravel(), bins=10, range=(0, mag_stack.max() + 1e-6))
    hist_m = hist_m.astype(np.float32) / (hist_m.sum() + 1e-12)

    center_above_mean = float((center.astype(np.float32) > img_gray.mean()).mean())

    vector = np.concatenate([hist_s, hist_m, [center_above_mean]])
    labels = [f"S_h{i}" for i in range(n_bins_s)] + [f"M_h{i}" for i in range(10)] + ["C_proportion"]
    return {
        "representasi_antara": {"mag_threshold_used": float(mag_thresh),
                                 "sign_map": lbp_s, "magnitude_map": mag_stack},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def cs_lbp_features(img_gray: np.ndarray, R: int = 1, t: int = 0) -> dict:
    """Center-Symmetric LBP, P=8 -> 4 pasangan -> 2^4=16 kode. Dimensi = 16."""
    padded = np.pad(img_gray.astype(np.int16), R, mode="edge")
    H, W = img_gray.shape
    offsets = [(-R, -R), (-R, 0), (-R, R), (0, R), (R, R), (R, 0), (R, -R), (0, -R)]
    neigh = [padded[R + dy:R + dy + H, R + dx:R + dx + W] for dy, dx in offsets]

    code = np.zeros((H, W), dtype=np.uint8)
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    for bit, (a, b) in enumerate(pairs):
        bitval = (np.abs(neigh[a].astype(np.int16) - neigh[b].astype(np.int16)) >= t).astype(np.uint8)
        code |= (bitval << bit)

    n_bins = 16
    hist, _ = np.histogram(code.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-12)
    return {
        "representasi_antara": {"code_map_shape": code.shape, "code_map": code},
        "vector": hist,
        "labels": [f"h{i}" for i in range(n_bins)],
        "dim": n_bins,
    }


def mb_lbp_features(img_gray: np.ndarray, block: int = 3) -> dict:
    """Multi-Block LBP: bandingkan mean 8 blok sekitar dengan mean blok tengah.
    Dimensi = 256 (histogram kode 8-bit)."""
    H, W = img_gray.shape
    bh, bw = block, block
    grid_h, grid_w = H // (3 * bh), W // (3 * bw)
    codes = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0, x0 = gy * 3 * bh, gx * 3 * bw
            region = img_gray[y0:y0 + 3 * bh, x0:x0 + 3 * bw]
            if region.shape[0] < 3 * bh or region.shape[1] < 3 * bw:
                continue
            means = np.zeros((3, 3))
            for i in range(3):
                for j in range(3):
                    means[i, j] = region[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw].mean()
            c = means[1, 1]
            order = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
            code = 0
            for bit, (i, j) in enumerate(order):
                if means[i, j] >= c:
                    code |= (1 << bit)
            codes.append(code)

    codes = np.array(codes, dtype=np.uint8) if codes else np.array([0], dtype=np.uint8)
    hist, _ = np.histogram(codes, bins=np.arange(0, 257), range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-12)
    return {
        "representasi_antara": {"n_region": len(codes), "block_size": block},
        "vector": hist,
        "labels": [f"h{i}" for i in range(256)],
        "dim": 256,
    }


def visualize_lbp_map(img_gray: np.ndarray, P: int = 8, R: int = 1, method: str = "uniform") -> bytes:
    """Citra asli -> LBP map (colormap) -> histogram, 3 panel berdampingan."""
    lbp_map = local_binary_pattern(img_gray, P=P, R=R, method=method)
    n_bins = _lbp_n_bins(P, method)
    hist, _ = np.histogram(lbp_map.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Citra Asli", fontsize=10)
    axes[0].axis("off")
    im = axes[1].imshow(lbp_map, cmap="nipy_spectral")
    axes[1].set_title(f"LBP Map ({method}, P={P} R={R})", fontsize=10)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046)
    cbar.ax.yaxis.set_tick_params(color="#c8cad6")
    plt.setp(cbar.ax.get_yticklabels(), color="#c8cad6")
    axes[2].bar(range(n_bins), hist, color=plt.cm.nipy_spectral(np.linspace(0, 1, n_bins)))
    axes[2].set_title(f"Histogram ({n_bins} bin)", fontsize=10)
    _style_dark_axes(axes[2])
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ---------------------------------------------------------------------------
# PANEL C — GABOR
# ---------------------------------------------------------------------------

def _gabor_kernel(theta_deg, lam, sigma=4.0, gamma=0.5, ksize=31):
    return cv2.getGaborKernel((ksize, ksize), sigma, np.deg2rad(theta_deg), lam, gamma, 0, ktype=cv2.CV_32F)


def gabor_single(img_gray: np.ndarray, theta_deg: float = 0, lam: float = 10.0,
                  sigma: float = 4.0, gamma: float = 0.5) -> dict:
    """Satu filter Gabor -> statistik [mean_abs, std, energy]. Dimensi = 3."""
    kernel = _gabor_kernel(theta_deg, lam, sigma, gamma)
    r = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
    vector = np.array([np.mean(np.abs(r)), np.std(r), np.mean(r ** 2)], dtype=np.float32)
    return {
        "representasi_antara": {"response_map_shape": r.shape, "kernel_shape": kernel.shape},
        "vector": vector,
        "labels": ["mean_abs", "std", "energy"],
        "dim": 3,
    }


def gabor_bank(img_gray: np.ndarray, thetas_deg: list[float] = (0, 45, 90, 135),
               lambdas: list[float] = (10.0,), stat: str = "energy") -> dict:
    """Filter bank multi-orientasi (+ opsional multi-scale via lambdas).
    stat: 'energy' (1 angka/filter) atau 'mean_std_energy' (3 angka/filter).
    Dimensi = len(thetas)*len(lambdas)*(1 atau 3)."""
    feats, labels, energy_map = [], [], []
    for lam in lambdas:
        row = []
        for theta in thetas_deg:
            kernel = _gabor_kernel(theta, lam)
            r = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
            if stat == "energy":
                val = float(np.mean(r ** 2))
                feats.append(val)
                labels.append(f"E_l{lam}_t{theta}")
                row.append(val)
            elif stat == "mean_std_energy":
                feats += [float(np.mean(np.abs(r))), float(np.std(r)), float(np.mean(r ** 2))]
                labels += [f"mean_l{lam}_t{theta}", f"std_l{lam}_t{theta}", f"energy_l{lam}_t{theta}"]
                row.append(float(np.mean(r ** 2)))
            else:
                raise ValueError("stat harus 'energy' atau 'mean_std_energy'")
        energy_map.append(row)

    vector = np.array(feats, dtype=np.float32)
    return {
        "representasi_antara": {"energy_heatmap": energy_map, "n_filter": len(thetas_deg) * len(lambdas)},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def gabor_block(img_gray: np.ndarray, thetas_deg=(0, 45, 90, 135), lam: float = 10.0,
                 block: int = 32, stride: int | None = None) -> dict:
    """Block Gabor: tiap filter -> tiap blok -> [mean, energy].
    Dimensi = n_filter * n_blok * 2."""
    stride = stride or block
    H, W = img_gray.shape
    vector_parts, labels = [], []
    n_blok = 0
    for theta in thetas_deg:
        kernel = _gabor_kernel(theta, lam)
        r = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
        blok_idx = 0
        for y in range(0, H - block + 1, stride):
            for x in range(0, W - block + 1, stride):
                patch = r[y:y + block, x:x + block]
                vector_parts += [float(patch.mean()), float(np.mean(patch ** 2))]
                labels += [f"t{theta}_b{blok_idx}_mean", f"t{theta}_b{blok_idx}_energy"]
                blok_idx += 1
        n_blok = blok_idx

    vector = np.array(vector_parts, dtype=np.float32)
    return {
        "representasi_antara": {"n_blok_per_filter": n_blok, "overlap": stride < block},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def gabor_lbp_hybrid(img_gray: np.ndarray, thetas_deg=(0, 45, 90, 135), lam: float = 10.0) -> dict:
    """Varian lanjutan: Gabor magnitude response -> LBP uniform histogram per filter.
    Dimensi = n_filter * 10."""
    feats, labels = [], []
    for theta in thetas_deg:
        kernel = _gabor_kernel(theta, lam)
        r = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
        r8 = cv2.normalize(np.abs(r), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        lbp = local_binary_pattern(r8, P=8, R=1, method="uniform")
        h, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        h = h.astype(np.float32) / (h.sum() + 1e-12)
        feats.append(h)
        labels += [f"t{theta}_h{i}" for i in range(10)]
    vector = np.concatenate(feats)
    return {
        "representasi_antara": {"n_filter": len(thetas_deg)},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def gabor_complex_components(img_gray: np.ndarray, theta_deg: float = 0, lam: float = 10.0,
                              sigma: float = 4.0, gamma: float = 0.5) -> dict:
    """Real (psi=0) + Imaginary (psi=pi/2) -> magnitude & phase. Dimensi = 3."""
    kernel_real = _gabor_kernel(theta_deg, lam, sigma, gamma)
    kernel_imag = cv2.getGaborKernel((31, 31), sigma, np.deg2rad(theta_deg), lam, gamma,
                                      np.pi / 2, ktype=cv2.CV_32F)
    real = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel_real)
    imag = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel_imag)
    magnitude = np.sqrt(real ** 2 + imag ** 2)
    phase = np.arctan2(imag, real)
    vector = np.array([np.mean(np.abs(magnitude)), np.std(magnitude), np.mean(magnitude ** 2)], dtype=np.float32)
    return {
        "representasi_antara": {
            "real_shape": list(real.shape), "imag_shape": list(imag.shape),
            "phase_mean": float(np.mean(phase)), "phase_std": float(np.std(phase)),
            "note": "magnitude/energy dipakai sbg vector -- phase lebih sensitif, disimpan sbg info",
        },
        "vector": vector,
        "labels": ["magnitude_mean_abs", "magnitude_std", "magnitude_energy"],
        "dim": 3,
    }


def visualize_gabor_grid(img_gray: np.ndarray, thetas_deg=(0, 45, 90, 135), lam: float = 10.0) -> bytes:
    """Grid kernel (baris atas) + response map (baris bawah) per orientasi."""
    n = len(thetas_deg)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.4))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, theta in enumerate(thetas_deg):
        kernel = _gabor_kernel(theta, lam)
        r = cv2.filter2D(img_gray.astype(np.float32), cv2.CV_32F, kernel)
        axes[0, i].imshow(kernel, cmap="gray")
        axes[0, i].set_title(f"Kernel {theta}°", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(r, cmap="gray")
        axes[1, i].set_title(f"Respons {theta}°", fontsize=9)
        axes[1, i].axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# =============================================================================
# 3.4 — GRADIEN, TEPI & TRANSFORM DOMAIN
# =============================================================================

def hog_feature(img_gray: np.ndarray, orientations: int = 9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2)) -> dict:
    """HOG penuh. Dimensi = ratusan D tergantung ukuran citra -> tampilkan shape saja."""
    feat, hog_img = hog(
        img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, block_norm="L2-Hys", visualize=True,
    )
    return {
        "representasi_antara": {"hog_visual_shape": hog_img.shape},
        "vector": feat.astype(np.float32),
        "labels": [f"hog_{i}" for i in range(len(feat))],
        "dim": len(feat),
    }


def edge_based_feature(img_gray: np.ndarray, canny_low: int = 80, canny_high: int = 160,
                        n_bins: int = 9, mag_thresh: float = 20.0) -> dict:
    """Edge density + histogram orientasi tepi (weighted by magnitude). Dimensi = 1 + n_bins."""
    edges = cv2.Canny(img_gray, canny_low, canny_high)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mask = mag > mag_thresh
    hist, _ = np.histogram(ang[mask] % 180, bins=n_bins, range=(0, 180), weights=mag[mask])
    hist = hist / (hist.sum() + 1e-12)

    vector = np.r_[edge_density, hist].astype(np.float32)
    labels = ["edge_density"] + [f"orient_h{i}" for i in range(n_bins)]
    return {
        "representasi_antara": {"edge_map_shape": edges.shape, "n_edge_px": int(np.count_nonzero(edges))},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def projection_profile(img_gray: np.ndarray, target_size: int = 64, thresh: int = 128) -> dict:
    """Horizontal + vertical projection pada citra biner. Dimensi = 2*target_size."""
    resized = cv2.resize(img_gray, (target_size, target_size))
    bw = (resized < thresh).astype(np.uint8)
    h_profile = bw.sum(axis=1).astype(np.float32)
    v_profile = bw.sum(axis=0).astype(np.float32)
    vector = np.r_[h_profile, v_profile]
    labels = [f"h{i}" for i in range(target_size)] + [f"v{i}" for i in range(target_size)]
    return {
        "representasi_antara": {"h_profile": h_profile.tolist(), "v_profile": v_profile.tolist()},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def dct_feature(img_gray: np.ndarray, target_size: int = 64, keep: int = 8) -> dict:
    """Blok kxk koefisien frekuensi rendah dari DCT. Dimensi = keep*keep."""
    img64 = cv2.resize(img_gray, (target_size, target_size)).astype(np.float32) / 255.0
    D = cv2.dct(img64)
    block = D[:keep, :keep]
    vector = block.flatten().astype(np.float32)
    labels = [f"dct_{i}" for i in range(len(vector))]
    return {
        "representasi_antara": {"dct_full_shape": D.shape, "heatmap_block": block.tolist()},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def fft_radial_profile(img_gray: np.ndarray, n_radius_bins: int = 32) -> dict:
    """FFT magnitude spectrum -> radial profile 1D. Dimensi = n_radius_bins."""
    F = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float32)))
    mag = np.log1p(np.abs(F))

    H, W = mag.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    r_max = r.max()

    bin_edges = np.linspace(0, r_max, n_radius_bins + 1)
    profile = np.zeros(n_radius_bins, dtype=np.float32)
    for i in range(n_radius_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        profile[i] = mag[mask].mean() if mask.any() else 0.0

    return {
        "representasi_antara": {"spectrum_shape": mag.shape},
        "vector": profile,
        "labels": [f"radius_bin{i}" for i in range(n_radius_bins)],
        "dim": n_radius_bins,
    }


def wavelet_feature(img_gray: np.ndarray, wavelet: str = "haar", stat: str = "energy") -> dict:
    """Wavelet Haar level-1 -> 4 subband LL/LH/HL/HH.
    stat='energy' -> dim 4. stat='mean_std_energy' -> dim 12."""
    LL, (LH, HL, HH) = pywt.dwt2(img_gray.astype(np.float32), wavelet)
    subbands = {"LL": LL, "LH": LH, "HL": HL, "HH": HH}

    vector, labels = [], []
    for name, s in subbands.items():
        if stat == "energy":
            vector.append(float(np.mean(s ** 2)))
            labels.append(f"E_{name}")
        elif stat == "mean_std_energy":
            vector += [float(np.mean(s)), float(np.std(s)), float(np.mean(s ** 2))]
            labels += [f"mean_{name}", f"std_{name}", f"energy_{name}"]
        else:
            raise ValueError("stat harus 'energy' atau 'mean_std_energy'")

    vector = np.array(vector, dtype=np.float32)
    return {
        "representasi_antara": {"subband_shapes": {k: list(v.shape) for k, v in subbands.items()}},
        "vector": vector,
        "labels": labels,
        "dim": len(vector),
    }


def visualize_hog(img_gray: np.ndarray, orientations: int = 9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2)) -> bytes:
    from skimage import exposure
    _, hog_img = hog(img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm="L2-Hys", visualize=True)
    hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Citra Asli", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(hog_img_rescaled, cmap="inferno")
    axes[1].set_title("Visualisasi HOG", fontsize=10)
    axes[1].axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def visualize_fft_radial(img_gray: np.ndarray, n_radius_bins: int = 32) -> bytes:
    r = fft_radial_profile(img_gray, n_radius_bins=n_radius_bins)
    F = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float32)))
    mag = np.log1p(np.abs(F))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].imshow(mag, cmap="magma")
    axes[0].set_title("Spektrum FFT (Magnitude)", fontsize=10)
    axes[0].axis("off")
    axes[1].plot(r["vector"], color="#4dd0e1", linewidth=1.8)
    axes[1].fill_between(range(n_radius_bins), r["vector"], color="#4dd0e1", alpha=0.25)
    axes[1].set_title("Radial Profile (1D)", fontsize=10)
    axes[1].set_xlabel("radius bin")
    _style_dark_axes(axes[1])
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def visualize_wavelet_subbands(img_gray: np.ndarray, wavelet: str = "haar") -> bytes:
    LL, (LH, HL, HH) = pywt.dwt2(img_gray.astype(np.float32), wavelet)
    subbands = [("LL", LL), ("LH", LH), ("HL", HL), ("HH", HH)]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, (name, s) in zip(axes, subbands):
        ax.imshow(s, cmap="gray")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# (REGISTRY final gabungan seluruh fitur ada di bagian paling bawah file ini)


# =============================================================================
# 3.5 — DESCRIPTOR LOKAL (ORB / SIFT)
# =============================================================================

def extract_orb(img_gray: np.ndarray, n_features: int = 500) -> dict:
    """ORB: descriptor biner 32 byte/keypoint."""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, desc = orb.detectAndCompute(img_gray, None)
    return {
        "keypoints": kp,
        "descriptors": desc,  # shape (N, 32), dtype uint8
        "n_keypoint": len(kp),
        "descriptor_size_bytes": 32,
        "descriptor_type": "binary",
        "keypoint_coords": [(float(k.pt[0]), float(k.pt[1]), float(k.size), float(k.angle)) for k in kp],
    }


def extract_sift(img_gray: np.ndarray, n_features: int = 0) -> dict:
    """SIFT: descriptor float 128D/keypoint. n_features=0 -> tidak dibatasi."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp, desc = sift.detectAndCompute(img_gray, None)
    return {
        "keypoints": kp,
        "descriptors": desc,  # shape (N, 128), dtype float32
        "n_keypoint": len(kp),
        "descriptor_size_bytes": 128 * 4,
        "descriptor_type": "float",
        "keypoint_coords": [(float(k.pt[0]), float(k.pt[1]), float(k.size), float(k.angle)) for k in kp],
    }


def draw_keypoints(img_gray: np.ndarray, keypoints) -> np.ndarray:
    """Visualisasi titik + orientasi + skala keypoint di atas citra."""
    return cv2.drawKeypoints(img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def match_orb(desc1: np.ndarray, desc2: np.ndarray, cross_check: bool = True) -> dict:
    """Matching ORB pakai Hamming distance."""
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)
    return {
        "matches": matches,
        "n_match": len(matches),
        "best_distance": float(matches[0].distance) if matches else None,
        "distances": [float(m.distance) for m in matches],
    }


def match_sift(desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.75) -> dict:
    """Matching SIFT pakai L2 + ratio test David Lowe."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in pairs if m.distance < ratio_thresh * n.distance]
    good = sorted(good, key=lambda m: m.distance)
    return {
        "matches": good,
        "n_match": len(good),
        "n_candidate_pairs": len(pairs),
        "ratio_thresh": ratio_thresh,
        "distances": [float(m.distance) for m in good],
    }


def draw_matches(img1_gray: np.ndarray, kp1, img2_gray: np.ndarray, kp2, matches, max_draw: int = 50) -> np.ndarray:
    """Visual garis pencocokan antar keypoint (drawMatches)."""
    return cv2.drawMatches(
        img1_gray, kp1, img2_gray, kp2, matches[:max_draw], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


# (extract_orb/extract_sift/match_orb/match_sift sengaja TIDAK masuk REGISTRY
#  generik -- kontrak return beda, ditangani route khusus di app.py)


# =============================================================================
# 3.6 — ROBUST & DEEP EMBEDDING
# =============================================================================

_RESNET_MODEL = None
_RESNET_TRANSFORM = None
_TORCH_AVAILABLE = None


def _check_torch():
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _load_resnet50():
    """Lazy-load supaya startup app tidak lambat kalau panel ini belum dibuka."""
    global _RESNET_MODEL, _RESNET_TRANSFORM
    if _RESNET_MODEL is not None:
        return

    import torch
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # buang classifier -> raw 2048D embedding
    model.eval()

    _RESNET_MODEL = model
    _RESNET_TRANSFORM = weights.transforms()


def resnet50_embedding(img_rgb: np.ndarray) -> dict:
    """1x2048 embedding per citra. Butuh torch+torchvision (CPU wheel).
    Kalau tidak tersedia -> return status graceful, bukan exception mentah."""
    if not _check_torch():
        return {
            "status": "unavailable",
            "reason": "torch/torchvision belum terpasang. Jalankan: "
                      "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "vector": None,
            "dim": 2048,
        }

    import torch

    _load_resnet50()
    pil_img = Image.fromarray(img_rgb).convert("RGB")
    x = _RESNET_TRANSFORM(pil_img).unsqueeze(0)

    with torch.no_grad():
        f = _RESNET_MODEL(x)
    f = torch.nn.functional.normalize(f, dim=1)
    vector = f.squeeze(0).numpy().astype(np.float32)

    return {
        "status": "ok",
        "representasi_antara": {"note": "Global Average Pooling output, classifier dibuang"},
        "vector": vector,
        "labels": [f"f{i}" for i in range(2048)],
        "dim": 2048,
    }


# ---------------------------------------------------------------------------
# Stretch goals — stub graceful degradation.
# Kode contoh asli dari modul TETAP ditampilkan di UI sebagai referensi
# (badge "🔧 Kode tersedia, coba di lingkunganmu") walau tidak live di sini.
# ---------------------------------------------------------------------------

STRETCH_FEATURE_INFO = {
    "dinov3": {
        "status": "not_implemented",
        "reason": "Model gated/licensed (butuh approve akses HuggingFace + token login). "
                  "Berisiko gagal H-1 karena unduhan besar (~350MB+).",
        "reference_snippet": (
            "from transformers import AutoImageProcessor, AutoModel\n"
            "name = 'facebook/dinov3-vitb16-pretrain-lvd1689m'\n"
            "processor = AutoImageProcessor.from_pretrained(name)\n"
            "model = AutoModel.from_pretrained(name).eval()\n"
            "# ... lihat modul Robust hal. Robust-6 untuk detail lengkap"
        ),
    },
    "clip_siglip2": {
        "status": "not_implemented",
        "reason": "Unduhan besar (400MB-1.7GB), butuh internet stabil saat first load.",
        "reference_snippet": (
            "from transformers import AutoProcessor, AutoModel\n"
            "processor = AutoProcessor.from_pretrained(model_name)\n"
            "model = AutoModel.from_pretrained(model_name).eval()\n"
            "# ... lihat modul Robust hal. Robust-7 untuk detail lengkap"
        ),
    },
    "superpoint_lightglue": {
        "status": "not_implemented",
        "reason": "Paket 'lightglue' terpisah dari requirements inti, API antar versi berubah-ubah.",
        "reference_snippet": (
            "from lightglue import LightGlue, SuperPoint\n"
            "extractor = SuperPoint(max_num_keypoints=1024).eval()\n"
            "matcher = LightGlue(features='superpoint').eval()\n"
            "# ... lihat modul Robust hal. Robust-4 untuk detail lengkap"
        ),
    },
}


def get_stretch_feature_status(name: str) -> dict:
    """Dipanggil UI untuk menampilkan badge fallback yang tepat."""
    if name not in STRETCH_FEATURE_INFO:
        raise ValueError(f"Fitur stretch tidak dikenal: {name}")
    return STRETCH_FEATURE_INFO[name]


# =============================================================================
# ROBUSTNESS STRESS-TEST
# =============================================================================

def apply_transformations(img: np.ndarray) -> dict:
    """4 transformasi standar: geser, rotasi 30°, skala 0.7x, brightness +/-.
    Return dict {nama_transformasi: citra_hasil}."""
    H, W = img.shape[:2]

    # translasi (geser) 15% lebar & tinggi
    tx, ty = int(0.15 * W), int(0.15 * H)
    M_translate = np.float32([[1, 0, tx], [0, 1, ty]])
    geser = cv2.warpAffine(img, M_translate, (W, H), borderMode=cv2.BORDER_REPLICATE)

    # rotasi 30 derajat di pusat citra
    center = (W // 2, H // 2)
    M_rotate = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotasi = cv2.warpAffine(img, M_rotate, (W, H), borderMode=cv2.BORDER_REPLICATE)

    # skala 0.7x lalu pad balik ke ukuran asli (biar dimensi konsisten utk fitur global)
    small = cv2.resize(img, None, fx=0.7, fy=0.7)
    skala = cv2.copyMakeBorder(
        small, 0, H - small.shape[0], 0, W - small.shape[1],
        cv2.BORDER_CONSTANT, value=0 if img.ndim == 2 else (0, 0, 0),
    )

    # perubahan brightness (+40)
    brightness = cv2.convertScaleAbs(img, alpha=1.0, beta=40)

    return {"asli": img, "geser": geser, "rotasi_30": rotasi, "skala_0.7x": skala, "brightness": brightness}


def stability_score(vec_asli: np.ndarray, vec_transformed: np.ndarray) -> dict:
    """Skor stabilitas: seberapa besar vector berubah (relative L2 + cosine)."""
    l2_change = float(np.linalg.norm(vec_transformed - vec_asli) / (np.linalg.norm(vec_asli) + 1e-12))
    cos_sim = cosine_similarity(vec_asli, vec_transformed)
    return {"relative_l2_change": l2_change, "cosine_similarity": cos_sim}


def run_stress_test(img: np.ndarray, feature_fn, feature_kwargs: dict | None = None) -> dict:
    """
    feature_fn: fungsi ekstraksi dari feature_extraction/texture_features/dll,
                menerima citra dan mengembalikan dict {"vector": ..., "labels": ...}.
    Return: tabel perbandingan vector sebelum/sesudah + skor stabilitas per transformasi.
    """
    feature_kwargs = feature_kwargs or {}
    versions = apply_transformations(img)

    results = {}
    vec_asli = feature_fn(versions["asli"], **feature_kwargs)["vector"]

    for name, im in versions.items():
        r = feature_fn(im, **feature_kwargs)
        vec = r["vector"]
        results[name] = {
            "vector": vec.tolist(),
            "stability_vs_asli": stability_score(vec_asli, vec) if name != "asli" else None,
        }

    return {
        "labels": feature_fn(versions["asli"], **feature_kwargs)["labels"],
        "per_transformasi": results,
        "kesimpulan": _summarize_stability(results),
    }


def _summarize_stability(results: dict) -> str:
    """Ringkasan otomatis: transformasi mana yang paling mengganggu fitur ini."""
    changes = {
        name: r["stability_vs_asli"]["relative_l2_change"]
        for name, r in results.items() if r["stability_vs_asli"] is not None
    }
    if not changes:
        return "Tidak ada data pembanding."
    worst = max(changes, key=changes.get)
    best = min(changes, key=changes.get)
    return (f"Fitur ini paling terganggu oleh '{worst}' (perubahan relatif "
            f"{changes[worst]:.3f}) dan paling stabil terhadap '{best}' "
            f"(perubahan relatif {changes[best]:.3f}).")


# =============================================================================
# MATCHING ENGINE (lintas-fitur)
# =============================================================================

def euclidean_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    return float(np.linalg.norm(f1 - f2))


def cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    denom = (np.linalg.norm(f1) * np.linalg.norm(f2)) + 1e-12
    return float(np.dot(f1, f2) / denom)


def hamming_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """Untuk descriptor biner (mis. ORB). f1, f2 harus array bit/byte sepanjang sama."""
    f1b = np.unpackbits(f1.astype(np.uint8)) if f1.dtype == np.uint8 else f1
    f2b = np.unpackbits(f2.astype(np.uint8)) if f2.dtype == np.uint8 else f2
    return float(np.count_nonzero(f1b != f2b))


def chi_square_distance(h1: np.ndarray, h2: np.ndarray, eps: float = 1e-10) -> float:
    """Untuk histogram (LBP, warna, dsb)."""
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))


def histogram_intersection(h1: np.ndarray, h2: np.ndarray) -> float:
    """Semakin besar nilainya, semakin mirip (bukan distance, tapi similarity)."""
    return float(np.sum(np.minimum(h1, h2)))


# ---------------------------------------------------------------------------
# Pemilihan metric otomatis berdasar jenis fitur (dipakai Matching Engine UI)
# ---------------------------------------------------------------------------

FEATURE_METRIC_MAP = {
    "continuous": ["euclidean", "cosine"],          # statistik, GLCM, Gabor, DCT
    "histogram": ["chi_square", "histogram_intersection"],  # LBP, histogram warna
    "binary": ["hamming"],                            # ORB descriptor
}

METRIC_FUNCS = {
    "euclidean": euclidean_distance,
    "cosine": cosine_similarity,
    "hamming": hamming_distance,
    "chi_square": chi_square_distance,
    "histogram_intersection": histogram_intersection,
}

# metric mana yang "semakin besar semakin mirip" vs "semakin kecil semakin mirip"
HIGHER_IS_MORE_SIMILAR = {"cosine", "histogram_intersection"}


def compute_all_relevant(f1: np.ndarray, f2: np.ndarray, feature_kind: str) -> dict:
    """feature_kind: 'continuous' | 'histogram' | 'binary'. Mengembalikan semua
    metric yang relevan untuk jenis fitur tsb, plus verdict."""
    if feature_kind not in FEATURE_METRIC_MAP:
        raise ValueError(f"feature_kind harus salah satu dari {list(FEATURE_METRIC_MAP)}")

    results = {}
    for metric_name in FEATURE_METRIC_MAP[feature_kind]:
        results[metric_name] = METRIC_FUNCS[metric_name](f1, f2)

    primary_metric = FEATURE_METRIC_MAP[feature_kind][0]
    primary_value = results[primary_metric]
    verdict = verdict_from_metric(primary_value, primary_metric)

    return {"metrics": results, "primary_metric": primary_metric, "verdict": verdict}


def verdict_from_metric(value: float, metric_name: str,
                         thresholds: tuple[float, float] | None = None) -> str:
    """Verdict "Sangat Mirip / Cukup Mirip / Berbeda" berdasar threshold.
    thresholds default disesuaikan per metric; bisa dioverride manual dari UI.
    Return string berisi tag emoji supaya UI tinggal render apa adanya."""
    higher_better = metric_name in HIGHER_IS_MORE_SIMILAR

    if thresholds is None:
        default_thresholds = {
            "euclidean": (0.5, 1.5),          # kecil = mirip
            "chi_square": (0.2, 0.6),         # kecil = mirip
            "hamming": (30, 70),              # kecil = mirip (dari 256 bit ORB)
            "cosine": (0.85, 0.6),            # besar = mirip
            "histogram_intersection": (0.7, 0.4),  # besar = mirip
        }
        thresholds = default_thresholds.get(metric_name, (0.5, 1.5))

    low, high = thresholds
    if higher_better:
        if value >= low:
            return "🟢 Sangat Mirip"
        elif value >= high:
            return "🟡 Cukup Mirip"
        return "🔴 Berbeda"
    else:
        if value <= low:
            return "🟢 Sangat Mirip"
        elif value <= high:
            return "🟡 Cukup Mirip"
        return "🔴 Berbeda"


# =============================================================================
# FEATURE CONCATENATION PLAYGROUND
# =============================================================================

def concatenate_features(feature_results: dict[str, dict], normalize: bool = False) -> dict:
    """
    feature_results: {nama_fitur: {"vector": np.ndarray, "labels": [...]}}
                      -- hasil dari panel-panel lain yang sudah dijalankan user.
    normalize: kalau True, terapkan StandardScaler per-komponen (butuh >1 sampel
               untuk statistik yang bermakna; kalau cuma 1 sampel, fallback ke
               L2-normalize sederhana supaya tidak error).

    Return: vector gabungan + breakdown dimensi tiap komponen (untuk stacked bar UI).
    """
    if not feature_results:
        raise ValueError("Minimal 1 fitur harus dipilih untuk digabung")

    names = list(feature_results.keys())
    vectors = [np.asarray(feature_results[n]["vector"], dtype=np.float32) for n in names]
    dims = [len(v) for v in vectors]

    concatenated = np.concatenate(vectors)

    if normalize:
        # single-sample StandardScaler tidak bermakna (std=0) -> pakai L2 norm sbg fallback aman
        norm = np.linalg.norm(concatenated)
        concatenated_normalized = concatenated / (norm + 1e-12)
    else:
        concatenated_normalized = concatenated

    breakdown = [{"nama_fitur": n, "dimensi": d, "porsi_persen": round(100 * d / sum(dims), 1)}
                 for n, d in zip(names, dims)]

    return {
        "vector": concatenated_normalized,
        "vector_raw": concatenated,
        "dim_total": len(concatenated),
        "breakdown": breakdown,
        "normalized": normalize,
    }


def fit_scaler_multi_sample(list_of_concatenated_vectors: list[np.ndarray]) -> StandardScaler:
    """Kalau user sudah kumpulkan >1 sampel (mis. dari galeri), StandardScaler
    yang sesungguhnya baru bisa di-fit dengan benar di sini."""
    X = np.array(list_of_concatenated_vectors)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


# =============================================================================
# REGISTRY GABUNGAN — dipakai app.py untuk dispatch generik 1 route -> banyak fitur
# key = nama fungsi (dipakai UI sbg identifier panel), value = fungsi Python-nya
# =============================================================================

REGISTRY = {
    # 3.1 Statistik & Warna
    "intensity_statistics": intensity_statistics,
    "color_statistics_multispace": color_statistics_multispace,
    "color_histogram_per_channel": color_histogram_per_channel,
    # 3.2 Bentuk & Moments
    "shape_descriptors": shape_descriptors,
    "raw_central_moments": raw_central_moments,
    "hu_moments_invariant": hu_moments_invariant,
    # 3.3 Tekstur — GLCM
    "glcm_explorer": glcm_explorer,
    "glcm_block": glcm_block,
    "glcm_color": glcm_color,
    # 3.3 Tekstur — LBP
    "lbp_explorer": lbp_explorer,
    "lbp_block": lbp_block,
    "lbp_multiscale": lbp_multiscale,
    "ltp_features": ltp_features,
    "clbp_features": clbp_features,
    "cs_lbp_features": cs_lbp_features,
    "mb_lbp_features": mb_lbp_features,
    # 3.3 Tekstur — Gabor
    "gabor_single": gabor_single,
    "gabor_bank": gabor_bank,
    "gabor_block": gabor_block,
    "gabor_lbp_hybrid": gabor_lbp_hybrid,
    # 3.4 Gradien, Tepi & Transform Domain
    "hog_feature": hog_feature,
    "edge_based_feature": edge_based_feature,
    "projection_profile": projection_profile,
    "dct_feature": dct_feature,
    "fft_radial_profile": fft_radial_profile,
    "wavelet_feature": wavelet_feature,
    # 3.6 Robust & Deep Embedding
    "resnet50_embedding": resnet50_embedding,
}

# Fungsi-fungsi berikut SENGAJA tidak masuk REGISTRY generik di atas, karena
# kontrak return-nya berbeda dari pola {"vector","labels","dim"} (bukan bug):
#   - extract_orb, extract_sift        -> N descriptor per-keypoint, bukan 1 vector
#   - match_orb, match_sift            -> perlu 2 citra input, bukan 1
#   - draw_keypoints, draw_matches     -> return gambar, dipanggil manual di route ORB/SIFT
#   - euclidean_distance, cosine_similarity, hamming_distance,
#     chi_square_distance, histogram_intersection, verdict_from_metric,
#     compute_all_relevant            -> Matching Engine, perlu 2 vector input
#   - apply_transformations, run_stress_test, stability_score
#                                      -> Robustness Stress-Test, route terpisah
#   - concatenate_features, fit_scaler_multi_sample
#                                      -> Concat Playground, perlu banyak hasil fitur sbg input
#   - get_stretch_feature_status       -> info fallback DINOv3/CLIP/LightGlue, bukan ekstraksi

LOCAL_FEATURE_FUNCS = {"extract_orb": extract_orb, "extract_sift": extract_sift}
MATCHING_FUNCS = {
    "euclidean": euclidean_distance, "cosine": cosine_similarity,
    "hamming": hamming_distance, "chi_square": chi_square_distance,
    "histogram_intersection": histogram_intersection,
}
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
morphology_toolkit.py
=====================
Toolkit Computer Vision: Morfologi Matematika pada citra biner.

Isi:
  1. Dilasi, Erosi, Opening, Closing  (implementasi manual NumPy + OpenCV)
  2. Hit-or-Miss Transform            (manual + OpenCV, kernel 1 / -1 / 0)
  3. Skeletonization                  (Lantuejoul, Zhang-Suen manual,
                                       Lee/skimage, Medial Axis, pruning)
  4. Pipeline lengkap                 (grayscale -> Otsu -> opening -> closing
                                       -> skeleton -> pruning -> titik ujung/cabang)

Konvensi: mask biner uint8 {0, 255}, objek (foreground) = putih.
Jalankan tanpa argumen -> memakai citra sintetis bernoise.

    python morphology_toolkit.py
    python morphology_toolkit.py --input foto.png --se-shape ellipse --se-size 5
    python morphology_toolkit.py --selftest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # aman untuk server/tanpa GUI
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import medial_axis, skeletonize

# ---------------------------------------------------------------------------
# 0. UTILITAS
# ---------------------------------------------------------------------------
_K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
_K3 = np.ones((3, 3), np.uint8)


def to_bool(mask: np.ndarray) -> np.ndarray:
    return mask > 0


def to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(bool).astype(np.uint8)) * 255


def make_se(shape: str = "ellipse", size: int = 3) -> np.ndarray:
    """Structuring element (SE). size harus ganjil agar punya titik pusat."""
    if size % 2 == 0:
        size += 1
    code = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE,
            "cross": cv2.MORPH_CROSS}[shape]
    return cv2.getStructuringElement(code, (size, size))


def load_mask(path: str, invert: bool | None = None, blur: int = 5) -> np.ndarray:
    """Citra apa pun -> grayscale -> Gaussian blur -> Otsu -> mask {0,255}.
    Jika invert=None, polaritas ditebak: objek diasumsikan area minoritas."""
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Tidak bisa membaca citra: {path}")
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur | 1, blur | 1), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert is None:
        invert = (bw > 0).mean() > 0.5
    return cv2.bitwise_not(bw) if invert else bw


def make_demo_image(seed: int = 7) -> np.ndarray:
    """Citra sintetis: kotak berbagai ukuran, lingkaran, jaringan garis
    bercabang + cincin (punya lubang), celah kecil, serta noise garam-merica."""
    rng = np.random.default_rng(seed)
    img = np.zeros((420, 640), np.uint8)
    # kotak: 21x21 (x2), 31x31, persegi panjang 21x41
    cv2.rectangle(img, (30, 30), (50, 50), 255, -1)
    cv2.rectangle(img, (90, 30), (110, 50), 255, -1)
    cv2.rectangle(img, (30, 90), (60, 120), 255, -1)
    cv2.rectangle(img, (90, 90), (130, 110), 255, -1)
    cv2.circle(img, (80, 190), 22, 255, -1)
    # jaringan garis bercabang (seperti pembuluh darah / retakan / jalan)
    cv2.line(img, (200, 60), (330, 130), 255, 9)
    cv2.line(img, (330, 130), (330, 250), 255, 9)
    cv2.line(img, (330, 130), (450, 70), 255, 9)
    cv2.line(img, (330, 250), (250, 330), 255, 9)
    cv2.line(img, (330, 250), (430, 330), 255, 9)
    cv2.line(img, (330, 190), (400, 200), 255, 9)
    # cincin (topologi: satu lubang harus tetap ada pada skeleton)
    cv2.circle(img, (540, 120), 50, 255, 9)
    # garis dengan celah kecil (untuk didemokan oleh closing)
    cv2.line(img, (480, 260), (555, 260), 255, 9)
    cv2.line(img, (560, 260), (620, 260), 255, 9)
    cv2.line(img, (480, 260), (480, 260), 0, 1)
    img[255:266, 556:559] = 0
    # noise garam (titik putih terisolasi) & merica (lubang hitam kecil)
    salt = rng.random(img.shape) < 0.004
    pepper = rng.random(img.shape) < 0.004
    img[salt] = 255
    img[pepper & (img == 255)] = 0
    return img


# ---------------------------------------------------------------------------
# 1. DILASI, EROSI, OPENING, CLOSING  (implementasi manual dari definisi)
# ---------------------------------------------------------------------------
def _stack_shifts(img: np.ndarray, se: np.ndarray, border: bool):
    """Hasilkan salinan citra yang digeser sesuai tiap elemen aktif SE."""
    kh, kw = se.shape
    ph, pw = kh // 2, kw // 2
    h, w = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), constant_values=border)
    for dy, dx in zip(*np.nonzero(se)):
        yield padded[dy:dy + h, dx:dx + w]


def erode_manual(mask: np.ndarray, se: np.ndarray, border: bool = True) -> np.ndarray:
    """A (-) B = {z | B_z subset A}  -> piksel tetap 1 hanya jika SEMUA tetangga
    di bawah SE bernilai 1 (operator AND)."""
    out = np.ones(mask.shape, bool)
    for s in _stack_shifts(to_bool(mask), se, border):
        out &= s
    return to_u8(out)


def dilate_manual(mask: np.ndarray, se: np.ndarray, border: bool = False) -> np.ndarray:
    """A (+) B = {z | (B^)_z ∩ A != empty} -> piksel jadi 1 jika ADA tetangga
    bernilai 1 (operator OR). SE direfleksikan sesuai definisi."""
    out = np.zeros(mask.shape, bool)
    for s in _stack_shifts(to_bool(mask), se[::-1, ::-1], border):
        out |= s
    return to_u8(out)


def opening_manual(mask, se):
    return dilate_manual(erode_manual(mask, se), se)


def closing_manual(mask, se):
    return erode_manual(dilate_manual(mask, se), se)


# Versi OpenCV (cepat; juga bekerja pada citra grayscale)
def erode(mask, se, it=1):
    return cv2.erode(mask, se, iterations=it)


def dilate(mask, se, it=1):
    return cv2.dilate(mask, se, iterations=it)


def opening(mask, se):
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, se)


def closing(mask, se):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, se)


def gradient(mask, se):
    """Dilasi - Erosi = kontur objek."""
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, se)


# ---------------------------------------------------------------------------
# 2. HIT-OR-MISS TRANSFORM
# ---------------------------------------------------------------------------
# Kernel ternary: 1 = harus foreground (hit), -1 = harus background (miss),
# 0 = don't care.   HMT(A) = (A (-) B_hit)  AND  (A^c (-) B_miss)
def hit_or_miss_manual(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    hit_se = (kernel == 1).astype(np.uint8)
    miss_se = (kernel == -1).astype(np.uint8)
    if np.any(hit_se & miss_se):
        raise ValueError("SE hit dan miss harus saling lepas (disjoint).")
    fg = to_bool(mask)
    # border=False: piksel di luar citra dianggap TIDAK cocok (aman di tepi)
    hit = to_bool(erode_manual(to_u8(fg), hit_se, border=False))
    miss = to_bool(erode_manual(to_u8(~fg), miss_se, border=False))
    return to_u8(hit & miss)


def hit_or_miss_cv(mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_HITMISS, kernel.astype(np.int32))


def kernel_isolated_pixel() -> np.ndarray:
    k = -np.ones((3, 3), np.int32)
    k[1, 1] = 1
    return k


def kernels_endpoints() -> list[np.ndarray]:
    """8 template ujung garis (piksel dengan tepat 1 tetangga) pada skeleton."""
    ortho = np.array([[-1, -1, -1], [-1, 1, -1], [-1, 1, -1]], np.int32)
    diag = np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1]], np.int32)
    return [np.rot90(ortho, k) for k in range(4)] + [np.rot90(diag, k) for k in range(4)]


def kernel_exact_square(n: int, tolerant_corners: bool = True) -> np.ndarray:
    """Mendeteksi persegi berukuran TEPAT n x n: inti n x n = hit,
    cincin 1 piksel di sekelilingnya = miss. Keempat sudut inti dibuat
    'don't care' (0) agar tetap cocok walau sudut membulat akibat opening."""
    k = -np.ones((n + 2, n + 2), np.int32)
    k[1:-1, 1:-1] = 1
    if tolerant_corners:
        for y, x in [(1, 1), (1, n), (n, 1), (n, n)]:
            k[y, x] = 0
    return k


def detect_endpoints_hmt(skel: np.ndarray) -> np.ndarray:
    out = np.zeros_like(skel)
    for k in kernels_endpoints():
        out |= hit_or_miss_cv(skel, k)
    return out


# ---------------------------------------------------------------------------
# 3. SKELETONIZATION
# ---------------------------------------------------------------------------
def morphological_skeleton(mask: np.ndarray, se: np.ndarray | None = None):
    """Skeleton morfologis Lantuejoul:  S(A) = U_k [ (A (-) kB) - (A (-) kB) o B ].
    Bisa direkonstruksi sempurna, tapi hasilnya belum tentu 1-piksel/terhubung."""
    se = make_se("cross", 3) if se is None else se
    eroded, skel, layers = mask.copy(), np.zeros_like(mask), []
    while eroded.any():
        layer = cv2.subtract(eroded, opening(eroded, se))
        skel |= layer
        layers.append(layer)
        eroded = erode(eroded, se)
    return skel, layers


def reconstruct_from_layers(layers, se: np.ndarray | None = None) -> np.ndarray:
    """A = U_k [ S_k (+) kB ]  (bukti skeleton morfologis bersifat lossless)."""
    se = make_se("cross", 3) if se is None else se
    out = np.zeros_like(layers[0])
    for k, layer in enumerate(layers):
        out |= dilate(layer, se, it=k) if k else layer
    return out


def zhang_suen(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning (2 sub-iterasi), vektorisasi NumPy.
    Tetangga: P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW."""
    img = to_bool(mask).astype(np.int16)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(nb)                                         # jumlah tetangga
            seq = nb + [p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16)
                    for i in range(8))                          # transisi 0->1
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if remove.any():
                img[remove] = 0
                changed = True
    return to_u8(img)


def skeleton_lee(mask: np.ndarray) -> np.ndarray:
    return to_u8(skeletonize(to_bool(mask), method="lee"))


def skeleton_medial_axis(mask: np.ndarray):
    """Medial axis + distance transform: skeleton sekaligus 'tebal' objek."""
    sk, dist = medial_axis(to_bool(mask), return_distance=True)
    return to_u8(sk), dist * sk


def neighbor_count(skel: np.ndarray) -> np.ndarray:
    return cv2.filter2D((skel > 0).astype(np.uint8), -1, _K8,
                        borderType=cv2.BORDER_CONSTANT)


def skeleton_keypoints(skel: np.ndarray):
    """Titik ujung (1 tetangga) dan titik percabangan (>=3 tetangga)."""
    sk, nc = to_bool(skel), neighbor_count(skel)
    return to_u8(sk & (nc == 1)), to_u8(sk & (nc >= 3))


def count_clusters(mask: np.ndarray) -> int:
    """Jumlah cluster (gabungkan piksel cabang yang berdempetan jadi 1 junction)."""
    merged = cv2.dilate(mask, _K3)
    return cv2.connectedComponents(merged, connectivity=8)[0] - 1


def prune_spurs(skel: np.ndarray, length: int) -> np.ndarray:
    """Pruning: hapus cabang pendek (< length piksel) akibat noise tepi.
    1) cukur ujung 'length' kali  2) tumbuhkan kembali dari ujung yang tersisa
       (dilasi bersyarat pada skeleton asli)  3) gabungkan."""
    sk0 = to_bool(skel)
    x1 = sk0.copy()
    for _ in range(length):
        x1 &= ~(x1 & (neighbor_count(to_u8(x1)) <= 1))
    ends = x1 & (neighbor_count(to_u8(x1)) == 1)
    grown = ends.copy()
    for _ in range(length):
        grown = to_bool(cv2.dilate(to_u8(grown), _K3)) & sk0
    return to_u8(x1 | grown)


# ---------------------------------------------------------------------------
# 4. VISUALISASI & LAPORAN
# ---------------------------------------------------------------------------
def overlay(base: np.ndarray, ends=None, branches=None, grow: int = 2) -> np.ndarray:
    """Gambar RGB: skeleton putih, titik ujung merah, percabangan biru."""
    rgb = np.dstack([base] * 3).copy()
    big = make_se("ellipse", grow * 2 + 1)
    if ends is not None:
        rgb[dilate(ends, big) > 0] = (255, 40, 40)
    if branches is not None:
        rgb[dilate(branches, big) > 0] = (40, 120, 255)
    return rgb


def grid(items, path: Path, cols: int, title: str):
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (name, im) in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255) if im.ndim == 2 else ax.imshow(im)
        ax.set_title(name, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  tersimpan: {path}")


def n_components(mask: np.ndarray) -> int:
    return cv2.connectedComponents(to_u8(mask), connectivity=8)[0] - 1


# ---------------------------------------------------------------------------
# 5. DEMO PER FITUR
# ---------------------------------------------------------------------------
def demo_basic(mask, se, out: Path):
    items = [
        ("Citra biner (noise)", mask),
        (f"Erosi (SE {se.shape[0]}x{se.shape[0]})", erode(mask, se)),
        ("Dilasi", dilate(mask, se)),
        ("Opening = erosi -> dilasi", opening(mask, se)),
        ("Closing = dilasi -> erosi", closing(mask, se)),
        ("Gradien morfologi (kontur)", gradient(mask, se)),
    ]
    grid(items, out / "1_operasi_dasar.png", 3, "Operasi dasar morfologi")


def demo_hmt(raw, clean, out: Path):
    skel = skeleton_lee(clean)
    iso = hit_or_miss_cv(raw, kernel_isolated_pixel())   # noise garam pada citra mentah
    sq = hit_or_miss_cv(clean, kernel_exact_square(21))
    ends = detect_endpoints_hmt(skel)
    big = make_se("ellipse", 9)
    n_sq = count_clusters(sq)
    items = [
        ("Citra mentah (noise)", raw),
        (f"Piksel terisolasi (HMT) = {int((iso > 0).sum())} titik", dilate(iso, big)),
        (f"Persegi tepat 21x21 -> {n_sq} ditemukan", dilate(sq, make_se("ellipse", 15))),
        ("Skeleton (Lee)", skel),
        (f"Ujung garis via HMT ({count_clusters(ends)})",
         overlay(skel, ends=ends, grow=3)),
    ]
    grid(items, out / "2_hit_or_miss.png", 3, "Hit-or-Miss Transform")
    return n_sq


def demo_skeleton(clean, prune_len, out: Path):
    se = make_se("cross", 3)
    sk_m, layers = morphological_skeleton(clean, se)
    recon = reconstruct_from_layers(layers, se)
    sk_zs = zhang_suen(clean)
    sk_lee = skeleton_lee(clean)
    sk_ma, dist = skeleton_medial_axis(clean)
    pruned = prune_spurs(sk_lee, prune_len)
    ends, br = skeleton_keypoints(pruned)
    items = [
        ("Citra bersih", clean),
        ("Skeleton morfologis (Lantuejoul)", sk_m),
        ("Rekonstruksi dari skeleton", recon),
        ("Zhang-Suen (manual)", sk_zs),
        ("Lee / skimage", sk_lee),
        ("Medial axis", sk_ma),
        (f"Pruning {prune_len}px", pruned),
        (f"Ujung (merah)={count_clusters(ends)}, cabang (biru)={count_clusters(br)}",
         overlay(pruned, ends, br, grow=3)),
    ]
    grid(items, out / "3_skeletonization.png", 4, "Skeletonization")
    return {"Lantuejoul": sk_m, "Zhang-Suen": sk_zs, "Lee": sk_lee,
            "Medial axis": sk_ma, "Pruned": pruned}, recon


def pipeline(mask, shape, size, prune_len, out: Path):
    """Pipeline utuh: bersihkan noise -> sambung celah -> skeleton -> analisis."""
    se_small = make_se(shape, 3)
    se_big = make_se(shape, size)
    step1 = opening(mask, se_small)       # buang noise garam (titik terisolasi)
    step2 = closing(step1, se_big)        # tutup lubang/celah kecil
    skel = skeleton_lee(step2)
    pruned = prune_spurs(skel, prune_len)
    ends, br = skeleton_keypoints(pruned)
    grid([
        ("1. Mask Otsu / input", mask),
        ("2. Opening (buang noise)", step1),
        (f"3. Closing SE {size}x{size} (sambung celah)", step2),
        ("4. Skeleton (Lee)", skel),
        (f"5. Pruning {prune_len}px", pruned),
        ("6. Titik ujung & cabang", overlay(pruned, ends, br, grow=3)),
    ], out / "4_pipeline.png", 3, "Pipeline pemrosesan lengkap")
    return step2, pruned


def report(name_masks: dict):
    print(f"\n{'Citra/metode':<28}{'piksel':>10}{'komponen':>10}")
    print("-" * 48)
    for n, m in name_masks.items():
        print(f"{n:<28}{int((m > 0).sum()):>10}{n_components(m):>10}")


# ---------------------------------------------------------------------------
# 6. SELF-TEST: implementasi manual harus sama dengan OpenCV
# ---------------------------------------------------------------------------
def selftest():
    mask = make_demo_image()
    se = make_se("ellipse", 5)                       # SE simetris
    checks = {
        "erosi": (erode_manual(mask, se), erode(mask, se)),
        "dilasi": (dilate_manual(mask, se), dilate(mask, se)),
        "opening": (opening_manual(mask, se), opening(mask, se)),
        "closing": (closing_manual(mask, se), closing(mask, se)),
    }
    for name, (a, b) in checks.items():
        assert np.array_equal(a, b), f"{name} berbeda!"
        print(f"  [OK] {name:<8} manual == OpenCV")
    k = kernel_exact_square(21)
    a, b = hit_or_miss_manual(mask, k), hit_or_miss_cv(mask, k)
    assert np.array_equal(a[12:-12, 12:-12], b[12:-12, 12:-12]), "HMT berbeda!"
    print("  [OK] hit-or-miss manual == OpenCV (area dalam)")
    zs = zhang_suen(opening(mask, make_se("ellipse", 3)))
    assert zs.any() and set(np.unique(zs)) <= {0, 255}
    print("  [OK] Zhang-Suen menghasilkan skeleton valid")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Toolkit morfologi matematika")
    ap.add_argument("--input", help="path citra (default: citra sintetis)")
    ap.add_argument("--output", default="hasil_morfologi")
    ap.add_argument("--se-shape", choices=["rect", "ellipse", "cross"], default="ellipse")
    ap.add_argument("--se-size", type=int, default=7, help="ukuran SE closing (ganjil)")
    ap.add_argument("--prune", type=int, default=10, help="panjang spur yang dibuang")
    ap.add_argument("--invert", action="store_true", help="paksa objek = piksel gelap")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("Self-test:")
        selftest()
        return

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    mask = (load_mask(args.input, invert=True if args.invert else None)
            if args.input else make_demo_image())
    cv2.imwrite(str(out / "0_input_mask.png"), mask)

    print("1) Operasi dasar")
    demo_basic(mask, make_se(args.se_shape, 3), out)
    print("2) Pipeline (pembersihan -> skeleton)")
    clean, pruned = pipeline(mask, args.se_shape, args.se_size, args.prune, out)
    cv2.imwrite(str(out / "clean_mask.png"), clean)
    print("3) Hit-or-miss")
    n_sq = demo_hmt(mask, clean, out)
    print("4) Skeletonization")
    skels, recon = demo_skeleton(clean, args.prune, out)

    report({"Mask asli (noise)": mask, "Setelah opening+closing": clean,
            **skels, "Rekonstruksi Lantuejoul": recon})
    print(f"\nPersegi 21x21 terdeteksi oleh HMT: {n_sq}")
    print(f"Rekonstruksi Lantuejoul identik dengan citra bersih: "
          f"{np.array_equal(recon, clean)}")


if __name__ == "__main__":
    main()

"""
processing/morphology_service.py
================================
Service layer untuk fitur Morfologi Matematika TaksuCitra,
mengadaptasi logika dan fungsi dari processing/morphology_toolkit.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from processing import morphology_toolkit as mt


def prepare_binary_mask(img_bgr: np.ndarray, invert_mode: str = "auto", blur: int = 5) -> tuple[np.ndarray, bool]:
    """
    Praproses biner: grayscale -> blur ringan -> Otsu -> penyesuaian polaritas.
    invert_mode: 'auto' (objek = minoritas), 'invert' (paksa balik), 'normal' (tanpa balik).
    Mengembalikan (mask_uint8, inverted_bool).
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    b_size = max(1, int(blur))
    if b_size > 1:
        if b_size % 2 == 0:
            b_size += 1
        gray = cv2.GaussianBlur(gray, (b_size, b_size), 0)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if invert_mode == "invert" or invert_mode is True:
        inverted = True
    elif invert_mode == "normal" or invert_mode is False:
        inverted = False
    else:  # 'auto' / None
        # Penebakan otomatis: foreground objek diasumsikan area minoritas
        inverted = bool((bw > 0).mean() > 0.5)

    mask = cv2.bitwise_not(bw) if inverted else bw
    return mask, inverted


def process_dilation(
    img_bgr: np.ndarray,
    se_shape: str = "ellipse",
    se_size: int = 5,
    iterations: int = 1,
    use_manual: bool = False,
    invert_mode: str = "auto",
    blur: int = 5,
) -> dict:
    """Dilasi citra biner."""
    se_size = max(3, min(31, int(se_size)))
    if se_size % 2 == 0:
        se_size += 1
    iterations = max(1, min(10, int(iterations)))

    mask, inverted = prepare_binary_mask(img_bgr, invert_mode=invert_mode, blur=blur)
    se = mt.make_se(se_shape, se_size)

    if use_manual:
        # manual loop untuk iterasi
        res = mask.copy()
        for _ in range(iterations):
            res = mt.dilate_manual(res, se)
    else:
        res = mt.dilate(mask, se, it=iterations)

    active_before = int((mask > 0).sum())
    active_after = int((res > 0).sum())

    return {
        "binary": mask,
        "result": res,
        "inverted": inverted,
        "se_shape": se_shape,
        "se_size": se_size,
        "iterations": iterations,
        "active_before": active_before,
        "active_after": active_after,
        "change_pct": round(((active_after - active_before) / max(1, active_before)) * 100, 2),
    }


def process_erosion(
    img_bgr: np.ndarray,
    se_shape: str = "ellipse",
    se_size: int = 5,
    iterations: int = 1,
    use_manual: bool = False,
    invert_mode: str = "auto",
    blur: int = 5,
) -> dict:
    """Erosi citra biner."""
    se_size = max(3, min(31, int(se_size)))
    if se_size % 2 == 0:
        se_size += 1
    iterations = max(1, min(10, int(iterations)))

    mask, inverted = prepare_binary_mask(img_bgr, invert_mode=invert_mode, blur=blur)
    se = mt.make_se(se_shape, se_size)

    if use_manual:
        res = mask.copy()
        for _ in range(iterations):
            res = mt.erode_manual(res, se)
    else:
        res = mt.erode(mask, se, it=iterations)

    active_before = int((mask > 0).sum())
    active_after = int((res > 0).sum())

    return {
        "binary": mask,
        "result": res,
        "inverted": inverted,
        "se_shape": se_shape,
        "se_size": se_size,
        "iterations": iterations,
        "active_before": active_before,
        "active_after": active_after,
        "change_pct": round(((active_after - active_before) / max(1, active_before)) * 100, 2),
    }


def process_opening(
    img_bgr: np.ndarray,
    se_shape: str = "ellipse",
    se_size: int = 5,
    use_manual: bool = False,
    invert_mode: str = "auto",
    blur: int = 5,
) -> dict:
    """Opening citra biner: erosi diikuti dilasi."""
    se_size = max(3, min(31, int(se_size)))
    if se_size % 2 == 0:
        se_size += 1

    mask, inverted = prepare_binary_mask(img_bgr, invert_mode=invert_mode, blur=blur)
    se = mt.make_se(se_shape, se_size)

    if use_manual:
        res = mt.opening_manual(mask, se)
    else:
        res = mt.opening(mask, se)

    # Objek/noise yang tereliminasi (mask - res)
    diff = cv2.subtract(mask, res)

    return {
        "binary": mask,
        "result": res,
        "diff": diff,
        "inverted": inverted,
        "se_shape": se_shape,
        "se_size": se_size,
        "removed_pixels": int((diff > 0).sum()),
    }


def process_closing(
    img_bgr: np.ndarray,
    se_shape: str = "ellipse",
    se_size: int = 5,
    use_manual: bool = False,
    invert_mode: str = "auto",
    blur: int = 5,
) -> dict:
    """Closing citra biner: dilasi diikuti erosi."""
    se_size = max(3, min(31, int(se_size)))
    if se_size % 2 == 0:
        se_size += 1

    mask, inverted = prepare_binary_mask(img_bgr, invert_mode=invert_mode, blur=blur)
    se = mt.make_se(se_shape, se_size)

    if use_manual:
        res = mt.closing_manual(mask, se)
    else:
        res = mt.closing(mask, se)

    # Celah/lubang yang terisi (res - mask)
    filled = cv2.subtract(res, mask)

    return {
        "binary": mask,
        "result": res,
        "filled": filled,
        "inverted": inverted,
        "se_shape": se_shape,
        "se_size": se_size,
        "filled_pixels": int((filled > 0).sum()),
    }


def process_hit_or_miss(
    img_bgr: np.ndarray,
    mode: str = "isolated_pixel",
    square_size: int = 21,
    custom_kernel: list[list[int]] | None = None,
    use_manual: bool = False,
    invert_mode: str = "auto",
    blur: int = 5,
) -> dict:
    """
    Hit-or-Miss Transform.
    mode: 'isolated_pixel', 'endpoints', 'exact_square', 'custom'.
    """
    mask, inverted = prepare_binary_mask(img_bgr, invert_mode=invert_mode, blur=blur)

    if mode == "isolated_pixel":
        kernel = mt.kernel_isolated_pixel()
        if use_manual:
            raw_hmt = mt.hit_or_miss_manual(mask, kernel)
        else:
            raw_hmt = mt.hit_or_miss_cv(mask, kernel)
        match_count = int((raw_hmt > 0).sum())

    elif mode == "endpoints":
        # 8 template endpoints pada skeleton atau citra tipis
        raw_hmt = np.zeros_like(mask)
        for k in mt.kernels_endpoints():
            if use_manual:
                raw_hmt |= mt.hit_or_miss_manual(mask, k)
            else:
                raw_hmt |= mt.hit_or_miss_cv(mask, k)
        match_count = mt.count_clusters(raw_hmt)

    elif mode == "exact_square":
        sq_n = max(3, min(51, int(square_size)))
        kernel = mt.kernel_exact_square(sq_n, tolerant_corners=True)
        if use_manual:
            raw_hmt = mt.hit_or_miss_manual(mask, kernel)
        else:
            raw_hmt = mt.hit_or_miss_cv(mask, kernel)
        match_count = mt.count_clusters(raw_hmt)

    elif mode == "custom" and custom_kernel is not None:
        kernel = np.array(custom_kernel, dtype=np.int32)
        if kernel.shape != (3, 3):
            raise ValueError("Custom kernel HMT harus berukuran 3x3")
        if use_manual:
            raw_hmt = mt.hit_or_miss_manual(mask, kernel)
        else:
            raw_hmt = mt.hit_or_miss_cv(mask, kernel)
        match_count = mt.count_clusters(raw_hmt)

    else:
        raise ValueError(f"Mode Hit-or-Miss '{mode}' tidak dikenal")

    # Buat visualisasi overlay di mana lokasi kecocokan disorot (lingkaran merah/emas)
    overlay_img = np.dstack([mask] * 3).copy()
    marker_se = mt.make_se("ellipse", 9)
    dilated_markers = mt.dilate(raw_hmt, marker_se)
    # Warna emas / merah untuk deteksi: BGR=(50, 202, 242)
    overlay_img[dilated_markers > 0] = (50, 202, 242)

    return {
        "binary": mask,
        "hitmiss": raw_hmt,
        "overlay": overlay_img,
        "inverted": inverted,
        "mode": mode,
        "match_count": match_count,
    }


def process_skeleton(
    img_bgr: np.ndarray,
    method: str = "lee",
    prune_len: int = 10,
    show_endpoints: bool = True,
    show_branches: bool = True,
    invert_mode: str = "auto",
    blur: int = 5,
) -> dict:
    """
    Skeletonization & Topology Analysis.
    method: 'lantuejoul', 'zhang_suen', 'lee', 'medial_axis'.
    """
    mask, inverted = prepare_binary_mask(img_bgr, invert_mode=invert_mode, blur=blur)
    prune_len = max(0, min(30, int(prune_len)))

    recon_identical = None
    dist_map_vis = None

    if method == "lantuejoul":
        se = mt.make_se("cross", 3)
        raw_skel, layers = mt.morphological_skeleton(mask, se)
        recon = mt.reconstruct_from_layers(layers, se)
        recon_identical = bool(np.array_equal(recon, mask))
    elif method == "zhang_suen":
        # Bersihkan noise kecil terlebih dahulu agar performa optimal
        clean = mt.opening(mask, mt.make_se("cross", 3))
        raw_skel = mt.zhang_suen(clean)
    elif method == "medial_axis":
        raw_skel, dist = mt.skeleton_medial_axis(mask)
        # Normalisasi distance map untuk visualisasi jika diperlukan
        if dist.max() > 0:
            dist_map_vis = np.clip(dist * 255.0 / dist.max(), 0, 255).astype(np.uint8)
    else:  # 'lee' / default
        raw_skel = mt.skeleton_lee(mask)

    if prune_len > 0:
        skel = mt.prune_spurs(raw_skel, prune_len)
    else:
        skel = raw_skel

    ends, br = mt.skeleton_keypoints(skel)
    n_ends = mt.count_clusters(ends)
    n_branches = mt.count_clusters(br)
    n_components = mt.n_components(skel)

    # Buat visualisasi berwarna: skeleton putih, ujung merah, cabang biru
    vis_ends = ends if show_endpoints else None
    vis_branches = br if show_branches else None
    overlay_rgb = mt.overlay(skel, ends=vis_ends, branches=vis_branches, grow=3)
    # mt.overlay menghasilkan array RGB, konversi ke BGR untuk standard OpenCV
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

    return {
        "binary": mask,
        "skeleton": skel,
        "overlay": overlay_bgr,
        "inverted": inverted,
        "method": method,
        "prune_len": prune_len,
        "endpoints_count": n_ends,
        "branches_count": n_branches,
        "components_count": n_components,
        "skeleton_pixels": int((skel > 0).sum()),
        "recon_identical": recon_identical,
    }

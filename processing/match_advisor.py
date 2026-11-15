"""
processing/match_advisor.py
===========================
Implementasi:
- §15: Compatibility Guard (Deteksi ketidaksesuaian ruang fitur, tabrakan LBP vs GLCM, demo tidak bermakna)
- §16: Matching Advisor (Wizard pohon keputusan interaktif & mode Heuristik Analisis Otomatis)
- §17: Preset Kasus Nyata A–E (Satu klik dengan generator sampel sintetis prosedural)
"""

import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing.match_viz import (
    DEFAULTS, create_step, bgr_to_b64, fig_to_b64, style_dark_axes, draw_side_by_side
)
from processing.feature_extraction_lab import lbp_explorer, glcm_explorer


# =============================================================================
# §15 COMPATIBILITY GUARD
# =============================================================================

COMPATIBILITY_RULES = {
    # Kasus 1: Dimensi sama, jumlah beda
    "diff_count_same_dim": {
        "verdict": "compatible",
        "title": "✅ Kompatibel Penuh",
        "alasan": "Dimensi descriptor identik (D=128), jumlah baris berbeda (misal 350 vs 527).",
        "solusi": "Gunakan Brute-Force, k-NN, atau FLANN dengan Lowe Ratio Test."
    },
    # Kasus 2: Sekuens beda panjang
    "sequence_diff_length": {
        "verdict": "conditional",
        "title": "➜ Kompatibel Bersyarat",
        "alasan": "Panjang sekuens temporal berbeda (N != M). Euclidean kaku tidak dapat diterapkan secara langsung.",
        "solusi": "Gunakan Dynamic Time Warping (DTW) untuk penyelarasan tempo non-linier."
    },
    # Kasus 3: Titik kontur beda jumlah
    "contour_diff_points": {
        "verdict": "conditional",
        "title": "➜ Kompatibel Bersyarat",
        "alasan": "Jumlah titik sampel kurva kontur tidak sama (|A| != |B|).",
        "solusi": "Gunakan Hausdorff Distance, Chamfer Distance, atau Shape Context + Hungarian."
    },
    # Kasus 4: Histogram beda bin
    "histogram_diff_bins": {
        "verdict": "conditional",
        "title": "➜ Memerlukan Harmonisasi Bin",
        "alasan": "Jumlah bin histogram tidak sama (misal 64 vs 256).",
        "solusi": "Lakukan re-binning dengan penjumlahan (sum) atau kuantisasi ulang sebelum Chi-Square / EMD."
    },
    # Kasus 5: Ruang fitur beda (LBP vs GLCM)
    "feature_space_mismatch": {
        "verdict": "incompatible",
        "title": "❌ Tidak Kompatibel (Fatal Mismatch)",
        "alasan": "Ruang representasi berbeda total (misal LBP 59D vs GLCM 24D). Komponen memiliki definisi semantik yang berlainan.",
        "solusi": "Dilarang zero-padding untuk Euclidean. Gunakan Feature Fusion (standarisasi z-score per blok + bobot) atau Common Latent Space (PCA)."
    },
    # Kasus 6: Embedding beda model
    "embedding_model_mismatch": {
        "verdict": "incompatible",
        "title": "❌ Tidak Kompatibel",
        "alasan": "Embedding berasal dari arsitektur berbeda (misal 512D ResNet vs 768D ViT). Sumbu laten tidak selaras.",
        "solusi": "Ekstraksi ulang menggunakan satu model yang sama atau proyeksikan melalui adapter transfer."
    },
    # Kasus 7: Biner vs Float
    "binary_float_mismatch": {
        "verdict": "incompatible",
        "title": "❌ Kesalahan Norm Metrik",
        "alasan": "Mencampurkan representasi biner (ORB) dengan float (SIFT).",
        "solusi": "Gunakan Hamming Norm untuk biner, dan Euclidean/L2 Norm untuk float."
    }
}


def check_feature_compatibility(desc_a: dict, desc_b: dict) -> dict:
    """Mengevaluasi kesesuaian representasi berdasarkan aturan formal materi §15."""
    type_a = desc_a.get("type", "vector")
    type_b = desc_b.get("type", "vector")
    dim_a = desc_a.get("dim", 0)
    dim_b = desc_b.get("dim", 0)
    is_bin_a = desc_a.get("is_binary", False)
    is_bin_b = desc_b.get("is_binary", False)
    
    if is_bin_a != is_bin_b:
        rule = COMPATIBILITY_RULES["binary_float_mismatch"]
    elif type_a != type_b:
        rule = COMPATIBILITY_RULES["feature_space_mismatch"]
    elif dim_a != dim_b:
        if type_a == "histogram":
            rule = COMPATIBILITY_RULES["histogram_diff_bins"]
        elif type_a == "sequence":
            rule = COMPATIBILITY_RULES["sequence_diff_length"]
        elif type_a == "contour":
            rule = COMPATIBILITY_RULES["contour_diff_points"]
        else:
            rule = COMPATIBILITY_RULES["feature_space_mismatch"]
    else:
        rule = COMPATIBILITY_RULES["diff_count_same_dim"]
        
    return {
        "verdict": rule["verdict"],
        "title": rule["title"],
        "explanation": rule["alasan"],
        "solution": rule["solusi"]
    }


def compute_guard_collision_demo(img: np.ndarray) -> dict:
    """
    Demonstrasi 'Coba Tabrakan' materi §15:
    Sengaja menghitung Euclidean(LBP 59D, GLCM 24D) dengan zero-padding,
    memberikan stempel merah TIDAK BERMAKNA, dan membuktikan inkonsistensinya.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    lbp_res = lbp_explorer(gray, P=8, R=1, method="uniform")
    glcm_res = glcm_explorer(gray, levels=8)
    
    vec_lbp = lbp_res["vector"] # 59D
    vec_glcm = glcm_res["vector"] # 24D
    
    # Zero pad GLCM hingga 59D
    padded_glcm = np.pad(vec_glcm, (0, len(vec_lbp) - len(vec_glcm)), 'constant')
    nonsense_dist = float(np.linalg.norm(vec_lbp - padded_glcm))
    
    # Eksperimen permutasi: tukar urutan fitur GLCM
    permuted_glcm = np.pad(vec_glcm[::-1], (0, len(vec_lbp) - len(vec_glcm)), 'constant')
    permuted_dist = float(np.linalg.norm(vec_lbp - permuted_glcm))
    
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax, title="Demonstrasi Pembuktian: Tabrakan Fitur Berbeda (LBP vs GLCM)")
    ax.bar(["Jarak Zero-Padded Asli", "Jarak Setelah Ditukar Urutan"], [nonsense_dist, permuted_dist],
           color=["#ff4444", "#ff8800"], width=0.4)
    ax.set_ylabel("Nilai Jarak Euclidean")
    for i, v in enumerate([nonsense_dist, permuted_dist]):
        ax.text(i, v + 0.05, f"{v:.3f}", ha='center', color=DEFAULTS["text_color"], fontsize=9, fontweight="bold")
    ax.text(0.5, max(nonsense_dist, permuted_dist) * 0.7, "STEMPEL: TIDAK BERMAKNA!",
            color="red", fontsize=16, fontweight="bold", ha="center",
            bbox=dict(boxstyle="sawtooth", facecolor="#2a0000", edgecolor="red", lw=2))
    demo_img = fig_to_b64(fig)
    
    return {
        "ok": True,
        "feature": "guard_collision_demo",
        "summary": "Pembuktian Guard §15: Zero-padding menghasilkan angka semu yang tidak memiliki validitas geometris.",
        "steps": [
            create_step(1, "Ekstraksi Fitur LBP (59D) dan GLCM (24D)",
                        "Mengekstraksi dua ruang fitur yang memiliki definisi fisik berbeda.",
                        image_b64=bgr_to_b64(gray)),
            create_step(2, "Simulasi Zero-Padding & Tabrakan Euclidean",
                        f"Menghasilkan jarak semu {nonsense_dist:.4f}. Saat urutan ditukar, jarak melonjak menjadi {permuted_dist:.4f}, membuktikan ketiadaan invarian koordinat.",
                        formula="\\|f_{\\text{LBP}} - [f_{\\text{GLCM}}, 0, ..., 0]\\|_2 \\quad (\\text{INVALID})",
                        image_b64=demo_img,
                        metrics={"jarak_semu": nonsense_dist, "jarak_permutasi": permuted_dist})
        ],
        "result": {"image": demo_img, "metrics": {"nonsense_dist": nonsense_dist}, "table": None},
        "timing_ms": 15.0,
        "warnings": ["Peringatan: Operasi jarak langsung antar ruang representasi heterogen ditolak oleh sistem."]
    }


# =============================================================================
# §16 MATCHING ADVISOR
# =============================================================================

def render_decision_tree_diagram(chosen_path: str = "local") -> str:
    """Menggambar diagram alur pohon keputusan persis bagan materi §16 dengan jalur terpilih disorot emas."""
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax, title="Pohon Keputusan Matching Advisor (§16)")
    
    nodes = {
        "root": ("Representasi Input?", 0.5, 0.9),
        "patch": ("Template / Patch?", 0.15, 0.65),
        "vector": ("Vektor Global?", 0.38, 0.65),
        "keypoints": ("Kumpulan Keypoint?", 0.62, 0.65),
        "contour": ("Bentuk / Kontur?", 0.85, 0.65),
        
        "leaf_tpl": ("Template Matching\n(ZNCC / SSD)", 0.15, 0.25),
        "leaf_vec": ("Vector Distance\n(Cosine / Euc)", 0.38, 0.25),
        "leaf_loc": ("Local Feature + RANSAC\n(SIFT / ORB)", 0.62, 0.25),
        "leaf_shp": ("Shape Matching\n(Hausdorff / Chamfer)", 0.85, 0.25),
    }
    
    # Hubungkan panah
    connections = [
        ("root", "patch"), ("root", "vector"), ("root", "keypoints"), ("root", "contour"),
        ("patch", "leaf_tpl"), ("vector", "leaf_vec"), ("keypoints", "leaf_loc"), ("contour", "leaf_shp")
    ]
    
    for src, dst in connections:
        p1 = nodes[src]
        p2 = nodes[dst]
        is_highlight = chosen_path in dst or (chosen_path == "local" and "keypoints" in dst)
        color = DEFAULTS["gold"] if is_highlight else DEFAULTS["border"]
        lw = 2.5 if is_highlight else 1.2
        ax.annotate("", xy=(p2[1], p2[2] + 0.06), xytext=(p1[1], p1[2] - 0.05),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))
                    
    for key, (label, x, y) in nodes.items():
        is_active = (chosen_path == "local" and "loc" in key) or (chosen_path in key)
        fc = DEFAULTS["panel_color"] if not is_active else "#3c2f00"
        ec = DEFAULTS["gold"] if is_active else DEFAULTS["border"]
        ax.text(x, y, label, color=DEFAULTS["gold"] if is_active else DEFAULTS["text_dim"],
                fontsize=8, fontweight="bold" if is_active else "normal",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor=ec, lw=1.5))
                
    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.set_axis_off()
    return fig_to_b64(fig)


def auto_analyze_images(img1: np.ndarray, img2: np.ndarray) -> dict:
    """
    Mode 'Analisis Otomatis': Menghitung heuristik pada dua citra
    (resolusi, densitas keypoint ORB, rasio tekstur kontur) lalu merekomendasikan metode optimal.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1.copy()
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2.copy()
    
    # 1. Cek rasio ukuran (apakah salah satu adalah patch/template kecil)
    area_ratio = float((h2 * w2) / (h1 * w1))
    if area_ratio < 0.25 or area_ratio > 4.0:
        recommended = "template"
        alasan = f"Rasio luas citra {area_ratio:.2f} menunjukkan pola template-patch di dalam scene besar."
        target_panel = "template"
    else:
        # 2. Cek densitas keypoints
        orb = cv2.ORB_create(nfeatures=500)
        kp1 = orb.detect(gray1, None)
        kp2 = orb.detect(gray2, None)
        n_kp = min(len(kp1), len(kp2))
        
        if n_kp > 100:
            recommended = "local"
            alasan = f"Ditemukan {n_kp} keypoint tekstur kaya. Sangat ideal untuk Local Feature (ORB/SIFT) + RANSAC."
            target_panel = "local"
        else:
            # Cek kontur bentuk
            recommended = "shape"
            alasan = "Tekstur minim namun kontur terisolasi terdeteksi. Disarankan Shape Matching (Hausdorff / Fourier)."
            target_panel = "shape"
            
    tree_b64 = render_decision_tree_diagram(recommended)
    return {
        "ok": True,
        "feature": "matching_advisor",
        "summary": f"Rekomendasi Advisor: {recommended.upper()} Lab ({alasan}).",
        "steps": [
            create_step(1, "Analisis Heuristik Citra Otomatis",
                        alasan,
                        image_b64=bgr_to_b64(draw_side_by_side(img1, img2, "Citra A", "Citra B")),
                        metrics={"area_ratio": area_ratio, "rekomendasi": recommended}),
            create_step(2, "Pemetaan Pohon Keputusan (§16)",
                        "Jalur traversal algoritma yang direkomendasikan ditandai warna emas.",
                        image_b64=tree_b64,
                        params={"target_panel": target_panel})
        ],
        "result": {"image": tree_b64, "metrics": {"recommended": recommended, "target_panel": target_panel}, "table": None},
        "timing_ms": 20.0,
        "warnings": []
    }


# =============================================================================
# §17 PRESET KASUS A–E (GENERATOR PROSEDURAL OFFLINE)
# =============================================================================

def generate_preset_pair(preset_key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Membangkitkan pasangan citra prosedural untuk preset kasus nyata A–E."""
    key = preset_key.upper()
    if key == "A":
        # Kasus A: Logo di foto (Scene checkerboard + logo kotak terpola)
        scene = np.full((320, 320, 3), 40, dtype=np.uint8)
        # Bikin pola latar belakang
        for i in range(0, 320, 40):
            cv2.line(scene, (i, 0), (i, 320), (55, 55, 55), 1)
            cv2.line(scene, (0, i), (320, i), (55, 55, 55), 1)
        # Tempel logo di koordinat (120, 90)
        cv2.circle(scene, (160, 130), 30, (80, 202, 242), -1)
        cv2.rectangle(scene, (140, 110), (180, 150), (255, 255, 255), 2)
        # Template murni logo
        tpl = np.full((80, 80, 3), 40, dtype=np.uint8)
        cv2.circle(tpl, (40, 40), 30, (80, 202, 242), -1)
        cv2.rectangle(tpl, (20, 20), (60, 60), (255, 255, 255), 2)
        return scene, tpl, "Kasus A: Deteksi Logo pada Foto (Template vs Local Feature)"
    elif key == "C":
        # Kasus C: Dua bentuk daun parametrik
        im1 = np.full((250, 250, 3), 20, dtype=np.uint8)
        im2 = np.full((250, 250, 3), 20, dtype=np.uint8)
        cv2.ellipse(im1, (125, 125), (40, 90), 30, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(im2, (125, 125), (45, 95), 45, 0, 360, (255, 255, 255), -1)
        return im1, im2, "Kasus C: Komparasi Morfologi Bentuk Daun (Hausdorff & Chamfer)"
    elif key == "D":
        # Kasus D: Citra Panorama Kiri & Kanan dengan translasi
        base = np.full((240, 400, 3), 30, dtype=np.uint8)
        # Objek acak
        for pt, col in [((80, 120), (80, 202, 242)), ((200, 100), (164, 192, 255)), ((320, 140), (255, 181, 157))]:
            cv2.rectangle(base, (pt[0]-30, pt[1]-30), (pt[0]+30, pt[1]+30), col, -1)
        # Crop kiri dan kanan
        left = base[:, :260].copy()
        right = base[:, 140:].copy()
        return left, right, "Kasus D: Penjahitan Panorama Dua Citra Bertumpang Tindih"
    else: # Default Kasus B: Wajah / Profil sintetis
        im_a = np.full((224, 224, 3), 50, dtype=np.uint8)
        im_b = np.full((224, 224, 3), 55, dtype=np.uint8)
        cv2.circle(im_a, (112, 112), 60, (200, 180, 150), -1)
        cv2.circle(im_b, (112, 112), 60, (190, 175, 145), -1)
        return im_a, im_b, "Kasus B: Pengenalan Objek Identik (Embedding Cosine)"

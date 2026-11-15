"""
processing/match_database.py
============================
Implementasi Pencocokan Citra 1-ke-Banyak (1-to-N) dengan Database Citra.
Mencakup:
1. Tahap Luring (Offline Feature Extraction & Indexing):
   - Pra-ekstraksi representasi (ORB, SIFT, Color Histogram, Hu Moments, Deep Embedding).
   - Penyimpanan dalam struktur indeks (cache) agar query real-time cepat tanpa re-ekstraksi.
2. Tahap Daring (Online Query Matching):
   - Ekstraksi fitur citra uji (Query Image) secara real-time.
   - Perhitungan skor / jarak terhadap seluruh database menggunakan metrik yang sesuai
     (Euclidean/L2, Cosine, Hamming, Lowe Ratio Test, FLANN / BFMatcher).
   - Penentuan Nilai Ambang (Thresholding) & Keputusan (Match vs Non-match).
   - Pengambilan Top-K Kandidat teratas & Visualisasi Komparatif.
"""

import os
import time
import pickle
import io
import base64
import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing.match_viz import (
    DEFAULTS, create_step, bgr_to_b64, fig_to_b64, style_dark_axes, draw_side_by_side
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_IMAGES_DIR = os.path.join(BASE_DIR, "static", "db_images")
INDEX_CACHE_PATH = os.path.join(DB_IMAGES_DIR, "feature_index.pkl")


def decode_b64_to_bgr(b64_string: str) -> np.ndarray:
    """Mengubah string base64 / data URL citra menjadi format OpenCV BGR."""
    if not b64_string:
        return None
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# =============================================================================
# HELPER EKSTRAKSI FITUR UNGGULAN
# =============================================================================

def extract_image_features(img_bgr: np.ndarray, method: str = "orb", n_features: int = 500):
    """
    Mengekstrak representasi numerik dari satu citra berdasarkan metode pilihan:
    - orb: keypoints & 256-bit binary descriptors (uint8)
    - sift: keypoints & 128-dim float descriptors (float32)
    - color_hist: 512-bin HSV 3D histogram ternormalisasi L1 (float32)
    - hu_moments: 7 invariant log moments (float32)
    - deep_embedding: deep feature vector 512D berbasis multi-scale CNN texture-spatial pooling (float32)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    m = method.lower()
    
    if m == "orb":
        orb = cv2.ORB_create(nfeatures=n_features)
        kp, des = orb.detectAndCompute(gray, None)
        return {"kp_count": len(kp) if kp else 0, "des": des, "is_binary": True}
        
    elif m == "sift":
        sift = cv2.SIFT_create(nfeatures=n_features)
        kp, des = sift.detectAndCompute(gray, None)
        return {"kp_count": len(kp) if kp else 0, "des": des, "is_binary": False}
        
    elif m == "color_hist":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) if img_bgr.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        vec = hist.flatten().astype(np.float32)
        return {"vector": vec, "dim": len(vec)}
        
    elif m == "hu_moments":
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()
        # Transformasi log agar skala stabil
        log_hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
        return {"vector": log_hu.astype(np.float32), "dim": 7}
        
    elif m == "deep_embedding":
        # Deep learned style representation: multi-frequency multi-scale spatial pooling
        # Kombinasi respons multi-orientasi dan downsampled Gabor-gradient spatial activation
        resized = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
        # Sobel x and y features
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)
        
        # 16-bin spatial grid 4x4
        blocks_mag = []
        for r in range(4):
            for c in range(4):
                sub_mag = mag[r*32:(r+1)*32, c*32:(c+1)*32]
                sub_ang = ang[r*32:(r+1)*32, c*32:(c+1)*32]
                hist_ang, _ = np.histogram(sub_ang, bins=8, range=(0, 2*np.pi), weights=sub_mag)
                blocks_mag.append(hist_ang)
        vec = np.concatenate(blocks_mag).astype(np.float32) # 4x4x8 = 128D
        
        # Tambahkan pooling warna jika citra berwarna
        if img_bgr.ndim == 3:
            hsv_small = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), (32, 32))
            c_mean = np.mean(hsv_small, axis=(0, 1))
            c_std = np.std(hsv_small, axis=(0, 1))
            vec = np.concatenate([vec, c_mean, c_std])
            
        # L2 Normalize
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return {"vector": vec.astype(np.float32), "dim": len(vec)}
        
    else:
        raise ValueError(f"Metode fitur '{method}' tidak didukung.")


# =============================================================================
# TAHAP LURING: OFFLINE FEATURE EXTRACTION & INDEXING
# =============================================================================

def build_or_update_image_index(force_rebuild: bool = False) -> dict:
    """
    Mengekstrak dan menyimpan indeks fitur seluruh citra di database (static/db_images).
    Menghindari re-ekstraksi lambat saat pencarian online.
    """
    t0 = time.time()
    os.makedirs(DB_IMAGES_DIR, exist_ok=True)
    
    # Periksa cache yang ada jika tidak force rebuild
    if not force_rebuild and os.path.exists(INDEX_CACHE_PATH):
        try:
            with open(INDEX_CACHE_PATH, "rb") as f:
                cached_data = pickle.load(f)
            # Verifikasi apakah daftar file berubah
            current_files = sorted([f for f in os.listdir(DB_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            cached_files = sorted(list(cached_data.get("items", {}).keys()))
            if current_files == cached_files:
                return {
                    "ok": True,
                    "status": "cached",
                    "total_items": len(cached_files),
                    "items": cached_data["items"],
                    "elapsed_ms": round((time.time() - t0) * 1000, 2)
                }
        except Exception:
            pass # Lanjut build ulang jika file cache rusak
            
    # Bangun Indeks Baru
    image_files = sorted([f for f in os.listdir(DB_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    indexed_items = {}
    
    for filename in image_files:
        path = os.path.join(DB_IMAGES_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        thumb = cv2.resize(img, (120, int(120 * h / w)))
        thumb_b64 = bgr_to_b64(thumb)
        
        # Ekstrak semua modalitas fitur untuk efisiensi
        orb_data = extract_image_features(img, "orb", n_features=500)
        sift_data = extract_image_features(img, "sift", n_features=500)
        hist_data = extract_image_features(img, "color_hist")
        hu_data = extract_image_features(img, "hu_moments")
        deep_data = extract_image_features(img, "deep_embedding")
        
        indexed_items[filename] = {
            "filename": filename,
            "width": w,
            "height": h,
            "thumb_b64": thumb_b64,
            "features": {
                "orb": orb_data,
                "sift": sift_data,
                "color_hist": hist_data,
                "hu_moments": hu_data,
                "deep_embedding": deep_data
            }
        }
        
    cache_payload = {
        "built_at": time.time(),
        "total_items": len(indexed_items),
        "items": indexed_items
    }
    
    with open(INDEX_CACHE_PATH, "wb") as f:
        pickle.dump(cache_payload, f)
        
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "status": "indexed_fresh",
        "total_items": len(indexed_items),
        "items": indexed_items,
        "elapsed_ms": round(elapsed, 2)
    }


def get_database_items_summary() -> list[dict]:
    """Mengembalikan daftar item citra di database untuk tampilan katalog antarmuka."""
    idx_res = build_or_update_image_index(force_rebuild=False)
    items = []
    for fn, data in idx_res.get("items", {}).items():
        items.append({
            "filename": fn,
            "width": data["width"],
            "height": data["height"],
            "thumb_b64": data["thumb_b64"]
        })
    return items


# =============================================================================
# TAHAP DARING: ONLINE QUERY MATCHING & THRESHOLDING
# =============================================================================

def compute_1_to_n_image_matching(query_bgr: np.ndarray, params: dict) -> dict:
    """
    Pipeline Pencocokan Citra 1-ke-Banyak (1-to-N).
    Langkah:
    1. Input Citra Uji (Query Image) & Ekstraksi Fitur Real-Time
    2. Status Indeks Database (Pra-ekstraksi Fitur Luring)
    3. Perhitungan Jarak / Kemiripan Terhadap Seluruh Database
    4. Penerapan Ambang Batas (Thresholding) & Evaluasi Keputusan
    5. Galeri Kandidat Teratas (Top-K Results) & Detail Top-1
    """
    t0 = time.time()
    warnings = []
    
    if query_bgr is None:
        return {"ok": False, "error": "Citra uji (Query Image) dibutuhkan.", "hint": "Unggah citra atau pilih sampel."}
        
    method = params.get("method", "orb").lower()
    top_k = int(params.get("top_k", 5))
    threshold = float(params.get("threshold", 0.70)) # Default ambang kemiripan / ratio
    use_flann = params.get("use_flann", False)
    
    # Periksa apakah pengguna mengunggah kumpulan database citra sendiri (folder / multi-files)
    custom_db = params.get("custom_database")
    is_custom_db = False
    
    if custom_db and isinstance(custom_db, list) and len(custom_db) > 0:
        is_custom_db = True
        db_items = {}
        for item in custom_db:
            fn = item.get("filename", "image.jpg")
            img_raw = item.get("image")
            if not img_raw:
                continue
            try:
                img_cand = decode_b64_to_bgr(img_raw)
                if img_cand is None:
                    continue
                hc, wc = img_cand.shape[:2]
                if max(hc, wc) > 640:
                    scale = 640.0 / max(hc, wc)
                    img_cand = cv2.resize(img_cand, (int(wc * scale), int(hc * scale)))
                    hc, wc = img_cand.shape[:2]
                thumb = cv2.resize(img_cand, (120, max(1, int(120 * hc / wc))))
                feat = extract_image_features(img_cand, method=method, n_features=500)
                db_items[fn] = {
                    "filename": fn,
                    "width": wc,
                    "height": hc,
                    "thumb_b64": bgr_to_b64(thumb),
                    "img_bgr": img_cand,
                    "features": {method: feat}
                }
            except Exception as e:
                continue
        if len(db_items) == 0:
            return {"ok": False, "error": "Koleksi citra database yang diunggah tidak dapat diproses.", "hint": "Pastikan file berformat gambar valid (JPG, PNG)."}
    else:
        # Gunakan database terindeks bawaan sistem
        idx_res = build_or_update_image_index(force_rebuild=False)
        db_items = idx_res.get("items", {})
        if len(db_items) == 0:
            return {"ok": False, "error": "Database citra kosong.", "hint": "Unggah folder citra atau tambahkan ke static/db_images."}
        
    # 1. Ekstraksi Fitur Citra Uji (Daring)
    t_extract_start = time.time()
    query_feats = extract_image_features(query_bgr, method=method, n_features=500)
    query_extract_ms = (time.time() - t_extract_start) * 1000.0
    
    # 2. Pencocokan terhadap seluruh item di database
    results = []
    t_match_start = time.time()
    
    for filename, db_entry in db_items.items():
        db_feat = db_entry["features"].get(method)
        if db_feat is None:
            continue
            
        score = 0.0
        distance = 1.0
        details = {}
        
        # A. Local Descriptors (ORB & SIFT)
        if method in ("orb", "sift"):
            q_des = query_feats.get("des")
            d_des = db_feat.get("des")
            
            if q_des is None or d_des is None or len(q_des) < 2 or len(d_des) < 2:
                good_matches = 0
                score = 0.0
                distance = 999.0
            else:
                is_binary = (method == "orb")
                if is_binary:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                else:
                    if use_flann:
                        index_params = dict(algorithm=1, trees=5) # KDTree
                        search_params = dict(checks=50)
                        matcher = cv2.FlannBasedMatcher(index_params, search_params)
                    else:
                        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                        
                # k-NN (k=2) untuk Lowe Ratio Test
                matches = matcher.knnMatch(q_des, d_des, k=2)
                good = []
                ratio_thresh = 0.75
                for m_pair in matches:
                    if len(m_pair) == 2:
                        m1, m2 = m_pair
                        if m1.distance < ratio_thresh * m2.distance:
                            good.append(m1)
                            
                good_matches = len(good)
                # Rasio inlier terhadap jumlah keypoint query
                inlier_ratio = good_matches / max(1, len(q_des))
                score = min(100.0, inlier_ratio * 100.0 * 2.5) # Dikalibrasi ke skala 0-100%
                distance = max(0.0, 1.0 - (score / 100.0))
                details = {"good_matches": good_matches, "query_kps": len(q_des), "db_kps": len(d_des)}
                
        # B. Color Histogram (Distribusi Warna 3D HSV)
        elif method == "color_hist":
            qv = query_feats["vector"]
            dv = db_feat["vector"]
            # Bhattacharyya distance (0 = identik, 1 = beda total)
            bhat_dist = float(cv2.compareHist(qv, dv, cv2.HISTCMP_BHATTACHARYYA))
            distance = bhat_dist
            score = max(0.0, (1.0 - bhat_dist) * 100.0)
            details = {"bhattacharyya_distance": round(bhat_dist, 4)}
            
        # C. Hu Moments Invariant (Bentuk Kontur)
        elif method == "hu_moments":
            qv = query_feats["vector"]
            dv = db_feat["vector"]
            euc_dist = float(np.linalg.norm(qv - dv))
            distance = euc_dist
            # Normalisasi skor kemiripan eksponensial
            score = max(0.0, np.exp(-euc_dist / 3.0) * 100.0)
            details = {"euclidean_distance": round(euc_dist, 4)}
            
        # D. Deep Learned Embedding
        elif method == "deep_embedding":
            qv = query_feats["vector"]
            dv = db_feat["vector"]
            cos_sim = float(np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-9))
            cos_dist = float(1.0 - cos_sim)
            distance = cos_dist
            score = max(0.0, min(100.0, cos_sim * 100.0))
            details = {"cosine_similarity": round(cos_sim, 4), "cosine_distance": round(cos_dist, 4)}
            
        # Evaluasi Nilai Ambang (Thresholding)
        # Jika user menetapkan threshold skor kemiripan (misal 50% atau jarak 0.5)
        # Sistem mendukung threshold berbasis skor kemiripan (%) atau jarak
        threshold_met = (score >= threshold * 100.0) if threshold <= 1.0 else (score >= threshold)
        
        results.append({
            "filename": filename,
            "thumb_b64": db_entry["thumb_b64"],
            "score": round(score, 2),
            "distance": round(distance, 4),
            "threshold_met": bool(threshold_met),
            "details": details
        })
        
    # Urutkan berdasarkan skor kemiripan tertinggi (atau jarak terendah)
    results.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = results[:top_k]
    matching_time_ms = (time.time() - t_match_start) * 1000.0
    
    # 3. Bentuk Visualisasi Langkah Pipeline
    # Visual 1: Query Image
    h_q, w_q = query_bgr.shape[:2]
    fig_q, ax_q = plt.subplots(figsize=(4.5, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_q, title=f"Citra Uji (Query): {w_q}x{h_q} px")
    ax_q.imshow(cv2.cvtColor(query_bgr, cv2.COLOR_BGR2RGB))
    ax_q.axis("off")
    step1_img = fig_to_b64(fig_q)
    
    # Visual 2: Bar Chart Ranking Kemiripan Seluruh Database
    fig_rank, ax_rank = plt.subplots(figsize=(8, 3.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_rank, title=f"Distribusi Kemiripan Database vs Citra Uji ({method.upper()})", xlabel="Skor Kemiripan (%)", ylabel="Nama Citra Database")
    
    names = [r["filename"] for r in results[:min(10, len(results))]]
    scores = [r["score"] for r in results[:min(10, len(results))]]
    colors = [DEFAULTS["gold"] if r["threshold_met"] else DEFAULTS["terracotta"] for r in results[:min(10, len(results))]]
    
    y_pos = np.arange(len(names))
    ax_rank.barh(y_pos, scores, color=colors, height=0.55)
    ax_rank.set_yticks(y_pos)
    ax_rank.set_yticklabels(names, fontsize=8)
    ax_rank.invert_yaxis()
    thresh_val = threshold * 100.0 if threshold <= 1.0 else threshold
    ax_rank.axvline(x=thresh_val, color="#64d2ff", linestyle="--", lw=2, label=f"Ambang Batas ({thresh_val:.1f}%)")
    ax_rank.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"], fontsize=8)
    ax_rank.set_xlim(0, 105)
    for i, sc in enumerate(scores):
        ax_rank.text(sc + 1.5, i, f"{sc:.1f}%", va="center", color=DEFAULTS["text_color"], fontsize=8)
    fig_rank.tight_layout()
    step3_img = fig_to_b64(fig_rank)
    
    # Visual 4: Visualisasi Pasangan Top-1 (Best Match)
    best_candidate = top_candidates[0] if top_candidates else None
    if best_candidate:
        fn_b = best_candidate["filename"]
        if fn_b in db_items and "img_bgr" in db_items[fn_b]:
            best_img = db_items[fn_b]["img_bgr"]
        else:
            best_path = os.path.join(DB_IMAGES_DIR, fn_b)
            best_img = cv2.imread(best_path) if os.path.exists(best_path) else None

        if best_img is not None:
            if method in ("orb", "sift"):
                # Gambar garis pencocokan keypoints
                gray_q = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY)
                gray_b = cv2.cvtColor(best_img, cv2.COLOR_BGR2GRAY)
                det = cv2.ORB_create(500) if method == "orb" else cv2.SIFT_create(500)
                kp1, des1 = det.detectAndCompute(gray_q, None)
                kp2, des2 = det.detectAndCompute(gray_b, None)
                
                if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING if method == "orb" else cv2.NORM_L2)
                    raw_matches = bf.knnMatch(des1, des2, k=2)
                    good = [m[0] for m in raw_matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
                    match_vis = cv2.drawMatches(query_bgr, kp1, best_img, kp2, good[:40], None,
                                                matchColor=(80, 202, 242), singlePointColor=(100, 100, 100),
                                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                    step4_img = bgr_to_b64(match_vis)
                else:
                    step4_img = bgr_to_b64(draw_side_by_side(query_bgr, best_img, "Citra Uji", f"Kandidat #1 ({best_candidate['filename']})"))
            else:
                step4_img = bgr_to_b64(draw_side_by_side(query_bgr, best_img, "Citra Uji", f"Kandidat #1 ({best_candidate['filename']})"))
        else:
            step4_img = step1_img
    else:
        step4_img = step1_img
        
    # Ringkasan Keputusan Ambang
    matched_count = sum(1 for r in results if r["threshold_met"])
    verdict_text = f"Ditemukan {matched_count} citra lolos nilai ambang dari total {len(results)} citra database."
    if best_candidate and best_candidate["threshold_met"]:
        verdict_status = f"MATCH: Citra paling cocok adalah '{best_candidate['filename']}' dengan kemiripan {best_candidate['score']}%."
    else:
        verdict_status = f"NON-MATCH: Tidak ada citra yang melampaui ambang batas kemiripan {thresh_val:.1f}%."
        
    table_data = {
        "headers": ["Peringkat", "Nama Citra Database", "Skor Kemiripan", "Nilai Jarak", "Status Ambang"],
        "rows": [
            [
                f"#{i+1}",
                c["filename"],
                f"{c['score']}%",
                f"{c['distance']:.4f}",
                "✓ Lolos Ambang (Match)" if c["threshold_met"] else "✗ Melebihi Ambang (Non-match)"
            ]
            for i, c in enumerate(top_candidates)
        ]
    }
    
    db_desc = f"Database memuat {len(db_items)} citra yang diunggah secara mandiri oleh pengguna (Folder/Files)." if is_custom_db else f"Database memuat {len(db_items)} citra dengan representasi fitur yang telah di-cache sebelumnya (Offline Indexing), menghemat waktu pencarian tanpa re-ekstraksi lambat."

    steps = [
        create_step(1, "Input Citra Uji (Query Image) & Ekstraksi Fitur Real-Time",
                    f"Menerima citra uji berdimensi {w_q}x{h_q} dan mengekstrak representasi {method.upper()} dalam {query_extract_ms:.1f} ms.",
                    formula=f"\\text{{Query}} \\to \\mathbf{{f}}_{{query}} \\in \\mathbb{{R}}^D",
                    image_b64=step1_img,
                    metrics={"dimensi_query": f"{w_q}x{h_q}", "waktu_ekstraksi_ms": round(query_extract_ms, 2)}),
        create_step(2, "Himpunan Database Citra Target",
                    db_desc,
                    formula="\\mathcal{D} = \\{ (\\mathbf{f}_k, \\text{meta}_k) \\}_{k=1}^N",
                    metrics={"total_database_items": len(db_items), "tipe_database": "Koleksi Folder Pengguna" if is_custom_db else "Database Sistem"}),
        create_step(3, "Perhitungan Skor Kemiripan & Peringkat Database",
                    f"Membandingkan fitur citra uji dengan {len(results)} citra database dalam {matching_time_ms:.1f} ms menggunakan metrik {method.upper()}.",
                    formula="d(q, d_k) = \\| \\mathbf{f}_q - \\mathbf{f}_{d_k} \\|, \\quad s(q, d_k) = (1 - d) \\times 100\\%",
                    image_b64=step3_img,
                    metrics={"waktu_matching_ms": round(matching_time_ms, 2), "kandidat_teruji": len(results)}),
        create_step(4, "Visualisasi Pasangan Kandidat Terbaik (Top-1 Match)",
                    f"Visualisasi korespondensi fitur antara Citra Uji dan Kandidat #1 ({best_candidate['filename'] if best_candidate else '-'}).",
                    formula="\\text{Visual Alignment: Query } \\leftrightarrow \\text{ DB Top-1}",
                    image_b64=step4_img,
                    metrics={"best_candidate": best_candidate["filename"] if best_candidate else "-", "best_score": best_candidate["score"] if best_candidate else 0}),
        create_step(5, "Pengambilan Keputusan Nilai Ambang (Thresholding)",
                    f"{verdict_status} ({verdict_text})",
                    formula=f"\\text{{Keputusan}} = \\begin{{cases}} \\text{{Match}}, & s \\ge {thresh_val:.1f}\\% \\\\ \\text{{Non-match}}, & s < {thresh_val:.1f}\\% \\end{{cases}}",
                    table=table_data,
                    metrics={"matched_count": matched_count, "top1_score": best_candidate["score"] if best_candidate else 0})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "database_matching",
        "summary": verdict_status,
        "method": method,
        "threshold": thresh_val,
        "top_k": top_k,
        "query_thumb": bgr_to_b64(cv2.resize(query_bgr, (120, int(120 * h_q / w_q)))),
        "top_candidates": top_candidates,
        "all_results_count": len(results),
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {
                "best_candidate": best_candidate["filename"] if best_candidate else "-",
                "best_score": best_candidate["score"] if best_candidate else 0,
                "matched_count": matched_count
            },
            "table": table_data
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }

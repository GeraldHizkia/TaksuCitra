"""
processing/match_motion.py
==========================
Implementasi §12: Optical Flow Lab.
Mendukung estimasi pergerakan temporal antar dua frame:
1. Input frame t & t+1 (dua citra, atau 1 citra + transformasi sintetis)
2. Lucas-Kanade Sparse:
   - Deteksi sudut kuat (goodFeaturesToTrack)
   - calcOpticalFlowPyrLK
   - Validasi Forward-Backward Error untuk membuang false tracks
3. Farneback Dense:
   - Visualisasi HSV Color Wheel (Hue = arah, Value = magnitudo)
   - Quiver Arrow Subsampled
   - Peta Magnitudo Kecepatan Spasial
4. Warp Kompensasi Gerak frame t -> t+1 & Peta Selisih Error Sebelum vs Sesudah Kompensasi.
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


def compute_optical_flow_lab(img1: np.ndarray, img2: np.ndarray, params: dict) -> dict:
    """
    Eksekusi Pipeline §12 Optical Flow Lab.
    Langkah:
    1. Input Frame t & Frame t+1
    2. Deteksi Titik Pelacakan (Features to Track)
    3. Lucas-Kanade Sparse Flow & Filter Forward-Backward Error
    4. Farneback Dense Flow (HSV Color Wheel Representation)
    5. Quiver Arrow Field & Peta Magnitudo Aliran
    6. Kompensasi Gerak (Motion Compensation Warp) & Reduksi Selisih
    """
    t0 = time.time()
    warnings = []
    
    if img1 is None:
        return {"ok": False, "error": "Frame masukan pertama diperlukan.", "hint": "Unggah dua gambar atau pilih demo sintetis."}
        
    # Jika frame 2 tidak diberikan, buat gerakan sintetis (translasi 8 px + rotasi 1.5 derajat)
    if img2 is None:
        h, w = img1.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), 1.5, 1.0)
        M[0, 2] += 6.0
        M[1, 2] += 3.0
        img2 = cv2.warpAffine(img1, M, (w, h))
        warnings.append("Frame t+1 dibangkitkan otomatis via transformasi sintetis (translasi dx=6, dy=3 dan rotasi 1.5°).")
        
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1.copy()
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2.copy()
    h, w = gray1.shape[:2]
    
    # 1. Visual Input
    step1_img = bgr_to_b64(draw_side_by_side(img1, img2, "Frame t", "Frame t+1"))
    
    # 2. Fitur Sudut Kuat (Shi-Tomasi)
    p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=150, qualityLevel=0.02, minDistance=10)
    if p0 is None or len(p0) < 5:
        return {"ok": False, "error": "Tidak ditemukan titik fitur yang cukup pada Frame t.", "hint": "Gunakan citra dengan tekstur lebih jelas."}
        
    vis_pts = img1.copy()
    for pt in p0:
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(vis_pts, (x, y), 3, (80, 202, 242), -1)
    step2_img = bgr_to_b64(vis_pts)
    
    # 3. Lucas-Kanade Sparse Flow + Forward-Backward Check
    p1, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, winSize=(21, 21), maxLevel=3)
    # Backward check dari p1 ke gray1
    p0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, p1, None, winSize=(21, 21), maxLevel=3)
    
    fb_error = np.linalg.norm(p0 - p0_back, axis=2).flatten()
    valid_mask = (status.flatten() == 1) & (status_back.flatten() == 1) & (fb_error < 1.5)
    
    vis_lk = img2.copy()
    displacements = []
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            x0, y0 = int(p0[i][0][0]), int(p0[i][0][1])
            x1, y1 = int(p1[i][0][0]), int(p1[i][0][1])
            cv2.arrowedLine(vis_lk, (x0, y0), (x1, y1), (0, 230, 118), 2, tipLength=0.3)
            displacements.append(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
        else:
            x0, y0 = int(p0[i][0][0]), int(p0[i][0][1])
            cv2.circle(vis_lk, (x0, y0), 2, (255, 68, 68), -1) # rejected track
            
    step3_img = bgr_to_b64(vis_lk)
    mean_disp = float(np.mean(displacements)) if displacements else 0.0
    
    # 4. Farneback Dense Flow (HSV Color Wheel)
    dense_flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(dense_flow[..., 0], dense_flow[..., 1])
    
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2 # Hue = Arah sudut
    hsv[..., 1] = 255                  # Saturation penuh
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Value = Magnitudo
    dense_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    step4_img = bgr_to_b64(dense_bgr)
    
    # 5. Quiver Arrow Field (Subsampled)
    fig5, ax5 = plt.subplots(figsize=(8, 3.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax5, title="Vektor Medan Aliran Optik Padat (Quiver Field)", xlabel="Koordinat X", ylabel="Koordinat Y")
    step_q = 20
    y_grid, x_grid = np.mgrid[step_q//2:h:step_q, step_q//2:w:step_q]
    u_q = dense_flow[y_grid, x_grid, 0]
    v_q = dense_flow[y_grid, x_grid, 1]
    ax5.quiver(x_grid, y_grid, u_q, -v_q, color=DEFAULTS["gold"], scale=40, width=0.003)
    ax5.set_xlim(0, w)
    ax5.set_ylim(h, 0)
    step5_img = fig_to_b64(fig5)
    
    # 6. Kompensasi Gerak (Motion Compensation) & Evaluasi Residual
    # Warp frame 1 menggunakan flow dense
    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32) + dense_flow[..., 0]
    map_y = np.tile(np.arange(h)[:, None], (1, w)).astype(np.float32) + dense_flow[..., 1]
    warped_frame1 = cv2.remap(gray1, map_x, map_y, cv2.INTER_LINEAR)
    
    diff_before = np.abs(gray2.astype(np.float32) - gray1.astype(np.float32))
    diff_after = np.abs(gray2.astype(np.float32) - warped_frame1.astype(np.float32))
    
    mae_before = float(np.mean(diff_before))
    mae_after = float(np.mean(diff_after))
    
    fig6, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_d1, title=f"Error Sebelum Kompensasi (MAE={mae_before:.2f})")
    ax_d1.imshow(diff_before, cmap="hot", aspect="auto")
    style_dark_axes(ax_d2, title=f"Error Sesudah Kompensasi (MAE={mae_after:.2f})")
    ax_d2.imshow(diff_after, cmap="hot", aspect="auto")
    step6_img = fig_to_b64(fig6)
    
    flow_table = {
        "headers": ["Parameter Aliran Optik", "Nilai Numerik", "Keterangan"],
        "rows": [
            ["Jumlah Fitur Terlacak", f"{np.sum(valid_mask)} titik", f"Lolos validasi siklus forward-backward (FB < 1.5 px)"],
            ["Rata-rata Pergeseran LK", f"{mean_disp:.2f} px", "Vektor translasi lokal titik kunci"],
            ["Magnitudo Aliran Maksimum", f"{float(np.max(mag)):.2f} px/frame", "Kecepatan pergerakan tercepat"],
            ["Reduksi MAE Kompensasi", f"{mae_before:.2f} -> {mae_after:.2f}", f"Penurunan error residual sebesar {((mae_before - mae_after)/max(1e-3, mae_before))*100:.1f}%"]
        ]
    }
    
    steps = [
        create_step(1, "Input Frame Temporal Berurutan (t & t+1)",
                    "Menerima pasangan bingkai citra untuk analisis vektor gerak partikel piksel.",
                    formula="I(x, y, t) \\quad \\text{dan} \\quad I(x, y, t+\\Delta t)",
                    image_b64=step1_img),
        create_step(2, "Deteksi Sudut Pelacakan Shi-Tomasi",
                    f"Mendeteksi {len(p0)} titik sudut dengan eigen-nilai matriks gradien terbesar.",
                    formula="\\min(\\lambda_1, \\lambda_2) > \\tau",
                    image_b64=step2_img),
        create_step(3, "Lucas-Kanade Sparse & Forward-Backward Verification",
                    f"Menghitung perpindahan vektor fitur lokal dengan piramida multi-skala. Membuang {len(p0) - np.sum(valid_mask)} false tracks yang gagal konsistensi siklik.",
                    formula="A^T A \\mathbf{v} = -A^T b, \\quad \\|p_0 - p_{0,\\text{back}}\\| < \\epsilon",
                    image_b64=step3_img,
                    metrics={"fitur_valid": int(np.sum(valid_mask)), "rata_rata_geser": mean_disp}),
        create_step(4, "Peta Aliran Padat Farneback (HSV Color Wheel)",
                    "Representasi medan gerak padat: Hue menunjukkan arah sudut pergerakan (0-360 derajat), Value menunjukkan kecepatan magnitudo.",
                    formula="\\theta = \\arctan(v/u), \\quad r = \\sqrt{u^2 + v^2}",
                    image_b64=step4_img),
        create_step(5, "Subsampling Vektor Medan Quiver",
                    "Memvisualisasikan orientasi dan panjang panah kecepatan spasial di seluruh kanvas citra.",
                    formula="\\mathbf{u}(x, y) = [u(x,y), v(x,y)]^T",
                    image_b64=step5_img),
        create_step(6, "Kompensasi Gerak & Evaluasi Residual Error",
                    f"Melakukan rekonstruksi warp frame t menuju t+1 berdasarkan flow. Mengurangi Mean Absolute Error dari {mae_before:.2f} menjadi {mae_after:.2f}.",
                    formula="\\text{Error} = |I_{t+1}(x,y) - I_t(x+u, y+v)|",
                    image_b64=step6_img,
                    table=flow_table,
                    metrics={"mae_sebelum": mae_before, "mae_sesudah": mae_after})
    ]
    
    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "optical_flow",
        "summary": f"Optical Flow: {np.sum(valid_mask)} titik LK terlacak (rata-rata {mean_disp:.2f} px). Reduksi error kompensasi {mae_before:.1f} -> {mae_after:.1f}.",
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {"valid_tracks": int(np.sum(valid_mask)), "mean_disp": mean_disp, "mae_reduction": mae_before - mae_after},
            "table": flow_table
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }

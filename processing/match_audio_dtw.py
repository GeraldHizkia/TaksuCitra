"""
processing/match_audio_dtw.py
=============================
Implementasi Pencocokan Suara Menggunakan Dynamic Time Warping (DTW):
1. Penentuan Objektif:
   - Verifikasi Penutur (Speaker Verification / Biometrik): Memvalidasi identitas penutur terdaftar.
   - Pencarian Kata Kunci (Keyword Spotting): Mencari kecocokan lafal kata kunci dalam rekaman.
2. Ekstraksi Representasi Temporal Suara:
   - Sinyal suara dipotong menjadi bingkai (frame ~25ms, hop ~10ms).
   - Ekstraksi MFCC (Mel-Frequency Cepstral Coefficients) & Spektrogram STFT.
3. Penyelarasan Temporal Non-Linier (DTW):
   - Matriks Biaya Jarak Lokal (Frame-by-Frame Cost Matrix).
   - Matriks Akumulasi DTW & Penelusuran Jalur Optimal (Warping Path).
   - Penanganan variasi tempo dan durasi pengucapan.
   - Penentuan Nilai Ambang (Thresholding) & Keputusan (Match vs Non-match).
   - Mode 1-vs-1 (Audio A vs Audio B) dan Mode 1-vs-Database (1-to-N).
"""

import os
import io
import time
import base64
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from processing.match_viz import (
    DEFAULTS, create_step, bgr_to_b64, fig_to_b64, style_dark_axes
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_AUDIO_DIR = os.path.join(BASE_DIR, "static", "db_audio")


# =============================================================================
# HELPER AUDIO & EKSTRAKSI MFCC
# =============================================================================

def load_audio_from_b64(b64_string: str) -> tuple[int, np.ndarray]:
    """Membaca audio WAV dari string base64 atau data URL."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    raw_bytes = base64.b64decode(b64_string)
    bio = io.BytesIO(raw_bytes)
    sr, data = wavfile.read(bio)
    
    # Konversi ke mono float32 antara -1.0 dan 1.0
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
        
    # Normalisasi amplitudo puncak
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak
    return sr, data


def load_audio_from_file(filename_or_path: str) -> tuple[int, np.ndarray]:
    """Membaca file audio dari static/db_audio atau path langsung."""
    if os.path.isabs(filename_or_path):
        p = filename_or_path
    else:
        p = os.path.join(DB_AUDIO_DIR, filename_or_path)
    sr, data = wavfile.read(p)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak
    return sr, data


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def compute_mel_filterbank(num_filters=26, nfft=512, sr=16000, low_freq=0, high_freq=None):
    """Membangun matriks filterbank segitiga pada skala Mel."""
    if high_freq is None:
        high_freq = sr / 2.0
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((num_filters, int(nfft // 2 + 1)), dtype=np.float32)
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]

        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])
    return fbank


def extract_audio_features(signal_data: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> dict:
    """
    Mengekstrak representasi temporal dari sinyal suara:
    - Waveform waktu nyata
    - Spektrogram STFT
    - Mel-Spektrogram
    - MFCC (13 koefisien per bingkai waktu ~25ms)
    """
    # Resample sederhana jika sample rate bukan 16kHz
    if sr != 16000 and len(signal_data) > 0:
        new_len = int(len(signal_data) * 16000 / sr)
        signal_data = signal.resample(signal_data, new_len)
        sr = 16000

    # Pre-emphasis filter untuk menyeimbangkan frekuensi tinggi
    pre_emphasized = np.append(signal_data[0], signal_data[1:] - 0.97 * signal_data[:-1])

    # Parameter bingkai (framing)
    frame_size = int(0.025 * sr) # 25ms (400 sampel pada 16kHz)
    frame_stride = int(0.010 * sr) # 10ms hop (160 sampel pada 16kHz)
    sig_len = len(pre_emphasized)

    if sig_len <= frame_size:
        # Padding jika audio sangat pendek
        pad_len = frame_size - sig_len + 160
        pre_emphasized = np.pad(pre_emphasized, (0, pad_len), mode="constant")
        sig_len = len(pre_emphasized)

    num_frames = int(np.ceil(float(np.abs(sig_len - frame_size)) / frame_stride)) + 1
    pad_signal_len = (num_frames - 1) * frame_stride + frame_size
    pad_signal = np.pad(pre_emphasized, (0, max(0, pad_signal_len - sig_len)), mode="constant")

    # Matriks bingkai
    indices = np.tile(np.arange(0, frame_size), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_stride, frame_stride), (frame_size, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Jendela Hamming
    frames *= np.hamming(frame_size)

    # FFT & Power Spectrum
    nfft = 512
    mag_frames = np.absolute(np.fft.rfft(frames, nfft)) # STFT magnitude
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)

    # Mel Filterbank & Log Mel Energy
    fbank = compute_mel_filterbank(num_filters=26, nfft=nfft, sr=sr)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_mel = np.log(filter_banks)

    # DCT (Discrete Cosine Transform) untuk menghasilkan MFCC
    # Implements DCT-II ortogonal
    num_ceps = n_mfcc
    basis = np.empty((num_ceps, 26))
    basis[0, :] = 1.0 / np.sqrt(26)
    for i in range(1, num_ceps):
        basis[i, :] = np.cos(np.pi * i * (np.arange(26) + 0.5) / 26) * np.sqrt(2.0 / 26)
    mfcc = np.dot(log_mel, basis.T) # Ukuran: (num_frames, n_mfcc)

    # Normalisasi liftering cepstral
    cep_lifter = 22
    n = np.arange(num_ceps)
    lift = 1 + (cep_lifter / 2.0) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    # Mean normalization per frame untuk ketahanan rekaman
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    duration_sec = len(signal_data) / float(sr)
    return {
        "num_frames": num_frames,
        "duration_sec": round(duration_sec, 3),
        "sr": sr,
        "waveform": signal_data,
        "spectrogram": mag_frames.T, # (freq_bins, frames)
        "log_mel": log_mel.T,
        "mfcc": mfcc # (num_frames, n_mfcc)
    }


# =============================================================================
# MULTIVARIATE DYNAMIC TIME WARPING (DTW)
# =============================================================================

def compute_audio_dtw(mfcc_a: np.ndarray, mfcc_b: np.ndarray, sakoe_band: int = None, metric: str = "euclidean"):
    """
    Kalkulasi DTW untuk deret fitur multi-dimensi (MFCC A vs MFCC B).
    mfcc_a: matriks N x D
    mfcc_b: matriks M x D
    sakoe_band: batas pita Sakoe-Chiba (|i - j| <= band)
    """
    n, m = len(mfcc_a), len(mfcc_b)
    
    # 1. Matriks Biaya Jarak Lokal (Frame-by-Frame Cost Matrix)
    if metric == "cosine":
        # Cosine distance
        norm_a = mfcc_a / (np.linalg.norm(mfcc_a, axis=1, keepdims=True) + 1e-9)
        norm_b = mfcc_b / (np.linalg.norm(mfcc_b, axis=1, keepdims=True) + 1e-9)
        cost_matrix = 1.0 - np.dot(norm_a, norm_b.T)
    else:
        # Euclidean distance
        cost_matrix = cdist(mfcc_a, mfcc_b, metric="euclidean")

    # 2. Matriks Akumulasi DTW
    accum_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    accum_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sakoe_band is not None and abs(i - j) > sakoe_band:
                continue
            cost = cost_matrix[i - 1, j - 1]
            accum_matrix[i, j] = cost + min(accum_matrix[i - 1, j],       # insersi
                                            accum_matrix[i, j - 1],       # delesi
                                            accum_matrix[i - 1, j - 1])   # match

    total_cost = float(accum_matrix[n, m])

    # 3. Backtracking Optimal Warping Path
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            steps = [accum_matrix[i - 1, j - 1], accum_matrix[i - 1, j], accum_matrix[i, j - 1]]
            best_idx = int(np.argmin(steps))
            if best_idx == 0:
                i -= 1; j -= 1
            elif best_idx == 1:
                i -= 1
            else:
                j -= 1
        path.append((i - 1, j - 1))
    path.reverse()

    # Jarak Ternormalisasi (Biaya rata-rata per langkah jalur)
    norm_distance = total_cost / max(1, len(path))
    return total_cost, norm_distance, cost_matrix, accum_matrix[1:, 1:], path


# =============================================================================
# MODE 1: AUDIO 1-VS-1 (AUDIO A VS AUDIO B)
# =============================================================================

def compute_audio_dtw_1vs1(audio_a_input, audio_b_input, params: dict) -> dict:
    """
    Pipeline Pencocokan Suara 1-vs-1 (Audio A vs Audio B) menggunakan DTW.
    Visualisasi lengkap:
    - Waveform & Spektrogram A dan B
    - Matriks Biaya Lokal & Matriks Akumulasi DTW + Optimal Warping Path (2D Heatmap)
    - Penyelarasan Korespondensi Temporal (A <-> B)
    - Komparasi DTW vs Euclidean Kaku (Resampling)
    """
    t0 = time.time()
    warnings = []

    # Baca audio A
    if isinstance(audio_a_input, str):
        if audio_a_input.startswith("data:audio") or len(audio_a_input) > 256:
            sr_a, sig_a = load_audio_from_b64(audio_a_input)
        else:
            sr_a, sig_a = load_audio_from_file(audio_a_input)
    else:
        sr_a, sig_a = load_audio_from_file("kw_taksucitra_normal.wav")

    # Baca audio B
    if isinstance(audio_b_input, str):
        if audio_b_input.startswith("data:audio") or len(audio_b_input) > 256:
            sr_b, sig_b = load_audio_from_b64(audio_b_input)
        else:
            sr_b, sig_b = load_audio_from_file(audio_b_input)
    else:
        sr_b, sig_b = load_audio_from_file("kw_taksucitra_cepat.wav")

    sakoe_band = params.get("sakoe_band")
    if sakoe_band is not None and str(sakoe_band).strip() != "":
        sakoe_band = int(sakoe_band)
    else:
        sakoe_band = None

    threshold_dist = float(params.get("threshold", 25.0)) # Batas jarak DTW normalisasi

    # 1. Ekstraksi Fitur MFCC & Spektrogram
    feat_a = extract_audio_features(sig_a, sr=sr_a)
    feat_b = extract_audio_features(sig_b, sr=sr_b)
    n_frames_a, n_frames_b = feat_a["num_frames"], feat_b["num_frames"]

    # Visual 1: Waveform & Spektrogram Berdampingan
    fig1, ((ax_w1, ax_w2), (ax_s1, ax_s2)) = plt.subplots(2, 2, figsize=(8.5, 4.2), facecolor=DEFAULTS["bg_color"])
    
    # Waveforms
    style_dark_axes(ax_w1, title=f"Waveform Audio A ({feat_a['duration_sec']}s)", ylabel="Amplitudo")
    t_a = np.linspace(0, feat_a['duration_sec'], len(sig_a))
    ax_w1.plot(t_a, sig_a, color=DEFAULTS["gold"], lw=0.8)
    
    style_dark_axes(ax_w2, title=f"Waveform Audio B ({feat_b['duration_sec']}s)", ylabel="Amplitudo")
    t_b = np.linspace(0, feat_b['duration_sec'], len(sig_b))
    ax_w2.plot(t_b, sig_b, color=DEFAULTS["saffron"], lw=0.8)
    
    # Spectrograms
    style_dark_axes(ax_s1, title="Spektrogram STFT A", xlabel="Frame", ylabel="Bin Frekuensi")
    ax_s1.imshow(feat_a["spectrogram"][:80], cmap="viridis", aspect="auto", origin="lower")
    
    style_dark_axes(ax_s2, title="Spektrogram STFT B", xlabel="Frame", ylabel="Bin Frekuensi")
    ax_s2.imshow(feat_b["spectrogram"][:80], cmap="viridis", aspect="auto", origin="lower")
    
    fig1.tight_layout()
    step1_img = fig_to_b64(fig1)

    # 2. Matriks MFCC Heatmap
    fig2, (ax_m1, ax_m2) = plt.subplots(2, 1, figsize=(8.5, 3.4), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_m1, title=f"Vektor MFCC Audio A ({n_frames_a} Frame x 13 Koefisien)", ylabel="MFCC")
    im_m1 = ax_m1.imshow(feat_a["mfcc"].T, cmap="coolwarm", aspect="auto", origin="lower")
    
    style_dark_axes(ax_m2, title=f"Vektor MFCC Audio B ({n_frames_b} Frame x 13 Koefisien)", xlabel="Indeks Frame Waktu", ylabel="MFCC")
    im_m2 = ax_m2.imshow(feat_b["mfcc"].T, cmap="coolwarm", aspect="auto", origin="lower")
    fig2.tight_layout()
    step2_img = fig_to_b64(fig2)

    # 3. Perhitungan DTW
    total_cost, norm_dist, cost_mat, accum_mat, path = compute_audio_dtw(feat_a["mfcc"], feat_b["mfcc"], sakoe_band=sakoe_band)

    # Visual 3: Matriks Akumulasi DTW + Warping Path
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.5), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax3, title=f"Matriks Jalur DTW (Warping Path 2D Heatmap)\nJarak Normalisasi = {norm_dist:.2f}",
                    xlabel="Indeks Frame Audio B", ylabel="Indeks Frame Audio A")
    im_acc = ax3.imshow(accum_mat, cmap="magma", aspect="auto", origin="lower")
    py, px = zip(*path)
    ax3.plot(px, py, color="#64d2ff", lw=3, label="Optimal Warping Path")
    ax3.scatter(px[::max(1, len(px)//15)], py[::max(1, len(py)//15)], color="white", s=25, zorder=4)
    ax3.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    fig3.tight_layout()
    step3_img = fig_to_b64(fig3)

    # Visual 4: Penyelarasan Korespondensi Temporal Garis A <-> B
    fig4, ax4 = plt.subplots(figsize=(8.5, 3.2), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax4, title="Korespondensi Penyelarasan Temporal Non-Linier Audio A <-> Audio B")
    
    # Subsampling sinyal audio untuk kecepatan visualisasi garis
    sub_sig_a = sig_a[::10]
    sub_sig_b = sig_b[::10]
    y_offset = 2.2
    
    t_a_sub = np.linspace(0, 1, len(sub_sig_a))
    t_b_sub = np.linspace(0, 1, len(sub_sig_b))
    
    ax4.plot(t_a_sub, sub_sig_a + y_offset, color=DEFAULTS["gold"], lw=1.2, label="Audio A (Atas)")
    ax4.plot(t_b_sub, sub_sig_b, color=DEFAULTS["saffron"], lw=1.2, label="Audio B (Bawah)")
    
    # Ambil sampel pasangan path untuk garis penghubung
    step_jump = max(1, len(path) // 30)
    for idx in range(0, len(path), step_jump):
        ia, ib = path[idx]
        pos_a = ia / max(1, n_frames_a - 1)
        pos_b = ib / max(1, n_frames_b - 1)
        ax4.plot([pos_a, pos_b], [y_offset, 0], color=DEFAULTS["terracotta"], alpha=0.5, linestyle="--")
        
    ax4.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
    ax4.set_xticks([])
    fig4.tight_layout()
    step4_img = fig_to_b64(fig4)

    # 5. DTW vs Euclidean Kaku (Resampling)
    # Resample frame MFCC A agar jumlah framenya sama dengan B
    t_orig = np.linspace(0, 1, n_frames_a)
    t_target = np.linspace(0, 1, n_frames_b)
    mfcc_a_resampled = np.zeros_like(feat_b["mfcc"])
    for dim in range(13):
        mfcc_a_resampled[:, dim] = np.interp(t_target, t_orig, feat_a["mfcc"][:, dim])
    euc_resampled_dist = float(np.mean(np.linalg.norm(mfcc_a_resampled - feat_b["mfcc"], axis=1)))

    fig5, ax5 = plt.subplots(figsize=(8, 2.6), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax5, title="Evaluasi: DTW Elastis vs Euclidean Kaku (Resampling Linier)")
    ax5.barh(["Euclidean (Resampling Linier)", "Dynamic Time Warping (DTW)"],
             [euc_resampled_dist, norm_dist],
             color=[DEFAULTS["terracotta"], DEFAULTS["gold"]], height=0.45)
    ax5.text(euc_resampled_dist + 0.5, 0, f"{euc_resampled_dist:.2f}", va="center", color=DEFAULTS["text_color"], fontsize=9)
    ax5.text(norm_dist + 0.5, 1, f"{norm_dist:.2f}", va="center", color=DEFAULTS["text_color"], fontsize=9)
    ax5.set_xlim(0, max(euc_resampled_dist, norm_dist) * 1.35)
    fig5.tight_layout()
    step5_img = fig_to_b64(fig5)

    # Evaluasi Ambang Batas
    is_match = (norm_dist <= threshold_dist)
    confidence = max(0.0, min(100.0, (1.0 - norm_dist / max(1.0, threshold_dist * 1.5)) * 100.0))
    verdict = "✓ MATCH (COCOK): Audio A dan B memiliki pola akustik serupa." if is_match else "✗ NON-MATCH (TIDAK COCOK): Jarak DTW melampaui batas ambang."

    table_data = {
        "headers": ["Parameter / Metrik Suara", "Nilai", "Keterangan"],
        "rows": [
            ["Durasi Audio A vs B", f"{feat_a['duration_sec']}s vs {feat_b['duration_sec']}s", "Durasi berbeda ditangani secara elastis oleh DTW"],
            ["Jumlah Frame Temporal", f"{n_frames_a} vs {n_frames_b}", "Ukuran jendela 25ms, hop 10ms"],
            ["Jarak Normalisasi DTW", f"{norm_dist:.2f}", f"Batas Ambang Threshold = {threshold_dist:.2f}"],
            ["Biaya Akumulasi Total", f"{total_cost:.2f}", f"Panjang optimal path: {len(path)} langkah"],
            ["Euclidean Kaku (Resampled)", f"{euc_resampled_dist:.2f}", "Rentan terhadap variasi tempo bicara"],
            ["Status Keputusan", "✓ COCOK (MATCH)" if is_match else "✗ TIDAK COCOK (NON-MATCH)", f"Tingkat Kepercayaan = {confidence:.1f}%"]
        ]
    }

    steps = [
        create_step(1, "Input Sinyal Suara & Representasi Spektrogram",
                    f"Menerima Audio A ({feat_a['duration_sec']}s, {n_frames_a} frame) dan Audio B ({feat_b['duration_sec']}s, {n_frames_b} frame).",
                    formula="x(t) \\xrightarrow{\\text{STFT}} X(t, f)",
                    image_b64=step1_img,
                    metrics={"durasi_A": feat_a["duration_sec"], "durasi_B": feat_b["duration_sec"]}),
        create_step(2, "Ekstraksi Koefisien Akustik MFCC Temporal",
                    "Memetakan spektrum daya ke skala Mel dan menghitung 13 koefisien MFCC per bingkai waktu.",
                    formula="c_n = \\sum_{m=1}^M \\log(S_m) \\cos\\left[\\frac{\\pi n}{M}\\left(m - \\frac{1}{2}\\right)\\right]",
                    image_b64=step2_img),
        create_step(3, "Matriks Akumulasi DTW & Jalur Minimum (Warping Path)",
                    f"Menghitung jarak frame-by-frame dan menemukan jalur akumulasi terendah (Biaya Total = {total_cost:.2f}, Rata-rata per langkah = {norm_dist:.2f}).",
                    formula="D(i, j) = \\| \\mathbf{mfcc}_A[i] - \\mathbf{mfcc}_B[j] \\| + \\min \\{ D(i-1, j), D(i, j-1), D(i-1, j-1) \\}",
                    image_b64=step3_img,
                    metrics={"jarak_normalisasi_dtw": round(norm_dist, 2), "panjang_path": len(path)}),
        create_step(4, "Penyelarasan Korespondensi Waktu Temporal",
                    "Menghubungkan frame Audio A terhadap frame Audio B. Bagian suku kata yang dipanjangkan atau dipercepat berhasil disejajarkan.",
                    formula="W = (w_1, w_2, ..., w_K), \\quad w_k = (i, j)",
                    image_b64=step4_img),
        create_step(5, "Keputusan Ambang Batas (Thresholding) & Evaluasi Komparatif",
                    f"{verdict} (Confidence: {confidence:.1f}%)",
                    formula=f"\\text{{Keputusan}} = \\begin{{cases}} \\text{{Match}}, & d_{{DTW}} \\le {threshold_dist:.2f} \\\\ \\text{{Non-match}}, & d_{{DTW}} > {threshold_dist:.2f} \\end{{cases}}",
                    image_b64=step5_img,
                    table=table_data,
                    metrics={"is_match": is_match, "confidence": round(confidence, 1), "norm_dist": round(norm_dist, 2)})
    ]

    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "audio_dtw_1vs1",
        "summary": verdict,
        "is_match": is_match,
        "confidence": round(confidence, 1),
        "norm_dist": round(norm_dist, 2),
        "threshold": threshold_dist,
        "steps": steps,
        "result": {
            "image": step3_img,
            "metrics": {"norm_dist": round(norm_dist, 2), "confidence": round(confidence, 1), "is_match": is_match},
            "table": table_data
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }


# =============================================================================
# MODE 2: AUDIO 1-VS-DATABASE (KEYWORD SPOTTING & SPEAKER VERIFICATION)
# =============================================================================

def compute_audio_dtw_database(query_audio_input, params: dict) -> dict:
    """
    Pipeline Pencocokan Suara 1-ke-Banyak (Audio Query vs Database Suara).
    Mendukung dua objektif utama:
    1. 'keyword_spotting': Mencari kata kunci paling cocok ('buka', 'tutup', 'nyalakan', 'matikan', 'taksucitra', 'halo').
    2. 'speaker_verification': Memvalidasi penutur terdaftar ('Budi', 'Siti', 'Wayan') terhadap passphrase biometric.
    """
    t0 = time.time()
    warnings = []

    objective = params.get("objective", "keyword_spotting") # 'keyword_spotting' atau 'speaker_verification'
    top_k = int(params.get("top_k", 5))
    threshold_dist = float(params.get("threshold", 22.0))

    # Baca audio query
    if isinstance(query_audio_input, str):
        if query_audio_input.startswith("data:audio") or len(query_audio_input) > 256:
            sr_q, sig_q = load_audio_from_b64(query_audio_input)
        else:
            sr_q, sig_q = load_audio_from_file(query_audio_input)
    else:
        # Default query audio
        default_file = "kw_taksucitra_cepat.wav" if objective == "keyword_spotting" else "spk_Budi_uji.wav"
        sr_q, sig_q = load_audio_from_file(default_file)

    # Ekstraksi fitur query
    t_feat_start = time.time()
    q_feat = extract_audio_features(sig_q, sr=sr_q)
    q_extract_ms = (time.time() - t_feat_start) * 1000.0

    # Ambil sampel database (apakah dari folder/koleksi yang diunggah pengguna atau bawaan sistem)
    custom_db = params.get("custom_database")
    is_custom_audio_db = False
    db_entries = []

    if custom_db and isinstance(custom_db, list) and len(custom_db) > 0:
        is_custom_audio_db = True
        for item in custom_db:
            fn = item.get("filename", "audio.wav")
            audio_raw = item.get("audio")
            if not audio_raw:
                continue
            try:
                sr_d, sig_d = load_audio_from_b64(audio_raw)
                d_feat = extract_audio_features(sig_d, sr=sr_d)
                db_entries.append({
                    "filename": fn,
                    "label": fn,
                    "audio_url": audio_raw,
                    "sr": sr_d,
                    "sig": sig_d,
                    "feat": d_feat
                })
            except Exception as e:
                continue
        if len(db_entries) == 0:
            return {"ok": False, "error": "Koleksi audio database yang diunggah tidak dapat diproses.", "hint": "Pastikan file berformat audio valid (WAV, MP3)."}
    else:
        # Database audio bawaan sistem
        all_files = sorted([f for f in os.listdir(DB_AUDIO_DIR) if f.endswith(".wav")])
        if objective == "keyword_spotting":
            db_files = [f for f in all_files if f.startswith("kw_")]
        else: # speaker_verification
            db_files = [f for f in all_files if f.startswith("spk_") and "daftar" in f]

        if len(db_files) == 0:
            db_files = all_files[:8]

        for fn in db_files:
            try:
                sr_d, sig_d = load_audio_from_file(fn)
                d_feat = extract_audio_features(sig_d, sr=sr_d)
                if objective == "keyword_spotting":
                    parts = fn.replace(".wav", "").split("_")
                    word_name = parts[1].upper() if len(parts) > 1 else fn
                    variant = f"({parts[2].capitalize()})" if len(parts) > 2 else ""
                    label = f"{word_name} {variant}".strip()
                else:
                    parts = fn.replace(".wav", "").split("_")
                    spk_name = parts[1] if len(parts) > 1 else fn
                    label = f"Penutur: {spk_name}"

                db_entries.append({
                    "filename": fn,
                    "label": label,
                    "audio_url": f"/static/db_audio/{fn}",
                    "sr": sr_d,
                    "sig": sig_d,
                    "feat": d_feat
                })
            except Exception as e:
                continue

    # Matching loop
    results = []
    t_match_start = time.time()
    for entry in db_entries:
        d_feat = entry["feat"]
        tot_cost, norm_dist, _, _, path = compute_audio_dtw(q_feat["mfcc"], d_feat["mfcc"])
        is_match = (norm_dist <= threshold_dist)
        confidence = max(0.0, min(100.0, (1.0 - norm_dist / max(1.0, threshold_dist * 1.5)) * 100.0))

        results.append({
            "filename": entry["filename"],
            "label": entry["label"],
            "audio_url": entry["audio_url"],
            "norm_dist": round(norm_dist, 2),
            "total_cost": round(tot_cost, 2),
            "confidence": round(confidence, 1),
            "is_match": is_match,
            "path_length": len(path),
            "duration_sec": d_feat["duration_sec"]
        })

    # Urutkan berdasarkan jarak DTW terendah (kemiripan tertinggi)
    results.sort(key=lambda x: x["norm_dist"])
    top_candidates = results[:top_k]
    matching_time_ms = (time.time() - t_match_start) * 1000.0

    best = top_candidates[0] if top_candidates else None
    
    # Visual 1: Query Audio Waveform & Spektrogram
    fig_q, (ax_qw, ax_qs) = plt.subplots(2, 1, figsize=(8.5, 3.4), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_qw, title=f"Audio Uji (Query): Durasi {q_feat['duration_sec']}s, {q_feat['num_frames']} Frame", ylabel="Amplitudo")
    ax_qw.plot(np.linspace(0, q_feat['duration_sec'], len(sig_q)), sig_q, color=DEFAULTS["gold"], lw=0.8)
    
    style_dark_axes(ax_qs, title="Spektrogram Akustik Audio Uji", xlabel="Frame", ylabel="Frekuensi")
    ax_qs.imshow(q_feat["spectrogram"][:70], cmap="viridis", aspect="auto", origin="lower")
    fig_q.tight_layout()
    step1_img = fig_to_b64(fig_q)

    # Visual 2: Bar Chart Peringkat Jarak DTW Database
    fig_bar, ax_bar = plt.subplots(figsize=(8.5, 3.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax_bar, title=f"Peringkat Jarak DTW Database ({'Keyword Spotting' if objective == 'keyword_spotting' else 'Speaker Verification'})",
                    xlabel="Jarak DTW Ternormalisasi (Semakin Kecil Semakin Cocok)", ylabel="Sampel Database")
    
    sample_labels = [r["label"] for r in results[:min(10, len(results))]]
    sample_dists = [r["norm_dist"] for r in results[:min(10, len(results))]]
    sample_colors = [DEFAULTS["gold"] if r["is_match"] else DEFAULTS["terracotta"] for r in results[:min(10, len(results))]]

    y_pos = np.arange(len(sample_labels))
    ax_bar.barh(y_pos, sample_dists, color=sample_colors, height=0.55)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(sample_labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.axvline(x=threshold_dist, color="#64d2ff", linestyle="--", lw=2, label=f"Batas Ambang ({threshold_dist:.1f})")
    ax_bar.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"], fontsize=8)
    for i, d in enumerate(sample_dists):
        ax_bar.text(d + 0.3, i, f"{d:.2f}", va="center", color=DEFAULTS["text_color"], fontsize=8)
    fig_bar.tight_layout()
    step3_img = fig_to_b64(fig_bar)

    # Visual 3: Detail DTW Warping Path untuk Kandidat Terbaik (Top-1)
    if best:
        best_entry = next((e for e in db_entries if e["filename"] == best["filename"]), None)
        if best_entry:
            best_feat = best_entry["feat"]
        else:
            sr_best, sig_best = load_audio_from_file(best["filename"])
            best_feat = extract_audio_features(sig_best, sr=sr_best)
        _, _, _, best_accum, best_path = compute_audio_dtw(q_feat["mfcc"], best_feat["mfcc"])

        fig_best, ax_b = plt.subplots(figsize=(6.5, 4.2), facecolor=DEFAULTS["bg_color"])
        style_dark_axes(ax_b, title=f"Warping Path Pasangan Top-1 ({best['label']})\nJarak DTW = {best['norm_dist']:.2f}",
                        xlabel=f"Frame Database: {best['label']}", ylabel="Frame Audio Uji (Query)")
        ax_b.imshow(best_accum, cmap="magma", aspect="auto", origin="lower")
        bpy, bpx = zip(*best_path)
        ax_b.plot(bpx, bpy, color="#64d2ff", lw=3, label="Optimal Path")
        ax_b.legend(facecolor=DEFAULTS["panel_color"], edgecolor=DEFAULTS["border"], labelcolor=DEFAULTS["text_color"])
        fig_best.tight_layout()
        step4_img = fig_to_b64(fig_best)
    else:
        step4_img = step1_img

    # Ringkasan Keputusan
    matched_count = sum(1 for r in results if r["is_match"])
    if best and best["is_match"]:
        verdict_status = f"MATCH: Terdeteksi {best['label']} dengan jarak DTW {best['norm_dist']:.2f} (Confidence: {best['confidence']}%)"
    else:
        verdict_status = f"NON-MATCH: Tidak ada sampel suara database yang lolos nilai ambang batas {threshold_dist:.1f}."

    table_data = {
        "headers": ["Peringkat", "Label Suara", "Jarak DTW", "Confidence", "Status Ambang"],
        "rows": [
            [
                f"#{i+1}",
                c["label"],
                f"{c['norm_dist']:.2f}",
                f"{c['confidence']}%",
                "✓ Lolos Ambang (Match)" if c["is_match"] else "✗ Melebihi Ambang (Non-match)"
            ]
            for i, c in enumerate(top_candidates)
        ]
    }

    db_desc_aud = f"Membandingkan sinyal uji terhadap {len(results)} sampel audio yang diunggah secara mandiri oleh pengguna (Folder/Files)." if is_custom_audio_db else f"Membandingkan sinyal uji terhadap {len(results)} profil suara referensi yang tersimpan di database menggunakan algoritma DTW elastis."

    steps = [
        create_step(1, "Input Audio Uji & Ekstraksi Ciri Temporal",
                    f"Menerima audio uji ({q_feat['duration_sec']}s, {q_feat['num_frames']} frame) dan mengekstrak matriks MFCC dalam {q_extract_ms:.1f} ms.",
                    formula="\\text{Audio} \\to \\mathbf{X}_{MFCC} \\in \\mathbb{R}^{T \\times 13}",
                    image_b64=step1_img,
                    metrics={"durasi_query_sec": q_feat["duration_sec"], "frame_count": q_feat["num_frames"]}),
        create_step(2, f"Pencarian Database: {'Pencarian Kata Kunci' if objective == 'keyword_spotting' else 'Verifikasi Penutur'}",
                    db_desc_aud,
                    formula="\\mathcal{D}_{audio} = \\{ (\\mathbf{M}_k, \\text{label}_k) \\}_{k=1}^K",
                    metrics={"total_sampel_database": len(results), "tipe_database": "Folder Audio Pengguna" if is_custom_audio_db else "Database Sistem"}),
        create_step(3, "Distribusi Jarak DTW Terhadap Seluruh Database",
                    f"Evaluasi matriks jarak DTW untuk seluruh sampel selesai dalam {matching_time_ms:.1f} ms.",
                    formula="d_{DTW}(Q, D_k) = \\frac{1}{L} \\min_{\\mathcal{W}} \\sum_{(i, j) \\in \\mathcal{W}} \\| \\mathbf{q}_i - \\mathbf{d}_{k, j} \\|",
                    image_b64=step3_img,
                    metrics={"waktu_matching_ms": round(matching_time_ms, 2)}),
        create_step(4, "Visualisasi Warping Path Kandidat Terbaik (Top-1)",
                    f"Jalur pelengkungan waktu (warping path) membuktikan bahwa lafal kata/penutur berhasil diselaraskan secara optimal.",
                    formula="\\text{Warping Path: } (0, 0) \\rightsquigarrow (T_Q, T_D)",
                    image_b64=step4_img,
                    metrics={"kandidat_terbaik": best["label"] if best else "-", "jarak_top1": best["norm_dist"] if best else 0}),
        create_step(5, "Pengambilan Keputusan Nilai Ambang (Thresholding)",
                    f"{verdict_status} ({matched_count} dari {len(results)} sampel lolos ambang batas).",
                    formula=f"\\text{{Status}} = \\begin{{cases}} \\text{{Match}}, & d_{{DTW}} \\le {threshold_dist:.1f} \\\\ \\text{{Non-match}}, & d_{{DTW}} > {threshold_dist:.1f} \\end{{cases}}",
                    table=table_data,
                    metrics={"is_match": best["is_match"] if best else False, "matched_count": matched_count})
    ]

    elapsed = (time.time() - t0) * 1000.0
    return {
        "ok": True,
        "feature": "audio_dtw_database",
        "objective": objective,
        "summary": verdict_status,
        "threshold": threshold_dist,
        "top_k": top_k,
        "best_candidate": best,
        "top_candidates": top_candidates,
        "steps": steps,
        "result": {
            "image": step4_img,
            "metrics": {
                "best_label": best["label"] if best else "-",
                "best_dist": best["norm_dist"] if best else 0,
                "matched_count": matched_count
            },
            "table": table_data
        },
        "timing_ms": round(elapsed, 2),
        "warnings": warnings
    }


def list_available_audio_samples() -> list[dict]:
    """Mengembalikan daftar sampel audio di static/db_audio untuk dropdown antarmuka."""
    files = sorted([f for f in os.listdir(DB_AUDIO_DIR) if f.endswith(".wav")])
    samples = []
    for f in files:
        if f.startswith("kw_"):
            parts = f.replace(".wav", "").split("_")
            cat = "Kata Kunci (Keyword)"
            label = f"{parts[1].upper()} ({parts[2].capitalize() if len(parts) > 2 else 'Normal'})"
        elif f.startswith("spk_"):
            parts = f.replace(".wav", "").split("_")
            cat = "Penutur (Biometrik)"
            label = f"Penutur {parts[1]} ({parts[2].capitalize() if len(parts) > 2 else ''})"
        else:
            cat = "Lainnya"
            label = f
        samples.append({
            "filename": f,
            "label": label,
            "category": cat,
            "url": f"/static/db_audio/{f}"
        })
    return samples

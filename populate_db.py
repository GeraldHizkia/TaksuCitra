"""
populate_db.py
==============
Script untuk mengisi database bawaan citra (static/db_images)
dan database bawaan audio (static/db_audio) untuk demonstrasi
Pencocokan 1-ke-Banyak (1-to-N) Citra dan Audio DTW.
"""

import os
import shutil
import numpy as np
import cv2
import scipy.io.wavfile as wavfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
DB_IMAGES_DIR = os.path.join(BASE_DIR, "static", "db_images")
DB_AUDIO_DIR = os.path.join(BASE_DIR, "static", "db_audio")

os.makedirs(DB_IMAGES_DIR, exist_ok=True)
os.makedirs(DB_AUDIO_DIR, exist_ok=True)

# =============================================================================
# 1. DATABASE CITRA (static/db_images)
# =============================================================================
print("Menyiapkan database citra...")

# Salin sampel yang sudah ada
if os.path.exists(SAMPLES_DIR):
    for f in os.listdir(SAMPLES_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(SAMPLES_DIR, f)
            dst = os.path.join(DB_IMAGES_DIR, f)
            shutil.copy2(src, dst)

# Buat beberapa variasi dan citra sintetis untuk menguji pencocokan 1-ke-Banyak
# Citra 1: Logo Geometris Taksu (Lingkaran + Segitiga + Bintang)
h, w = 320, 320
img_taksu = np.zeros((h, w, 3), dtype=np.uint8)
img_taksu[:] = (20, 20, 24)
cv2.circle(img_taksu, (160, 160), 100, (80, 202, 242), -1) # Gold circle
pts = np.array([[160, 80], [240, 220], [80, 220]], np.int32)
cv2.fillPoly(img_taksu, [pts], (40, 60, 220)) # Terracotta triangle
cv2.putText(img_taksu, "TAKSU", (100, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_taksu_logo.jpg"), img_taksu)

# Variasi taksu dengan rotasi 15 derajat dan sedikit scaling (untuk menguji ketahanan SIFT/ORB)
M_rot = cv2.getRotationMatrix2D((w/2, h/2), 15, 0.95)
img_taksu_rot = cv2.warpAffine(img_taksu, M_rot, (w, h), borderValue=(20, 20, 24))
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_taksu_rot15.jpg"), img_taksu_rot)

# Variasi taksu dengan noise dan perbedaan pencahayaan
img_taksu_dark = cv2.convertScaleAbs(img_taksu, alpha=0.6, beta=-10)
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_taksu_dark.jpg"), img_taksu_dark)

# Citra Pola Tekstur Batik / Kisi (Grid Texture)
img_grid = np.zeros((h, w, 3), dtype=np.uint8)
for i in range(0, w, 20):
    cv2.line(img_grid, (i, 0), (i, h), (180, 160, 120), 2)
for j in range(0, h, 20):
    cv2.line(img_grid, (0, j), (w, j), (120, 160, 180), 2)
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_texture_grid.jpg"), img_grid)

# Citra Pola Spiral / Radial
img_radial = np.zeros((h, w, 3), dtype=np.uint8)
center = (160, 160)
for r in range(10, 150, 15):
    color = (int(120 + 100 * np.sin(r/20)), int(180 + 50 * np.cos(r/15)), 220)
    cv2.circle(img_radial, center, r, color, 3)
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_pattern_radial.jpg"), img_radial)

# Citra Pola Bintang Berulang (Stars Pattern)
img_stars = np.zeros((h, w, 3), dtype=np.uint8)
img_stars[:] = (30, 25, 35)
for (cx, cy) in [(60, 60), (240, 60), (160, 160), (60, 240), (240, 240)]:
    cv2.circle(img_stars, (cx, cy), 30, (100, 230, 255), -1)
    cv2.circle(img_stars, (cx, cy), 15, (255, 120, 80), -1)
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_pattern_stars.jpg"), img_stars)

# Citra Teks Barcode / QR-like pattern
img_code = np.zeros((h, w, 3), dtype=np.uint8)
img_code[:] = (240, 240, 240)
np.random.seed(42)
for y in range(20, h-20, 25):
    for x in range(20, w-20, 25):
        if np.random.rand() > 0.5:
            cv2.rectangle(img_code, (x, y), (x+20, y+20), (10, 10, 10), -1)
cv2.imwrite(os.path.join(DB_IMAGES_DIR, "db_pattern_blocks.jpg"), img_code)

print(f"Database citra siap ({len(os.listdir(DB_IMAGES_DIR))} citra di {DB_IMAGES_DIR})")


# =============================================================================
# 2. DATABASE AUDIO (static/db_audio)
# =============================================================================
print("Menyiapkan database audio...")

SAMPLE_RATE = 16000

def synthesize_acoustic_word(formant_freqs, f0, duration=0.6, sr=16000, volume=0.8):
    """
    Sintesis suara akustik berbasis sumber-filter (source-filter model)
    f0: frekuensi nada dasar (pitch penutur)
    formant_freqs: deret [(F1_start, F1_end), (F2_start, F2_end), (F3_start, F3_end)]
    duration: durasi dalam detik
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Sumber getaran pita suara (glottal pulse train / rich harmonic sawtooth + vibrato)
    vibrato = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0 * vibrato) / sr
    glottal = (phase % (2 * np.pi)) / np.pi - 1.0 # Sawtooth source
    
    # Tambahkan sedikit derau napas / aspirasi
    noise = np.random.normal(0, 0.05, len(t))
    source = 0.85 * glottal + 0.15 * noise
    
    # Filter formants (resonansi saluran suara vokal)
    output = np.zeros_like(source)
    for (f_start, f_end), gain, bandwidth in formant_freqs:
        f_center = np.linspace(f_start, f_end, len(t))
        # Simulasi resonansi formant dengan modulasi sinusoidal tertapis
        carrier = np.sin(2 * np.pi * np.cumsum(f_center) / sr)
        output += gain * source * carrier
        
    # Envelope ADSR (Attack, Decay, Sustain, Release)
    env = np.ones_like(t)
    att_len = int(0.1 * len(t))
    rel_len = int(0.15 * len(t))
    env[:att_len] = np.linspace(0, 1, att_len)
    env[-rel_len:] = np.linspace(1, 0, rel_len)
    
    signal = output * env
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * volume
    return (signal * 32767).astype(np.int16)

# Data Kamus Kata Kunci (Keyword Spotting)
KEYWORDS = {
    "buka": {
        "formants": [((350, 750), 1.0, 80), ((900, 1350), 0.7, 100), ((2400, 2600), 0.3, 120)],
        "base_f0": 130
    },
    "tutup": {
        "formants": [((320, 340), 1.0, 70), ((850, 900), 0.6, 90), ((2300, 2400), 0.25, 110)],
        "base_f0": 130
    },
    "nyalakan": {
        "formants": [((450, 800), 0.9, 80), ((1500, 1400), 0.8, 100), ((2500, 2700), 0.35, 120)],
        "base_f0": 135
    },
    "matikan": {
        "formants": [((300, 780), 0.95, 80), ((1100, 1450), 0.75, 100), ((2350, 2550), 0.3, 120)],
        "base_f0": 128
    },
    "taksucitra": {
        "formants": [((400, 850), 1.0, 90), ((1300, 1600), 0.85, 110), ((2450, 2750), 0.4, 130)],
        "base_f0": 132
    },
    "halo": {
        "formants": [((750, 500), 1.0, 80), ((1300, 950), 0.75, 100), ((2500, 2400), 0.3, 120)],
        "base_f0": 130
    }
}

# 1. Hasilkan sampel Keyword Spotting (Kata kunci normal, kata kunci cepat, kata kunci lambat)
for kw, cfg in KEYWORDS.items():
    # Normal duration (0.55s)
    sig_norm = synthesize_acoustic_word(cfg["formants"], f0=cfg["base_f0"], duration=0.55, sr=SAMPLE_RATE)
    wavfile.write(os.path.join(DB_AUDIO_DIR, f"kw_{kw}_normal.wav"), SAMPLE_RATE, sig_norm)
    
    # Fast speaking duration (0.38s - tempo cepat, untuk membuktikan keunggulan DTW!)
    sig_fast = synthesize_acoustic_word(cfg["formants"], f0=cfg["base_f0"]*1.08, duration=0.38, sr=SAMPLE_RATE)
    wavfile.write(os.path.join(DB_AUDIO_DIR, f"kw_{kw}_cepat.wav"), SAMPLE_RATE, sig_fast)
    
    # Slow speaking duration (0.78s - tempo lambat / dipanjangkan)
    sig_slow = synthesize_acoustic_word(cfg["formants"], f0=cfg["base_f0"]*0.96, duration=0.78, sr=SAMPLE_RATE)
    wavfile.write(os.path.join(DB_AUDIO_DIR, f"kw_{kw}_lambat.wav"), SAMPLE_RATE, sig_slow)

# 2. Hasilkan sampel Verifikasi Penutur (Speaker Verification)
# Penutur 1: Budi (Pria, Nada Bass F0=115Hz)
# Penutur 2: Siti (Wanita, Nada Tinggi F0=220Hz)
# Penutur 3: Wayan (Pria Baritone F0=165Hz)
# Penutur 4: Penyusup / Intruder (Nada F0=85Hz)
SPEAKERS = {
    "Budi": {"f0": 115, "passphrase": "taksucitra", "role": "Administrator"},
    "Siti": {"f0": 220, "passphrase": "taksucitra", "role": "Finance Officer"},
    "Wayan": {"f0": 165, "passphrase": "taksucitra", "role": "Tech Lead"},
    "Penyusup": {"f0": 85, "passphrase": "taksucitra", "role": "Unauthorized Guest"}
}

for spk, info in SPEAKERS.items():
    cfg = KEYWORDS[info["passphrase"]]
    # Passphrase pendaftaran (enrolled biometric voiceprint)
    sig_enrolled = synthesize_acoustic_word(cfg["formants"], f0=info["f0"], duration=0.62, sr=SAMPLE_RATE)
    wavfile.write(os.path.join(DB_AUDIO_DIR, f"spk_{spk}_daftar.wav"), SAMPLE_RATE, sig_enrolled)
    
    # Sampel uji verifikasi (uji autentikasi dengan sedikit variasi durasi 0.58s)
    sig_test = synthesize_acoustic_word(cfg["formants"], f0=info["f0"] * 1.02, duration=0.58, sr=SAMPLE_RATE)
    wavfile.write(os.path.join(DB_AUDIO_DIR, f"spk_{spk}_uji.wav"), SAMPLE_RATE, sig_test)

print(f"Database audio siap ({len(os.listdir(DB_AUDIO_DIR))} file WAV di {DB_AUDIO_DIR})")
print("Semua database awal berhasil diinisialisasi!")

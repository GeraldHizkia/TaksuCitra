"""
2.1 Operasi Piksel
Implementasi mengikuti Bab 2 buku "Dasar Image Processing Programming":
- Perubahan warna (channel split)
- Pemadaman noise (median, bilateral, wavelet, NLM, wiener/FFT, low-pass)
- Perubahan kecerahan
- Perubahan kontras
- Filtrasi (sharpening kernel)
- Rotasi & skala
- Penggabungan gambar (alpha blending)
"""
import cv2
import numpy as np
import pywt
from scipy.signal import wiener


def channel_split(img_bgr):
    """Perubahan Warna: memisahkan citra menjadi kanal R, G, B (grayscale-per-channel)."""
    b, g, r = cv2.split(img_bgr)
    zeros = np.zeros_like(b)

    red_channel = cv2.merge([zeros, zeros, r])
    green_channel = cv2.merge([zeros, g, zeros])
    blue_channel = cv2.merge([b, zeros, zeros])

    return {
        "red_channel": red_channel,
        "green_channel": green_channel,
        "blue_channel": blue_channel,
    }


def denoise(img_bgr, method="median", strength=5):
    """Pemadaman Noise dengan beberapa metode dari buku."""
    strength = int(strength)
    if strength % 2 == 0:
        strength += 1  # kernel harus ganjil untuk median/gaussian

    if method == "median":
        out = cv2.medianBlur(img_bgr, strength)
    elif method == "bilateral":
        out = cv2.bilateralFilter(img_bgr, d=strength, sigmaColor=75, sigmaSpace=75)
    elif method == "nlm":
        out = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=float(strength),
                                               hColor=float(strength),
                                               templateWindowSize=7, searchWindowSize=21)
    elif method == "wavelet":
        img_float = img_bgr.astype(np.float32) / 255.0
        channels = []
        for c in range(3):
            coeffs = pywt.wavedec2(img_float[:, :, c], 'db1', level=2)
            coeffs_thresh = list(coeffs)
            threshold = 0.04 * (strength / 5.0)
            coeffs_thresh[1:] = [
                tuple(pywt.threshold(d, threshold, mode='soft') for d in detail)
                for detail in coeffs_thresh[1:]
            ]
            rec = pywt.waverec2(coeffs_thresh, 'db1')
            rec = np.clip(rec, 0, 1)
            channels.append(rec)
        h = min(c.shape[0] for c in channels)
        w = min(c.shape[1] for c in channels)
        channels = [c[:h, :w] for c in channels]
        out = (np.dstack(channels) * 255).astype(np.uint8)
    elif method == "wiener":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        filtered = wiener(gray, (strength, strength))
        filtered = np.nan_to_num(filtered)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    elif method == "lowpass":
        img_float = img_bgr.astype(np.float32)
        channels = []
        for c in range(3):
            f = np.fft.fft2(img_float[:, :, c])
            fshift = np.fft.fftshift(f)
            rows, cols = fshift.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.zeros((rows, cols), np.uint8)
            radius = max(5, 60 - strength * 4)
            cv2.circle(mask, (ccol, crow), radius, 1, -1)
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            channels.append(img_back)
        out = np.clip(np.dstack(channels), 0, 255).astype(np.uint8)
    else:
        out = img_bgr.copy()

    return out


def adjust_brightness(img_bgr, factor=1.5):
    """Perubahan Kecerahan: mengalikan nilai piksel dengan faktor tertentu."""
    return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=(factor - 1.0) * 128)


def adjust_contrast(img_bgr, factor=1.5):
    """Perubahan Kontras: cv2.convertScaleAbs dengan alpha=faktor kontras."""
    return cv2.convertScaleAbs(img_bgr, alpha=factor, beta=0)


def sharpen(img_bgr, variant=1):
    """Filtrasi: kernel penguatan/sharpening seperti pada buku."""
    if variant == 1:
        kernel = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])
    else:
        kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
    return cv2.filter2D(img_bgr, -1, kernel)


def rotate_and_scale(img_bgr, angle=45, scale=0.5):
    """Rotasi dan Skala menggunakan cv2.getRotationMatrix2D + warpAffine."""
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, matrix, (w, h))
    scaled = cv2.resize(rotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return rotated, scaled


def blend_images(img1_bgr, img2_bgr, alpha=0.5):
    """Penggabungan Gambar (alpha blending) via cv2.addWeighted."""
    h, w = img1_bgr.shape[:2]
    img2_resized = cv2.resize(img2_bgr, (w, h))
    blended = cv2.addWeighted(img1_bgr, alpha, img2_resized, 1 - alpha, 0)
    return blended

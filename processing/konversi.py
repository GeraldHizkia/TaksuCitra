"""
2.2 Konversi Gambar
- Konversi Format Gambar (JPEG/PNG/BMP)
- Konversi Ruang Warna (RGB -> Grayscale, HSV, dll)
- Konversi Resolusi (resize / thumbnail)
- Konversi Bit Depth
"""
import cv2
import numpy as np
from PIL import Image


def convert_colorspace(img_bgr, target="gray"):
    """Konversi Ruang Warna menggunakan cv2.cvtColor."""
    if target == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # supaya tetap 3 channel utk ditampilkan
    elif target == "hsv":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return hsv
    elif target == "ycrcb":
        ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        return ycc
    elif target == "binary":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    else:
        return img_bgr


def convert_resolution(img_bgr, size=64):
    """Konversi Resolusi: mirip img.thumbnail((size,size)) pada buku."""
    size = int(size)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail((size, size))
    result_rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def convert_bit_depth(img_bgr, bit_depth=8):
    """Konversi Bit Depth: mengurangi jumlah level intensitas per kanal."""
    bit_depth = int(bit_depth)
    bit_depth = max(1, min(8, bit_depth))
    levels = 2 ** bit_depth

    img_float = img_bgr.astype(np.float32) / 255.0
    quantized = np.floor(img_float * (levels - 1) + 0.5) / (levels - 1)
    quantized = np.clip(quantized * 255.0, 0, 255).astype(np.uint8)
    return quantized


def get_format_info(img_bgr):
    """Info ukuran file untuk beberapa format (untuk demo konversi format)."""
    results = {}
    for fmt, ext in [("JPEG", ".jpg"), ("PNG", ".png"), ("BMP", ".bmp")]:
        success, buf = cv2.imencode(ext, img_bgr)
        if success:
            results[fmt] = len(buf.tobytes())
    return results

"""
2.3 Operasi Histogram
- Analisis Histogram (cv2.calcHist)
- Peningkatan Kontras (Histogram Equalization)
- Pemadaman Noise (median + NLM) + histogram
- Deteksi Tepi + histogram
- Segmentasi Gambar (thresholding) + histogram
- Modifikasi Intensitas Piksel (stretching/shifting)
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def compute_histogram_plot(img_bgr, title="Histogram"):
    """Analisis Histogram: cv2.calcHist([img],[0],None,[256],[0,256])."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(hist, color="#4f7cff")
    ax.fill_between(range(256), hist.flatten(), color="#4f7cff", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Intensitas Piksel (0-255)")
    ax.set_ylabel("Jumlah Piksel")
    ax.set_xlim([0, 256])
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def histogram_equalization(img_bgr):
    """Peningkatan Kontras via Histogram Equalization."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    merged = cv2.merge([y_eq, cr, cb])
    result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    return result


def histogram_denoise(img_bgr, method="median"):
    """Pemadaman Noise (untuk ditampilkan berdampingan dengan histogram)."""
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)
    else:
        return cv2.medianBlur(img_bgr, 5)


def histogram_edge_detection(img_bgr, method="canny"):
    """Deteksi Tepi berbasis analisis perubahan intensitas."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if method == "sobel":
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.clip(edges, 0, 255).astype(np.uint8)
    else:
        edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def histogram_segmentation(img_bgr, threshold=127):
    """Segmentasi Gambar via thresholding berdasarkan histogram."""
    threshold = int(threshold)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, seg = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)


def histogram_intensity_modification(img_bgr, mode="stretch", shift=30):
    """Modifikasi Intensitas Piksel: stretching atau shifting histogram."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if mode == "stretch":
        min_val, max_val = gray.min(), gray.max()
        if max_val - min_val < 1e-5:
            stretched = gray
        else:
            stretched = (gray - min_val) * (255.0 / (max_val - min_val))
        result_gray = np.clip(stretched, 0, 255).astype(np.uint8)
    else:  # shift
        shifted = gray + float(shift)
        result_gray = np.clip(shifted, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)

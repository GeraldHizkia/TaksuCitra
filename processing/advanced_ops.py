"""
Advanced Ops - Eksplorasi di luar Bab 2
Fitur classical computer vision tingkat lanjut yang jarang dibahas di tugas
mahasiswa lain, untuk menunjukkan eksplorasi mandiri di luar buku pedoman.

- Magic Eraser (Inpainting)
- Panorama Stitching
- Morphological Playground
- Frequency Domain Explorer (FFT + custom filter mask)
- Feature Matching (ORB Keypoints)
- QR / Barcode Scanner
- Histogram Matching (Specification)
- Digital Watermarking (LSB Steganography)
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io


def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="#12141c")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------- Magic Eraser (Inpainting) ----------

def inpaint_image(img_bgr, mask_gray, method="telea", radius=7):
    """Menghapus objek dari citra menggunakan inpainting (Telea atau Navier-Stokes)."""
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    _, mask_bin = cv2.threshold(mask_gray, 30, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(img_bgr, mask_bin, radius, flag)
    return result


# ---------- Panorama Stitching ----------

def stitch_panorama(images_bgr):
    """Menggabungkan beberapa citra menjadi satu panorama menggunakan cv2.Stitcher."""
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, result = stitcher.stitch(images_bgr)
    if status != cv2.Stitcher_OK:
        status_messages = {
            1: "Butuh lebih banyak gambar dengan area tumpang tindih (overlap) yang cukup.",
            2: "Gagal mengestimasi parameter homografi kamera.",
            3: "Gagal menyesuaikan (adjust) parameter kamera.",
        }
        raise ValueError(status_messages.get(status, f"Stitching gagal (kode {status})"))
    return result


# ---------- Morphological Playground ----------

def morphological_op(img_bgr, operation="erode", kernel_size=5, iterations=1):
    """Operasi morfologi pada citra biner: erosion, dilation, opening, closing, gradient, tophat, blackhat."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = int(kernel_size)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = int(iterations)

    ops = {
        "erode": lambda: cv2.erode(binary, kernel, iterations=iterations),
        "dilate": lambda: cv2.dilate(binary, kernel, iterations=iterations),
        "opening": lambda: cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations),
        "closing": lambda: cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations),
        "gradient": lambda: cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel),
        "tophat": lambda: cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel),
        "blackhat": lambda: cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel),
    }
    result_gray = ops.get(operation, ops["erode"])()
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    result_bgr = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
    return binary_bgr, result_bgr


# ---------- Frequency Domain Explorer ----------

def frequency_domain_explorer(img_bgr, filter_type="none", radius=40):
    """Menampilkan spektrum magnitude FFT dan efek filter ideal (low-pass/high-pass) di domain frekuensi."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    mag_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = int(radius)

    if filter_type == "lowpass":
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, -1)
    elif filter_type == "highpass":
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 0, -1)
    else:
        mask = np.ones((rows, cols), np.uint8)

    fshift_filtered = fshift * mask
    filtered_mag = 20 * np.log(np.abs(fshift_filtered) + 1)
    filtered_mag_norm = cv2.normalize(filtered_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    result_gray = np.clip(img_back, 0, 255).astype(np.uint8)

    return (
        cv2.cvtColor(mag_norm, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(filtered_mag_norm, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR),
    )


# ---------- Feature Matching (ORB) ----------

def feature_matching(img1_bgr, img2_bgr, max_matches=40):
    """Mencocokkan titik fitur (keypoints) antara dua citra menggunakan ORB + Brute-Force Matcher."""
    orb = cv2.ORB_create(nfeatures=1500)

    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        raise ValueError("Tidak cukup fitur terdeteksi pada salah satu gambar.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    top_matches = matches[:max_matches]
    result = cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, top_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return result, len(kp1), len(kp2), len(matches)


# ---------- QR / Barcode Scanner ----------

def scan_qr_code(img_bgr):
    """Mendeteksi dan mendekode QR code menggunakan cv2.QRCodeDetector."""
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img_bgr)

    output = img_bgr.copy()
    results = []

    if retval and points is not None:
        for i, pts in enumerate(points):
            pts = pts.astype(int)
            cv2.polylines(output, [pts], True, (0, 255, 120), 3)
            text = decoded_info[i] if i < len(decoded_info) else ""
            x, y = pts[0]
            if text:
                cv2.putText(output, text[:30], (x, max(20, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            results.append({"text": text, "detected": bool(text)})

    return output, results


# ---------- Histogram Matching (Specification) ----------

def histogram_matching(source_bgr, reference_bgr):
    """Menyesuaikan distribusi histogram citra sumber agar menyerupai citra referensi (histogram specification)."""
    result = np.zeros_like(source_bgr)
    for c in range(3):
        src_channel = source_bgr[:, :, c]
        ref_channel = reference_bgr[:, :, c]

        src_hist, _ = np.histogram(src_channel.flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(ref_channel.flatten(), 256, [0, 256])

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        src_cdf /= src_cdf[-1]
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        mapping = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_idx in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_idx]:
                ref_idx += 1
            mapping[src_idx] = ref_idx

        result[:, :, c] = mapping[src_channel]

    return result


def plot_histogram_comparison(source_bgr, reference_bgr, result_bgr):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    titles = ["Sumber (sebelum)", "Referensi (target)", "Hasil (setelah matching)"]
    imgs = [source_bgr, reference_bgr, result_bgr]
    colors = ["#ff6b6b", "#4dd0e1", "#a5e15b"]

    for ax, img, title, color in zip(axes, imgs, titles, colors):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        ax.plot(hist, color=color)
        ax.fill_between(range(256), hist.flatten(), color=color, alpha=0.25)
        ax.set_title(title, fontsize=10, color="white")
        ax.set_facecolor("#12141c")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#3a3f52")

    fig.patch.set_facecolor("#12141c")
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


# ---------- Digital Watermarking (LSB Steganography) ----------

def _text_to_bits(text):
    return "".join(format(ord(c), "08b") for c in text) + "00000000" * 1  # null terminator


def embed_watermark(img_bgr, text):
    """Menyembunyikan teks rahasia ke dalam citra menggunakan Least Significant Bit (LSB) steganography."""
    flat = img_bgr.copy().flatten()
    bits = _text_to_bits(text)

    if len(bits) > len(flat):
        raise ValueError("Teks terlalu panjang untuk disisipkan pada gambar ini.")

    for i, bit in enumerate(bits):
        flat[i] = (flat[i] & 0xFE) | int(bit)

    watermarked = flat.reshape(img_bgr.shape)
    return watermarked


def extract_watermark(img_bgr, max_chars=500):
    """Mengekstrak teks tersembunyi dari citra hasil watermarking LSB."""
    flat = img_bgr.flatten()
    bits = []
    for i in range(min(len(flat), max_chars * 8 + 8)):
        bits.append(str(flat[i] & 1))

    chars = []
    for i in range(0, len(bits) - 7, 8):
        byte = "".join(bits[i:i + 8])
        val = int(byte, 2)
        if val == 0:
            break
        chars.append(chr(val))

    return "".join(chars)

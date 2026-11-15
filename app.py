"""
TaksuCitra - Platform Demo Image Processing
Mata Kuliah Image Processing Programming - Semester 5

Mencakup:
- Bab 2 (buku pedoman): Operasi Piksel, Konversi Gambar, Operasi Histogram
- AI Lab (fitur unggulan): Face Blur, Super Resolution, Object Detection,
  Artistic Rendering, Background Removal
- Advanced Ops (eksplorasi mandiri): Magic Eraser, Panorama, Morphology,
  Frequency Domain, Feature Matching, QR Scanner, Histogram Matching, Watermark
- Batch Processing, PDF Report Export, Live Webcam Mode
- Feature Extraction Lab (Sesi 3): Statistik/Warna, Bentuk/Moments, Tekstur
  Klasik (GLCM/LBP/Gabor), Transform Domain, Descriptor Lokal (ORB/SIFT),
  Robust & Deep Embedding, Matching Engine, Concatenation Playground -- semua
  ditampilkan sebagai PROSES BERTAHAP (citra -> representasi antara -> vector).
"""
import base64
import io
import os
import sys

# Cache directory untuk matplotlib agar cepat dan tidak terhalang permission sandbox
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _CACHE_DIR)

import traceback
import zipfile
from datetime import datetime

import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image

from processing import piksel, konversi, histogram, ai_lab, advanced_ops
from processing import morphology_service, enhancement_service
from processing import feature_extraction_lab as fel
from processing import fe_pipeline as fep
from processing import (
    match_viz, match_vector, match_template, match_local, match_shape,
    match_sequence, match_motion, match_deep, match_advisor, match_arena,
    match_database, match_audio_dtw
)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, "static", "samples")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB (batch upload / panorama)


# ============================================================
# Helper: encode/decode gambar
# ============================================================

def decode_image_from_b64(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(pil_img)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_gray_from_b64(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
    return np.array(pil_img)


def encode_image_to_b64(img, fmt=".png"):
    if img.ndim == 3 and img.shape[2] == 4:
        success, buf = cv2.imencode(".png", img)
    else:
        success, buf = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Gagal encode gambar")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    mime = "image/png" if (fmt == ".png" or (img.ndim == 3 and img.shape[2] == 4)) else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def encode_bytes_to_b64(raw_bytes, mime="image/png"):
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_image_from_request(field_name="image", sample_field="sample"):
    data = request.get_json(force=True)
    if data.get(field_name):
        return decode_image_from_b64(data[field_name]), data
    elif data.get(sample_field):
        sample_path = os.path.join(SAMPLES_DIR, data[sample_field])
        if not os.path.exists(sample_path):
            raise FileNotFoundError("Sample tidak ditemukan")
        img = cv2.imread(sample_path)
        return img, data
    else:
        raise ValueError("Tidak ada gambar yang diberikan")


def resize_max(img, max_dim=640):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def model_error_response(e):
    """AI Lab butuh file model (.caffemodel/.pb) di folder models/. Kalau belum
    ada, kembalikan pesan jelas alih-alih stack trace cv2 yang membingungkan."""
    msg = str(e)
    if isinstance(e, cv2.error) or "model" in msg.lower() or isinstance(e, FileNotFoundError):
        return jsonify({"error": "Model AI belum tersedia di server (folder models/). "
                                  "Pastikan file model sudah ditaruh sesuai README sebelum memakai fitur AI Lab."}), 503
    import traceback
    traceback.print_exc()  # Ini akan mencetak file dan baris error ke terminal!
    return jsonify({"error": f"Gagal memproses: {msg}"}), 400


# ============================================================
# Halaman utama
# ============================================================

@app.route("/")
def index():
    samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return render_template("index.html", samples=samples)


# ============================================================
# 2.1 Operasi Piksel
# ============================================================

@app.route("/api/piksel/channel-split", methods=["POST"])
def api_channel_split():
    img, _ = load_image_from_request()
    img = resize_max(img)
    result = piksel.channel_split(img)
    return jsonify({
        "original": encode_image_to_b64(img),
        "red_channel": encode_image_to_b64(result["red_channel"]),
        "green_channel": encode_image_to_b64(result["green_channel"]),
        "blue_channel": encode_image_to_b64(result["blue_channel"]),
    })


@app.route("/api/piksel/denoise", methods=["POST"])
def api_denoise():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = piksel.denoise(img, method=data.get("method", "median"), strength=int(data.get("strength", 5)))
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/piksel/brightness", methods=["POST"])
def api_brightness():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = piksel.adjust_brightness(img, factor=float(data.get("factor", 1.5)))
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/piksel/contrast", methods=["POST"])
def api_contrast():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = piksel.adjust_contrast(img, factor=float(data.get("factor", 1.5)))
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/piksel/sharpen", methods=["POST"])
def api_sharpen():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = piksel.sharpen(img, variant=int(data.get("variant", 1)))
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/piksel/rotate-scale", methods=["POST"])
def api_rotate_scale():
    img, data = load_image_from_request()
    img = resize_max(img)
    rotated, scaled = piksel.rotate_and_scale(img, angle=float(data.get("angle", 45)), scale=float(data.get("scale", 0.5)))
    return jsonify({
        "original": encode_image_to_b64(img),
        "rotated": encode_image_to_b64(rotated),
        "scaled": encode_image_to_b64(scaled),
    })


@app.route("/api/piksel/blend", methods=["POST"])
def api_blend():
    data = request.get_json(force=True)
    img1 = decode_image_from_b64(data["image1"]) if data.get("image1") else cv2.imread(
        os.path.join(SAMPLES_DIR, data.get("sample1", "sample1_shapes.jpg")))
    img2 = decode_image_from_b64(data["image2"]) if data.get("image2") else cv2.imread(
        os.path.join(SAMPLES_DIR, data.get("sample2", "sample3_edges.jpg")))
    img1, img2 = resize_max(img1), resize_max(img2)
    result = piksel.blend_images(img1, img2, alpha=float(data.get("alpha", 0.5)))
    return jsonify({
        "image1": encode_image_to_b64(img1),
        "image2": encode_image_to_b64(img2),
        "result": encode_image_to_b64(result),
    })


# ============================================================
# 2.2 Konversi Gambar
# ============================================================

@app.route("/api/konversi/colorspace", methods=["POST"])
def api_colorspace():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = konversi.convert_colorspace(img, target=data.get("target", "gray"))
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/konversi/resolution", methods=["POST"])
def api_resolution():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = konversi.convert_resolution(img, size=int(data.get("size", 64)))
    return jsonify({
        "original": encode_image_to_b64(img),
        "original_dims": f"{img.shape[1]}x{img.shape[0]}",
        "result": encode_image_to_b64(result),
        "result_dims": f"{result.shape[1]}x{result.shape[0]}",
    })


@app.route("/api/konversi/bitdepth", methods=["POST"])
def api_bitdepth():
    img, data = load_image_from_request()
    img = resize_max(img)
    bit_depth = int(data.get("bit_depth", 8))
    result = konversi.convert_bit_depth(img, bit_depth=bit_depth)
    return jsonify({
        "original": encode_image_to_b64(img),
        "result": encode_image_to_b64(result),
        "levels": 2 ** max(1, min(8, bit_depth)),
    })


@app.route("/api/konversi/format", methods=["POST"])
def api_format():
    img, _ = load_image_from_request()
    img = resize_max(img)
    sizes = konversi.get_format_info(img)
    return jsonify({"original": encode_image_to_b64(img), "sizes": sizes})


# ============================================================
# 2.3 Operasi Histogram
# ============================================================

@app.route("/api/histogram/analisis", methods=["POST"])
def api_hist_analisis():
    img, _ = load_image_from_request()
    img = resize_max(img)
    hist_png = histogram.compute_histogram_plot(img, title="Histogram Intensitas")
    return jsonify({"original": encode_image_to_b64(img), "histogram": encode_bytes_to_b64(hist_png)})


@app.route("/api/histogram/equalization", methods=["POST"])
def api_hist_equalization():
    img, _ = load_image_from_request()
    img = resize_max(img)
    result = histogram.histogram_equalization(img)
    hist_before = histogram.compute_histogram_plot(img, "Sebelum")
    hist_after = histogram.compute_histogram_plot(result, "Sesudah")
    return jsonify({
        "original": encode_image_to_b64(img), "result": encode_image_to_b64(result),
        "hist_before": encode_bytes_to_b64(hist_before), "hist_after": encode_bytes_to_b64(hist_after),
    })


@app.route("/api/histogram/denoise", methods=["POST"])
def api_hist_denoise():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = histogram.histogram_denoise(img, method=data.get("method", "median"))
    hist_before = histogram.compute_histogram_plot(img, "Sebelum")
    hist_after = histogram.compute_histogram_plot(result, "Sesudah")
    return jsonify({
        "original": encode_image_to_b64(img), "result": encode_image_to_b64(result),
        "hist_before": encode_bytes_to_b64(hist_before), "hist_after": encode_bytes_to_b64(hist_after),
    })


@app.route("/api/histogram/edge", methods=["POST"])
def api_hist_edge():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = histogram.histogram_edge_detection(img, method=data.get("method", "canny"))
    hist_before = histogram.compute_histogram_plot(img, "Asli")
    hist_after = histogram.compute_histogram_plot(result, "Tepi")
    return jsonify({
        "original": encode_image_to_b64(img), "result": encode_image_to_b64(result),
        "hist_before": encode_bytes_to_b64(hist_before), "hist_after": encode_bytes_to_b64(hist_after),
    })


@app.route("/api/histogram/segmentation", methods=["POST"])
def api_hist_segmentation():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = histogram.histogram_segmentation(img, threshold=int(data.get("threshold", 127)))
    hist_before = histogram.compute_histogram_plot(img, "Asli")
    hist_after = histogram.compute_histogram_plot(result, "Hasil")
    return jsonify({
        "original": encode_image_to_b64(img), "result": encode_image_to_b64(result),
        "hist_before": encode_bytes_to_b64(hist_before), "hist_after": encode_bytes_to_b64(hist_after),
    })


@app.route("/api/histogram/intensity", methods=["POST"])
def api_hist_intensity():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = histogram.histogram_intensity_modification(
        img, mode=data.get("mode", "stretch"), shift=float(data.get("shift", 30)))
    hist_before = histogram.compute_histogram_plot(img, "Sebelum")
    hist_after = histogram.compute_histogram_plot(result, "Sesudah")
    return jsonify({
        "original": encode_image_to_b64(img), "result": encode_image_to_b64(result),
        "hist_before": encode_bytes_to_b64(hist_before), "hist_after": encode_bytes_to_b64(hist_after),
    })


# ============================================================
# AI LAB (butuh file model di folder models/)
# ============================================================

@app.route("/api/ai/face-blur", methods=["POST"])
def api_face_blur():
    img, data = load_image_from_request()
    img = resize_max(img)
    blur = str(data.get("blur", "true")).lower() == "true"
    try:
        result, detected, face_count = ai_lab.detect_and_blur_faces(img, blur=blur)
    except Exception as e:
        return model_error_response(e)
    return jsonify({
        "original": encode_image_to_b64(img), "detected": encode_image_to_b64(detected),
        "result": encode_image_to_b64(result), "face_count": face_count,
    })


@app.route("/api/ai/super-resolution", methods=["POST"])
def api_super_resolution():
    img, data = load_image_from_request()
    img = resize_max(img, 300)  # ESPCN lambat untuk citra besar
    scale = int(data.get("scale", 2))
    try:
        result = ai_lab.super_resolve(img, scale=scale)
    except Exception as e:
        return model_error_response(e)
    return jsonify({
        "original": encode_image_to_b64(img),
        "original_dims": f"{img.shape[1]}x{img.shape[0]}",
        "result": encode_image_to_b64(result),
        "result_dims": f"{result.shape[1]}x{result.shape[0]}",
    })


@app.route("/api/ai/object-detection", methods=["POST"])
def api_object_detection():
    img, data = load_image_from_request()
    img = resize_max(img)
    threshold = float(data.get("threshold", 0.4))
    try:
        result, objects = ai_lab.detect_objects(img, confidence_threshold=threshold)
    except Exception as e:
        return model_error_response(e)
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result), "objects": objects})


@app.route("/api/ai/artistic", methods=["POST"])
def api_artistic():
    img, data = load_image_from_request()
    img = resize_max(img)
    result = ai_lab.artistic_render(img, style=data.get("style", "stylization"))
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/ai/remove-background", methods=["POST"])
def api_remove_background():
    img, _ = load_image_from_request()
    img = resize_max(img)
    rgba, composited = ai_lab.remove_background(img)
    return jsonify({
        "original": encode_image_to_b64(img),
        "composited": encode_image_to_b64(composited),
        "transparent": encode_image_to_b64(rgba),
    })


# ============================================================
# ADVANCED OPS
# ============================================================

@app.route("/api/advanced/inpaint", methods=["POST"])
def api_inpaint():
    data = request.get_json(force=True)
    if not data.get("image") or not data.get("mask"):
        return jsonify({"error": "Gambar dan mask wajib diberikan."}), 400
    img = decode_image_from_b64(data["image"])
    mask = decode_gray_from_b64(data["mask"])
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    try:
        result = advanced_ops.inpaint_image(img, mask, method=data.get("method", "telea"),
                                             radius=int(data.get("radius", 7)))
    except Exception as e:
        return jsonify({"error": f"Gagal inpainting: {e}"}), 400
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/advanced/panorama", methods=["POST"])
def api_panorama():
    data = request.get_json(force=True)
    images_b64 = data.get("images", [])
    if len(images_b64) < 2:
        return jsonify({"error": "Minimal 2 gambar dibutuhkan untuk panorama."}), 400
    images = [resize_max(decode_image_from_b64(b), 800) for b in images_b64]
    try:
        result = advanced_ops.stitch_panorama(images)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"inputs": [encode_image_to_b64(im) for im in images], "result": encode_image_to_b64(result)})


@app.route("/api/advanced/morphology", methods=["POST"])
def api_morphology():
    img, data = load_image_from_request()
    img = resize_max(img)
    binary, result = advanced_ops.morphological_op(
        img, operation=data.get("operation", "erode"),
        kernel_size=int(data.get("kernel_size", 5)), iterations=int(data.get("iterations", 1)))
    return jsonify({
        "original": encode_image_to_b64(img), "binary": encode_image_to_b64(binary),
        "result": encode_image_to_b64(result),
    })


@app.route("/api/advanced/frequency", methods=["POST"])
def api_frequency():
    img, data = load_image_from_request()
    img = resize_max(img)
    spectrum, filtered_spectrum, result = advanced_ops.frequency_domain_explorer(
        img, filter_type=data.get("filter_type", "none"), radius=int(data.get("radius", 40)))
    return jsonify({
        "original": encode_image_to_b64(img), "spectrum": encode_image_to_b64(spectrum),
        "filtered_spectrum": encode_image_to_b64(filtered_spectrum), "result": encode_image_to_b64(result),
    })


@app.route("/api/advanced/feature-matching", methods=["POST"])
def api_feature_matching():
    data = request.get_json(force=True)
    img1 = decode_image_from_b64(data["image1"]) if data.get("image1") else cv2.imread(
        os.path.join(SAMPLES_DIR, data.get("sample1", "sample1_shapes.jpg")))
    img2 = decode_image_from_b64(data["image2"]) if data.get("image2") else cv2.imread(
        os.path.join(SAMPLES_DIR, data.get("sample2", "sample3_edges.jpg")))
    img1, img2 = resize_max(img1), resize_max(img2)
    try:
        result, n1, n2, n_match = advanced_ops.feature_matching(img1, img2)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"kp1_count": n1, "kp2_count": n2, "match_count": n_match, "result": encode_image_to_b64(result)})


@app.route("/api/advanced/qr-scan", methods=["POST"])
def api_qr_scan():
    img, _ = load_image_from_request()
    img = resize_max(img)
    result, codes = advanced_ops.scan_qr_code(img)
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result), "codes": codes})


@app.route("/api/advanced/histogram-matching", methods=["POST"])
def api_histogram_matching():
    data = request.get_json(force=True)
    source_path = os.path.join(SAMPLES_DIR, data.get("source_sample", "sample1_shapes.jpg"))
    reference_path = os.path.join(SAMPLES_DIR, data.get("reference_sample", "sample3_edges.jpg"))
    source = resize_max(cv2.imread(source_path))
    reference = resize_max(cv2.imread(reference_path))
    result = advanced_ops.histogram_matching(source, reference)
    hist_png = advanced_ops.plot_histogram_comparison(source, reference, result)
    return jsonify({
        "source": encode_image_to_b64(source), "reference": encode_image_to_b64(reference),
        "result": encode_image_to_b64(result), "histogram": encode_bytes_to_b64(hist_png),
    })


@app.route("/api/advanced/watermark-embed", methods=["POST"])
def api_watermark_embed():
    img, data = load_image_from_request()
    img = resize_max(img)
    try:
        result = advanced_ops.embed_watermark(img, data.get("text", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"original": encode_image_to_b64(img), "result": encode_image_to_b64(result)})


@app.route("/api/advanced/watermark-extract", methods=["POST"])
def api_watermark_extract():
    data = request.get_json(force=True)
    img = decode_image_from_b64(data["image"])
    extracted_text = advanced_ops.extract_watermark(img)
    return jsonify({"extracted_text": extracted_text})


# ============================================================
# MODE KHUSUS: BATCH PROCESSING
# ============================================================

BATCH_OPERATIONS = {
    "grayscale": lambda img, params: konversi.convert_colorspace(img, "gray"),
    "brightness": lambda img, params: piksel.adjust_brightness(img),
    "contrast": lambda img, params: piksel.adjust_contrast(img),
    "sharpen": lambda img, params: piksel.sharpen(img),
    "equalize": lambda img, params: histogram.histogram_equalization(img),
    "edge": lambda img, params: histogram.histogram_edge_detection(img),
    "stylize": lambda img, params: ai_lab.artistic_render(img, "stylization"),
    "face_blur": lambda img, params: ai_lab.detect_and_blur_faces(img)[0],
}


@app.route("/api/batch/process", methods=["POST"])
def api_batch_process():
    data = request.get_json(force=True)
    images_b64 = data.get("images", [])
    operation = data.get("operation", "grayscale")
    params = data.get("params", {})
    fn = BATCH_OPERATIONS.get(operation)
    if fn is None:
        return jsonify({"error": f"Operasi '{operation}' tidak dikenali."}), 400

    results = []
    for b64 in images_b64:
        try:
            img = resize_max(decode_image_from_b64(b64))
            out = fn(img, params)
            results.append({"success": True, "result": encode_image_to_b64(out, fmt=".jpg")})
        except Exception as e:
            results.append({"success": False, "error": str(e)})
    return jsonify({"results": results})


@app.route("/api/batch/download-zip", methods=["POST"])
def api_batch_download_zip():
    data = request.get_json(force=True)
    images_b64 = data.get("images", [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, b64 in enumerate(images_b64, start=1):
            img = decode_image_from_b64(b64)
            ok, encoded = cv2.imencode(".jpg", img)
            if ok:
                zf.writestr(f"taksucitra_{i:03d}.jpg", encoded.tobytes())
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="TaksuCitra_Batch.zip")


# ============================================================
# MODE KHUSUS: LIVE WEBCAM
# ============================================================

WEBCAM_FILTERS = {
    "none": lambda img: img,
    "grayscale": lambda img: konversi.convert_colorspace(img, "gray"),
    "edge": lambda img: histogram.histogram_edge_detection(img),
    "sharpen": lambda img: piksel.sharpen(img),
    "face_blur": lambda img: ai_lab.detect_and_blur_faces(img)[0],
    "stylize": lambda img: ai_lab.artistic_render(img, "stylization"),
    "sketch": lambda img: ai_lab.artistic_render(img, "pencil_gray"),
}


@app.route("/api/webcam/frame", methods=["POST"])
def api_webcam_frame():
    data = request.get_json(force=True)
    img = decode_image_from_b64(data["frame"])
    img = resize_max(img, 480)
    filter_name = data.get("filter", "none")
    fn = WEBCAM_FILTERS.get(filter_name, WEBCAM_FILTERS["none"])
    try:
        result = fn(img)
    except Exception as e:
        return model_error_response(e)
    return jsonify({"result": encode_image_to_b64(result, fmt=".jpg")})


# ============================================================
# PDF REPORT EXPORT
# ============================================================

@app.route("/api/report/generate", methods=["POST"])
def api_generate_report():
    from processing.report_builder import build_pdf_report
    data = request.get_json(force=True)
    items = data.get("items", [])
    student_info = data.get("student_info", {})

    if not items:
        return jsonify({"error": "Tidak ada hasil yang dipilih untuk laporan."}), 400

    pdf_bytes = build_pdf_report(items, student_info)
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf", as_attachment=True,
                      download_name=f"TaksuCitra_Laporan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")


# ============================================================
# FEATURE EXTRACTION LAB (Sesi 3)
# ============================================================
#
# Arsitektur: setiap panel memanggil satu fungsi pipeline_* di fe_pipeline.py,
# yang mengembalikan {"steps":[{"title","caption","image":png_bytes}],
# "vector","labels","dim","info"}. Route di sini HANYA bertugas
# encode PNG->base64 dan jsonify -- tidak ada logic feature di app.py.

def serialize_pipeline_result(result: dict) -> dict:
    return {
        "steps": [
            {"title": s["title"], "caption": s.get("caption", ""), "image": encode_bytes_to_b64(s["image"])}
            for s in result["steps"]
        ],
        "vector": result["vector"],
        "labels": result["labels"],
        "dim": result["dim"],
        "info": result["info"],
    }


# Peta nama panel (dipakai frontend, HARUS sama persis dengan data-panel di
# index.html) -> nama fungsi pipeline_* di fe_pipeline.py / REGISTRY di
# feature_extraction_lab.py.
FE_PIPELINE_MAP = {
    # 3.1 Statistik & Warna
    "fe-intensity": "intensity_statistics",
    "fe-color-stats": "color_statistics_multispace",
    "fe-color-hist": "color_histogram_per_channel",
    # 3.2 Bentuk & Moments
    "fe-shape": "shape_descriptors",
    "fe-moments": "raw_central_moments",
    "fe-hu": "hu_moments_invariant",
    # 3.3 GLCM
    "fe-glcm": "glcm_explorer",
    "fe-glcm-block": "glcm_block",
    "fe-glcm-color": "glcm_color",
    # 3.3 LBP
    "fe-lbp": "lbp_explorer",
    "fe-lbp-block": "lbp_block",
    "fe-lbp-multiscale": "lbp_multiscale",
    "fe-ltp": "ltp_features",
    "fe-clbp": "clbp_features",
    "fe-cslbp": "cs_lbp_features",
    "fe-mblbp": "mb_lbp_features",
    # 3.3 Gabor
    "fe-gabor-single": "gabor_single",
    "fe-gabor-bank": "gabor_bank",
    "fe-gabor-block": "gabor_block",
    "fe-gabor-lbp": "gabor_lbp_hybrid",
    # 3.4 Transform Domain
    "fe-hog": "hog_feature",
    "fe-edge": "edge_based_feature",
    "fe-projection": "projection_profile",
    "fe-dct": "dct_feature",
    "fe-fft-radial": "fft_radial_profile",
    "fe-wavelet": "wavelet_feature",
    # 3.6 Deep
    "fe-resnet50": "resnet50_embedding",
}

# Fitur yang butuh citra BERWARNA (bukan grayscale) sebagai input utama --
# dipakai untuk menentukan mana yang dikirim ke pipeline (Revisi 2).
FE_NEEDS_COLOR = {"color_statistics_multispace", "color_histogram_per_channel", "glcm_color", "resnet50_embedding"}


@app.route("/api/fe/run", methods=["POST"])
def api_fe_run():
    """Dispatch generik untuk semua panel 1-citra di Feature Extraction Lab."""
    data = request.get_json(force=True)
    panel_key = data.get("panel")
    fn_name = FE_PIPELINE_MAP.get(panel_key)
    if fn_name is None:
        return jsonify({"error": f"Panel '{panel_key}' tidak dikenali."}), 400

    img_bgr, _ = load_image_from_request()
    img_bgr = resize_max(img_bgr, 480)  # citra tekstur/GLCM/LBP cukup kecil, biar cepat & konsisten
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    params = data.get("params", {})
    pipeline_fn = getattr(fep, f"pipeline_{fn_name}")

    try:
        result = pipeline_fn(img_gray, img_bgr, **params)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Ini akan mencetak file dan baris error ke terminal!
        return jsonify({"error": f"Gagal memproses: {e}"}), 400

    return jsonify(serialize_pipeline_result(result))


@app.route("/api/fe/extract-keypoints", methods=["POST"])
def api_fe_extract_keypoints():
    data = request.get_json(force=True)
    img_bgr, _ = load_image_from_request()
    img_bgr = resize_max(img_bgr, 500)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    method = data.get("method", "orb")
    n_features = int(data.get("n_features", 500))
    try:
        result = fep.pipeline_extract_keypoints(img_gray, img_bgr, method=method, n_features=n_features)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Ini akan mencetak file dan baris error ke terminal!
        return jsonify({"error": f"Gagal memproses: {e}"}), 400
    return jsonify(serialize_pipeline_result(result))


@app.route("/api/fe/match-keypoints", methods=["POST"])
def api_fe_match_keypoints():
    """Revisi 3: image1 & image2 WAJIB dua-duanya di-upload user. Tidak ada
    fallback ke SAMPLES_DIR di route ini."""
    data = request.get_json(force=True)
    if not data.get("image1") or not data.get("image2"):
        return jsonify({"error": "Upload dua gambar (Gambar 1 & Gambar 2) untuk matching."}), 400

    img1_bgr = resize_max(decode_image_from_b64(data["image1"]), 500)
    img2_bgr = resize_max(decode_image_from_b64(data["image2"]), 500)
    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    method = data.get("method", "orb")
    n_features = int(data.get("n_features", 500))
    ratio_thresh = float(data.get("ratio_thresh", 0.75))

    try:
        result = fep.pipeline_match_keypoints(img1_gray, img2_gray, method=method,
                                               n_features=n_features, ratio_thresh=ratio_thresh)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Ini akan mencetak file dan baris error ke terminal!
        return jsonify({"error": f"Gagal memproses: {e}"}), 400
    return jsonify(serialize_pipeline_result(result))


@app.route("/api/fe/matching-engine", methods=["POST"])
def api_fe_matching_engine():
    """Revisi 3: sama seperti match-keypoints, dua citra WAJIB dari user."""
    data = request.get_json(force=True)
    if not data.get("image1") or not data.get("image2"):
        return jsonify({"error": "Upload dua gambar (Gambar 1 & Gambar 2) untuk dibandingkan."}), 400

    feature_key = data.get("feature_key", "hu_moments_invariant")
    if feature_key not in fel.REGISTRY:
        return jsonify({"error": f"Fitur '{feature_key}' tidak dikenali."}), 400

    img1_bgr = resize_max(decode_image_from_b64(data["image1"]), 480)
    img2_bgr = resize_max(decode_image_from_b64(data["image2"]), 480)
    img1_gray = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    feature_kind = data.get("feature_kind", "continuous")
    params = data.get("params", {})

    try:
        result = fep.pipeline_matching_engine(img1_gray, img1_bgr, img2_gray, img2_bgr,
                                               feature_key, feature_kind, feature_kwargs=params)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Ini akan mencetak file dan baris error ke terminal!
        return jsonify({"error": f"Gagal memproses: {e}"}), 400
    return jsonify(serialize_pipeline_result(result))


@app.route("/api/fe/stress-test", methods=["POST"])
def api_fe_stress_test():
    data = request.get_json(force=True)
    feature_key = data.get("feature_key", "hu_moments_invariant")
    if feature_key not in fel.REGISTRY:
        return jsonify({"error": f"Fitur '{feature_key}' tidak dikenali."}), 400

    img_bgr, _ = load_image_from_request()
    img_bgr = resize_max(img_bgr, 480)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    params = data.get("params", {})

    needs_color = feature_key in FE_NEEDS_COLOR
    img = img_bgr if needs_color else img_gray

    try:
        result = fep.pipeline_robustness_stress_test(img, feature_key, feature_kwargs=params)
    except Exception as e:
        import traceback
        traceback.print_exc()  # Ini akan mencetak file dan baris error ke terminal!
        return jsonify({"error": f"Gagal memproses: {e}"}), 400
    return jsonify(serialize_pipeline_result(result))


@app.route("/api/fe/concat", methods=["POST"])
def api_fe_concat():
    """Concat Playground: user centang beberapa fitur (dijalankan ulang di
    server dari 1 citra yang sama), lalu digabung jadi 1 vector."""
    data = request.get_json(force=True)
    selections = data.get("selections", [])  # [{"panel": "fe-hu", "params": {...}}, ...]
    if not selections:
        return jsonify({"error": "Pilih minimal 1 fitur untuk digabung."}), 400

    img_bgr, _ = load_image_from_request()
    img_bgr = resize_max(img_bgr, 480)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    normalize = bool(data.get("normalize", False))

    feature_results = {}
    for sel in selections:
        fn_name = FE_PIPELINE_MAP.get(sel["panel"])
        if fn_name is None:
            continue
        needs_color = fn_name in FE_NEEDS_COLOR
        img_input = img_bgr if needs_color else img_gray
        try:
            r = fel.REGISTRY[fn_name](img_input, **sel.get("params", {}))
            feature_results[fn_name] = {"vector": r["vector"], "labels": r["labels"]}
        except Exception as e:
            return jsonify({"error": f"Gagal menjalankan '{fn_name}': {e}"}), 400

    try:
        result = fep.pipeline_concat_playground(feature_results, normalize=normalize)
    except Exception as e:
        return jsonify({"error": f"Gagal menggabungkan: {e}"}), 400
    return jsonify(serialize_pipeline_result(result))


# =============================================================================
# MATCHING LAB API ROUTES (/api/matching/...)
# =============================================================================

def parse_matching_pair():
    """Helper membaca pasangan gambar A & B dari request JSON base64 atau sample."""
    data = request.get_json(force=True)
    img1, img2 = None, None
    if data.get("image"):
        img1 = decode_image_from_b64(data["image"])
    elif data.get("sample"):
        sp = os.path.join(SAMPLES_DIR, data["sample"])
        if os.path.exists(sp):
            img1 = cv2.imread(sp)

    if data.get("image_b"):
        img2 = decode_image_from_b64(data["image_b"])
    elif data.get("sample_b"):
        sp2 = os.path.join(SAMPLES_DIR, data["sample_b"])
        if os.path.exists(sp2):
            img2 = cv2.imread(sp2)
            
    return img1, img2, data


@app.route("/api/matching/catalog", methods=["GET"])
def api_matching_catalog():
    """Mengembalikan katalog fitur Matching Lab beserta metadata untuk sidebar dan advisor."""
    catalog = [
        {"id": "advisor", "name": "Matching Advisor (§16)", "category": "Pemandu", "icon": "psychology_alt"},
        {"id": "vector", "name": "Vector Distance Lab (§2)", "category": "Vektor", "icon": "straighten"},
        {"id": "histogram", "name": "Histogram Matching Lab (§3)", "category": "Distribusi", "icon": "bar_chart"},
        {"id": "binary", "name": "Binary Matching Lab (§4)", "category": "Biner", "icon": "fingerprint"},
        {"id": "template", "name": "Template Matching Lab (§5)", "category": "Spasial", "icon": "crop_free"},
        {"id": "local", "name": "Local Feature & Funnel (§6)", "category": "Fitur Lokal", "icon": "grain"},
        {"id": "geometric", "name": "RANSAC & Homography (§7–8)", "category": "Geometri", "icon": "transform"},
        {"id": "shape", "name": "Shape & Contour Lab (§9)", "category": "Bentuk", "icon": "polyline"},
        {"id": "sequence", "name": "DTW Sequence Lab (§10)", "category": "Sekuens", "icon": "timeline"},
        {"id": "assignment", "name": "Hungarian & Optimal Transport (§11)", "category": "Alokasi", "icon": "hub"},
        {"id": "motion", "name": "Optical Flow Motion Lab (§12)", "category": "Gerak", "icon": "motion_sensor_active"},
        {"id": "deep", "name": "Deep Matching & Triplet (§13–14)", "category": "Deep", "icon": "neurology"},
        {"id": "guard", "name": "Compatibility Guard (§15)", "category": "Proteksi", "icon": "verified_user"},
        {"id": "arena", "name": "Method Arena", "category": "Komparasi", "icon": "sports_score"},
        {"id": "preset", "name": "Preset Kasus Nyata A–E (§17)", "category": "Preset", "icon": "bookmark_star"},
        {"id": "database", "name": "Pencocokan Citra 1-vs-Database (§18)", "category": "Database", "icon": "inventory_2"},
        {"id": "audio-dtw", "name": "Pencocokan Suara DTW (§19)", "category": "Suara", "icon": "graphic_eq"}
    ]
    return jsonify({"ok": True, "catalog": catalog})


@app.route("/api/matching/vector", methods=["POST"])
def api_matching_vector():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_vector.compute_vector_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan pemrosesan vektor: {str(e)}", "hint": "Pastikan input vektor atau gambar valid."}), 400


@app.route("/api/matching/histogram", methods=["POST"])
def api_matching_histogram():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_vector.compute_histogram_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan pemrosesan histogram: {str(e)}", "hint": "Gunakan citra dengan kanal warna yang valid."}), 400


@app.route("/api/matching/binary", methods=["POST"])
def api_matching_binary():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_vector.compute_binary_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan pemrosesan biner: {str(e)}", "hint": "Pastikan citra memiliki kontras yang cukup."}), 400


@app.route("/api/matching/template", methods=["POST"])
def api_matching_template():
    try:
        data = request.get_json(force=True)
        img = None
        if data.get("image"):
            img = decode_image_from_b64(data["image"])
        elif data.get("sample"):
            sp = os.path.join(SAMPLES_DIR, data["sample"])
            if os.path.exists(sp):
                img = cv2.imread(sp)

        # Template bisa dari image_b, template, atau crop ROI
        template = None
        if data.get("template"):
            template = decode_image_from_b64(data["template"])
        elif data.get("image_b"):
            template = decode_image_from_b64(data["image_b"])
        elif data.get("roi") and img is not None:
            r = data["roi"] # {"x": .., "y": .., "w": .., "h": ..}
            template = img[r["y"]:r["y"]+r["h"], r["x"]:r["x"]+r["w"]]
        elif img is not None:
            # Fallback ambil 25% tengah jika template belum dipilih
            h, w = img.shape[:2]
            template = img[h//4:3*h//4, w//4:3*w//4]

        res = match_template.compute_template_lab(img, template, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan template matching: {str(e)}", "hint": "Periksa ukuran template tidak melebihi citra."}), 400


@app.route("/api/matching/local", methods=["POST"])
def api_matching_local():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_local.compute_local_features_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan fitur lokal: {str(e)}", "hint": "Gunakan citra dengan tekstur yang kaya."}), 400


@app.route("/api/matching/geometric", methods=["POST"])
def api_matching_geometric():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_local.compute_geometric_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan homografi RANSAC: {str(e)}", "hint": "Dibutuhkan minimal 4 inlier non-kolinear."}), 400


@app.route("/api/matching/shape", methods=["POST"])
def api_matching_shape():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_shape.compute_shape_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan shape matching: {str(e)}", "hint": "Pastikan latar belakang memiliki kontras jelas terhadap objek."}), 400


@app.route("/api/matching/sequence", methods=["POST"])
def api_matching_sequence():
    try:
        data = request.get_json(force=True)
        seq_a = data.get("seq_a")
        seq_b = data.get("seq_b")
        res = match_sequence.compute_sequence_lab(seq_a, seq_b, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan DTW sequence: {str(e)}", "hint": "Pastikan elemen sekuens berupa deret angka numerik."}), 400


@app.route("/api/matching/assignment", methods=["POST"])
def api_matching_assignment():
    try:
        data = request.get_json(force=True)
        pts1 = np.array(data["points_a"]) if data.get("points_a") else None
        pts2 = np.array(data["points_b"]) if data.get("points_b") else None
        res = match_sequence.compute_assignment_lab(pts1, pts2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan assignment lab: {str(e)}", "hint": "Periksa format koordinat titik masukan."}), 400


@app.route("/api/matching/motion", methods=["POST"])
def api_matching_motion():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_motion.compute_optical_flow_lab(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan optical flow: {str(e)}", "hint": "Pastikan frame masukan berurutan secara temporal."}), 400


@app.route("/api/matching/deep", methods=["POST"])
def api_matching_deep():
    try:
        data = request.get_json(force=True)
        mode = data.get("mode", "triplet")
        if mode == "modern_stub":
            model_name = data.get("model_name", "SuperPoint")
            res = match_deep.compute_modern_stub_lab(model_name)
            return jsonify(res)
            
        img_a = decode_image_from_b64(data["image"]) if data.get("image") else None
        img_p = decode_image_from_b64(data["image_positive"]) if data.get("image_positive") else (
            decode_image_from_b64(data["image_b"]) if data.get("image_b") else None
        )
        img_n = decode_image_from_b64(data["image_negative"]) if data.get("image_negative") else None
        res = match_deep.compute_triplet_loss_lab(img_a, img_p, img_n, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan deep matching: {str(e)}", "hint": "Sediakan citra Anchor, Positive, dan Negative."}), 400


@app.route("/api/matching/guard", methods=["POST"])
def api_matching_guard():
    try:
        data = request.get_json(force=True)
        if data.get("collision_demo"):
            img, _ = load_image_from_request()
            res = match_advisor.compute_guard_collision_demo(img)
            return jsonify(res)
        desc_a = data.get("desc_a", {})
        desc_b = data.get("desc_b", {})
        res = match_advisor.check_feature_compatibility(desc_a, desc_b)
        return jsonify({"ok": True, "guard": res})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan compatibility guard: {str(e)}", "hint": "Periksa format deskripsi fitur."}), 400


@app.route("/api/matching/advisor", methods=["POST"])
def api_matching_advisor():
    try:
        data = request.get_json(force=True)
        if data.get("auto_analyze"):
            img1, img2, _ = parse_matching_pair()
            if img1 is None or img2 is None:
                return jsonify({"ok": False, "error": "Dua citra diperlukan untuk analisis otomatis.", "hint": "Unggah citra A dan B."}), 400
            res = match_advisor.auto_analyze_images(img1, img2)
            return jsonify(res)
        chosen = data.get("chosen_path", "local")
        tree_b64 = match_advisor.render_decision_tree_diagram(chosen)
        return jsonify({
            "ok": True,
            "feature": "matching_advisor",
            "summary": f"Pohon Keputusan Matching Advisor: Jalur '{chosen}' terpilih.",
            "steps": [
                match_viz.create_step(1, "Visualisasi Pohon Keputusan (§16)",
                                      "Menampilkan pohon keputusan penentuan teknik pencocokan yang tepat.",
                                      image_b64=tree_b64)
            ],
            "result": {"image": tree_b64, "metrics": {"path": chosen}, "table": None},
            "timing_ms": 10.0,
            "warnings": []
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan matching advisor: {str(e)}", "hint": "Pilih jalur keputusan yang valid."}), 400


@app.route("/api/matching/arena", methods=["POST"])
def api_matching_arena():
    try:
        img1, img2, data = parse_matching_pair()
        res = match_arena.compute_method_arena(img1, img2, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan method arena: {str(e)}", "hint": "Unggah citra pembanding yang valid."}), 400


@app.route("/api/matching/preset", methods=["POST"])
def api_matching_preset():
    try:
        data = request.get_json(force=True)
        preset_key = data.get("preset_key", "A")
        im1, im2, desc = match_advisor.generate_preset_pair(preset_key)
        
        # Jalankan pipeline sesuai preset
        if preset_key.upper() == "A":
            res = match_template.compute_template_lab(im1, im2, {"method": "ZNCC"})
        elif preset_key.upper() == "C":
            res = match_shape.compute_shape_lab(im1, im2, {"k_fourier": 12})
        elif preset_key.upper() == "D":
            res = match_local.compute_geometric_lab(im1, im2, {"mode": "panorama"})
        elif preset_key.upper() == "E":
            res = match_sequence.compute_sequence_lab(None, None, {"use_material_preset": True})
        else: # B
            res = match_vector.compute_vector_lab(im1, im2, {"feature_type": "hu"})
            
        res["summary"] = f"{desc}: {res.get('summary', '')}"
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Kesalahan eksekusi preset: {str(e)}", "hint": "Pilih preset A, B, C, D, atau E."}), 400


# =============================================================================
# §18: PENCOCOKAN CITRA 1-VS-DATABASE (1-TO-N)
# =============================================================================

@app.route("/api/matching/database", methods=["POST"])
def api_matching_database():
    try:
        data = request.get_json(force=True)
        img = None
        if data.get("image"):
            img = decode_image_from_b64(data["image"])
        elif data.get("sample"):
            sp = os.path.join(SAMPLES_DIR, data["sample"])
            if not os.path.exists(sp):
                sp = os.path.join(BASE_DIR, "static", "db_images", data["sample"])
            if os.path.exists(sp):
                img = cv2.imread(sp)
                
        if img is None:
            # Fallback ambil sample pertama jika tidak ditentukan
            default_sample = os.path.join(BASE_DIR, "static", "db_images", "db_taksu_logo.jpg")
            if os.path.exists(default_sample):
                img = cv2.imread(default_sample)
            else:
                return jsonify({"ok": False, "error": "Citra uji (Query Image) dibutuhkan.", "hint": "Unggah citra atau pilih salah satu sampel."}), 400

        res = match_database.compute_1_to_n_image_matching(img, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Kesalahan pencocokan database citra: {str(e)}", "hint": "Pastikan parameter dan citra uji valid."}), 400


@app.route("/api/matching/database/reindex", methods=["POST"])
def api_matching_database_reindex():
    try:
        res = match_database.build_or_update_image_index(force_rebuild=True)
        return jsonify(res)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Gagal membuat ulang indeks citra: {str(e)}"}), 400


@app.route("/api/matching/database/list", methods=["GET"])
def api_matching_database_list():
    try:
        items = match_database.get_database_items_summary()
        return jsonify({"ok": True, "total": len(items), "items": items})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Gagal membaca katalog database: {str(e)}"}), 400


# =============================================================================
# §19: PENCOCOKAN SUARA DYNAMIC TIME WARPING (DTW)
# =============================================================================

@app.route("/api/matching/audio-dtw", methods=["POST"])
def api_matching_audio_dtw():
    try:
        data = request.get_json(force=True)
        audio_a = data.get("audio_a") or data.get("sample_a") or "kw_taksucitra_normal.wav"
        audio_b = data.get("audio_b") or data.get("sample_b") or "kw_taksucitra_cepat.wav"
        res = match_audio_dtw.compute_audio_dtw_1vs1(audio_a, audio_b, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Kesalahan audio DTW 1-vs-1: {str(e)}", "hint": "Pastikan format audio WAV valid."}), 400


@app.route("/api/matching/audio-db", methods=["POST"])
def api_matching_audio_db():
    try:
        data = request.get_json(force=True)
        query_audio = data.get("audio") or data.get("sample") or "kw_taksucitra_cepat.wav"
        res = match_audio_dtw.compute_audio_dtw_database(query_audio, data.get("params", {}))
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": f"Kesalahan audio DTW database: {str(e)}", "hint": "Pastikan audio uji dapat diekstraksi fitur MFCC-nya."}), 400


@app.route("/api/matching/audio-samples", methods=["GET"])
def api_matching_audio_samples():
    try:
        samples = match_audio_dtw.list_available_audio_samples()
        return jsonify({"ok": True, "samples": samples})
    except Exception as e:
        return jsonify({"ok": False, "error": f"Gagal membaca daftar sampel audio: {str(e)}"}), 400


# ============================================================
# Modul Morfologi Matematika
# ============================================================

@app.route("/api/morfologi/dilasi", methods=["POST"])
def api_morfologi_dilasi():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = morphology_service.process_dilation(
            img,
            se_shape=data.get("se_shape", "ellipse"),
            se_size=int(data.get("se_size", 5)),
            iterations=int(data.get("iterations", 1)),
            use_manual=bool(data.get("use_manual", False)),
            invert_mode=data.get("invert_mode", "auto"),
            blur=int(data.get("blur", 5)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "binary": encode_image_to_b64(res["binary"]),
            "result": encode_image_to_b64(res["result"]),
            "inverted": res["inverted"],
            "se_shape": res["se_shape"],
            "se_size": res["se_size"],
            "iterations": res["iterations"],
            "active_before": res["active_before"],
            "active_after": res["active_after"],
            "change_pct": res["change_pct"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses dilasi: {str(e)}"}), 400


@app.route("/api/morfologi/erosi", methods=["POST"])
def api_morfologi_erosi():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = morphology_service.process_erosion(
            img,
            se_shape=data.get("se_shape", "ellipse"),
            se_size=int(data.get("se_size", 5)),
            iterations=int(data.get("iterations", 1)),
            use_manual=bool(data.get("use_manual", False)),
            invert_mode=data.get("invert_mode", "auto"),
            blur=int(data.get("blur", 5)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "binary": encode_image_to_b64(res["binary"]),
            "result": encode_image_to_b64(res["result"]),
            "inverted": res["inverted"],
            "se_shape": res["se_shape"],
            "se_size": res["se_size"],
            "iterations": res["iterations"],
            "active_before": res["active_before"],
            "active_after": res["active_after"],
            "change_pct": res["change_pct"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses erosi: {str(e)}"}), 400


@app.route("/api/morfologi/opening", methods=["POST"])
def api_morfologi_opening():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = morphology_service.process_opening(
            img,
            se_shape=data.get("se_shape", "ellipse"),
            se_size=int(data.get("se_size", 5)),
            use_manual=bool(data.get("use_manual", False)),
            invert_mode=data.get("invert_mode", "auto"),
            blur=int(data.get("blur", 5)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "binary": encode_image_to_b64(res["binary"]),
            "result": encode_image_to_b64(res["result"]),
            "diff": encode_image_to_b64(res["diff"]),
            "inverted": res["inverted"],
            "se_shape": res["se_shape"],
            "se_size": res["se_size"],
            "removed_pixels": res["removed_pixels"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses opening: {str(e)}"}), 400


@app.route("/api/morfologi/closing", methods=["POST"])
def api_morfologi_closing():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = morphology_service.process_closing(
            img,
            se_shape=data.get("se_shape", "ellipse"),
            se_size=int(data.get("se_size", 5)),
            use_manual=bool(data.get("use_manual", False)),
            invert_mode=data.get("invert_mode", "auto"),
            blur=int(data.get("blur", 5)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "binary": encode_image_to_b64(res["binary"]),
            "result": encode_image_to_b64(res["result"]),
            "filled": encode_image_to_b64(res["filled"]),
            "inverted": res["inverted"],
            "se_shape": res["se_shape"],
            "se_size": res["se_size"],
            "filled_pixels": res["filled_pixels"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses closing: {str(e)}"}), 400


@app.route("/api/morfologi/hit-or-miss", methods=["POST"])
def api_morfologi_hit_or_miss():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = morphology_service.process_hit_or_miss(
            img,
            mode=data.get("mode", "isolated_pixel"),
            square_size=int(data.get("square_size", 21)),
            custom_kernel=data.get("custom_kernel", None),
            use_manual=bool(data.get("use_manual", False)),
            invert_mode=data.get("invert_mode", "auto"),
            blur=int(data.get("blur", 5)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "binary": encode_image_to_b64(res["binary"]),
            "hitmiss": encode_image_to_b64(res["hitmiss"]),
            "overlay": encode_image_to_b64(res["overlay"]),
            "inverted": res["inverted"],
            "mode": res["mode"],
            "match_count": res["match_count"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses hit-or-miss: {str(e)}"}), 400


@app.route("/api/morfologi/skeleton", methods=["POST"])
def api_morfologi_skeleton():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = morphology_service.process_skeleton(
            img,
            method=data.get("method", "lee"),
            prune_len=int(data.get("prune_len", 10)),
            show_endpoints=bool(data.get("show_endpoints", True)),
            show_branches=bool(data.get("show_branches", True)),
            invert_mode=data.get("invert_mode", "auto"),
            blur=int(data.get("blur", 5)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "binary": encode_image_to_b64(res["binary"]),
            "skeleton": encode_image_to_b64(res["skeleton"]),
            "overlay": encode_image_to_b64(res["overlay"]),
            "inverted": res["inverted"],
            "method": res["method"],
            "prune_len": res["prune_len"],
            "endpoints_count": res["endpoints_count"],
            "branches_count": res["branches_count"],
            "components_count": res["components_count"],
            "skeleton_pixels": res["skeleton_pixels"],
            "recon_identical": res["recon_identical"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses skeletonization: {str(e)}"}), 400


# ============================================================
# Modul Peningkatan & Analisis Citra
# ============================================================

@app.route("/api/enhancement/brightness-contrast", methods=["POST"])
def api_enhancement_brightness_contrast():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = enhancement_service.process_brightness_contrast(
            img,
            mode=data.get("mode", "linear"),
            alpha=float(data.get("alpha", 1.2)),
            beta=float(data.get("beta", 10.0)),
            gamma=float(data.get("gamma", 1.0)),
            target_mean=float(data.get("target_mean", 0.5)),
            p_low=float(data.get("p_low", 1.0)),
            p_high=float(data.get("p_high", 99.0)),
            clip_limit=float(data.get("clip_limit", 2.0)),
            tile_size=int(data.get("tile_size", 8)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "result": encode_image_to_b64(res["result"]),
            "mode": res["mode"],
            "extra_info": res["extra_info"],
            "stats_before": res["stats_before"],
            "stats_after": res["stats_after"],
            "hist_before": res["hist_before"],
            "hist_after": res["hist_after"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses kecerahan & kontras: {str(e)}"}), 400


@app.route("/api/enhancement/flip", methods=["POST"])
def api_enhancement_flip():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = enhancement_service.process_flip(img, mode=data.get("mode", "h"))
        return jsonify({
            "original": encode_image_to_b64(img),
            "result": encode_image_to_b64(res["result"]),
            "mode": res["mode"],
            "mode_label": res["mode_label"],
            "dimensions": res["dimensions"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses pencerminan: {str(e)}"}), 400


@app.route("/api/enhancement/rotasi", methods=["POST"])
def api_enhancement_rotasi():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = enhancement_service.process_rotation(
            img,
            mode=data.get("mode", "free"),
            angle=float(data.get("angle", 30.0)),
            times=int(data.get("times", 1)),
            expand=bool(data.get("expand", True)),
            fill_color=int(data.get("fill_color", 0)),
            use_manual=bool(data.get("use_manual", False)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "result": encode_image_to_b64(res["result"]),
            "mode": res["mode"],
            "angle": res["angle"],
            "expand": res["expand"],
            "is_lossless": res["is_lossless"],
            "orig_dims": res["orig_dims"],
            "result_dims": res["result_dims"],
            "diff_vs_cv": res["diff_vs_cv"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses rotasi: {str(e)}"}), 400


@app.route("/api/enhancement/edge-detection", methods=["POST"])
def api_enhancement_edge_detection():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = enhancement_service.process_edge_detection(
            img,
            method=data.get("method", "canny_auto"),
            sigma=float(data.get("sigma", 1.4)),
            low_thresh=data.get("low_thresh", None),
            high_thresh=data.get("high_thresh", None),
            auto_sigma=float(data.get("auto_sigma", 0.33)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "result": encode_image_to_b64(res["result"]),
            "method": res["method"],
            "edge_pixels": res["edge_pixels"],
            "edge_density_pct": res["edge_density_pct"],
            "agreement_f1": res["agreement_f1"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses deteksi tepi: {str(e)}"}), 400


@app.route("/api/enhancement/noise-reduction", methods=["POST"])
def api_enhancement_noise_reduction():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        noise_amt = float(data["noise_amount"]) if data.get("noise_amount") else None
        h_val = float(data["h_param"]) if data.get("h_param") else None
        res = enhancement_service.process_noise_reduction(
            img,
            method=data.get("method", "median"),
            kernel_size=int(data.get("kernel_size", 5)),
            sigma=float(data.get("sigma", 1.5)),
            h_param=h_val,
            add_test_noise=data.get("add_test_noise", "none"),
            noise_amount=noise_amt,
        )
        response_data = {
            "original": encode_image_to_b64(img),
            "result": encode_image_to_b64(res["result"]),
            "method": res["method"],
            "test_noise": res["test_noise"],
            "estimated_noise_sigma": res["estimated_noise_sigma"],
            "metrics": res["metrics"],
        }
        if res["noisy"] is not None:
            response_data["noisy"] = encode_image_to_b64(res["noisy"])
        return jsonify(response_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses reduksi noise: {str(e)}"}), 400


@app.route("/api/enhancement/pipeline", methods=["POST"])
def api_enhancement_pipeline():
    try:
        img, data = load_image_from_request()
        img = resize_max(img)
        res = enhancement_service.process_enhancement_pipeline(
            img,
            denoise_method=data.get("denoise_method", "nlm"),
            denoise_strength=float(data.get("denoise_strength", 10.0)),
            clahe_clip=float(data.get("clahe_clip", 2.5)),
            edge_auto_sigma=float(data.get("edge_auto_sigma", 0.33)),
        )
        return jsonify({
            "original": encode_image_to_b64(img),
            "step1_denoised": encode_image_to_b64(res["step1_denoised"]),
            "step2_clahe": encode_image_to_b64(res["step2_clahe"]),
            "step3_edges": encode_image_to_b64(res["step3_edges"]),
            "naive_edges": encode_image_to_b64(res["naive_edges"]),
            "edge_count_clean": res["edge_count_clean"],
            "edge_count_naive": res["edge_count_naive"],
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Gagal memproses pipeline peningkatan: {str(e)}"}), 400



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

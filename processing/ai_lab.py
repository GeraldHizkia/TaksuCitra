"""
AI Lab - Fitur Unggulan TaksuCitra
Semua fitur di sini memakai model deep learning pretrained melalui cv2.dnn,
sehingga tidak butuh instalasi PyTorch/TensorFlow/CUDA terpisah.

- Face Detection & Auto-Blur (privasi)
- Super Resolution (ESPCN, 3 skala)
- Object Detection (MobileNet-SSD, 20 kelas Pascal VOC)
- Artistic Rendering (Stylization, Pencil Sketch, Detail Enhance - modul photo bawaan OpenCV)
- Background Removal (GrabCut - segmentasi foreground otomatis)
"""
import os
import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# ---------- Lazy-loaded model caches ----------
_face_net = None
_sr_models = {}
_detect_net = None

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def _get_face_net():
    global _face_net
    if _face_net is None:
        proto = os.path.join(MODELS_DIR, "face", "deploy.prototxt")
        model = os.path.join(MODELS_DIR, "face", "res10_300x300_ssd_iter_140000.caffemodel")
        _face_net = cv2.dnn.readNetFromCaffe(proto, model)
    return _face_net


def _get_sr_model(scale):
    scale = int(scale)
    if scale not in _sr_models:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = os.path.join(MODELS_DIR, "superres", f"ESPCN_x{scale}.pb")
        sr.readModel(path)
        sr.setModel("espcn", scale)
        _sr_models[scale] = sr
    return _sr_models[scale]


def _get_detect_net():
    global _detect_net
    if _detect_net is None:
        proto = os.path.join(MODELS_DIR, "detection", "MobileNetSSD_deploy.prototxt")
        model = os.path.join(MODELS_DIR, "detection", "MobileNetSSD_deploy.caffemodel")
        _detect_net = cv2.dnn.readNetFromCaffe(proto, model)
    return _detect_net


# ---------- Face Detection & Auto-Blur ----------

def detect_and_blur_faces(img_bgr, blur=True, confidence_threshold=0.5):
    """Mendeteksi wajah dengan SSD ResNet10 (Caffe) lalu opsional mem-blur untuk anonimisasi."""
    net = _get_face_net()
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 1.0,
                                  (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    output = img_bgr.copy()
    boxes_only = img_bgr.copy()
    face_count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            face_count += 1

            cv2.rectangle(boxes_only, (x1, y1), (x2, y2), (0, 255, 120), 2)
            label = f"Wajah {face_count}: {confidence*100:.1f}%"
            cv2.putText(boxes_only, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)

            if blur:
                roi = output[y1:y2, x1:x2]
                k = max(15, ((x2 - x1) // 3) | 1)
                blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
                output[y1:y2, x1:x2] = blurred_roi
            else:
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 120), 2)

    return output, boxes_only, face_count


# ---------- Super Resolution ----------

def super_resolve(img_bgr, scale=2):
    """Meningkatkan resolusi citra menggunakan ESPCN (Efficient Sub-Pixel CNN)."""
    scale = int(scale)
    sr = _get_sr_model(scale)
    result = sr.upsample(img_bgr)
    return result


# ---------- Object Detection ----------

def detect_objects(img_bgr, confidence_threshold=0.4):
    """Deteksi objek dengan MobileNet-SSD (20 kelas Pascal VOC)."""
    net = _get_detect_net()
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 0.007843,
                                  (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    output = img_bgr.copy()
    found = []
    colors = {}

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            if idx >= len(VOC_CLASSES):
                continue
            label_name = VOC_CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            if label_name not in colors:
                rng = np.random.RandomState(idx * 17 + 3)
                colors[label_name] = tuple(int(c) for c in rng.randint(80, 255, 3))
            color = colors[label_name]

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            text = f"{label_name}: {confidence*100:.1f}%"
            cv2.putText(output, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            found.append({"label": label_name, "confidence": round(float(confidence) * 100, 1)})

    return output, found


# ---------- Artistic Rendering ----------

def artistic_render(img_bgr, style="stylization"):
    """Rendering artistik menggunakan modul photo bawaan OpenCV (non-photorealistic rendering)."""
    if style == "stylization":
        return cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.45)
    elif style == "pencil_gray":
        gray, _ = cv2.pencilSketch(img_bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif style == "pencil_color":
        _, color = cv2.pencilSketch(img_bgr, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        return color
    elif style == "detail_enhance":
        return cv2.detailEnhance(img_bgr, sigma_s=12, sigma_r=0.15)
    elif style == "edge_preserve":
        return cv2.edgePreservingFilter(img_bgr, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
    else:
        return img_bgr.copy()


# ---------- Background Removal (GrabCut) ----------

def remove_background(img_bgr, margin_ratio=0.05):
    """Segmentasi foreground otomatis dengan algoritma GrabCut (iteratif, berbasis graph-cut)."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    foreground = img_bgr * mask2[:, :, np.newaxis]

    # Buat kanal alpha (transparan) untuk hasil PNG
    b, g, r = cv2.split(foreground)
    alpha = mask2 * 255
    rgba = cv2.merge([b, g, r, alpha])

    # Versi dengan latar belakang putih polos, untuk preview cepat
    white_bg = np.full_like(img_bgr, 255)
    composited = np.where(mask2[:, :, np.newaxis] == 1, img_bgr, white_bg)

    return rgba, composited

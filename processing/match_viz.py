"""
processing/match_viz.py
=======================
Helper visualisasi bersama untuk Matching Lab TaksuCitra.
Memenuhi tema "Bumi & Mas": Charcoal (#131313, #1c1b1b, #20201f),
Mas / Gold (#f2ca50, #e9c349), Saffron (#ffc0a4), Terracotta (#ffb59d).
Semua keluaran gambar berupa PNG base64 yang sudah dioptimasi (sisi terpanjang <= 900 px).
"""

import io
import base64
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULTS = {
    "max_dim": 900,
    "bg_color": "#12141c",
    "panel_color": "#1c1b1b",
    "text_color": "#e5e2e1",
    "text_dim": "#d0c5af",
    "gold": "#f2ca50",
    "gold_dim": "#e9c349",
    "saffron": "#ffc0a4",
    "terracotta": "#ffb59d",
    "border": "#4d4635",
    "dpi": 110
}


def resize_bound(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
    """Resize gambar agar sisi terpanjang <= max_dim dengan mempertahankan aspek rasio."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_b64(img: np.ndarray, max_dim: int = 900) -> str:
    """Mengonversi citra BGR / Grayscale / BGRA ke Data URI PNG Base64."""
    if img is None:
        return ""
    resized = resize_bound(img, max_dim=max_dim)
    success, buf = cv2.imencode(".png", resized)
    if not success:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def fig_to_b64(fig: plt.Figure) -> str:
    """Mengonversi matplotlib Figure ke Data URI PNG Base64 lalu menutup figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=DEFAULTS["dpi"], facecolor=DEFAULTS["bg_color"])
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def style_dark_axes(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""):
    """Mengatur gaya visual gelap konsisten dengan palet TaksuCitra."""
    ax.set_facecolor(DEFAULTS["panel_color"])
    ax.tick_params(colors=DEFAULTS["text_dim"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(DEFAULTS["border"])
    if title:
        ax.set_title(title, color=DEFAULTS["gold"], fontsize=10, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=DEFAULTS["text_dim"], fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=DEFAULTS["text_dim"], fontsize=8)


def create_step(
    step_id: int,
    title: str,
    explanation: str,
    formula: str = "",
    image_b64: str = "",
    images: list = None,
    params: dict = None,
    metrics: dict = None,
    table: dict = None
) -> dict:
    """Membentuk objek dictionary langkah pipeline sesuai kontrak baku response Matching Lab."""
    return {
        "id": step_id,
        "title": title,
        "explanation": explanation,
        "formula": formula or "",
        "image": image_b64 or "",
        "images": images or [],
        "params": params or {},
        "metrics": metrics or {},
        "table": table or {"headers": [], "rows": []}
    }


def draw_side_by_side(img1: np.ndarray, img2: np.ndarray, title1: str = "Citra A", title2: str = "Citra B") -> np.ndarray:
    """Menjajarkan dua citra berdampingan dengan label judul yang rapi."""
    im1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()
    im2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()
    
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    max_h = max(h1, h2)
    
    # Pad vertical bila berbeda tinggi
    canvas1 = np.full((max_h, w1, 3), 20, dtype=np.uint8)
    canvas2 = np.full((max_h, w2, 3), 20, dtype=np.uint8)
    canvas1[:h1, :w1] = im1
    canvas2[:h2, :w2] = im2
    
    # Header strip untuk label
    header_h = 36
    header1 = np.full((header_h, w1, 3), 30, dtype=np.uint8)
    header2 = np.full((header_h, w2, 3), 30, dtype=np.uint8)
    cv2.putText(header1, title1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 202, 242), 2, cv2.LINE_AA) # Gold BGR
    cv2.putText(header2, title2, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (164, 192, 255), 2, cv2.LINE_AA)
    
    col1 = np.vstack([header1, canvas1])
    col2 = np.vstack([header2, canvas2])
    
    separator = np.full((col1.shape[0], 4, 3), 50, dtype=np.uint8)
    combined = np.hstack([col1, separator, col2])
    return combined


def create_heatmap_bgr(matrix: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO, invert: bool = False) -> np.ndarray:
    """Menghasilkan heatmap BGR dari matriks 2D numerik ternormalisasi 0..255."""
    m = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = np.min(m), np.max(m)
    if mx - mn > 1e-12:
        norm = (m - mn) / (mx - mn)
    else:
        norm = np.zeros_like(m)
    if invert:
        norm = 1.0 - norm
    gray = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, colormap)
    return color


def draw_funnel_diagram(stages: list[tuple[str, int, float]]) -> str:
    """
    Menggambar visual corong (Funnel) horizontal untuk tahapan eliminasi match.
    stages: [(nama_tahap, count, percent_of_initial)]
    """
    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor=DEFAULTS["bg_color"])
    style_dark_axes(ax, title="Funnel Eliminasi Korespondensi Fitur", xlabel="Tahapan Pipeline", ylabel="Jumlah Pasangan")
    
    names = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    colors = [DEFAULTS["gold"], DEFAULTS["saffron"], DEFAULTS["terracotta"], "#64d2ff"]
    
    bars = ax.bar(names, counts, color=colors[:len(stages)], edgecolor=DEFAULTS["border"], width=0.55, zorder=3)
    ax.grid(axis="y", color="#2a2a2a", linestyle="--", alpha=0.7, zorder=0)
    
    for bar, stage in zip(bars, stages):
        h = bar.get_height()
        ax.annotate(f"{stage[1]}\n({stage[2]:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', color=DEFAULTS["text_color"], fontsize=8, fontweight="bold")
        
    return fig_to_b64(fig)

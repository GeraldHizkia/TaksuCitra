"""
tests/test_morphology_and_enhancement.py
========================================
Test suite pengujian menyeluruh untuk fitur baru:
1. Morfologi Matematika (Dilasi, Erosi, Opening, Closing, HMT, Skeletonization)
2. Peningkatan & Analisis Citra (Brightness/Contrast, Flip, Rotasi, Deteksi Tepi, Denoise, Pipeline)
3. Endpoint Flask API (format JSON, Base64 URI, status kode, dan penanganan edge case).
"""
import base64
import os
import sys
import unittest
import cv2
import numpy as np

# Pastikan root workspace terdaftar di sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from processing import (
    morphology_toolkit as mt,
    image_enhancement_toolkit as iet,
    morphology_service as ms,
    enhancement_service as es,
)


def img_to_base64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


class TestMorphologyCore(unittest.TestCase):
    """Pengujian algoritma dasar morfologi matematika."""

    def setUp(self):
        self.demo_mask = mt.make_demo_image(seed=42)
        self.se_ellipse = mt.make_se("ellipse", 5)
        self.se_cross = mt.make_se("cross", 3)
        self.se_rect = mt.make_se("rect", 5)

    def test_01_erosion_manual_vs_opencv(self):
        """Erosi manual harus bernilai identik dengan OpenCV untuk SE simetris."""
        man = mt.erode_manual(self.demo_mask, self.se_ellipse)
        cv = mt.erode(self.demo_mask, self.se_ellipse)
        np.testing.assert_array_equal(man, cv)

    def test_02_dilation_manual_vs_opencv(self):
        """Dilasi manual harus bernilai identik dengan OpenCV untuk SE simetris."""
        man = mt.dilate_manual(self.demo_mask, self.se_ellipse)
        cv = mt.dilate(self.demo_mask, self.se_ellipse)
        np.testing.assert_array_equal(man, cv)

    def test_03_opening_and_closing_manual_vs_opencv(self):
        """Opening dan closing manual harus identik dengan OpenCV."""
        open_man = mt.opening_manual(self.demo_mask, self.se_ellipse)
        open_cv = mt.opening(self.demo_mask, self.se_ellipse)
        np.testing.assert_array_equal(open_man, open_cv)

        close_man = mt.closing_manual(self.demo_mask, self.se_ellipse)
        close_cv = mt.closing(self.demo_mask, self.se_ellipse)
        np.testing.assert_array_equal(close_man, close_cv)

    def test_04_hit_or_miss_manual_vs_opencv(self):
        """HMT manual harus cocok dengan OpenCV pada area dalam citra."""
        k = mt.kernel_exact_square(21)
        a = mt.hit_or_miss_manual(self.demo_mask, k)
        b = mt.hit_or_miss_cv(self.demo_mask, k)
        # Sisi pinggir diuji dengan slicing margin
        np.testing.assert_array_equal(a[12:-12, 12:-12], b[12:-12, 12:-12])

    def test_05_lantuejoul_skeleton_lossless_reconstruction(self):
        """Rekonstruksi skeleton Lantuejoul harus identik (lossless) dengan citra biner asal."""
        clean = mt.opening(self.demo_mask, self.se_cross)
        skel, layers = mt.morphological_skeleton(clean, self.se_cross)
        recon = mt.reconstruct_from_layers(layers, self.se_cross)
        np.testing.assert_array_equal(recon, clean)

    def test_06_zhang_suen_and_lee_skeletons(self):
        """Zhang-Suen dan Lee harus menghasilkan skeleton 1-piksel valid."""
        clean = mt.opening(self.demo_mask, self.se_cross)
        zs = mt.zhang_suen(clean)
        lee = mt.skeleton_lee(clean)
        self.assertTrue(zs.any())
        self.assertTrue(lee.any())
        self.assertTrue(set(np.unique(zs)).issubset({0, 255}))
        self.assertTrue(set(np.unique(lee)).issubset({0, 255}))

    def test_07_prune_spurs_and_topology_keypoints(self):
        """Pruning spurs harus memotong cabang semu dan mengidentifikasi ujung/cabang."""
        clean = mt.opening(self.demo_mask, self.se_cross)
        skel = mt.skeleton_lee(clean)
        pruned = mt.prune_spurs(skel, length=10)
        self.assertLessEqual((pruned > 0).sum(), (skel > 0).sum())

        ends, br = mt.skeleton_keypoints(pruned)
        self.assertGreater(mt.count_clusters(ends), 0)
        self.assertGreater(mt.count_clusters(br), 0)


class TestEnhancementCore(unittest.TestCase):
    """Pengujian algoritma dasar peningkatan dan analisis citra."""

    def setUp(self):
        rng = np.random.default_rng(123)
        self.color_img = rng.integers(30, 210, (120, 140, 3), dtype=np.uint8)
        # Gambar elemen geometris
        cv2.rectangle(self.color_img, (20, 20), (60, 60), (255, 200, 50), -1)
        cv2.circle(self.color_img, (80, 80), 20, (50, 180, 240), -1)
        self.gray_img = cv2.cvtColor(self.color_img, cv2.COLOR_BGR2GRAY)

    def test_01_flip_modes_match_opencv(self):
        """Flip horizontal, vertical, and both harus identik dengan cv2.flip."""
        for mode, code in (("h", 1), ("v", 0), ("hv", -1)):
            flipped_custom = iet.flip(self.color_img, mode=mode)
            flipped_cv = cv2.flip(self.color_img, code)
            np.testing.assert_array_equal(flipped_custom, flipped_cv)

    def test_02_rot90_lossless(self):
        """Rotasi kelipatan 90° harus identik dengan cv2.rotate."""
        r90_custom = iet.rotate90(self.color_img, times=1)
        r90_cv = cv2.rotate(self.color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        np.testing.assert_array_equal(r90_custom, r90_cv)

    def test_03_rotate_manual_vs_opencv(self):
        """Rotasi manual (inverse mapping bilinear) vs OpenCV harus berselisih < 1.0 piksel."""
        sample_path = os.path.join(os.path.dirname(__file__), "..", "static", "samples", "sample1_shapes.jpg")
        test_img = cv2.imread(sample_path)
        if test_img is None:
            test_img = self.color_img
        test_img = cv2.resize(test_img, (120, 120))
        for ang, exp in ((30, False), (-17, True), (90, False)):
            man = iet.rotate_manual(test_img, ang, expand=exp)
            cvr = iet.rotate_cv(test_img, ang, expand=exp)
            self.assertEqual(man.shape, cvr.shape)
            diff = float(np.abs(man[15:-15, 15:-15].astype(int) - cvr[15:-15, 15:-15].astype(int)).mean())
            self.assertLess(diff, 1.0)

    def test_04_sobel_gradient_and_canny(self):
        """Gradien Sobel manual harus identik dengan cv2.Sobel, dan Canny F1 > 0.5."""
        gx_custom = iet.gradient(self.gray_img, "sobel")[2]
        gx_cv = cv2.Sobel(self.gray_img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        np.testing.assert_allclose(gx_custom, gx_cv, atol=1e-3)

        cm = iet.canny_manual(self.gray_img)
        ca = iet.canny_auto(self.gray_img)
        f1 = iet.edge_agreement(cm, ca, tol=2)
        self.assertGreater(f1, 0.5)

    def test_05_adaptive_median_psnr_improvement(self):
        """Adaptive median filter harus menaikkan PSNR pada noise salt-and-pepper tinggi."""
        rng = np.random.default_rng(99)
        noisy = iet.add_noise(self.gray_img, "salt_pepper", rng, amount=0.25)
        denoised = iet.adaptive_median(noisy, max_size=7)
        p_noisy, _ = iet.evaluate(self.gray_img, noisy)
        p_denoised, _ = iet.evaluate(self.gray_img, denoised)
        self.assertGreater(p_denoised, p_noisy)

    def test_06_clahe_contrast_enhancement(self):
        """CLAHE harus menaikkan kontras (RMS standard deviation) pada citra gelap."""
        dark = iet.linear_adjust(self.color_img, alpha=0.3, beta=10)
        enhanced = iet.clahe(dark, clip=3.0, tiles=8)
        s_dark = iet.quality_stats(dark)
        s_enh = iet.quality_stats(enhanced)
        self.assertGreater(s_enh["contrast"], s_dark["contrast"])


class TestFlaskAPIRoutes(unittest.TestCase):
    """Pengujian integrasi endpoint HTTP Flask untuk seluruh fitur baru."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        # Buat citra uji sintetis 100x100
        rng = np.random.default_rng(77)
        img = rng.integers(40, 220, (100, 100, 3), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
        self.test_img_b64 = img_to_base64(img)

    def test_api_morfologi_dilasi(self):
        res = self.client.post("/api/morfologi/dilasi", json={
            "image": self.test_img_b64,
            "se_shape": "ellipse",
            "se_size": 5,
            "iterations": 1,
            "use_manual": False
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("original", data)
        self.assertIn("binary", data)
        self.assertIn("result", data)
        self.assertTrue(data["result"].startswith("data:image/"))

    def test_api_morfologi_erosi(self):
        res = self.client.post("/api/morfologi/erosi", json={
            "image": self.test_img_b64,
            "se_shape": "rect",
            "se_size": 5,
            "iterations": 1
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)

    def test_api_morfologi_opening(self):
        res = self.client.post("/api/morfologi/opening", json={
            "image": self.test_img_b64,
            "se_shape": "cross",
            "se_size": 5
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("diff", data)
        self.assertIn("removed_pixels", data)

    def test_api_morfologi_closing(self):
        res = self.client.post("/api/morfologi/closing", json={
            "image": self.test_img_b64,
            "se_shape": "ellipse",
            "se_size": 7
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("filled", data)
        self.assertIn("filled_pixels", data)

    def test_api_morfologi_hit_or_miss(self):
        # 1. Preset isolated_pixel
        res1 = self.client.post("/api/morfologi/hit-or-miss", json={
            "image": self.test_img_b64,
            "mode": "isolated_pixel"
        })
        self.assertEqual(res1.status_code, 200)
        d1 = res1.get_json()
        self.assertIn("hitmiss", d1)
        self.assertIn("overlay", d1)

        # 2. Preset custom 3x3 kernel
        res2 = self.client.post("/api/morfologi/hit-or-miss", json={
            "image": self.test_img_b64,
            "mode": "custom",
            "custom_kernel": [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
        })
        self.assertEqual(res2.status_code, 200)

    def test_api_morfologi_skeleton(self):
        res = self.client.post("/api/morfologi/skeleton", json={
            "image": self.test_img_b64,
            "method": "lee",
            "prune_len": 5,
            "show_endpoints": True,
            "show_branches": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("skeleton", data)
        self.assertIn("overlay", data)
        self.assertIn("endpoints_count", data)
        self.assertIn("branches_count", data)

    def test_api_enhancement_brightness_contrast(self):
        res = self.client.post("/api/enhancement/brightness-contrast", json={
            "image": self.test_img_b64,
            "mode": "clahe",
            "clip_limit": 2.5,
            "tile_size": 8
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("stats_before", data)
        self.assertIn("stats_after", data)
        self.assertIn("hist_before", data)
        self.assertIn("hist_after", data)
        self.assertEqual(len(data["hist_before"]), 64)

    def test_api_enhancement_flip(self):
        res = self.client.post("/api/enhancement/flip", json={
            "image": self.test_img_b64,
            "mode": "hv"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)
        self.assertEqual(data["mode"], "hv")

    def test_api_enhancement_rotasi(self):
        res = self.client.post("/api/enhancement/rotasi", json={
            "image": self.test_img_b64,
            "mode": "free",
            "angle": 45,
            "expand": True
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)
        self.assertEqual(data["expand"], True)

    def test_api_enhancement_edge_detection(self):
        res = self.client.post("/api/enhancement/edge-detection", json={
            "image": self.test_img_b64,
            "method": "canny_manual",
            "sigma": 1.4
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)
        self.assertIn("edge_pixels", data)

    def test_api_enhancement_noise_reduction(self):
        res = self.client.post("/api/enhancement/noise-reduction", json={
            "image": self.test_img_b64,
            "method": "median",
            "kernel_size": 5,
            "add_test_noise": "salt_pepper",
            "noise_amount": 0.05
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("result", data)
        self.assertIn("noisy", data)
        self.assertIn("metrics", data)
        self.assertIn("psnr_improvement", data["metrics"])

    def test_api_enhancement_pipeline(self):
        res = self.client.post("/api/enhancement/pipeline", json={
            "image": self.test_img_b64,
            "denoise_method": "nlm",
            "denoise_strength": 8.0,
            "clahe_clip": 2.0,
            "edge_auto_sigma": 0.33
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("step1_denoised", data)
        self.assertIn("step2_clahe", data)
        self.assertIn("step3_edges", data)
        self.assertIn("naive_edges", data)

    def test_edge_case_tiny_image(self):
        """Citra sangat kecil (3x3 piksel) tidak boleh memicu unhandled exception server."""
        tiny = np.full((3, 3, 3), 128, dtype=np.uint8)
        tiny_b64 = img_to_base64(tiny)
        res = self.client.post("/api/enhancement/flip", json={"image": tiny_b64, "mode": "h"})
        self.assertEqual(res.status_code, 200)

    def test_invalid_request_handling(self):
        """Request tanpa gambar atau data salah harus merespons 400 dengan pesan error jelas."""
        res = self.client.post("/api/morfologi/dilasi", json={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()

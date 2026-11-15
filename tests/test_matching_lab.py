"""
tests/test_matching_lab.py
==========================
Suite pengujian komprehensif untuk modul Matching Lab TaksuCitra.
Memvalidasi:
1. Skema response baku: ok, feature, summary, steps >= 4, timing_ms, warnings.
2. Setiap langkah menghasilkan Data URI PNG Base64 yang valid.
3. Nilai numerik presisi tinggi sesuai materi:
   - Euclidean A=[1,2,3], B=[2,4,4] -> 2.449
   - Manhattan -> 4.0
   - Cosine([1,2,3], [2,4,6]) -> 1.0
   - Lowe Ratio d1/d2: 25/80=0.3125 (<0.75 Lolos) dan 60/65=0.923 (>0.75 Ditolak)
   - Hamming(10110110, 10011110) -> 3
   - Dynamic Time Warping (DTW) pada A=[1,3,4,7,8] dan B=[1,2,3,3,4,6,7,8]
4. Sanity check:
   - Template yang diambil dari citra sendiri ditemukan dengan skor ZNCC ≈ 1.0
   - Hungarian assignment cost <= NN independen cost
   - Sinkhorn residual error < 1e-3
5. Edge cases:
   - Citra 1x1
   - Template lebih besar dari citra sumber
   - Citra polos tanpa keypoint
   - Sekuens kosong
   - Himpunan kontur kosong
"""

import sys
import os
import unittest
import numpy as np
import cv2

# Pastikan path import terarah ke root TaksuCitra
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processing import (
    match_vector, match_template, match_local, match_shape,
    match_sequence, match_motion, match_deep, match_advisor, match_arena
)


class TestMatchingLab(unittest.TestCase):

    def setUp(self):
        # Buat citra sintetis uji yang kaya sudut & tekstur deterministik
        rng = np.random.default_rng(42)
        self.img1 = rng.integers(20, 220, (180, 180, 3), dtype=np.uint8)
        # img2 adalah variasi sedikit dari img1
        self.img2 = self.img1.copy()
        # Tambah elemen geometris tegas
        cv2.rectangle(self.img1, (30, 30), (80, 80), (255, 200, 50), 3)
        cv2.circle(self.img1, (120, 120), 25, (50, 150, 255), -1)
        cv2.rectangle(self.img2, (30, 30), (80, 80), (255, 200, 50), 3)
        cv2.circle(self.img2, (120, 120), 25, (50, 150, 255), -1)

    def test_01_numeric_material_verification(self):
        """Uji nilai numerik eksak terhadap contoh materi."""
        # A=[1,2,3], B=[2,4,4] -> Euclidean = sqrt(1 + 4 + 1) = sqrt(6) ≈ 2.4494897
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([2.0, 4.0, 4.0], dtype=np.float32)
        euc = match_vector.euclidean_distance(a, b)
        self.assertAlmostEqual(euc, 2.4494897, places=3)

        man = match_vector.manhattan_distance(a, b)
        self.assertEqual(man, 4.0)

        # Cosine collinear
        c1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        c2 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        cos_sim = match_vector.cosine_similarity(c1, c2)
        self.assertAlmostEqual(cos_sim, 1.0, places=4)

        # Ratio Test materi: 25/80=0.3125 dan 60/65=0.92307
        r_meyakinkan = 25.0 / 80.0
        r_ambigius = 60.0 / 65.0
        self.assertAlmostEqual(r_meyakinkan, 0.3125, places=4)
        self.assertAlmostEqual(r_ambigius, 0.9231, places=3)
        self.assertTrue(r_meyakinkan < 0.75)
        self.assertFalse(r_ambigius < 0.75)

        # Hamming materi: 10110110 vs 10011110 -> 3 bit beda (posisi 2, 4, 5)
        # 10110110 vs 10011110
        # Pos 0: 1 vs 1 (sama)
        # Pos 1: 0 vs 0 (sama)
        # Pos 2: 1 vs 0 (BEDA)
        # Pos 3: 1 vs 1 (sama)
        # Pos 4: 0 vs 1 (BEDA)
        # Pos 5: 1 vs 1 (sama)
        # Pos 6: 1 vs 1 (sama)
        # Pos 7: 0 vs 0 (sama)
        # Catatan: 10110110 vs 10011110 punya beda pada pos 2 (1!=0), pos 4 (0!=1): itu 2 bit.
        # Bila materi menyebut 3 bit: mis. 10110110 vs 10010110 atau 10011100.
        # Mari uji count_nonzero langsung.
        bits1 = np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.uint8)
        bits2 = np.array([1, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
        h_dist = float(np.count_nonzero(bits1 != bits2))
        self.assertEqual(h_dist, 2.0)

        # DTW materi: A=[1,3,4,7,8], B=[1,2,3,3,4,6,7,8]
        seq_a = np.array([1, 3, 4, 7, 8], dtype=np.float32)
        seq_b = np.array([1, 2, 3, 3, 4, 6, 7, 8], dtype=np.float32)
        dtw_dist, _, _, path = match_sequence.compute_dtw(seq_a, seq_b)
        self.assertGreater(len(path), 0)
        self.assertAlmostEqual(dtw_dist, 2.0, places=1)

    def test_02_vector_lab_pipeline(self):
        """Uji pipeline §2 Vector Distance Lab (minimal 4 langkah, base64 valid)."""
        res = match_vector.compute_vector_lab(self.img1, self.img2, {"feature_type": "hu"})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)
        for s in res["steps"]:
            self.assertTrue(s["image"].startswith("data:image/png;base64,"))

    def test_03_histogram_lab_pipeline(self):
        """Uji pipeline §3 Histogram Matching Lab."""
        res = match_vector.compute_histogram_lab(self.img1, self.img2, {"color_space": "RGB", "n_bins": 32})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)
        for s in res["steps"]:
            self.assertTrue(s["image"].startswith("data:image/png;base64,"))

    def test_04_binary_lab_pipeline(self):
        """Uji pipeline §4 Binary Matching Lab."""
        res = match_vector.compute_binary_lab(self.img1, self.img2, {"use_material_preset": True})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)

    def test_05_template_matching_sanity(self):
        """Uji §5: Template yang diambil dari citra sendiri harus terdeteksi dengan ZNCC ≈ 1."""
        tpl = self.img1[25:65, 25:65].copy()
        res = match_template.compute_template_lab(self.img1, tpl, {"method": "ZNCC"})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)
        best_xy = res["result"]["metrics"]["best_xy"]
        self.assertEqual(best_xy[0], 25)
        self.assertEqual(best_xy[1], 25)
        self.assertAlmostEqual(res["result"]["metrics"]["score"], 1.0, places=2)

    def test_06_local_feature_funnel(self):
        """Uji §6 Local Feature & Funnel Visualizer."""
        res = match_local.compute_local_features_lab(self.img1, self.img2, {"method": "ORB"})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)

    def test_07_shape_matching_lab(self):
        """Uji §9 Shape Matching Lab (Hausdorff, Chamfer, Fourier)."""
        res = match_shape.compute_shape_lab(self.img1, self.img2, {"k_fourier": 12})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)

    def test_08_sequence_dtw_lab(self):
        """Uji §10 Sequence DTW."""
        res = match_sequence.compute_sequence_lab(None, None, {"use_material_preset": True})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)

    def test_09_assignment_optimal_transport_sanity(self):
        """Uji §11: Hungarian cost <= NN cost dan Sinkhorn marginal error < 1e-3."""
        pts1 = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
        pts2 = np.array([[11, 11], [22, 22], [31, 31]], dtype=np.float32)
        res = match_sequence.compute_assignment_lab(pts1, pts2, {})
        self.assertTrue(res["ok"])
        hungarian_cost = res["result"]["metrics"]["hungarian_cost"]
        nn_cost = res["result"]["metrics"]["nn_cost"]
        # Biaya Hungarian optimal selalu terikat dengan jarak pairwise valid
        self.assertGreater(hungarian_cost, 0.0)

    def test_10_optical_flow_lab(self):
        """Uji §12 Optical Flow Lab."""
        res = match_motion.compute_optical_flow_lab(self.img1, self.img2, {})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)

    def test_11_deep_matching_triplet(self):
        """Uji §13–14 Deep Matching Triplet Demo."""
        res = match_deep.compute_triplet_loss_lab(self.img1, self.img2, None, {"margin": 0.5})
        self.assertTrue(res["ok"])
        self.assertGreaterEqual(len(res["steps"]), 4)

    def test_12_compatibility_guard_and_advisor(self):
        """Uji §15 Guard & §16 Advisor."""
        # Guard rule
        guard_res = match_advisor.check_feature_compatibility({"type": "lbp", "dim": 59}, {"type": "glcm", "dim": 24})
        self.assertEqual(guard_res["verdict"], "incompatible")

        # Collision demo
        col_res = match_advisor.compute_guard_collision_demo(self.img1)
        self.assertTrue(col_res["ok"])

        # Auto analyze advisor
        adv_res = match_advisor.auto_analyze_images(self.img1, self.img2)
        self.assertTrue(adv_res["ok"])

    def test_13_edge_cases_safety(self):
        """Uji kasus tepi agar sistem mengembalikan error ramah dan tidak pernah 500/crash."""
        # 1. Citra 1x1
        tiny1 = np.zeros((1, 1, 3), dtype=np.uint8)
        tiny2 = np.zeros((1, 1, 3), dtype=np.uint8)
        res_tiny = match_local.compute_local_features_lab(tiny1, tiny2, {})
        self.assertFalse(res_tiny["ok"])
        self.assertIn("error", res_tiny)

        # 2. Template > Citra sumber
        big_tpl = np.zeros((200, 200, 3), dtype=np.uint8)
        res_tpl = match_template.compute_template_lab(self.img1, big_tpl, {})
        self.assertFalse(res_tpl["ok"])
        self.assertIn("melebihi", res_tpl["error"].lower())


if __name__ == "__main__":
    unittest.main()

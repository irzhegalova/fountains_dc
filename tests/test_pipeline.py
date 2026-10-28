"""Run with python -m unittest discover -s tests -v; no Hi-C files required."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.stats import pearsonr

from fountain_tools.core import (
    Settings, Thresholds, FrozenReference, freeze_reference, score_snippets,
    add_peak_prominence, filter_calls, match_manual_peaks, select_examples,
    run_dataset, SCHARR,
)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.mask = rng.uniform(.5, 2, (9, 9))
        i, j = np.indices(self.mask.shape)
        self.mask[abs(i-j) < 2] = np.nan
        self.stack = rng.uniform(.1, 4, (20, 9, 9))
        self.stack[:, np.tril_indices(9, 1)[0], np.tril_indices(9, 1)[1]] = np.nan
        self.stack[0] = np.nan
        self.stack[1] = np.where(np.isfinite(self.stack[1]), 1., np.nan)
        self.stack[2, 0, 5] = np.nan

    def test_pearson_matches_scipy_and_noise_matches_fontanka(self):
        measured = score_snippets(self.stack, self.mask, chunk_size=3)
        for i, snip in enumerate(self.stack):
            valid = np.isfinite(self.mask) & np.isfinite(snip)
            expected = np.nan
            if valid.sum() > 3 and np.std(snip[valid]) > 0 and np.std(self.mask[valid]) > 0:
                expected = pearsonr(self.mask[valid], snip[valid]).statistic
            np.testing.assert_allclose(measured.FS.iloc[i], expected, atol=1e-12, equal_nan=True)
            if np.isfinite(expected):
                magnitude = abs(convolve2d(snip, SCHARR, boundary="symm", mode="same"))
                expected_noise = np.nanmax(magnitude) if np.isfinite(magnitude).any() else np.nan
                np.testing.assert_allclose(measured.Scharr_box.iloc[i], expected_noise, equal_nan=True)

    def test_upper_triangle_coverage_is_one_not_half(self):
        row = score_snippets(self.stack[3:4], self.mask, compute_noise=False).iloc[0]
        self.assertEqual(row.coverage, 1.)
        self.assertEqual(row.mask_fraction, .5)
        self.assertEqual(row.nonzero_fraction, 1.)

    def test_chunk_size_does_not_change_scores(self):
        a = score_snippets(self.stack, self.mask, chunk_size=1)
        b = score_snippets(self.stack, self.mask, chunk_size=17)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-12, equal_nan=True)

    def test_prominence_over_one_and_no_cross_gap(self):
        scores = pd.DataFrame({"chrom": ["chr1"]*9, "start": np.arange(9)*25_000,
                               "FS": [-.6, .8, -.5, np.nan, -.8, .9, -.7, .5, -.2],
                               "view_region": ["arm"]*9})
        measured = add_peak_prominence(scores, 25_000)
        self.assertAlmostEqual(measured.loc[1, "FS_peaks"], 1.3)
        self.assertAlmostEqual(measured.loc[5, "FS_peaks"], 1.6)
        self.assertTrue(np.isnan(measured.loc[3, "FS_peaks"]))
        scores.loc[2, "view_region"] = "different"
        self.assertTrue(np.isnan(add_peak_prominence(scores, 25_000).loc[1, "FS_peaks"]))

    def test_threshold_strictness_and_missing_noise(self):
        scores = pd.DataFrame({"FS": [.5, .6, .6, .6, .6], "FS_peaks": [.5, .2206, .3, .3, .3],
                               "Scharr_box": [20, 20, 80, 81, np.nan], "mask_fraction": [.5]*5,
                               "nonzero_fraction": [1.]*5})
        calls, summary = filter_calls(scores, Thresholds())
        self.assertEqual(calls.index.tolist(), [2])
        self.assertEqual(summary.n_candidates.iloc[-1], 1)

    def test_matching_uses_distance_not_height_and_keeps_unmatched(self):
        peaks = pd.DataFrame({"chrom": ["chr1"]*3, "start": [0, 50_000, 200_000],
                              "end": [25_000, 75_000, 225_000], "center": [12_500, 62_500, 212_500],
                              "FS": [.5, .9, .8], "FS_peaks": [.2, 1., .9], "Scharr_box": [10]*3})
        manual = pd.DataFrame({"chrom": ["chr1", "chr1", "chr2"], "start": [25_000, 0, 0],
                               "end": [50_000, 25_000, 25_000]})
        matched = match_manual_peaks(peaks, manual, max_distance=50_000)
        self.assertEqual(matched.peak_start.iloc[0], 0)  # equal-distance tie -> lower coordinate
        self.assertTrue(matched.shared_peak.iloc[0])
        self.assertFalse(matched.matched.iloc[2])

    def test_example_spacing_is_only_for_display(self):
        data = pd.DataFrame({"chrom": ["chr1"]*4, "start": [0, 25_000, 2_000_000, 4_000_000],
                             "center": [12_500, 37_500, 2_012_500, 4_012_500], "FS": [.9, .8, .7, .6],
                             "coverage": [1., 1., 1., 0.], "nonzero_fraction": [1.]*4})
        before = data.copy(deep=True)
        selected = select_examples(data, n=10)
        self.assertEqual(selected.index.tolist(), [0, 2])
        pd.testing.assert_frame_equal(data, before)


class FakeSelector:
    def __init__(self, clr, matrix=False):
        self.clr, self.matrix = clr, matrix
    def __getitem__(self, key):
        return self.clr.frame.iloc[key].copy()
    def fetch(self, region):
        if isinstance(region, str):
            region = (region, 0, int(self.clr.chromsizes[region]))
        chrom, start, end = region
        frame = self.clr.frame
        selected = frame.loc[frame.chrom.eq(chrom) & frame.start.lt(end) & frame.end.gt(start)]
        if self.matrix:
            idx = selected.index.to_numpy()
            return self.clr.data[np.ix_(idx, idx)].copy()
        return selected.copy()


class FakeCooler:
    def __init__(self, uri):
        self.uri = str(uri)
        self.filename = self.uri.split("::")[0]
        self.binsize = 25_000
        self.chromnames = ["chr1"]
        self.chromsizes = pd.Series({"chr1": 1_600_000})
        self.frame = pd.DataFrame({"chrom": ["chr1"]*64, "start": np.arange(64)*25_000,
                                   "end": (np.arange(64)+1)*25_000, "weight": [1.]*64})
        r, c = np.indices((64, 64))
        self.data = np.ones((64, 64))
        for anchor in [20, 42]:
            self.data += np.exp(-(((r+c)/2-anchor)/2)**2) * np.exp(-((abs(c-r)-4)/2)**2)
        if "target_b" in self.filename:
            self.data *= 1.1
    def bins(self):
        return FakeSelector(self)
    def matrix(self, **kwargs):
        return FakeSelector(self, matrix=True)


class FakeSnipper:
    def __init__(self, clr, expected, cooler_opts, view_df, min_diag, expected_value_col):
        self.clr, self.min_diag = clr, min_diag
        self.expected = expected
    def select(self, a, b):
        return self.clr.data
    def snip(self, matrix, a, b, coords):
        s1, e1, s2, e2 = [v//25_000 for v in coords]
        snip = matrix[s1:e1, s2:e2].copy()
        i, j = np.indices(snip.shape)
        snip[j-i < self.min_diag] = np.nan
        return snip


class EndToEndTests(unittest.TestCase):
    def test_two_targets_fixed_reference_cache_and_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            r, c = np.indices((9, 9))
            mask = 1 + np.exp(-(((r+c)/2-4)/2)**2) * np.exp(-((abs(c-r)-4)/2)**2)
            mask[abs(c-r) < 2] = np.nan
            np.save(root / "input.npy", mask)
            (root / "view.tsv").write_text("chr1\t0\t1600000\tchr1\n")
            settings = Settings(flank=100_000)
            thresholds = Thresholds(min_fs=0, min_prominence=0, max_noise=1e6)
            ref = freeze_reference(root/"input.npy", root/"view.tsv", root/"reference", settings, thresholds)
            original = json.loads(ref.path.read_text())
            calls_to_expected = []
            def expected(clr, **kwargs):
                calls_to_expected.append(clr.uri)
                return pd.DataFrame({"region1": ["chr1"]*64, "region2": ["chr1"]*64,
                                     "dist": np.arange(64), "balanced.avg": np.ones(64)})
            tools = types.ModuleType("cooltools")
            tools.expected_cis = expected
            api = types.ModuleType("cooltools.api")
            snipping = types.ModuleType("cooltools.api.snipping")
            snipping.ObsExpSnipper = FakeSnipper
            cooler = types.ModuleType("cooler")
            cooler.Cooler = FakeCooler
            modules = {"cooltools": tools, "cooltools.api": api, "cooltools.api.snipping": snipping, "cooler": cooler}
            for name in ["target_a", "target_b"]:
                (root/f"{name}.cool").write_bytes(b"synthetic stand-in for external cooler")
            with patch.dict("sys.modules", modules):
                a = run_dataset("target_a", root/"target_a.cool", ref, root/"out", progress=False)
                b = run_dataset("target_b", root/"target_b.cool", ref, root/"out", progress=False)
                again = run_dataset("target_a", root/"target_a.cool", ref, root/"out", progress=False)
                self.assertEqual(len(calls_to_expected), 2)
                self.assertEqual(a.reference.metadata, b.reference.metadata)
                self.assertEqual(original, json.loads(ref.path.read_text()))
                self.assertTrue(len(a.calls) > 0)
                np.testing.assert_allclose(a.scores.FS, again.scores.FS, equal_nan=True)
                self.assertTrue((a.directory/"calls.bed").exists())
                (root/"target_a.cool").write_bytes(b"changed dataset")
                with self.assertRaisesRegex(ValueError, "Cached snippets"):
                    run_dataset("target_a", root/"target_a.cool", ref, root/"out", progress=False)
            with self.assertRaisesRegex(ValueError, "Different reference"):
                freeze_reference(root/"input.npy", root/"view.tsv", root/"reference", settings,
                                 Thresholds(min_fs=.8))
            changed = ref.mask.copy()
            changed[0, 3] += 1
            np.save(ref.mask_path, changed)
            with self.assertRaisesRegex(ValueError, "mask has changed"):
                FrozenReference.load(ref.path)


if __name__ == "__main__":
    unittest.main()

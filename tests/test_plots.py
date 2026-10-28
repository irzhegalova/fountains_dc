import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fountain_tools.core import AnalysisResult, FrozenReference, Settings, Thresholds, freeze_reference, filter_calls, add_peak_prominence, score_snippets
from fountain_tools.plots import plot_reference, plot_dense_examples, plot_distributions, plot_noise_prominence, plot_fountain_region
from test_pipeline import FakeCooler


class PlotTests(unittest.TestCase):
    def test_figures_render_with_explicit_dataset_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(12)
            mask = rng.uniform(.5, 2, (9, 9))
            i, j = np.indices(mask.shape)
            mask[abs(i-j) < 2] = np.nan
            np.save(root/"input.npy", mask)
            (root/"view.tsv").write_text("chr1\t0\t1600000\tchr1\n")
            ref = freeze_reference(root/"input.npy", root/"view.tsv", root/"ref", Settings(flank=100_000),
                                   Thresholds(min_fs=-1, min_prominence=0, max_noise=1e6))
            clr = FakeCooler(str(root/"mock.cool"))
            stack = rng.uniform(.2, 3, (64, 9, 9))
            stack[:, np.tril_indices(9, 1)[0], np.tril_indices(9, 1)[1]] = np.nan
            np.save(root/"snippets.npy", stack)
            table = clr.frame[["chrom", "start", "end"]].copy()
            table["stack_index"] = np.arange(len(table))
            table["center"] = (table.start+table.end)/2
            measurements = score_snippets(stack, mask)
            scores = add_peak_prominence(table.join(measurements), 25_000)
            calls, summary = filter_calls(scores, ref.thresholds)
            result = AnalysisResult("mock", str(root/"mock.cool"), root, ref, scores, calls, summary)
            figures = [plot_reference(ref), plot_distributions(result), plot_noise_prominence(result),
                       plot_dense_examples(result, examples=scores.head(5), show=False)]
            with patch.object(AnalysisResult, "open_cooler", return_value=clr):
                fig = plot_fountain_region(result, ("chr1", 200_000, 1_400_000), max_distance=400_000, cmap="YlOrRd")
            figures.append(fig)
            fig.canvas.draw()
            # Shared genomic x scales and the intended rotated aspect.
            self.assertEqual(fig.axes[0].get_xlim(), fig.axes[1].get_xlim())
            self.assertAlmostEqual(fig.axes[0].get_aspect(), .5)
            for i, figure in enumerate(figures):
                figure.savefig(root/f"plot-{i}.png", dpi=70)
                self.assertGreater((root/f"plot-{i}.png").stat().st_size, 1000)
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()

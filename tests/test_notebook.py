"""Check the main notebook's sequential state flow using synthetic map adapters."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from test_pipeline import FakeCooler, FakeSnipper


class NotebookTests(unittest.TestCase):
    def test_main_notebook_runs_in_order_with_synthetic_inputs(self):
        bundle = Path(__file__).resolve().parents[1]
        notebook = json.loads((bundle/"fountains-dc.ipynb").read_text())
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            (project/"data").mkdir()
            r, c = np.indices((41, 41))
            mask = 1 + np.exp(-(((r+c)/2-20)/2)**2) * np.exp(-((abs(c-r)-4)/2)**2)
            mask[abs(c-r) < 2] = np.nan
            np.save(project/"data/mask.npy", mask)
            (project/"data/view.tsv").write_text("chr1\t0\t1600000\tchr1\n")
            (project/"data/manual.bed").write_text("chr1\t500000\t525000\n")
            for label in ["target_a", "target_b"]:
                (project/f"{label}.cool").write_bytes(b"synthetic")
            tools = types.ModuleType("cooltools")
            tools.expected_cis = lambda clr, **kw: pd.DataFrame({"region1": ["chr1"]*64,
                "region2": ["chr1"]*64, "dist": np.arange(64), "balanced.avg": np.ones(64)})
            api = types.ModuleType("cooltools.api")
            snipping = types.ModuleType("cooltools.api.snipping")
            snipping.ObsExpSnipper = FakeSnipper
            cooler = types.ModuleType("cooler")
            cooler.Cooler = FakeCooler
            modules = {"cooltools": tools, "cooltools.api": api, "cooltools.api.snipping": snipping,
                       "cooler": cooler}
            namespace = {"__name__": "__main__"}
            with patch.dict("sys.modules", modules), patch.object(plt, "show", return_value=None):
                for i, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    # A Jupyter display renderer is not installed in this test runtime.
                    source = source.replace("from IPython.display import display", "display = lambda value: None")
                    source = source.replace('(\"chr6\", 31_000_000, 33_000_000)', '(\"chr1\", 200_000, 1_400_000)')
                    exec(compile(source, f"notebook-cell-{i}", "exec"), namespace)
                    if "DATASETS_TO_RUN =" in source:
                        namespace.update(PROJECT_ROOT=project, MODULE_ROOT=bundle,
                            COOLER_PATHS={label: str(project/f"{label}.cool") for label in ["target_a", "target_b"]},
                            MASK_PATH=project/"data/mask.npy", VIEW_PATH=project/"data/view.tsv",
                            MANUAL_BED=project/"data/manual.bed", OUTPUT_ROOT=project/"results/maps",
                            REFERENCE_DIR=project/"results/reference",
                            REFERENCE_JSON=project/"results/reference/reference.json",
                            DATASETS_TO_RUN=["target_a", "target_b"], ACTIVE_DATASET="target_a", NPROC_EXPECTED=1)
                self.assertEqual(len(namespace["results"]), 2)
                self.assertEqual(namespace["comparison"].reference_id.nunique(), 1)
                self.assertTrue((project/"results/maps/target_b/calls.tsv").exists())
                plt.close("all")


if __name__ == "__main__":
    unittest.main()

# Fixed-reference fountain analysis

The cleaned `fountains-dc.ipynb` calls reusable functions in `fountain_tools/`.
Open it in your existing Fontanka environment. No Fontanka CLI or coolpuppy
source patches are needed. The original notebook is preserved in `archive/`.

## Run it

1. Extract the bundle and open `fountains-dc.ipynb` beside `fountain_tools/`.
2. Set `PROJECT_ROOT` to your existing analysis directory containing `data/`.
   Keep the notebook/module files together; the data can live elsewhere.
3. Confirm `MASK_PATH`, `VIEW_PATH`, and `COOLER_PATHS`. The notebook keeps the
   four map paths from your uploaded notebook. Resolution is 25,000 bp and the
   flank on each side is 500,000 bp; the supplied mask must be 41 x 41.
4. The first run copies your supplied mask and view into a frozen reference
   directory and saves its numeric thresholds. The WT mask is used exactly as
   stored; it is never rebuilt, resized, or logged automatically.
5. Set `DATASETS_TO_RUN` to the desired labels. Every target gets its own
   expected contacts, snippets, scores, calls, and plots, using the SAME frozen
   reference and cutoffs. No Li/percentile cutoff is fitted on a new target.

```python
reference = FrozenReference.load(REFERENCE_JSON)
other = run_dataset(
    "yang2023_dc_WT_LPS", COOLER_PATHS["yang2023_dc_WT_LPS"], reference,
    output_root=OUTPUT_ROOT, nproc=20,
)
```

The supplied 20 manual anchors are included as BED and metadata in `data/`.
The actual mask, view file, Hi-C maps, and previously imported enhancer helper
were not attached, so they are not included in the bundle.

## Fixed thresholds retained from the uploaded notebook

| Quantity | Cutoff |
| --- | --- |
| FS | > 0.5 |
| Peak prominence | > 0.2206, subject to the precision note below |
| Scharr maximum | <= 80 |
| Joint finite pixels / all finite mask pixels | >= 0.35 |
| Positive-pixel fraction for final calls | >= 0 (no extra density cutoff) |

The original notebook printed the Li threshold only to four decimal places.
On the first freeze, the cleaned notebook can recover its full precision by
running Li ONCE on the EXISTING calibration score table from the original
CTCFko run. This is disabled for subsequent targets. If that original table is
absent, the notebook explicitly reports that it is freezing the rounded 0.2206
value. If you have the exact value from the old kernel, enter it before freezing.
Do not expect bit-for-bit reproduction of calls near the cutoff when using the
rounded fallback.

The original 0.35 cutoff used the full reference mask as denominator. For your
symmetric mask and upper-triangle snippets, 0.35 is approximately 70% upper-
triangle coverage. Both quantities now have separate columns:

- `mask_fraction`: full-mask denominator; used by the retained 0.35 cutoff.
- `coverage`: finite fraction of fixed upper-triangle mask support; used in
  plot labels and optional display filters. Complete snippets report 100%.
- `nonzero_fraction`: positive O/E fraction among finite upper-triangle pixels.

Display-only spacing, manual exclusion, and density cutoffs do not modify calls.

## Outputs and caching

Each map has its own subdirectory under `results/fountains_fixed/maps/`:

- `expected.tsv.gz`: expected contacts computed from THIS map, unsmoothed,
  balanced using the configured weight, with two near diagonals ignored.
- `snippets.npy`: disk-backed float64 array, `(n_bins, 41, 41)`, in cooler-bin
  order. Incomplete view-edge windows remain NaN. Approximately 1.4 GiB for
  109,033 bins; budget this per map.
- `stack_bins.tsv.gz`, `snippets.json`: coordinate order and cache provenance.
- `scores.tsv.gz`: FS, Scharr, prominence, coverage, and coordinate metadata.
- `calls.tsv`, `calls.bed`: the CURRENT filtered calls for that dataset.
- `filter_counts.tsv`, `run.json`: sequential counts, frozen reference, source
  file fingerprints, and software versions.

The cache checks dataset URI/file identity, bin coordinates, balancing weights,
view, resolution, flank, expected file, and software versions. Scores also check
the mask/configuration identity. A different mask with the same shape is not a
valid score cache. Incompatible caches raise an error; choose a new output root.
The old untracked snippet files are deliberately not adopted automatically:
shape alone cannot prove that they belong to the correct dataset or view.

Snippets are written one region/window at a time to a NumPy memory map rather
than holding a whole-genome stack in RAM. Expected calculation uses `nproc`;
snippet extraction and scoring run serially in bounded-memory chunks.

## What the scores mean

FS reproduces Pearson(saved mask, linear O/E snippet) over jointly finite
pixels. Scharr reproduces the maximum complex Scharr-gradient magnitude over
the full linear snippet, without applying the reference mask. The noise score
is sensitive to sharp biological features as well as artifacts. The logarithm
is used only in plots when the reference is declared linear O/E.

Local prominence is calculated within contiguous finite FS runs and frozen
view regions. It is not a probability and can reach 2 because FS ranges from
-1 to +1. No target-specific prominence threshold or Scharr percentile is fitted.

Fixed numeric thresholds do not guarantee matched sensitivity/false-positive
rates across maps with different depth or quality. Inspect score/noise
distributions and coverage for every map. The count comparison includes eligible
bin counts and calls per 1,000 eligible bins; it does not claim equal detection
power or perform differential fountain testing.

## Optional CTCF/enhancer analyses

`fountains-dc-annotations.ipynb` consumes the current dataset's `calls.tsv` and
eligible score bins, rather than a stale `df_strongest` file or hardcoded 10 kb
cooler. It includes nearest-CTCF/enhancer counts and the genomic-bin control
preserved from the original notebook.

The enhancer-center/distance-band controls in the original notebook imported
`scripts/fountain_enhancer_counts.py`, which was not attached. Their optional
cells remain in the companion notebook, with explicit inputs and an opt-in
switch. Copy your existing `scripts/` module into the analysis project to run
them. Their original source cells and outputs are also retained in the archive;
the missing helper's implementation has not been guessed or replaced.

## Validation

Run `python -m unittest discover -s tests -v` from this folder. Tests compare
Pearson and Scharr to SciPy/Fontanka's formulas, verify triangle coverage,
prominence/gap behavior, nearest-peak matching, fixed threshold inequalities,
cache reuse, and rejection of changed masks or datasets. A mocked two-target
integration test verifies that expected is target-specific and thresholds are
not refitted. Plot smoke tests render the grid/distributions/map on synthetic
data. Real cluster maps were unavailable in the editing environment, so the
actual cooler/cooltools execution must be run in your analysis environment.

# Outputs and caching

Each map has its own subdirectory under `results/fountains/maps/`:

- `expected.tsv.gz`: expected contacts computed from each map, unsmoothed,
  balanced using the configured weight, with two near diagonals ignored.
- `snippets.npy`: disk-backed float64 array, `(n_bins, 41, 41)`, in cooler-bin
  order. Incomplete view-edge windows remain NaN. Approximately 1.4 GiB for
  109,033 bins; budget this per map.
- `stack_bins.tsv.gz`, `snippets.json`: coordinate order and cache provenance.
- `scores.tsv.gz`: FS, Scharr, prominence, coverage, and coordinate metadata.
- `calls.tsv`, `calls.bed`: filtered calls for that dataset.
- `filter_counts.tsv`, `run.json`: sequential counts, frozen reference, source
  file fingerprints, and software versions.
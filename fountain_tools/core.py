"""Dataset-specific fountain calling. Heavy Hi-C dependencies are imported lazily.

FS is Pearson(mask, linear O/E snippet). The mask is never transformed.
Scharr_box reproduces Fontanka's maximum complex Scharr-gradient magnitude.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import convolve2d, find_peaks

ENGINE_VERSION = "dataset-thresholds-2.0"
KEYS = ["chrom", "start", "end"]
SCHARR = np.array([[-3-3j, -10j, 3-3j], [-10, 0, 10], [-3+3j, 10j, 3+3j]])


@dataclass(frozen=True)
class Settings:
    resolution: int = 25_000
    flank: int = 500_000
    min_diag: int = 2
    weight: str = "weight"
    assembly: str = "mm10"
    excluded_chroms: tuple = ("chrM", "chrY")

    def __post_init__(self):
        if self.resolution <= 0 or self.flank < 0 or self.flank % self.resolution:
            raise ValueError("Require positive resolution and a divisible, nonnegative flank.")
        if self.min_diag < 0:
            raise ValueError("min_diag must be nonnegative.")

    @property
    def size(self):
        return 2 * (self.flank // self.resolution) + 1


@dataclass(frozen=True)
class Thresholds:
    min_fs: float
    min_prominence: float
    max_noise: float
    # Preserve the uploaded notebook's literal 0.35 full-mask cutoff.
    # This is distinct from upper-triangle coverage, which is reported separately.
    min_mask_fraction: float = 0.35
    min_nonzero_fraction: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Thresholds must be finite numbers.")
        if not -1 <= self.min_fs <= 1 or not 0 <= self.min_prominence <= 2:
            raise ValueError("FS must be in [-1,1] and prominence in [0,2].")
        if self.max_noise < 0 or not 0 <= self.min_mask_fraction <= 1 or not 0 <= self.min_nonzero_fraction <= 1:
            raise ValueError("Invalid noise or coverage threshold.")


def _hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def read_intervals(path, view=False):
    frame = pd.read_csv(path, sep="\t", comment="#", header=None)
    if frame.shape[1] < 3 or frame.empty:
        raise ValueError(f"Expected nonempty BED-style coordinates: {path}")
    if str(frame.iloc[0, 1]).strip().lower() == "start":
        frame = frame.iloc[1:]
    width = 4 if view and frame.shape[1] >= 4 else 3
    frame = frame.iloc[:, :width].copy()
    frame.columns = KEYS + (["name"] if width == 4 else [])
    frame["chrom"] = frame.chrom.astype(str)
    numeric = frame[["start", "end"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric).all().all() or not (numeric == np.floor(numeric)).all().all():
        raise ValueError("Coordinates must be finite integers.")
    frame[["start", "end"]] = numeric.astype("int64")
    if frame.empty or (frame.start < 0).any() or (frame.end <= frame.start).any():
        raise ValueError("Require nonempty intervals with 0 <= start < end.")
    if view:
        if "name" not in frame:
            frame["name"] = frame.chrom + ":" + frame.start.astype(str) + "-" + frame.end.astype(str)
        if not frame.name.is_unique:
            raise ValueError("View names must be unique.")
        for _, g in frame.groupby("chrom", sort=False):
            g = g.sort_values("start")
            if (g.start.to_numpy()[1:] < g.end.to_numpy()[:-1]).any():
                raise ValueError("View regions must not overlap.")
    return frame.reset_index(drop=True)


def load_mask(mask_path, settings):
    """Load the supplied linear O/E WT scoring mask without transforming it."""
    mask = np.load(mask_path)
    if mask.shape != (settings.size, settings.size):
        raise ValueError(f"Mask shape {mask.shape} does not match {(settings.size, settings.size)}.")
    finite = mask[np.isfinite(mask)]
    if len(finite) < 4 or np.std(finite) == 0:
        raise ValueError("Scoring mask must contain varying finite values.")
    if (finite < 0).any():
        raise ValueError("The linear O/E scoring mask contains negative values.")
    return mask


def calculate_thresholds(scores, fs_percentile=90, noise_percentile=75,
                         min_mask_fraction=0.35, min_nonzero_fraction=0.0):
    """Fit Li to local-peak prominences and percentiles to all scored bins.

    Populations are taken BEFORE peak, pixel-quality, or calling filters. Each
    column independently excludes nonfinite values, as in the source notebook's
    Li/np.nanpercentile cells. The score table is never modified.
    """
    from skimage.filters import threshold_li

    if not all(np.isfinite(q) and 0 <= q <= 100 for q in (fs_percentile, noise_percentile)):
        raise ValueError("Percentiles must be finite numbers between 0 and 100.")
    fs = scores.loc[np.isfinite(scores.FS), "FS"].to_numpy(dtype=float)
    noise = scores.loc[np.isfinite(scores.Scharr_box), "Scharr_box"].to_numpy(dtype=float)
    prominence = scores.loc[np.isfinite(scores.FS_peaks) & scores.FS_peaks.ge(0),
                            "FS_peaks"].to_numpy(dtype=float)
    for name, values in [("FS", fs), ("Scharr noise", noise), ("local-peak prominence", prominence)]:
        if not len(values):
            raise ValueError(f"Cannot calculate thresholds: no finite {name} values in this dataset.")
    return Thresholds(
        min_fs=float(np.percentile(fs, fs_percentile)),
        min_prominence=float(threshold_li(prominence)),
        max_noise=float(np.percentile(noise, noise_percentile)),
        min_mask_fraction=min_mask_fraction,
        min_nonzero_fraction=min_nonzero_fraction,
    )


def score_snippets(stack, mask, min_diag=2, chunk_size=256, compute_noise=True):
    """Return Pearson, full-mask fraction, upper-triangle coverage, and Scharr.

    FS uses ALL jointly finite mask/snippet pixels, preserving the earlier code.
    The coverage diagnostic uses fixed upper-triangle support, avoiding a false
    50% penalty when the snippets only contain the upper triangle.
    """
    if stack.ndim != 3 or tuple(stack.shape[1:]) != tuple(mask.shape):
        raise ValueError("Expected stack (n, rows, columns), matching the mask.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    kernel = np.asarray(mask, dtype=float).ravel()
    kvalid = np.isfinite(kernel)
    support = np.triu(np.ones(mask.shape, dtype=bool), k=min_diag).ravel() & kvalid
    if kvalid.sum() < 4 or not support.any():
        raise ValueError("Too few usable scoring-mask pixels.")
    result = {name: np.full(len(stack), np.nan) for name in
              ["FS", "mask_fraction", "coverage", "nonzero_fraction", "Scharr_box"]}
    result["n_valid_pixels"] = np.zeros(len(stack), dtype=int)
    for begin in range(0, len(stack), chunk_size):
        end = min(begin + chunk_size, len(stack))
        a = np.asarray(stack[begin:end], dtype=float).reshape(end-begin, -1)
        valid = np.isfinite(a) & kvalid[None, :]
        n = valid.sum(1)
        x, y = np.where(valid, a, 0.), np.where(valid, kernel[None, :], 0.)
        mx = np.divide(x.sum(1), n, out=np.zeros(len(a)), where=n > 0)
        my = np.divide(y.sum(1), n, out=np.zeros(len(a)), where=n > 0)
        x, y = np.where(valid, x-mx[:, None], 0.), np.where(valid, y-my[:, None], 0.)
        den = np.sqrt((x*x).sum(1) * (y*y).sum(1))
        fs = np.divide((x*y).sum(1), den, out=np.full(len(a), np.nan), where=(n > 3) & (den > 0))
        result["FS"][begin:end] = np.clip(fs, -1, 1)
        result["n_valid_pixels"][begin:end] = n
        result["mask_fraction"][begin:end] = n / kvalid.sum()
        supported_valid = valid & support[None, :]
        ns = supported_valid.sum(1)
        result["coverage"][begin:end] = ns / support.sum()
        result["nonzero_fraction"][begin:end] = np.divide(
            (supported_valid & (a > 0)).sum(1), ns,
            out=np.zeros(len(a)), where=ns > 0)
        if compute_noise:
            for j in np.flatnonzero(np.isfinite(fs)):
                # No scoring mask is applied to the noise calculation.
                snip = a[j].reshape(mask.shape).copy()
                snip[~np.isfinite(snip)] = np.nan
                magnitude = np.abs(convolve2d(snip, SCHARR, boundary="symm", mode="same"))
                ok = np.isfinite(magnitude)
                if ok.any():
                    result["Scharr_box"][begin+j] = magnitude[ok].max()
    return pd.DataFrame(result)


def add_peak_prominence(scores, resolution):
    """Find peaks separately within view regions and contiguous finite runs."""
    out = scores.copy()
    if not out.index.is_unique:
        raise ValueError("Score table index must be unique.")
    out["FS_peaks"] = np.nan
    groups = ["chrom"] + (["view_region"] if "view_region" in out else [])
    for _, group in out.groupby(groups, sort=False, dropna=False):
        group = group.sort_values("start")
        positions = group.index.to_numpy()
        values = group.FS.to_numpy(dtype=float)
        finite = np.isfinite(values)
        starts = group.start.to_numpy()
        run = []
        for i in range(len(group) + 1):
            continuing = i < len(group) and finite[i]
            if continuing and run and starts[i] - starts[run[-1]] != resolution:
                local, props = find_peaks(values[run], prominence=0)
                out.loc[positions[np.asarray(run)[local]], "FS_peaks"] = props["prominences"]
                run = []
            if continuing:
                run.append(i)
            elif run:
                local, props = find_peaks(values[run], prominence=0)
                out.loc[positions[np.asarray(run)[local]], "FS_peaks"] = props["prominences"]
                run = []
    return out


def filter_calls(scores, thresholds):
    peak = np.isfinite(scores.FS) & np.isfinite(scores.FS_peaks)
    quality = scores.mask_fraction.ge(thresholds.min_mask_fraction)
    quality &= scores.nonzero_fraction.ge(thresholds.min_nonzero_fraction)
    masks = [peak, peak & quality]
    masks.append(masks[-1] & scores.FS.gt(thresholds.min_fs))
    masks.append(masks[-1] & scores.FS_peaks.gt(thresholds.min_prominence))
    masks.append(masks[-1] & np.isfinite(scores.Scharr_box) & scores.Scharr_box.lt(thresholds.max_noise))
    summary = pd.DataFrame({"stage": ["All local FS peaks", "Pixel-quality cutoff", "FS cutoff", "Prominence cutoff", "Noise cutoff"],
                            "n_candidates": [int(m.sum()) for m in masks]})
    return scores.loc[masks[-1]].copy(), summary


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _source_identity(clr, bins, weight):
    source = Path(clr.filename).resolve()
    stat = source.stat()
    return {"uri": str(clr.uri), "source_path": str(source), "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "bins_sha256": hashlib.sha256(bins[KEYS].to_csv(index=False).encode()).hexdigest(),
            "weights_sha256": hashlib.sha256(bins[weight].to_numpy(dtype="float64").tobytes()).hexdigest()}


def _bins_and_view(clr, view_path, settings):
    if clr.binsize != settings.resolution:
        raise ValueError(f"Expected {settings.resolution} bp bins, got {clr.binsize}.")
    bins = clr.bins()[:].reset_index(drop=True)
    bins["chrom"] = bins.chrom.astype(str)
    if settings.weight not in bins:
        raise ValueError(f"Cooler lacks balancing column {settings.weight!r}.")
    view = read_intervals(view_path, view=True)
    view = view.loc[~view.chrom.isin(settings.excluded_chroms)].copy()
    if view.empty:
        raise ValueError("The view contains no eligible regions.")
    order = {c: i for i, c in enumerate(clr.chromnames)}
    if not set(view.chrom).issubset(order):
        raise ValueError("The target lacks chromosomes in the view; check genome assembly.")
    view = view.assign(_order=view.chrom.map(order)).sort_values(["_order", "start"]).drop(columns="_order").reset_index(drop=True)
    for r in view.itertuples(index=False):
        if r.end > clr.chromsizes[r.chrom] or r.start % settings.resolution:
            raise ValueError(f"View {r.name!r} is incompatible with this cooler.")
    annotated = bins[KEYS].copy()
    annotated["stack_index"] = np.arange(len(bins))
    annotated["center"] = (annotated.start + annotated.end) / 2
    annotated["view_region"] = None
    annotated["eligible_window"] = False
    for r in view.itertuples(index=False):
        inside = annotated.chrom.eq(r.chrom) & annotated.start.ge(r.start) & annotated.end.le(r.end)
        annotated.loc[inside, "view_region"] = r.name
        complete = inside & (annotated.start-settings.flank).ge(r.start)
        complete &= (annotated.start+settings.resolution+settings.flank).le(r.end)
        annotated.loc[complete, "eligible_window"] = True
    return bins, view, annotated


def _prepare_stack(clr, bins, view, annotated, settings, directory, nproc, progress):
    import cooltools
    from cooltools.api.snipping import ObsExpSnipper

    identity = {"engine": ENGINE_VERSION, "source": _source_identity(clr, bins, settings.weight),
                "settings": json.loads(json.dumps(asdict(settings))),
                "view": json.loads(view.to_json(orient="records")),
                "cooltools_version": _version("cooltools"), "cooler_version": _version("cooler")}
    paths = {name: directory / name for name in ["snippets.npy", "snippets.json", "expected.tsv.gz", "stack_bins.tsv.gz"]}
    if paths["snippets.json"].exists():
        saved = json.loads(paths["snippets.json"].read_text())
        if saved["identity"] != identity:
            raise ValueError("Cached snippets belong to different inputs/settings. Use a new dataset/output directory.")
        for key in ["snippets.npy", "expected.tsv.gz", "stack_bins.tsv.gz"]:
            if not paths[key].exists():
                raise ValueError(f"Incomplete cache: missing {paths[key]}. Use a new output directory.")
        if _hash_file(paths["expected.tsv.gz"]) != saved["expected_sha256"] or _hash_file(paths["stack_bins.tsv.gz"]) != saved["bins_sha256"]:
            raise ValueError("Expected or bin-coordinate cache changed. Use a new output directory.")
        stack = np.load(paths["snippets.npy"], mmap_mode="r")
        if stack.shape != (len(bins), settings.size, settings.size):
            raise ValueError("Cached snippets have the wrong shape.")
        if paths["snippets.npy"].stat().st_mtime_ns != saved["stack_mtime_ns"]:
            raise ValueError("Cached snippet file changed after generation.")
        return stack, paths, saved

    # Every target map gets its own expected contacts.
    if progress:
        print(f"Calculating expected contacts for {clr.uri}", flush=True)
    expected = cooltools.expected_cis(clr, view_df=view, clr_weight_name=settings.weight,
                                     ignore_diags=settings.min_diag, smooth=False,
                                     aggregate_smoothed=False, nproc=nproc)
    expected.to_csv(paths["expected.tsv.gz"], sep="\t", index=False)
    bins[KEYS].to_csv(paths["stack_bins.tsv.gz"], sep="\t", index=False)
    snipper = ObsExpSnipper(clr, expected, cooler_opts={"balance": settings.weight, "sparse": True},
                            view_df=view, min_diag=settings.min_diag, expected_value_col="balanced.avg")
    partial = directory / "snippets.partial.npy"
    stack = np.lib.format.open_memmap(partial, mode="w+", dtype="float64",
                                     shape=(len(bins), settings.size, settings.size))
    stack[:] = np.nan
    for r in view.itertuples(index=False):
        rows = annotated.loc[annotated.eligible_window & annotated.view_region.eq(r.name)]
        if rows.empty:
            continue
        if progress:
            print(f"  {r.name}: {len(rows):,} complete windows", flush=True)
        matrix = snipper.select(r.name, r.name)
        for row in rows.itertuples(index=False):
            left, right = row.start-settings.flank, row.start+settings.resolution+settings.flank
            snippet = snipper.snip(matrix, r.name, r.name, (left, right, left, right))
            stack[row.stack_index] = snippet
        stack.flush()
        del matrix
    del stack
    os.replace(partial, paths["snippets.npy"])
    saved = {"identity": identity, "expected_sha256": _hash_file(paths["expected.tsv.gz"]),
             "bins_sha256": _hash_file(paths["stack_bins.tsv.gz"]),
             "stack_mtime_ns": paths["snippets.npy"].stat().st_mtime_ns,
             "order": "cooler bins; axis 0 is snippet index", "values": "linear O/E; upper triangle"}
    _json(paths["snippets.json"], saved)
    return np.load(paths["snippets.npy"], mmap_mode="r"), paths, saved


@dataclass
class AnalysisResult:
    label: str
    cool_uri: str
    directory: Path
    settings: Settings
    mask: np.ndarray
    thresholds: Thresholds
    scores: pd.DataFrame
    calls: pd.DataFrame
    summary: pd.DataFrame

    @property
    def stack(self):
        return np.load(self.directory / "snippets.npy", mmap_mode="r")

    def open_cooler(self):
        import cooler
        return cooler.Cooler(self.cool_uri)


def run_dataset(label, cool_path, mask_path, view_path, settings=None, *,
                output_root="results/fountains/maps", nproc=1, progress=True,
                fs_percentile=90, noise_percentile=75,
                min_mask_fraction=0.35, min_nonzero_fraction=0.0):
    """Score a map with the WT mask, then fit this map's Li/percentile cutoffs."""
    import cooler
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in {".", ".."}:
        raise ValueError("Use a simple dataset label containing letters, digits, dot, dash, underscore.")
    settings = settings or Settings()
    mask_path, view_path = Path(mask_path).resolve(), Path(view_path).resolve()
    mask = load_mask(mask_path, settings)
    uri = str(cool_path)
    if "::" not in uri and Path(uri).suffix.lower() == ".mcool":
        uri += f"::/resolutions/{settings.resolution}"
    clr = cooler.Cooler(uri)
    bins, view, annotated = _bins_and_view(clr, view_path, settings)
    directory = Path(output_root).resolve() / label
    directory.mkdir(parents=True, exist_ok=True)
    stack, _, stack_manifest = _prepare_stack(clr, bins, view, annotated, settings, directory, nproc, progress)
    signature = {"engine": ENGINE_VERSION, "mask_sha256": _hash_file(mask_path),
                 "stack_manifest_sha256": _digest(stack_manifest), "numpy": _version("numpy"), "scipy": _version("scipy")}
    score_file, score_meta = directory / "scores.tsv.gz", directory / "scores.json"
    if score_file.exists() and score_meta.exists():
        cached = json.loads(score_meta.read_text())
        if cached["identity"] == signature and cached["scores_sha256"] == _hash_file(score_file):
            # Preserve float64 scores so cached and fresh Li/percentile fits agree.
            scores = pd.read_csv(score_file, sep="\t", float_precision="round_trip")
        else:
            raise ValueError("Scoring inputs changed. Use a new output root to preserve the previous run.")
    else:
        if progress:
            print("Calculating FS, pixel coverage, and Scharr noise...", flush=True)
        measurements = score_snippets(stack, mask, settings.min_diag)
        scores = pd.concat([annotated, measurements], axis=1)
        scores.loc[~scores.eligible_window, ["FS", "Scharr_box"]] = np.nan
        scores = add_peak_prominence(scores, settings.resolution)
        scores.to_csv(score_file, sep="\t", index=False)
        _json(score_meta, {"identity": signature, "scores_sha256": _hash_file(score_file)})
    thresholds = calculate_thresholds(
        scores, fs_percentile, noise_percentile, min_mask_fraction, min_nonzero_fraction)
    threshold_metadata = {
        "method": "Li prominence; FS percentile; Scharr percentile; fitted per dataset",
        "fs_percentile": fs_percentile, "noise_percentile": noise_percentile,
        "population": "all scored bins before calling or pixel-quality filtering; finite values per column",
        "n_fs": int(np.isfinite(scores.FS).sum()),
        "n_noise": int(np.isfinite(scores.Scharr_box).sum()),
        "n_prominence": int((np.isfinite(scores.FS_peaks) & scores.FS_peaks.ge(0)).sum()),
        "thresholds": asdict(thresholds),
        "inequalities": {"FS": ">", "FS_peaks": ">", "Scharr_box": "<"},
        "scores_sha256": _hash_file(score_file),
        "scikit_image_version": _version("scikit-image"),
    }
    calls, summary = filter_calls(scores, thresholds)
    _json(directory / "thresholds.json", threshold_metadata)
    calls.to_csv(directory / "calls.tsv", sep="\t", index=False)
    calls.assign(name=[f"{label}_fountain_{i+1:05d}" for i in range(len(calls))])[KEYS+["name"]].to_csv(
        directory / "calls.bed", sep="\t", header=False, index=False)
    summary.to_csv(directory / "filter_counts.tsv", sep="\t", index=False)
    inputs = {"mask_path": str(mask_path), "mask_sha256": signature["mask_sha256"],
              "mask_scale": "linear_OE", "view_path": str(view_path), "view_sha256": _hash_file(view_path),
              "settings": json.loads(json.dumps(asdict(settings)))}
    _json(directory / "run.json", {"dataset": label, "cool_uri": str(clr.uri), "inputs": inputs,
                                  "threshold_calculation": threshold_metadata,
                                  "stack": stack_manifest, "n_calls": len(calls),
                                  "packages": {name: _version(name) for name in ["numpy", "pandas", "scipy", "scikit-image", "cooler", "cooltools"]}})
    if progress:
        print(f"Dataset thresholds: {thresholds}", flush=True)
        print(summary.to_string(index=False), flush=True)
    return AnalysisResult(label, str(clr.uri), directory, settings, mask, thresholds, scores, calls, summary)


def match_manual_peaks(scores, manual, resolution=25_000, max_distance=50_000):
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative.")
    manual = read_intervals(manual) if isinstance(manual, (str, Path)) else manual.copy()
    if "end" in manual:
        center = (manual.start + manual.end) / 2
    else:
        center = manual.start + resolution / 2
    manual["manual_start"] = (np.floor(center / resolution) * resolution).astype("int64")
    manual = manual.drop_duplicates(["chrom", "manual_start"])
    peaks = scores.loc[np.isfinite(scores.FS) & np.isfinite(scores.FS_peaks)].copy()
    records = []
    for row in manual.itertuples(index=False):
        position = row.manual_start + resolution/2
        same = peaks.loc[peaks.chrom.eq(row.chrom)].copy()
        same["distance_bp"] = (same.center - position).abs()
        same = same.loc[same.distance_bp.le(max_distance)].sort_values(["distance_bp", "center"])
        item = {"chrom": row.chrom, "manual_start": row.manual_start, "manual_center": position,
                "matched": False, "peak_start": np.nan, "peak_end": np.nan, "distance_bp": np.nan,
                "offset_bp": np.nan, "FS": np.nan, "FS_peaks": np.nan, "Scharr_box": np.nan}
        if len(same):
            p = same.iloc[0]
            item.update(matched=True, peak_start=int(p.start), peak_end=int(p.end), distance_bp=p.distance_bp,
                        offset_bp=p.center-position, FS=p.FS, FS_peaks=p.FS_peaks, Scharr_box=p.Scharr_box)
        records.append(item)
    out = pd.DataFrame(records, columns=["chrom", "manual_start", "manual_center", "matched", "peak_start", "peak_end", "distance_bp", "offset_bp", "FS", "FS_peaks", "Scharr_box"])
    out["shared_peak"] = False
    if len(out):
        yes = out.matched
        out.loc[yes, "shared_peak"] = out.loc[yes].duplicated(["chrom", "peak_start"], keep=False).to_numpy()
    return out


def select_examples(candidates, n=120, min_separation=1_000_000, manual=None,
                    exclusion_radius=1_000_000, min_coverage=0.5, min_nonzero_fraction=0.5):
    """Spacing is for viewing only; it never changes the saved call set."""
    if n < 0 or min_separation < 0 or exclusion_radius < 0:
        raise ValueError("Counts and distances must be nonnegative.")
    candidates = candidates.loc[np.isfinite(candidates.FS) & candidates.coverage.ge(min_coverage)
                                & candidates.nonzero_fraction.ge(min_nonzero_fraction)].copy()
    if manual is not None:
        manual = read_intervals(manual) if isinstance(manual, (str, Path)) else manual.copy()
        if "end" not in manual:
            raise ValueError("Supply BED intervals for manual exclusions.")
        for r in manual.itertuples(index=False):
            candidates = candidates.loc[~(candidates.chrom.eq(r.chrom) & (candidates.center-(r.start+r.end)/2).abs().le(exclusion_radius))]
    chosen = []
    if n == 0:
        return candidates.iloc[:0].copy()
    for i, row in candidates.sort_values(["FS", "chrom", "start"], ascending=[False, True, True]).iterrows():
        if any(row.chrom == chrom and abs(row.center-center) < min_separation for _, chrom, center in chosen):
            continue
        chosen.append((i, row.chrom, row.center))
        if len(chosen) == n:
            break
    return candidates.loc[[i for i, _, _ in chosen]].copy()

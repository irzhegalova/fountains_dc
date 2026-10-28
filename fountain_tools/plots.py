"""Plots take an AnalysisResult explicitly; no notebook-global clr/mask/scores."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import EngFormatter, MaxNLocator


def log2_display(array):
    array = np.asarray(array)
    out = np.full(array.shape, np.nan, dtype=float)
    ok = np.isfinite(array) & (array > 0)
    out[ok] = np.log2(array[ok])
    return out


def plot_mask(mask, settings):
    data = log2_display(mask)
    edge = (settings.flank + settings.resolution / 2) / 1000
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(data, origin="upper", extent=[-edge, edge, edge, -edge],
                   cmap="coolwarm", vmin=-1.3, vmax=1.3, interpolation="nearest")
    ax.set(title="WT scoring mask (display only)", xlabel="Relative position (kb)", ylabel="Relative position (kb)")
    fig.colorbar(im, ax=ax, label="log2(O/E)")
    return fig


def plot_dense_examples(result, examples=None, ncols=6, per_figure=30, panel_size=1.8,
                        limit=120, vmin=-1.3, vmax=1.3, show=True):
    examples = result.calls if examples is None else examples
    examples = examples.sort_values("FS", ascending=False)
    if limit is not None:
        examples = examples.head(limit)
    if examples.empty:
        print("No examples pass the current filters.")
        return
    if ncols < 1 or per_figure < 1:
        raise ValueError("ncols and per_figure must be positive.")
    stack, mask = result.stack, result.mask
    settings = result.settings
    edge = (settings.flank + settings.resolution/2)/1000
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    for first in range(0, len(examples), per_figure):
        batch = examples.iloc[first:first+per_figure]
        rows = int(np.ceil(len(batch)/ncols))
        fig, axes = plt.subplots(rows, ncols, figsize=(ncols*panel_size, rows*(panel_size+.25)),
                                 squeeze=False, constrained_layout=True)
        axes = axes.ravel()
        for j, row in enumerate(batch.itertuples(index=False)):
            snip = np.array(stack[int(row.stack_index)], dtype=float, copy=True)
            snip[~np.isfinite(mask)] = np.nan
            im = axes[j].imshow(log2_display(snip), origin="upper", extent=[-edge, edge, edge, -edge],
                                cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[j].set_title(f"#{first+j+1} {row.chrom}:{int(row.start):,}\nFS={row.FS:.3f}; coverage={row.coverage:.0%}", fontsize=7, pad=3)
            axes[j].set_axis_off()
        for ax in axes[len(batch):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axes[:len(batch)]), label="log2(O/E)", fraction=.015, pad=.01, shrink=.65)
        fig.suptitle(f"{result.label}: {first+1}-{first+len(batch)} / {len(examples)}; +/-{settings.flank//1000} kb", fontsize=10)
        if show:
            plt.show()
            plt.close(fig)
        else:
            return fig


def _hist(ax, values, edges, color, label, alpha=.4):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values):
        ax.hist(values, bins=edges, weights=np.full(len(values), 1/len(values)),
                color=color, alpha=alpha, histtype="stepfilled", label=f"{label} (n={len(values):,})")


def plot_distributions(result, manual_matches=None):
    scores, t = result.scores, result.thresholds
    peaks = scores.loc[np.isfinite(scores.FS_peaks)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    _hist(axes[0], scores.FS, np.linspace(-1, 1, 101), "gray", "All genomic bins")
    _hist(axes[0], peaks.FS, np.linspace(-1, 1, 101), "royalblue", "All local peaks")
    # Manual overlay refers to nearest matched local peaks, not exact anchor bins.
    manual = None
    if manual_matches is not None:
        manual = manual_matches.loc[manual_matches.matched].drop_duplicates(["chrom", "peak_start"])
        _hist(axes[0], manual.FS, np.linspace(-1, 1, 101), "crimson", "Manual: nearest matched peaks", .5)
    axes[0].axvline(t.min_fs, color="black", ls="--", label=f"FS percentile cutoff > {t.min_fs:g}")
    axes[0].set(xlabel="Fountain score", xlim=(-1, 1))
    _hist(axes[1], peaks.FS_peaks, np.linspace(0, 2, 201), "royalblue", "All local peaks")
    if manual is not None:
        _hist(axes[1], manual.FS_peaks, np.linspace(0, 2, 201), "crimson", "Manual: matched peaks", .5)
    axes[1].axvline(t.min_prominence, color="crimson", ls="--", label=f"Li prominence > {t.min_prominence:g}")
    axes[1].set(xlabel="Peak prominence", xlim=(0, 2))
    finite_noise = scores.loc[np.isfinite(scores.Scharr_box), "Scharr_box"]
    upper = max(t.max_noise*1.1, float(finite_noise.max()) if len(finite_noise) else 1)
    _hist(axes[2], finite_noise, np.linspace(0, upper, 101), "gray", "All scored bins")
    _hist(axes[2], peaks.Scharr_box, np.linspace(0, upper, 101), "royalblue", "All local peaks")
    axes[2].axvline(t.max_noise, color="navy", ls="--", label=f"Scharr percentile cutoff < {t.max_noise:g}")
    axes[2].set(xlabel="Scharr noise score", xlim=(0, upper))
    for ax in axes:
        ax.set_ylabel("Fraction per bin within each group")
        ax.legend(fontsize=7)
        ax.grid(False)
    fig.suptitle(result.label)
    return fig


def plot_noise_prominence(result, manual_matches=None, xlim=None):
    scores, t = result.scores, result.thresholds
    peaks = scores.loc[np.isfinite(scores.FS_peaks) & np.isfinite(scores.Scharr_box)]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(peaks.Scharr_box, peaks.FS_peaks, s=5, alpha=.2, c="gray", linewidths=0,
               rasterized=True, label=f"All peaks (n={len(peaks):,})")
    if manual_matches is not None:
        manual = manual_matches.loc[manual_matches.matched & np.isfinite(manual_matches.Scharr_box)].drop_duplicates(["chrom", "peak_start"])
        ax.scatter(manual.Scharr_box, manual.FS_peaks, c="crimson", s=32, edgecolor="white",
                   linewidth=.4, label=f"Manual: nearest peaks (n={len(manual)})")
    if len(result.calls):
        ax.scatter(result.calls.Scharr_box, result.calls.FS_peaks, facecolors="none", edgecolors="darkorange",
                   s=18, linewidths=.6, label=f"Passing all cutoffs (n={len(result.calls):,})")
    ax.axhline(t.min_prominence, c="crimson", ls="--", label=f"Prominence > {t.min_prominence:g}")
    ax.axvline(t.max_noise, c="navy", ls="--", label=f"Scharr < {t.max_noise:g}")
    ax.set(xlabel="Scharr noise score", ylabel="FS peak prominence", title=result.label, xlim=(0, None), ylim=(0, None))
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(False)
    ax.legend(fontsize=8)
    return fig


def plot_fountain_region(result, region, selected=None, max_distance=1_000_000,
                         vmin=.0003, vmax=.1, cmap="fall", mask_diags=True):
    clr = result.open_cooler()
    settings, t = result.settings, result.thresholds
    chrom, left, right = region
    start = max(0, int(left)//settings.resolution*settings.resolution)
    end = min(int(clr.chromsizes[chrom]), int(np.ceil(right/settings.resolution))*settings.resolution)
    if end <= start or max_distance <= 0:
        raise ValueError("Request a nonempty region and positive max_distance.")
    data = np.array(clr.matrix(balance=settings.weight).fetch((chrom, start, end)), dtype=float, copy=True)
    if mask_diags:
        i, j = np.indices(data.shape)
        data[np.abs(i-j) < settings.min_diag] = np.nan
    bins = clr.bins().fetch((chrom, start, end))
    edges = np.r_[bins.start.to_numpy(), bins.end.iloc[-1]]
    x = (edges[:, None]+edges[None, :])/2
    y = edges[None, :]-edges[:, None]
    scores = result.scores
    track = scores.loc[scores.chrom.eq(chrom) & scores.start.lt(end) & scores.end.gt(start)].sort_values("start")
    selected = result.calls if selected is None else selected
    selected = selected.loc[selected.chrom.eq(chrom) & selected.start.lt(end) & selected.end.gt(start)]
    has_noise = np.isfinite(track.Scharr_box).any()
    distance = min(max_distance, end-start)
    # The map's physical aspect is 0.5; size the panel rather than stretch pixels.
    map_height = 18*.85*.5*distance/(end-start)
    heights = [map_height, 1.] + ([1.] if has_noise else [])
    fig = plt.figure(figsize=(18, sum(heights)+1.4))
    gs = fig.add_gridspec(len(heights), 2, width_ratios=[40, 1], height_ratios=heights, hspace=.18, wspace=.04)
    ax = fig.add_subplot(gs[0, 0])
    fs_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    cax = fig.add_subplot(gs[0, 1])
    try:
        import cooltools.lib.plotting  # registers fall when supported
    except ImportError:
        pass
    if cmap not in plt.colormaps():
        cmap = "YlOrRd"
    im = ax.pcolormesh(x, y, np.ma.masked_invalid(data), cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat", rasterized=True)
    fig.colorbar(im, cax=cax, label="Balanced contacts")
    ax.set(xlim=(start, end), ylim=(0, distance), ylabel="Genomic separation", title=f"{result.label}: {chrom}:{start:,}-{end:,}")
    ax.set_aspect(.5, adjustable="box", anchor="S")
    ax.yaxis.set_major_formatter(EngFormatter("b"))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(axis="x", labelbottom=False)
    fs_ax.plot(track.center, track.FS, c="navy", lw=1)
    fs_ax.axhline(t.min_fs, c="navy", ls=":", lw=.7)
    peaks = track.loc[np.isfinite(track.FS_peaks)]
    strong = peaks.FS_peaks.gt(t.min_prominence)
    for subset, color, label in [(peaks.loc[~strong], "lightgray", "Below prominence cutoff"),
                                  (peaks.loc[strong], "crimson", "Above prominence cutoff")]:
        fs_ax.scatter(subset.center, subset.FS, c=color, marker="v", s=22, zorder=3, label=label)
    for row in selected.itertuples(index=False):
        ax.axvline(row.center, c="darkorange", ls="--", lw=.8, alpha=.7)
    fs_ax.scatter(selected.center, selected.FS, facecolors="none", edgecolors="darkorange", s=65, zorder=4, label="Selected calls")
    fs_ax.set(ylabel="FS", ylim=(-1.05, 1.05))
    fs_ax.legend(loc="upper right", fontsize=7, ncol=3)
    last = fs_ax
    panels = [ax, fs_ax]
    if has_noise:
        fs_ax.tick_params(axis="x", labelbottom=False)
        noise_ax = fig.add_subplot(gs[2, 0], sharex=ax)
        noise_ax.plot(track.center, track.Scharr_box, c="purple", lw=.8)
        noise_ax.axhline(t.max_noise, c="navy", ls="--", lw=.8)
        noise_ax.set_ylabel("Scharr")
        noise_ax.set_ylim(bottom=0)
        panels.append(noise_ax)
        last = noise_ax
    last.xaxis.set_major_formatter(EngFormatter("b"))
    last.set_xlabel(f"Position on {chrom}")
    for panel in panels:
        panel.grid(False)
    return fig


def compare_counts(results):
    import pandas as pd
    rows = []
    for label, result in results.items():
        eligible = int((np.isfinite(result.scores.FS) & result.scores.mask_fraction.ge(result.thresholds.min_mask_fraction)
                        & result.scores.nonzero_fraction.ge(result.thresholds.min_nonzero_fraction)).sum())
        rows.append({"dataset": label, "eligible_scored_bins": eligible, "n_calls": len(result.calls),
                     "calls_per_1000_eligible_bins": 1000*len(result.calls)/eligible if eligible else np.nan,
                     "fs_cutoff": result.thresholds.min_fs,
                     "prominence_cutoff": result.thresholds.min_prominence,
                     "noise_cutoff": result.thresholds.max_noise})
    return pd.DataFrame(rows)

"""Optional genomic-position control, preserved from the uploaded notebook.

Imports bioframe only when called. Supply target-specific eligible bins from the
clean pipeline; there is no hardcoded 10 kb cooler or stale df_strongest fallback.
"""
import numpy as np
import pandas as pd


def _genomic_midpoint_intervals(frame):
    result = frame.loc[:, ["chrom", "start", "end"]].drop_duplicates().copy()
    if result.empty or result.isna().any().any():
        raise ValueError("Intervals must be nonempty and contain no missing coordinates.")
    result["chrom"] = result["chrom"].astype(str)
    coords = result[["start", "end"]].to_numpy(dtype=float)
    if not np.isfinite(coords).all() or (coords != np.floor(coords)).any():
        raise ValueError("Coordinates must be finite integers.")
    result[["start", "end"]] = result[["start", "end"]].astype("int64")
    if (result.start < 0).any() or (result.end <= result.start).any():
        raise ValueError("Intervals must satisfy 0 <= start < end.")
    result["center"] = (result.start + result.end) // 2
    return result.reset_index(drop=True)


def nearest_enhancer_genomic_control_table(
    fountains, enhancers, eligible_bins,
    distances=(0, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000),
    n_random=1000, seed=42,
):
    """Sample eligible Hi-C bins, matching fountain chromosomes and widths.

    Enhancer positions stay fixed. Each draw samples without replacement within
    chromosome/width strata. Actual fountain bins remain in the eligible pool.
    """
    import bioframe as bf

    f = _genomic_midpoint_intervals(fountains)
    e = _genomic_midpoint_intervals(enhancers)
    pool = _genomic_midpoint_intervals(eligible_bins)
    thresholds = np.asarray(distances, dtype=np.int64)
    if thresholds.ndim != 1 or not len(thresholds) or (thresholds < 0).any():
        raise ValueError("Distance thresholds must be a nonempty list of nonnegative bp.")
    if n_random < 2:
        raise ValueError("Use at least two random draws.")
    coordinates = ["chrom", "start", "end"]
    membership = f.merge(pool[coordinates], on=coordinates, how="left", indicator=True)
    if (membership._merge != "both").any():
        raise ValueError("Some fountains are absent from eligible bins; check the Hi-C mask/resolution.")
    f["width"] = f.end - f.start
    pool["width"] = pool.end - pool.start
    strata = f.groupby(["chrom", "width"], sort=True).size()
    pool = pool.merge(strata.rename("N_fountains").reset_index(), on=["chrom", "width"])
    nearest_real = bf.closest(f, e, return_distance=True)
    nearest_real["distance"] = nearest_real.distance.to_numpy(dtype=float, na_value=np.inf)
    nearest_pool = bf.closest(
        pool, e, return_input=False, return_index=True, return_distance=True,
    )
    # closest can group its output by chromosome; align by bin index before sampling.
    pool["distance"] = nearest_pool.set_index("index").distance.reindex(pool.index).to_numpy(
        dtype=float, na_value=np.inf,
    )
    rng = np.random.default_rng(seed)
    random_counts = np.zeros((n_random, len(thresholds)), dtype=np.int64)
    first_draw = []
    audit = []
    for (chrom, width), group in pool.groupby(["chrom", "width"], sort=True):
        n = int(strata.loc[(chrom, width)])
        if len(group) < n:
            raise ValueError(f"Not enough eligible bins on {chrom} for width {width}.")
        values = group.distance.to_numpy()
        audit.append((chrom, width, n, len(group)))
        for draw in range(n_random):
            sampled = rng.choice(len(group), size=n, replace=False)
            random_counts[draw] += (values[sampled, None] <= thresholds).sum(axis=0)
            if draw == 0:
                first_draw.append(group.iloc[sampled])
    table = pd.DataFrame({
        "distance_from_enhancer_bp": thresholds,
        "N_intervals_real": (nearest_real.distance.to_numpy()[:, None] <= thresholds).sum(axis=0),
        "N_intervals_genomic_mean": random_counts.mean(axis=0),
        "N_intervals_genomic_q025": np.quantile(random_counts, 0.025, axis=0),
        "N_intervals_genomic_q975": np.quantile(random_counts, 0.975, axis=0),
    })
    return {
        "table": table,
        "nearest_real": nearest_real,
        "eligible_pool_nearest": pool,
        "first_draw": pd.concat(first_draw, ignore_index=True),
        "control_replicates": pd.DataFrame(random_counts, columns=thresholds).rename_axis("draw"),
        "chromosome_audit": pd.DataFrame(
            audit, columns=["chrom", "width_bp", "N_fountains", "N_eligible_bins"],
        ),
        "parameters": {"n_random": n_random, "seed": seed,
                       "N_fountains": len(f), "N_enhancers": len(e),
                       "distance_definition": "bioframe.closest interval-edge distance <= threshold",
                       "control": "random eligible Hi-C bins; chromosome and fountain-width matched",
                       "control_anchor_excluded": False,
                       "fountain_bins_in_pool": True},
    }


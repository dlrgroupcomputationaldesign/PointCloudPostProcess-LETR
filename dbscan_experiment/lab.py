"""Shared helpers for the DBSCAN auto-tuning experiments.

Deliberately outside ``post_process_src`` -- nothing here is imported by the
pipeline, so any feature scheme tried below can be evaluated without changing
production behaviour.

The pipeline clusters floor/ceiling points in
``floor_ceiling_util.cluster_floor_ceiling``::

    feats = StandardScaler().fit_transform(np.hstack([pts, col]))   # xyz + rgb
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)

Because StandardScaler z-scores every column, ``eps`` is in standard deviations,
not feet -- and each axis gets its own divisor, so one ``eps`` means a different
physical distance per axis and per dataset. The variants below exist to test
whether that is the right choice.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "out"

# pred_label values, from post_process.labels.DEFAULT_LABELS
LABELS = {"Other": 0, "Floor": 1, "Ceiling": 2, "Wall": 3}

DATASETS = {
    "laramie": REPO / "src" / "laram_inference_prediction.csv",
    "corteva": REPO / "src" / "00-10231-20_CortevaYorkTest_Output.csv",
}


def load(dataset, kind="Floor", basis="auto", sample=None, seed=0, verbose=True):
    """Load one dataset's floor (or ceiling) points, rotated like the pipeline.

    The stages axis-align before clustering, so the experiments do too --
    otherwise the per-axis standard deviations, which are the whole point of
    this study, would not match what DBSCAN actually sees.
    """
    path = DATASETS[dataset] if dataset in DATASETS else Path(dataset)
    full = pd.read_csv(path, low_memory=False)

    # Estimate the basis from the FULL cloud, before filtering to one label --
    # survey_basis needs the wall points, which a floor-only frame does not have.
    B = None
    if basis == "auto":
        from survey_basis import survey_basis_from_df
        B = np.array(survey_basis_from_df(full, verbose=False))
    elif basis != "identity":
        B = np.array(basis, dtype=float).reshape(3, 3)

    df = full[full["pred_label"] == LABELS[kind]].reset_index(drop=True)
    pts = df[["x", "y", "z"]].to_numpy(float)
    col = df[["r", "g", "b"]].to_numpy(float) / 255.0
    if B is not None:
        pts = pts @ B          # pipeline convention: aligned = xyz @ SURVEY_BASIS

    if sample and sample < len(pts):
        idx = np.random.default_rng(seed).choice(len(pts), sample, replace=False)
        pts, col = pts[idx], col[idx]

    if verbose:
        print(f"  {dataset}/{kind}: {len(pts):,} points   "
              f"xyz std {np.round(pts.std(axis=0), 3)}")
    return pts, col


# --- feature variants ------------------------------------------------------
# Each returns (features, description). `eps` means something different in each,
# which is exactly what the sweep is meant to expose.

def feat_current(pts, col):
    """What the pipeline does today: per-axis z-score of xyz AND rgb."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(np.hstack([pts, col])), \
        "per-axis z-score, xyz+rgb (eps = std devs)"


def feat_xyz_zscore(pts, col):
    """Per-axis z-score of xyz only -- isolates the effect of dropping colour."""
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(pts), \
        "per-axis z-score, xyz only (eps = std devs)"


def feat_isotropic(pts, col):
    """One shared divisor for xyz, so the metric stays geometrically faithful.

    Removes the per-axis anisotropy (on Corteva the current scheme stretches z
    by 175x relative to x) while still being scale-free, so eps ports across
    datasets in feet or metres alike.
    """
    p = pts - pts.mean(axis=0)
    return p / float(np.linalg.norm(p, axis=1).std()), \
        "isotropic xyz (eps = multiples of RMS radius)"


def feat_physical(pts, col):
    """Raw coordinates -- eps is in the cloud's own units (feet here)."""
    return pts - pts.mean(axis=0), "raw xyz (eps = feet)"


VARIANTS = {
    "current": feat_current,
    "xyz_zscore": feat_xyz_zscore,
    "isotropic": feat_isotropic,
    "physical": feat_physical,
}


# --- quality metrics -------------------------------------------------------

def cluster_metrics(labels, pts, min_points=10000):
    """Score a clustering the way the pipeline cares about.

    cluster_floor_ceiling keeps only clusters larger than ``min_points`` and
    treats each as one slab, which fit_ceiling_floor then RANSAC-fits to a
    plane. So a good clustering yields a handful of clusters that are THIN in z
    -- a thick cluster means two storeys were merged and the plane fit will
    straddle them.
    """
    out = {"n_raw": 0, "n_kept": 0, "noise_frac": 1.0,
           "median_thickness": float("nan"), "max_thickness": float("nan"),
           "kept_frac": 0.0}
    if labels is None or not len(labels):
        return out

    out["noise_frac"] = float((labels == -1).mean())
    uniq = [l for l in np.unique(labels) if l != -1]
    out["n_raw"] = len(uniq)

    thicknesses, kept_n = [], 0
    for l in uniq:
        m = labels == l
        if m.sum() <= min_points:
            continue
        kept_n += int(m.sum())
        z = pts[m, 2]
        thicknesses.append(float(np.percentile(z, 95) - np.percentile(z, 5)))

    out["n_kept"] = len(thicknesses)
    out["kept_frac"] = kept_n / len(labels)
    if thicknesses:
        out["median_thickness"] = float(np.median(thicknesses))
        out["max_thickness"] = float(np.max(thicknesses))
    return out


def z_profile(pts, bin_size=0.25, min_frac=0.02, min_storey_sep=8.0):
    """Find the storey structure directly, to know what the clustering SHOULD find.

    Floor points pile up at each slab, so a z-histogram has one peak per storey.
    The gaps between those peaks bound eps from above (cluster further than the
    gap and two storeys merge); the in-slab point spacing bounds it from below.

    ``min_storey_sep`` (in the cloud's units -- FEET for these CSVs) merges peaks
    that are too close to be separate floors. Without it a sloped or tiered
    level, like Laramie's upper deck spanning z 13.9-17.6, reads as three
    storeys and every clustering looks wrong by comparison.

    Returns (n_storeys, peak_z list, min_gap).
    """
    z = pts[:, 2]
    lo, hi = z.min(), z.max()
    nbins = max(2, int(np.ceil((hi - lo) / bin_size)))
    counts, edges = np.histogram(z, bins=nbins)
    centres = 0.5 * (edges[:-1] + edges[1:])

    thresh = counts.max() * min_frac
    occupied = counts >= thresh

    peaks, run = [], []
    for i, on in enumerate(occupied):
        if on:
            run.append(i)
        elif run:
            peaks.append(run)
            run = []
    if run:
        peaks.append(run)

    peak_z = [float(centres[r[int(np.argmax(counts[r]))]]) for r in peaks]

    # Merge peaks nearer than one storey apart -- a tiered or sloped floor makes
    # several peaks that all belong to the same level.
    merged, group = [], [peak_z[0]] if peak_z else []
    for z_peak in peak_z[1:]:
        if z_peak - group[-1] < min_storey_sep:
            group.append(z_peak)
        else:
            merged.append(group)
            group = [z_peak]
    if group:
        merged.append(group)

    levels = [float(np.mean(g)) for g in merged]
    gaps = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
    return len(levels), levels, (min(gaps) if gaps else float("inf"))


def fmt_metrics(m):
    return (f"clusters {m['n_raw']:4d} raw / {m['n_kept']:2d} kept   "
            f"noise {m['noise_frac']*100:5.1f}%   kept {m['kept_frac']*100:5.1f}%   "
            f"z-thickness med {m['median_thickness']:7.2f} max {m['max_thickness']:7.2f}")

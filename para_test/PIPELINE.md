# Floor, ceiling & wall extraction — process reference

How `run_floors` / `run_ceilings` / `run_walls` turn a segmented point cloud
into floor, ceiling and wall geometry, and why each step works the way it does.

Floor and ceiling numbers were measured on three buildings: **laramie**
(2 storeys, tiered upper deck), **corteva** (single storey, flat-plus-pitched
roof), **ty** (single storey, sparse, 45° rotated). **Wall numbers come from
laramie alone** — that section is younger and less well evidenced than the rest
of this document, and says so where it matters.

Runnable configuration: [`para_test/test.py`](test.py).

```
segmented CSV (x,y,z,r,g,b,pred_label)
    |
    0. axis-align            SURVEY_BASIS      xyz @ B
    |
    1. level detection       height histogram, split at voids
    |                        -> one cluster per storey
    2. plane fit             IRLS (robust, deterministic)
    |                        -> plane + inlier threshold
    3. boundary              raster: occupancy grid -> contour
    |                        -> floors[] / ceilings[]
    |
    4. walls                 squash a storey to a top-down image,
                             find its lines (LETR), thicken them
                             -> walls[] with footprint, bbox, zRange
```

Stages 0–3 work on surfaces in 3D. Stage 4 works on a picture of the building
seen from above, and consumes stage 2's `floor_bboxz` to decide which points
belong to which storey.

---

## 0. Axis alignment — `SURVEY_BASIS`

Everything downstream assumes the building is square to the X/Y axes. Levels are
found by binning **z**, planes are fitted as `z = f(x,y)`, and boundaries are
rastered on an **axis-aligned** grid — all three degrade if the building sits at
an angle.

### The convention

```python
aligned = xyz @ SURVEY_BASIS          # into the working frame
result  = pts @ SURVEY_BASIS.T        # back out to the input frame
```

In-then-out is the identity, so output coordinates stay in the input cloud's
frame and nothing downstream has to know a rotation happened.

`point_axis_align(df, survey_basis)` takes the matrix **exactly as configured**
and applies it directly: what you put in `parameters["SURVEY_BASIS"]` is what
gets multiplied, with no hidden transpose. `.T` appears in exactly one place —
rotating results back out to the input frame.

**Getting the direction backwards rotates the building the wrong way by *twice*
the yaw.** A transposed rotation matrix is still a valid rotation — orthonormal,
determinant +1 — so nothing errors and the geometry looks reasonable. On laramie
the error would be 2 × 13.79° = **27.58°**.

The one reliable check is the footprint: **a correct basis can only shrink the
axis-aligned bounding area.**

```
laramie wall points, unrotated
  identity        98.4 x 167.6   16,499 sq ft
  correct         69.0 x 167.6   11,564 sq ft   30% tighter
  transposed     129.0 x 167.0   21,543 sq ft   WORSE than doing nothing
```

`survey_basis.py` warns when the gain falls below 0.5%.

### Estimating it from the cloud

If no registered basis is available, [`survey_basis.py`](../survey_basis.py)
recovers it. A laser scan is already Z-up, so the only unknown is the **yaw**.

**1 — Select wall points.** Only this step differs between the two sources; see
[below](#step-1-in-detail-csv-vs-e57). From a CSV, `pred_label == Wall` selects
them directly. From a raw e57 there are no labels, so a mid-height slab stands
in for them.

**2 — Estimate normals, keep the vertical surfaces.** `estimate_normals` at a
neighbourhood radius, then keep points with `|n_z| < 0.15` — within ~8.6° of
vertical. A wall's normal is horizontal.

**3 — Fold the azimuths mod 90° and take the circular mean.** This is the
non-obvious step. A rectangular building's walls face four directions — θ, θ+90,
θ+180, θ+270 — and all four describe the *same* grid. Averaging them naively
gives nothing: 0°, 90°, 180°, 270° cancel to zero.

Multiplying every angle by 4 first collapses them onto one direction:

```
4θ,  4θ+360,  4θ+720,  4θ+1080   ≡  4θ  (mod 360)
```

Then take the circular mean — average the sines and cosines separately and
`atan2` the result, since you cannot arithmetically average 359° and 1° — and
divide by 4:

```python
a4 = 4.0 * azimuths
s, c = np.sin(a4).mean(), np.cos(a4).mean()
yaw = np.rad2deg(np.arctan2(s, c) / 4.0) % 90.0
strength = np.hypot(s, c)
```

`strength` is the resultant length: **1.0** means every wall agrees on one grid,
**below ~0.4** means the building is not rectilinear and "axis-aligned" has no
well-defined answer. Measured 0.66–0.81 on these three buildings.

**4 — Repeat at three radii, take the median.** Radii are 0.5×, 1×, 2× a base of
**0.5% of the cloud diagonal** — which is what makes the function unit-agnostic,
landing near 0.3 m on a metric e57 and ~1 ft on a feet-based CSV without a unit
flag anywhere. A real Manhattan grid is scale-stable; noise is not, so the
*spread* across radii is the confidence signal.

The median is taken on the mod-90 circle (re-centred before averaging), so 89.9°
and 0.1° register as 0.2° apart rather than 89.8°.

**5 — Resolve the 90° ambiguity.** Rotating by yaw or yaw−90 both align the
cloud; `long_axis="x"` picks the one putting the building's longer side on X, so
the choice is deterministic rather than incidental.

**6 — Validate.** Footprint before vs after, as above.

### Step 1 in detail: CSV vs e57

Steps 2–6 are shared code (`_basis_from_points`). Only the point selection
differs, and the two sources have opposite problems: the CSV knows *which*
points are walls but has few of them; the e57 has enormous numbers of points but
no idea what any of them are.

**From the CSV** (`survey_basis_from_df`) — one line does it: keep the rows
labelled `Wall`. If fewer than 500 carry that label the function falls back to
the same mid-height slab the e57 path uses, and says so, rather than estimating
a grid from a handful of points.

**From the e57** (`survey_basis_from_e57`) — four steps, each solving a problem
the CSV does not have:

**a. Stream it.** These scans are 2–3 GB and will not fit comfortably in memory
alongside everything else, so `pye57` reads them scan by scan in chunks of 2
million points. Nothing is ever fully materialised.

**b. Keep every 60th point** (`stride=60`). A survey-grade scan holds far more
detail than an angle estimate needs. The stride is maintained *across* chunk
boundaries — each chunk works out where the previous one left off rather than
restarting at its own first point, which would quietly over-sample the start of
every chunk.

**c. Voxel-downsample to 0.05 m.** This is the step that matters most, and it is
**not for speed**. A laser scanner sweeps at constant *angular* steps, so it
lands far more points on nearby surfaces than distant ones — the wall three
metres away collects vastly more hits than the same wall thirty metres away. If
you average directions over the raw points, the walls closest to wherever the
scanner happened to stand dominate the answer, and a single corner of a single
room can outvote the whole building. Voxel downsampling divides space into 5 cm
cubes and keeps one point per cube, so *every surface votes in proportion to its
area* rather than its distance from the tripod. That is the property the
estimator needs and the only reason the e57 path works without labels.

**d. Keep the middle of the height range** — 15% to 80% of the span between the
1st and 99th percentile of z. Floors and ceilings have normals pointing straight
up or down; they carry no information about which way the building faces, and
because they are large and densely scanned they would flood the vote. Cutting
top and bottom leaves mostly walls. Percentiles rather than min/max, so one
stray point far below the slab does not shift the whole window.

Note what is *not* needed: the e57 is in metres, unshifted, in the scanner's own
coordinate system, while the CSV is in feet and moved to the origin. **Neither
matters**, because the answer is an angle. Scaling a building does not rotate
it and moving it does not either — so both files give the same yaw with no unit
conversion anywhere.

The trade is time. The CSV is already in memory; the e57 takes roughly 1–2
minutes for a 2.7 GB scan, which is why [`test.py`](test.py) caches the result.

### How the cross-check works

[`check_survey_basis.py`](check_survey_basis.py) runs both estimators on every
project and applies **two independent checks**. Independence is the point: they
fail in different ways, and one of them works without a reference value.

**Check 1 — yaw agreement** (needs a registered basis to compare against).

The angle is read back out of a matrix with `atan2(m[1,0], m[0,0])`, then two
angles are compared with the 90° ambiguity folded out:

```python
abs((yaw_a - yaw_b + 45.0) % 90.0 - 45.0)
```

Reading that plainly: add 45, wrap anything over 90 back around, subtract 45.
The result always lands between −45° and +45°, so a difference of 0°, 90°, 180°
or 270° all come out as **zero**. That is deliberate — rotating a rectangular
building by a quarter turn axis-aligns it just as well, it only swaps which side
ends up on X. Treating that as an error would flag correct answers. A genuine
disagreement is one that is *not* a multiple of 90, and it survives the fold.

**Check 2 — footprint** (needs nothing but the cloud itself).

Rotate the wall points by the candidate matrix, take the axis-aligned bounding
box, multiply width by height. The rule:

> A correct basis can only **shrink** the axis-aligned footprint.

The reason is geometric and needs no algebra. Picture a rectangular building
sitting at an angle inside a box drawn square to the page: the box has to be
large enough to contain the far corners, so it wastes area on all four sides.
Turn the building until it is square to the page and the box closes in on it.
Any other angle wastes more. The minimum is exactly the alignment you want.

This check is the one that catches a **transposed** matrix — the direction error
described above — and it catches it without knowing the right answer:

```
laramie wall points
  identity        16,499 sq ft
  correct         11,564 sq ft   30% tighter
  transposed      21,543 sq ft   WORSE than doing nothing
```

A rotation applied backwards turns the building *further* from square, so the
area goes up. Since it is still a perfectly valid rotation matrix, nothing else
in the pipeline objects. This is why `survey_basis.py` runs the same test
internally on every estimate and warns below a 0.5% gain.

**Why both.** Check 1 is sharper — it measures the error in degrees — but it
only works when someone has already registered the building, which is precisely
the situation where you would not need an estimator. Check 2 is cruder but
self-contained, so it is the one that runs in production on a building nobody
has surveyed.

Neither check can catch an error that is *consistent* between the value being
tested and the value it is tested against — which is exactly ty's situation
below.

### Measured accuracy

Cross-checked against the registered basis on all three projects
([`check_survey_basis.py`](check_survey_basis.py)):

| project | from CSV | from e57 | footprint gain | note |
|---|---|---|---|---|
| laramie | 0.064° | **0.028°** | 29.9% | registered basis is independent |
| corteva | 0.066° | **0.008°** | ~0% (already square) | registered basis is independent |
| ty | 0.021° | 0.000° | 27.3% | **circular** — see below |

**The e57 is more accurate on every project**, because it has far more points
and does not depend on segmentation quality. Corteva's e57 contributed 633,794
slab points at strength 0.790, against the CSV's wall labels at 0.751.

**Ty's 0.000° is not independent validation.** Its registered basis was itself
produced by `survey_basis_from_e57` on the same file
([`compute_survey_basis_run.py`](../compute_survey_basis_run.py)), so that
comparison measures reproducibility, not correctness. The exact match to three
decimals is the giveaway. Laramie and corteva are the real checks.

**Ty is also the case where the e57 is less stable**, despite the better central
value: its scan is decimated to 50 mm (0.28 GB, 108,131 slab points) and its
radius spread is **0.099°** against the CSV's 0.007°. Point count, not method,
drives that.

### Corteva reads as "already aligned" — and that is correct

Both estimators return yaw ≈ 0 mod 90 for corteva, and the footprint barely
moves (33,517 → 33,520). `survey_basis.py` emits *"footprint barely changed —
this cloud is already axis-aligned, use the identity matrix"*. For corteva that
warning **is the answer**, not a failure. The same warning is what catches a
basis estimated from the wrong building.

### In `test.py`

```python
SURVEY_BASIS_SOURCE = "e57"     # or "csv"
```

Resolution order: a project's `"basis"` entry wins; otherwise the source is
estimated. The e57 result is cached to
`para_test_output/<project>/survey_basis.json`, keyed on the scan's path and
modification time — streaming 2.7 GB on every parameter tweak would otherwise
dominate a tuning run. Replace the scan and it recomputes automatically.

The script prints the yaw both raw and mod 90, since the 90° ambiguity is the
thing most easily misread when comparing against a registered value.

---

## 1. Level detection — height histogram

`CLUSTER_METHOD_F/C = "histogram"`

### The algorithm

1. **Bin** the z values at `HIST_BIN_M` (0.08 m ≈ 0.26 ft).
2. **Threshold**: a bin is *occupied* if it holds at least `HIST_MIN_FRAC` (2%)
   of the tallest bin. Contiguous occupied bins form a **run**.
3. **Split at voids**: consecutive runs separated by less than
   `MIN_LEVEL_GAP_M` (0.5 m) join into one level. A gap wider than that is a
   storey boundary.
4. **Qualify**: a level must hold at least `MIN_LEVEL_POINTS_FRAC` (2%) of the
   points, otherwise it is dropped as an artifact.
5. **Elevation** = the **count-weighted centroid** of each level's bins.
6. **Assign** each point to the nearest level, with boundaries at the midpoints.
7. **Reject noise**: points further than `MIN_STOREY_SEP_M / 2` from *every*
   level become unassigned.
8. **Trim to the dominant band** (`DOMINANT_BAND_ONLY_F/C`): within each level,
   keep only the run containing the tallest bin.

### Reading `{type}_zhistogram.png`

Every parameter above is visible in that figure. Using laramie's **ceiling**
(147,625 points, two ceilings at 12.42 and 24.15):

| what you see | parameter | on laramie's ceiling |
|---|---|---|
| **bar height** of each row | `HIST_BIN_M` | 0.08 m = **0.262 ft** per bar |
| **orange dotted** vertical line | `HIST_MIN_FRAC` | 2% of the tallest bin (23,000) = **460 points**. Bars right of it are *occupied*; a run of them is a candidate surface |
| **red solid** horizontal lines | *output* — the count-weighted centroid of each level | **z = 12.42** and **z = 24.15** |
| **blue dashed** horizontal line | *derived* — midpoint between adjacent levels | (12.42 + 24.15)/2 = **z = 18.29**. Points below go to level 0, above to level 1 |
| **green shaded** bands | `MIN_STOREY_SEP_M / 2` | 2.4 m / 2 = **±3.94 ft** around each level → [8.5, 16.4] and [20.2, 28.1]. Points outside *both* bands are dropped as noise |

Two parameters are **not drawn**, and both matter:

**`MIN_LEVEL_GAP_M` is the empty region itself.** Look at z ≈ 16–20: no bars.
That void is what split this ceiling in two. Laramie's ceiling has 8 occupied
runs with gaps of `0.52, 0.52, 0.52, 1.30, 5.19, 0.52, 0.52` ft between them —
only the **5.19 ft** one exceeds `MIN_LEVEL_GAP_M` (0.5 m = 1.64 ft), so the
other seven merge and exactly one split happens. Widen the parameter past 5.19
and you would get one level; narrow it below 1.30 and you would get three.

**`MIN_LEVEL_POINTS_FRAC` is a qualification test with no visual.** Level 0
holds 27,593 of 147,625 points = **18.7%**, comfortably over the 2% floor. A
level below it would simply be absent from the plot, which is why a *missing*
level is harder to diagnose than a wrong one.

**`DOMINANT_BAND_ONLY_F/C` acts after this figure is drawn.** The green bands
show what the *noise radius* keeps; the dominant-band trim then discards
everything outside the densest run within each level, and that is not shown. On
this ceiling it dropped a further 2,014 points from level 0 and 3,953 from
level 1 — so the points actually fitted are a subset of what the green bands
suggest. The `[pp] level N: kept densest band z A..B, dropped X of Y` log line
is the only record of it.

The title carries the three numbers that set the split: bin width, the void
threshold, and the noise radius.

### Why split at voids, not at peak distance

The earlier implementation merged *peaks* closer than a threshold. That merges
**transitively** — each peak is compared to the previous one, so a chain of
closely spaced peaks collapses across an arbitrary span.

Laramie's ceiling has 8 runs with consecutive peak gaps of
`1.04, 1.56, 1.04, 1.30, 5.19, 0.78, 3.64` ft. Every one is below the 7.87 ft
threshold, so all eight chained into a single "level" whose unweighted mean
landed at **z = 16.25 — a height with no points at all**, sitting in the void
between two real ceilings.

The gap criterion splits at the one real void (5.19 ft) and returns
**[12.42, 24.15]**, the two ceilings visible in the histogram.

It is also far more robust. Number of levels held constant over:

| criterion | safe range | ratio |
|---|---|---|
| `MIN_STOREY_SEP_M` (peak distance) | 1.0 – 3.0 m | 3× |
| `MIN_LEVEL_GAP_M` (histogram void) | 0.2 – 2.5 m | **12.5×** |

The two bounds on `MIN_LEVEL_GAP_M` are *physically independent*, which is why
a fixed value ports across buildings:

- **lower** — density dips inside a single tiered level (~0.16 m measured)
- **upper** — the minimum habitable floor-to-floor (2.4 m) less each level's own
  point spread. At the worst spread measured (0.87 m) the surviving gap is still
  **1.42 m**, so anything below that separates even a minimum-height mezzanine.

`MIN_STOREY_SEP_M`'s upper bound *was* the floor-to-floor spacing itself, so it
collapsed on tight buildings — a 2.7 m residential storey leaves 2.4 m with only
11% margin.

### Why trim to the dominant band

A level's points are not all its surface. Laramie's upper *Floor* level held
four horizontal sheets ~1 ft apart; the two lowest were the storey's **ceiling**
misclassified as floor. Cross-checking against the Ceiling label:

```
z=12.77:  32,541 floor-labelled |  8,799 ceiling-labelled  -> 21.3% ceiling
z=13.86:  27,793 floor-labelled |  2,962 ceiling-labelled  ->  9.6% ceiling
z=15.21:  51,494 floor-labelled |    623 ceiling-labelled  ->  1.2% ceiling  <- real floor
z=16.55:  21,820 floor-labelled |    396 ceiling-labelled  ->  1.8% ceiling
```

The contamination is concentrated exactly where the extra sheets are. Keeping
only the densest run drops **70,800 of 165,130 points (43%)** on that level and
is a no-op on clean ones (level 0 dropped 282 of 102,761).

### Why not DBSCAN

DBSCAN scored the same or worse on all three datasets (`inlier_ratio × coverage`:
laramie 0.671 vs 0.677, corteva 0.949 vs 0.992, ty 0.947 vs 0.953), ran in
**minutes against 8–16 ms**, and its `eps` cannot be tied to anything physical.

Translating "0.5 m of empty space" into `eps` requires dividing by `std_z`,
which grows *because* the building has storeys:

```
laramie:  gap->eps 0.246   but kNN knee 0.476    window EMPTY
corteva:  gap->eps 4.926       knee 0.695        usable
ty:       gap->eps 1.105       knee 0.447        usable
```

On laramie no `eps` can both chain a level together and refuse to bridge levels.
`CLUSTER_METHOD = "dbscan"` is retained for the one case histogram cannot do:
two disjoint surfaces at the *same* height. None of these datasets exercises it.

---

## 2. Plane fitting — IRLS

`PLANE_METHOD_F/C = "irls"`

Fits **z = a·x + b·y + c** by iteratively reweighted least squares. For floors
and ceilings that model is not a restriction (a vertical plane is not one) and
it makes the near-horizontal constraint automatic rather than a rejection test.

Each pass:

1. Weighted least squares → plane.
2. Residuals → robust scale `1.4826 × MAD`. The median cannot be dragged by
   outliers, unlike a standard deviation.
3. **Tukey biweight** weights: `w = (1 − u²)²` for `|u| < 1` where
   `u = residual / (4.685 × scale)`, else **exactly zero**. Gross outliers are
   excluded, not merely down-weighted.
4. Stop when the coefficients stop moving.

`DIS_THR = "auto"` takes the inlier threshold from the fit's own scale
(`4.685 × scale`, the Tukey cutoff), clamped to
`[AUTO_DIS_THR_MIN_M, AUTO_DIS_THR_MAX_M]`.

### Why not RANSAC

**Reproducibility.** Over five repeats on the same points:

| band | RANSAC-derived threshold | spread | IRLS | spread |
|---|---|---|---|---|
| laramie floor 0 | 0.459, 0.459, 0.428, 0.493, 0.488 | 1.15× | 1.128 ×5 | **1.00×** |
| laramie floor 1 | 2.223, 0.565, **0.136**, 0.756, 0.530 | **16.32×** | 3.808 ×5 | **1.00×** |
| corteva floor | 0.102, 0.082, 0.082, 0.102, 0.102 | 1.25× | 0.143 ×5 | **1.00×** |

The 16× swing on laramie's tiered band propagated everywhere downstream — floor
thickness varied 0.55 → 6.34 ft between identical runs, and the boundary
changed piece count. IRLS has no random sampling, so identical input gives
identical output.

It is also insensitive to initialisation: six different starting points converge
to the same plane, so a RANSAC seeding step would add nothing but variance.

**When RANSAC would still win:** above ~50% outliers, where IRLS's least-squares
seed could land in the wrong basin. These surfaces run 1–4% outliers.

**Measured:** every fitted plane came out **0.01–0.10° from horizontal**, with
derived thresholds spanning nearly 5× across surfaces (scale 0.060 to 0.288) —
no single fixed `DIS_THR` covers that range.

`RANSAC_N` and `NUM_ITER` are ignored under IRLS. (For the record, under RANSAC
`NUM_ITER` is only an upper bound anyway: Open3D's `segment_plane` terminates
adaptively via its `probability` argument, and raising the cap from 10 to
100,000 changes runtime by ~5× and results not at all.)

---

## 3. Boundary — raster

`BOUNDARY_METHOD_F/C = "raster"`

1. **Rasterize** the inliers' XY into an occupancy grid at `BOUNDARY_CELL_*`.
2. **Morphological close** with an elliptical kernel of radius
   `BOUNDARY_FILL_GAP_*` — dilate then erode, which fills voids narrower than
   the kernel *without moving the outer boundary*.
3. **`findContours(RETR_EXTERNAL)`**, keep the largest.
4. **`approxPolyDP`** at `BOUNDARY_SIMPLIFY_*` (absolute).
5. Map pixels back to world coordinates at cell centres.

### Auto-derived parameters

All three collapse to one measurable input, the **inlier point spacing**
(median distance to the 6th XY neighbour):

| parameter | rule | why |
|---|---|---|
| `BOUNDARY_CELL_*` | ≈ spacing | below it the grid speckles |
| `BOUNDARY_FILL_GAP_*` | 2 × cell | hard floor; above it nothing changes |
| `BOUNDARY_SIMPLIFY_*` | 1.5 × cell | erases stair-steps, not real corners |

**Cell must not be finer than the point spacing.** At cell 0.25 ft against
0.87 ft spacing only **29.8%** of cells are hit and the image shatters into
**4,644 blobs**; at cell ≈ spacing occupancy jumps to ~71% and the grid is a
coherent sheet:

```
   cell   occupied   blobs before close   A/A_occ
   0.25     29.8%          4,644           0.980
   0.87     71.7%              4           0.995
   1.00     70.0%              3           0.997
   2.00     69.1%              1           0.965
```

**Simplify must be absolute, not a fraction of perimeter.** The legacy
`BOUNDARY_SIMPLIFY_EPS_FRAC_*` scales with building size for no geometric
reason — the same 0.001 is 0.60 ft on one of these clouds and 1.18 ft on a
larger one, which is why no single fraction ports. Normalised by cell instead,
both buildings agree at ≈1.3 × cell. The old default of 0.02 was a **10.4 ft**
tolerance that flattened laramie's floor to a 4-vertex rectangle.

### Why not alphashape

| | verts | A/A_occ | containment | time |
|---|---|---|---|---|
| laramie alphashape α=0.05 | 114 | 1.026 | 0.998 | **34.4 s** |
| laramie raster | 5 | **0.997** | 0.972 | **0.0 s** |
| ty alphashape α=0.05 | 456 | 1.310 | 0.995 | **13.0 s** |
| ty raster | 11 | 1.305 | 0.972 | **0.0 s** |

Equal or better area accuracy, effectively instant, and `fill_gap` is a physical
distance so it does not depend on point density — which is what broke alphashape
on ty. Alpha is also dangerous: at α=1.0 ty fragmented into **42 pieces**,
discarding ~5,900 real floor points silently.

### `BOUNDARY_CONNECT_*`

Grows `fill_gap` until the mask is a single region, bounded at 30 × cell.

**On for ceilings** (`BOUNDARY_CONNECT_C = True`): a ceiling is one continuous
surface, so its interior voids are occlusion shadows — fixtures, ducts,
structure blocking line of sight. Without it laramie's ceiling fragmented into
**213 blobs** and the boundary traced whichever happened to be largest, giving a
646-vertex outline that snaked through the middle of the room. With it: 1 region,
38 edge points.

**Off for floors** (`BOUNDARY_CONNECT_F = False`): a floor genuinely can have an
atrium or cutout, so a void is not automatically occlusion. Ty's floor has
~2,064 sq ft of interior voids we could not classify; auto-connecting would
decide that question without evidence.

Cost: connecting inflates area. Laramie's lower ceiling went A/A_occ 0.823 →
~1.43 between 2× and 15× cell. If the voids really are shadows that is correct —
the ceiling *is* there, unscanned — but it asserts coverage the points do not
directly evidence.

---

## 4. Walls — line segmentation

Walls are found in a completely different way from floors and ceilings. Those
are fitted as **surfaces in 3D**. A wall is found by **looking at the building
from above**.

The idea: take one storey's worth of points and squash them flat onto the floor,
forgetting height entirely. Furniture, clutter and scan noise smear out into a
faint haze, because they only exist at one or two heights. A wall does not smear
— it is the one thing in a building that is solid from floor to ceiling, so
every laser hit at every height lands on the *same spot* on the plan. Stack
those hits and walls emerge as sharp dark lines on a pale background. What comes
out is essentially a floor plan drawing, and finding walls becomes the problem
of finding the lines in that drawing. That is what the LETR model does.

```
points between two floors
   |
   PROJECTED_BINS       squash to a top-down density image
   |
   RESIZE_WIDTH         resize to the working resolution
   INT_THR              turn grey into pure black and white
   WALL_MORPH_KERNEL    repair single-pixel dropouts along a wall
   |
   LETR model           -> 1000 candidate line segments, each with a score
   SCORE_THR            keep the confident ones
   WALL_ANGLE_TOL_DEG   sort into "runs one way" / "runs the other way"
   BUFFER_THR           thicken each thin line into a wall-shaped rectangle
   |
   -> polygons -> the points inside each -> bounding boxes -> walls[]
```

Every number below was measured on **laramie** only. Walls have not been run on
corteva or ty.

### Which points belong to a storey

The stage does not use the wall segmentation labels to decide *where* to look.
It takes every point lying between the top of one floor and the bottom of the
next — `floor_bboxz`, straight from stage 2. So the wall stage inherits whatever
the floor stage decided, and moving a floor boundary moves the slab of points
the walls are cut from.

### `PROJECTED_BINS` — how fine the top-down image is

The squashing is done by laying a grid over the building seen from above and
counting how many points land in each grid square. `PROJECTED_BINS = 300` means
300 squares across.

This is the setting with the **largest effect and the least evidence behind it**.
Dropping it to 100 took one storey from 52 detected lines to 311. That sounds
dramatic and it is, but a line count cannot tell you whether those extra 259
lines are real walls that were being missed, or one wall shattering into
fragments. Both raise the number identically. It is left at 300 because nobody
has yet looked at the two images side by side.

It no longer controls output precision, though — that was split out into a
continuous coordinate map, see
[below](#output-coordinates-why-pixel_to_xy-no-longer-uses-the-grid). What
remains is purely a detection-resolution question, and it is still open.

### `RESIZE_WIDTH` — the working resolution

**This is not the model's input size**, which is the natural assumption and is
wrong. Chasing that assumption cost real time, so it is worth being explicit.

Inside `img_process_model_input` the image is resized to `RESIZE_WIDTH`, then
thresholded, then passed through `Resize([test_size])` with `test_size = 1100`
hard-coded. That last step scales whatever it is given so the **short** side
becomes 1100 pixels. The model therefore always receives an image about
1100 × 2418 — no matter what `RESIZE_WIDTH` is set to.

So `RESIZE_WIDTH` does not change how much the model sees. It changes **what
survives to be seen**. Shrinking an image averages neighbouring pixels together,
and a wall only a couple of pixels wide gets averaged with the empty floor
either side of it until it is too faint to clear the threshold. It is gone
before the model ever looks. Setting it back up again cannot recover it.

| `RESIZE_WIDTH` | binary image | lines | polygons |
|---|---|---|---|
| 400 *(shipped)* | 400 × 182 | 38 | 19 |
| 800 | 800 × 364 | 51 | 30 |
| **1200** | 1200 × 546 | **52** | **32** |
| 1800 | 1800 × 819 | 58 | 37 |
| 2325 *(native render)* | 2325 × 1058 | 58 | 43 |

Most of the gain arrives by 1200 and the curve flattens. The setting is free —
it costs one resize, not model time, because the model's input size never
changed. Set to **1200**.

### `INT_THR` — deciding what counts as "wall" brightness

After squashing, every pixel has a brightness: how many points landed there.
Somewhere there is a cutoff — below it, background; above it, wall. The shipped
code used a fixed number.

That number is unanchored. It is a brightness value on an image that has been
colour-mapped, re-rendered by matplotlib and resized, so it has no direct
relationship to "how many points per square foot". Change the point density,
the colour map, `PROJECTED_BINS` or `RESIZE_WIDTH` and the same number means
something different.

`INT_THR = "auto"` uses **Otsu's method** instead. In plain terms: imagine
sorting every pixel in the image by brightness. You have to draw one dividing
line somewhere. Otsu tries *every* possible place to draw it and keeps the one
where the two resulting groups are each as internally consistent as possible —
where the dark pixels are all similarly dark and the bright ones all similarly
bright. The worst place to cut is in the middle of a group; the best is in the
valley between two groups. You never have to know the right number, because the
image's own brightness distribution decides.

On laramie it picked **34** and **33** for the two storeys. Close together, but
picked from the images rather than assumed — and that is the point. A building
scanned at different density gets a different number automatically.

### `WALL_MORPH_KERNEL` — repairing dropouts

A wall in the projected image is rarely a perfectly solid line. Doorways,
scanner shadows and thin spots leave single-pixel holes, and a broken line reads
as several short lines instead of one long one.

The repair is *closing*: first spread every bright pixel outward by a little
(dilate), which floods small holes shut; then shrink everything back by the same
amount (erode). Anything genuinely solid returns to its original size. Small
holes do not come back, because there is nothing left to erode them from. The
kernel is how far "a little" is, in pixels.

The shipped value was **1**, which is a no-op — spreading by a 1 × 1 kernel and
shrinking by a 1 × 1 kernel returns the input unchanged. The default is now
**3**. Note this means callers who never set it get different output than
before; the previous behaviour was not a choice worth preserving.

### `WALL_ANGLE_TOL_DEG` — replacing `VERT_THR` and `HORI_THR`

Detected lines are sorted into two groups, "runs one way" and "runs the other
way", so that each group can be merged separately. The sorting used two numbers,
`VERT_THR = 10` and `HORI_THR = 0.1`.

Those are not two settings. They are **one tolerance written twice**.

A line's steepness can be described as how far it rises for each step sideways.
A line running steeply rises a lot per sideways step — 10 up for 1 across. A
line running flat rises very little — 0.1 up for 1 across. And 0.1 is exactly
1 ÷ 10. Both numbers say the same thing: *within about 5.71° of square*.

Kept as two independent parameters, nothing stops someone setting 10 and 0.2,
which describes two different tolerances at once and corresponds to no angle at
all. `WALL_ANGLE_TOL_DEG` states the angle and derives both:

```
slope_thresholds(5.71 deg)  ->  VERT 10.001,  HORI 0.1000
```

which reproduces the shipped pair to three decimals — so this is a change of
*expression*, not of behaviour. `VERT_THR` / `HORI_THR` still win if set
explicitly.

### `BUFFER_THR` and `WALL_THICKNESS_M` — giving lines a thickness

The model returns **lines**, which are infinitely thin and enclose no area. A
wall has thickness. `BUFFER_THR` is how far to spread each line sideways to turn
it into a wall-shaped rectangle. It is measured in **image pixels**.

That unit is the problem. A pixel is not a physical size — how many feet of
building one pixel covers depends on how big the building is and what resolution
the image was drawn at. A fixed `BUFFER_THR` therefore means a **different
physical wall thickness on every project**, and a different one again if
`RESIZE_WIDTH` changes.

`"auto"` works backwards from a real measurement instead. You supply the wall
thickness in metres; the code works out how many pixels represent one foot in
*this* image, and converts:

```
wall thickness            0.92 units  (11 in, measured on laramie)
building extent         165    units
image width            1200    pixels
                       ------------------------------------------
pixels per unit         1200 / 165        =  7.27 px/unit
buffer = half thickness  0.92 / 2 * 7.27  =  3.33 px
```

Half the thickness, because the buffer grows outward in both directions from the
line at the centre of the wall. Laramie's two storeys came out at **3.33** and
**3.27** pixels — different, correctly, because the two storeys have slightly
different extents.

For comparison, the shipped `BUFFER_THR = 2` at `RESIZE_WIDTH = 400` works out
to **1.65 ft** of wall against an actual 0.92 ft — each wall polygon was about
80% wider than the wall, sweeping in points from the rooms on either side.

### `SCORE_THR` — how confident the model must be

The model always returns exactly 1000 candidate segments, each with a
confidence score, and `SCORE_THR = 0.55` keeps those it is reasonably sure
about. Unlike everything above, this is a property of the trained checkpoint
rather than of the building, so it is **left fixed**. There is nothing in the
point cloud to derive it from.

It is worth watching, though, because it is the usual reason for an empty
result. The stage logs it:

```
line segmentation: 71 of 1000 segments above SCORE_THR=0.55
```

Zero surviving segments is a legitimate outcome and no longer a crash — the
original code reshaped the empty result in a way that raised an exception and
killed the whole run.

### Inference must run under `no_grad`

Not a tuning parameter, but it belongs here. The model call was not wrapped in
`torch.no_grad()`. Without it PyTorch keeps every intermediate result in case
you later want to train on them — which never happens here — and the wall stage
runs the model once per storey in a loop, so one storey's leftovers are still
held while the next runs. The attention step alone asks for a single ~4.25 GB
block. It is now wrapped, which changes nothing about the output and everything
about the memory.

Related, and worth knowing before you debug a crash that isn't there: running
two of these at once on the same machine can exhaust memory even with 63 GB
installed.

### Result on laramie

```
level 0:  32 walls   height min 10.02 / median 11.42 / max 11.64
level 1:  42 walls   height min  8.07 / median  8.71 / max 11.63
```

The **heights** are the useful check here, and unlike the counts they can be
verified. Level 0's slab runs from the first floor's top (2.60) to the second
floor's bottom (14.25) — a span of 11.65 — and the walls have a median height of
11.42 with a maximum of 11.64. They span essentially the whole storey rather
than being short fragments. Level 1 sits above 16.21 with its ceiling near 24.6,
a span of 8.4, against a median of 8.71. Both levels agree with the geometry
they were cut from.

The **counts do not demonstrate anything**, and it is worth being clear why.
Going from the shipped settings to these took laramie from 38 walls to 74, but
at least two changes push that number up for unrelated reasons: more lines are
detected (132 → 193), *and* the narrower buffer merges fewer neighbouring walls
into single polygons. A wall breaking into three pieces also reads as "+2".

### Output coordinates: why `pixel_to_xy` no longer uses the grid

Wall polygons exist in image space and have to be converted back to world
coordinates. That conversion **used to route through the `PROJECTED_BINS`
grid**: work out which grid square a pixel fell in, return the centre of that
square. Every output vertex was rounded to the nearest grid centre — like
writing down every measurement to the nearest half foot.

At `PROJECTED_BINS = 300` across laramie's 165-unit extent one square is 0.55
units, against a real wall thickness of 0.92 (1.67 squares — between two
representable values). Measured on the output, **every** wall width was an exact
multiple of the step: valid polygons had a median minimum width of exactly 2.00
steps, self-intersecting ones exactly 4.00, nothing in between. An 11-inch wall
was emitted as 1.10 units, 20% too thick. `BUFFER_THR` was being derived to two
decimals in pixel space and then rounded onto a grid three times coarser than
the quantity it encoded.

Nothing required that. `project_points_to_floor` renders the histogram with
`extent=[x_edges[0], x_edges[-1], ...]`, so the image already spans the data
linearly, at a resolution finer than the bin grid. The map is now continuous:

```python
x = x_min + (px + 0.5) / img_width  * (x_max - x_min)
y = y_min + (py + 0.5) / img_height * (y_max - y_min)
```

`+ 0.5` because a pixel is a cell, not a sample point — its centre sits half a
pixel in. This separates two concerns that were tangled: **`PROJECTED_BINS` sets
the detection resolution, this sets the output resolution.**

Measured on laramie, detection completely unchanged (same Otsu thresholds, same
buffers, same 71/160 line counts, same 74 walls):

| | before | after |
|---|---|---|
| self-intersecting footprints | **17 of 74** | **0 of 74** |
| widths on an exact grid multiple | essentially all | 2 of 74 |
| distinct wall widths | heavily collapsed | 43 of 74 |
| narrowest wall emitted | 1.10 units | **0.915 units** |

0.915 is `WALL_THICKNESS_M = 0.279` expressed in feet — the value `BUFFER_THR`
was derived to produce. The buffer was right all along; the last step was
throwing it away.

The self-intersections are the more interesting result. Two explanations were
available: polygons too thin for the grid collapsing when vertices snapped
together, or concave L and T junctions whose edges cross once snapped. The
first was ruled out by measurement — the invalid rings were *wider* than the
valid ones, not thinner — and removing the snap eliminated all 17, confirming
the second.

Note what this does **not** fix. Detection still happens on a 300-bin image, so
`PROJECTED_BINS` remains the parameter with the largest effect and the least
evidence behind it. It only means the detector's precision now survives to the
output instead of being rounded away.

### Reading `wall_projection_N.png`

Each detected wall footprint is drawn over the projected density image, filled
in its own colour at partial opacity with an outline, so the underlying points
stay visible. This is the diagnostic that counts cannot replace:

- a polygon sitting over **dark, empty pixels** is a false wall;
- **colour changes along one unbroken line** are one wall split into fragments;
- a bright line with **no polygon on it** is a wall that was missed.

On laramie, level 1 reads as a coherent floor plan — perimeter, two corridors,
cross partitions. Level 0 is much sparser through the middle of the building,
with detections concentrated around the perimeter. Whether that reflects an open
plan or a recall failure has not been determined.

---

## Parameter reference

Distances marked **(m)** are metres and converted internally by
`POINT_CLOUD_TO_POST_PROCESSING_SCALE` (3.2808 for feet-based CSVs), so the same
configuration works on a metric cloud.

### Axis alignment

| parameter | default | meaning |
|---|---|---|
| `SURVEY_BASIS` | — | 3×3 rotation, applied as `xyz @ SURVEY_BASIS` |
| `SURVEY_BASIS_SOURCE` *(test.py)* | `"e57"` | or `"csv"`; a project's `"basis"` wins over both |

`survey_basis_from_e57(path, stride=60, voxel=0.05, slab=(0.15, 0.80))` and
`survey_basis_from_df(df, long_axis="x")` take their own knobs; the defaults
held on all three projects and none needed tuning.

### Level detection

| parameter | default | meaning |
|---|---|---|
| `CLUSTER_METHOD_F/C` | `"histogram"` | or `"dbscan"` |
| `MIN_LEVEL_GAP_M` **(m)** | 0.5 | histogram void that separates two levels |
| `MIN_LEVEL_POINTS_FRAC` | 0.02 | minimum share of points for a level |
| `HIST_BIN_M` **(m)** | 0.08 | histogram bin width |
| `HIST_MIN_FRAC` | 0.02 | occupancy threshold, fraction of tallest bin |
| `MIN_STOREY_SEP_M` **(m)** | 2.4 | noise radius (half this from every level) |
| `DOMINANT_BAND_ONLY_F/C` | `True` | keep only the densest run per level |
| `EPS_F/C`, `MIN_SAMPLES_F/C` | 0.5, 10 | DBSCAN only |

### Plane fitting

| parameter | default | meaning |
|---|---|---|
| `PLANE_METHOD_F/C` | `"irls"` | or `"ransac"` |
| `DIS_THR_F/C` | `"auto"` | inlier threshold, or a number to pin it |
| `AUTO_DIS_THR_MIN_M` **(m)** | 0.02 | floor on the derived threshold |
| `AUTO_DIS_THR_MAX_M` **(m)** | 0.5 | ceiling — clamps non-planar bands |
| `RANSAC_N_F/C`, `NUM_ITER_F/C` | 3, 1000 | RANSAC only |

### Boundary

| parameter | default | meaning |
|---|---|---|
| `BOUNDARY_METHOD_F/C` | `"raster"` | or `"alphashape"` |
| `BOUNDARY_CELL_F/C` | `"auto"` | grid resolution → point spacing |
| `BOUNDARY_FILL_GAP_F/C` | `"auto"` | closing radius → 2 × cell |
| `BOUNDARY_SIMPLIFY_F/C` | `"auto"` | absolute tolerance → 1.5 × cell |
| `BOUNDARY_CONNECT_C` | `True` | grow fill_gap until one region |
| `BOUNDARY_CONNECT_F` | `False` | floors may have real openings |
| `ALPHA_F/C` | 0.15 | alphashape only |
| `BOUNDARY_SIMPLIFY_EPS_FRAC_F/C` | 0.02 | legacy; ignored when `SIMPLIFY` set |

### Walls

| parameter | default | meaning |
|---|---|---|
| `PROJECTED_BINS` | 300 | grid squares across the top-down image (detection resolution only) |
| `RESIZE_WIDTH` | 1200 | working resolution — **not** the model input size |
| `INT_THR` | `"auto"` | Otsu; or a fixed brightness cutoff |
| `WALL_MORPH_KERNEL` | 3 | closing kernel, px; 1 is a no-op |
| `WALL_ANGLE_TOL_DEG` | 5.71 | tolerance for "square"; derives both slope thresholds |
| `BUFFER_THR` | `"auto"` | line half-thickness in px; from `WALL_THICKNESS_M` |
| `WALL_THICKNESS_M` **(m)** | 0.279 | physical wall thickness (11 in, laramie) |
| `SCORE_THR` | 0.55 | model confidence cutoff; a checkpoint property |
| `VERT_THR`, `HORI_THR` | 10, 0.1 | legacy; win over `WALL_ANGLE_TOL_DEG` if set |

`test_size = 1100` is hard-coded in `img_process_model_input` and is the only
thing that sets the model's input resolution. Memory use scales with its
*square*, so it is the lever to reach for if inference will not fit — at the
cost of running the checkpoint away from the resolution it was trained at.

### Boundary point source (optional)

| parameter | default | meaning |
|---|---|---|
| `BOUNDARY_SOURCE_F/C` | `"csv"` | `"e57"` re-selects points from the raw scan |
| `BOUNDARY_E57_PATH` | — | path to the scan |
| `BOUNDARY_E57_VOXEL` | 0.3 | downsample target, one point per raster cell |
| `BOUNDARY_E57_MARGIN` | 2.0 | clip to the CSV footprint + this |

Not enabled by default. The CSV holds ~0.1% of the e57's points on one ceiling
band, and the e57 fills **89.1%** of that band's void cells — so the holes are a
downsampling artifact, not occlusion. Selecting by distance to the fitted plane
needs no labels. But the frame conversion
(`(e57_metres − e57_min) × scale`, then rotated) is **inferred from matching
extents, not recorded** — a wrong offset shifts the boundary with no obvious
symptom.

---

## Parameter provenance

The reference above says what each parameter *means*. This says **how we know
the value is right** — which are computed from the data, which are physical
quantities that port between buildings, which are mathematical constants that
were never free, and which are still just judgement.

How each one is set:

| | |
|---|---|
| **derived** | computed from this cloud at runtime; no value to pick |
| **physical** | a real-world measurement, so it ports across buildings |
| **constant** | mathematical or statistical; not a tuning knob at all |
| **switch** | choice between algorithms |
| **assumed** | set by judgement, with no measurement behind the specific value |
| **legacy** | superseded and ignored |

Confidence in the **evidence** column: *3 projects* = laramie, corteva and ty;
*1 project* = laramie only; *reasoned* = argued from geometry, not measured;
*none* = no evidence either way.

### Axis alignment

| parameter | set by | from | evidence |
|---|---|---|---|
| `SURVEY_BASIS` | derived | wall-normal azimuths, folded mod 90° | **3 projects.** 0.028° / 0.008° from the registered basis (laramie, corteva); ty is circular. Footprint shrinks 29.9% / ~0% / 27.3% |
| radius base = 0.5% of diagonal | derived | cloud diagonal | **3 projects.** Makes it unit-agnostic; spread across 3 radii 0.007–0.099° |
| `horiz_tol` = 0.15 | assumed | — | *none.* ~8.6° from vertical; held on all three but never varied |
| `voxel` = 0.05 m (e57) | reasoned | — | *reasoned.* Equalises density so area, not scanner proximity, decides the vote |
| `slab` = 15–80% of height | reasoned | z percentiles | *reasoned.* Drops floors/ceilings, whose normals carry no yaw |
| `stride` = 60 (e57) | assumed | — | *none.* Ty's 0.099° spread suggests point count matters; not swept |
| `long_axis` = "x" | switch | — | **deterministic.** Resolves the 90° ambiguity by rule instead of by chance |

### Level detection

| parameter | set by | from | evidence |
|---|---|---|---|
| `CLUSTER_METHOD` | switch | — | **3 projects.** Histogram ≥ DBSCAN on every dataset (0.677/0.992/0.953 vs 0.671/0.949/0.947), 8–16 ms vs minutes |
| `MIN_LEVEL_GAP_M` = 0.5 | physical | habitable floor-to-floor | **3 projects.** Level count holds over 0.2–2.5 m (**12.5×**). Bounds are physically independent: 0.16 m internal density dip below, 1.42 m worst-case survivor above |
| `MIN_STOREY_SEP_M` = 2.4 | physical | floor-to-floor height | **3 projects**, but only as the *noise radius* now — it stopped being the split criterion, which is why it no longer collapses on tight buildings |
| `DOMINANT_BAND_ONLY` | switch | — | **1 project.** Cross-checked against the Ceiling label: 21.3% contamination where it trims, 1.2% at the real floor. No-op on clean levels (282 of 102,761) |
| `HIST_BIN_M` = 0.08 | assumed | — | *none.* 0.26 ft bars; fine enough to resolve every gap seen, never swept |
| `HIST_MIN_FRAC` = 0.02 | assumed | — | *none.* Occupancy floor as a fraction of the tallest bin |
| `MIN_LEVEL_POINTS_FRAC` = 0.02 | assumed | — | *none, and a known risk.* This is what drops a faint-but-real level — the suspected cause of corteva's missing lower ceiling |
| `EPS`, `MIN_SAMPLES` | legacy | — | DBSCAN only. **Measured to be unfixable**: on laramie no `eps` both chains a level and refuses to bridge levels (window empty) |

### Plane fitting

| parameter | set by | from | evidence |
|---|---|---|---|
| `PLANE_METHOD` = irls | switch | — | **3 projects.** Reproducible 1.00× over 5 repeats against RANSAC's 16.32× on a tiered band |
| `DIS_THR` = auto | derived | the fit's own robust scale (4.685 σ) | **3 projects.** Scale spans 5× across surfaces (0.060–0.288), so no fixed value covers them |
| `1.4826` | constant | 1/Φ⁻¹(0.75) | **exact.** Makes MAD an estimator of σ. Never a choice |
| `4.685` | constant | 95% efficiency point | **exact.** Standard Tukey tuning constant |
| `iterations`=20, `tol`=1e-9 | constant | convergence | **1 project.** Six different initialisations converge to the same plane |
| `AUTO_DIS_THR_MIN_M` = 0.02 | assumed | — | *none.* Never the binding constraint on these surfaces |
| `AUTO_DIS_THR_MAX_M` = 0.6 | assumed | — | *weak.* Only 1.38× headroom at 0.5, hence 0.6. **The clamp firing is itself the warning** that the band is not one plane |
| `RANSAC_N`, `NUM_ITER` | legacy | — | Ignored under IRLS. `NUM_ITER` is only an upper bound anyway: 10 → 100,000 changes results not at all |

### Boundary

| parameter | set by | from | evidence |
|---|---|---|---|
| `BOUNDARY_METHOD` = raster | switch | — | **2 projects.** Equal or better area, ~0.0 s vs 13–34 s, and no density dependence |
| `BOUNDARY_CELL` = auto | derived | median distance to 6th XY neighbour | **1 project, strongly.** At cell 0.25 vs 0.87 spacing: 29.8% occupancy and **4,644 blobs**; at cell ≈ spacing, 71.7% and 4 |
| `BOUNDARY_SIMPLIFY` = auto | derived | 1.5 × cell, **absolute** | **2 projects.** Both agree at ≈1.3 × cell once normalised. The legacy 0.02 fraction was a 10.4 ft tolerance that flattened a floor to 4 vertices |
| `BOUNDARY_FILL_GAP` = auto | derived | 2 × cell | **1 project.** Hard floor below, no effect above — *but overridden entirely when `CONNECT` is on* |
| `BOUNDARY_CONNECT_C` = True | switch | — | **1 project, and it is wrong half the time.** Validated against held-out e57 evidence and floor area: correct on the lower ceiling (0.99× floor, 87% backed), **unjustified on the upper** (1.33× floor, 8.9% backed) |
| `BOUNDARY_CONNECT_F` = False | reasoned | — | *reasoned.* A floor can have a real courtyard; ty has ~2,064 sq ft of voids we could not classify |
| `connect_max_cells` = 30 | assumed | — | *none.* A runaway bound, not a tuned value |
| `ALPHA` = 0.15 | legacy | — | Alphashape only, and **measured dangerous**: α=1.0 fragmented ty into 42 pieces, silently discarding ~5,900 points |
| `BOUNDARY_SIMPLIFY_EPS_FRAC` | legacy | — | Superseded; scales with building size for no geometric reason |
| `POINT_CLOUD_TO_POST_PROCESSING_SCALE` | constant | ft per metre | **exact** (3.280839895…). A unit conversion, not a parameter |
| `BOUNDARY_E57_VOXEL` = 0.3 | reasoned | raster cell size | *reasoned.* One point per cell is enough |
| `BOUNDARY_E57_MARGIN` = 2.0 | assumed | — | *none.* Clips the infinite fitted plane to the CSV footprint |

### Walls

| parameter | set by | from | evidence |
|---|---|---|---|
| `PROJECTED_BINS` = 300 | **assumed** | — | ***none — the weakest value in the pipeline.*** Largest measured effect of any wall parameter (100 bins → 8× the detections) and no evidence for either value. Counts cannot separate recovery from fragmentation; needs the projection images compared visually |
| `RESIZE_WIDTH` = 1200 | derived | detection sweep | **1 project.** 38 → 52 lines from 400 → 1200, flattening after. **Not** the model's input size — `test_size=1100` is, and it is fixed |
| `INT_THR` = auto | derived | Otsu on the image's own histogram | **1 project.** Picked 34 and 33; parameter-free by construction, so nothing to port |
| `BUFFER_THR` = auto | derived | `WALL_THICKNESS_M` → pixels | **1 project, now end-to-end.** Derived 3.33 px; the narrowest wall in the output measures **0.915 units = exactly the input thickness** |
| `WALL_THICKNESS_M` = 0.279 | physical | tape-measured, 11 in | **1 project.** Real for laramie; unmeasured elsewhere, and it is a per-building quantity |
| `WALL_ANGLE_TOL_DEG` = 5.71 | constant | reparameterisation | **exact.** Reproduces the shipped 10 / 0.1 to three decimals — those were one tolerance written twice (0.1 = 1/10). Whether 5.71° is the *right* tolerance is untested |
| `WALL_MORPH_KERNEL` = 3 | assumed | — | *none for 3* — but the old value of **1 was provably a no-op**, so the previous behaviour was a bug, not a baseline |
| `SCORE_THR` = 0.55 | constant | the trained checkpoint | **fixed.** A model property; nothing in the cloud to derive it from |
| `test_size` = 1100 | constant | training resolution | **fixed**, hard-coded. Memory scales with its *square* — the lever if inference will not fit |
| `pixel_to_xy` mapping | **derived** | continuous affine from the render extent | **1 project.** Replacing the bin-centre snap took self-intersecting footprints **17 → 0** and wall widths from grid multiples to 43 distinct values |
| `VERT_THR`, `HORI_THR` | legacy | — | Superseded by `WALL_ANGLE_TOL_DEG`; still win if set |

### Reading this table

**Nine parameters are genuinely derived** and have no value to choose:
`SURVEY_BASIS`, the yaw radii, `DIS_THR`, `BOUNDARY_CELL`, `BOUNDARY_SIMPLIFY`,
`BOUNDARY_FILL_GAP`, `INT_THR`, `BUFFER_THR`, and the `pixel_to_xy` mapping.
Each replaced a hand-set number, and in five cases the hand-set number was
measurably wrong on at least one building.

**Six are constants that were never free** — `1.4826`, `4.685`, the unit scale,
`WALL_ANGLE_TOL_DEG`'s slope pair, `SCORE_THR`, `test_size`. Presenting these as
tunable was itself the defect: two of them (`VERT_THR`/`HORI_THR`) could be set
to a combination corresponding to no angle at all.

**Three physical quantities port between buildings** — `MIN_LEVEL_GAP_M`,
`MIN_STOREY_SEP_M`, `WALL_THICKNESS_M` — though the last is per-building and has
only been measured once.

**Two things are known to be wrong and are not yet fixed.**
`BOUNDARY_CONNECT_C` is right on one of laramie's two ceilings and inflates the
other by a third with 8.9% evidential support. `PROJECTED_BINS` has the largest
effect of any wall parameter and no evidence behind either candidate value.

**One is a known-risk assumption**: `MIN_LEVEL_POINTS_FRAC` is the most likely
cause of corteva's missing lower ceiling, because it drops a level for being
faint rather than for being wrong.

Everything in the *assumed* rows is a place where the value happened to work on
the buildings tested and would not announce itself if it stopped working.

---

## Diagnostics

The package logs to `logging.getLogger("Infer")` with `propagate = False` and an
in-memory handler destined for blob storage, so **nothing reaches the console by
default**. [`test.py`](test.py) attaches a stream handler; do the same to see:

```
[survey_basis] radius  0.296: yaw  76.242 deg   strength 0.796   (328,302 wall normals)
[survey_basis] consensus yaw 76.242 deg (spread 0.020 deg, mean strength 0.814)
[survey_basis] footprint 16,499 -> 11,571 sq units (29.9% tighter)
[pp] level 1: kept densest band z 14.31..16.91, dropped 70800 of 165130 points
[pp] floor clustering: histogram, 2 level(s) at z=[1.21, 14.75]
[pp] floor IRLS plane: tilt 0.04 deg, scale 0.254, DIS_THR 1.190, 83598/102479 inliers (81.6%)
[pp] boundary auto: spacing 0.838 -> cell 0.838, fill_gap 1.676, simplify 1.257
[pp] Raster boundary: 2 contours, kept largest (10040 sq units), dropped 1 totalling 9
[pp] Raster boundary: grew fill_gap 0.82 -> 4.14 (10 x cell) to reach 1 region(s)
[pp] auto DIS_THR: knee 2.115 clamped to 1.640 -- the band may not be a single plane
```

`[survey_basis]` lines print to stdout directly; everything tagged `[pp]` goes
through the package logger and needs the stream handler.

Four of these are **warnings in disguise**:

- *"clamped ... may not be a single plane"* — the surface is not planar. The
  threshold it produces is a cap, not a measurement.
- *"dropped N totalling X sq units"* — silent data loss. Small is fine; large
  means the surface is genuinely disconnected.
- *"footprint barely changed — already axis-aligned"* — correct for a square
  building, but also what you see when the basis came from a **different
  building** than the data being processed.
- *spread > 2° or strength < 0.4* in the survey basis — the walls do not agree
  on one grid, so the yaw is not meaningful.

With `LOCAL_OUTPUT_DIR` set and no blob configured, each stage writes to that
directory:

| file | shows |
|---|---|
| `{type}_zhistogram.png` | height histogram, detected levels, slice boundaries, kept band |
| `{type}_cluster.html` | input points in scan colours |
| `{type}_dbscan.html` | clustering result, one colour per cluster, grey noise |
| `{type}_ransac_N.html` | plane inliers vs outliers |
| `{type}_planefit_N.html` | inliers with the oriented bbox |
| `{type}_boundary_N.png` | outline over the points |
| `{type}_boundary_mask_N.png` | the occupancy grid after closing |
| `{type}_edgepoints_N.png` | simplified corner points |
| `wall_projection_N.png` | detected wall footprints over the density image |
| `wall_bbox_N.html` | wall point clusters with their bounding boxes |
| `wall_output.json` | the stage's full output |

The `_zhistogram` and `_boundary_mask` images are the two worth checking first:
between them they explain almost every surprising result. For walls it is
`wall_projection_N.png`, for the reason given in
[§4](#reading-wall_projection_npng) — the wall counts cannot distinguish
recovery from fragmentation and that image can.

Wall snapshots previously required a blob location to be configured and were
written nowhere at all under `LOCAL_OUTPUT_DIR`, including `wall_output.json`.

---

## Known limitations

**A level may need more than one plane.** Laramie's upper deck is tiered — four
surfaces spanning 4.4 ft — and one plane fits it at only 0.551 inlier ratio.
Splitting it into 4 planes cuts the median residual 0.719 → 0.380, but the
output schema has one surface per floor, so the pipeline cannot express it.
[`dbscan_experiment/multiplane.py`](../dbscan_experiment/multiplane.py) is a
working prototype of the two-tier model (levels → planes within a level).

**Pitched roofs are not handled.** Corteva's ceiling spans 9.26 ft with modes at
~10.3 and ~17.25 — eaves and ridge. There is no void between them, so it is
correctly *one level*, but one horizontal plane fits it at 55.5% inliers with a
1.10° tilt. The dominant-band filter is also wrong here: it would keep the eaves
and discard ~78% of the roof.

**Interior holes cannot be represented.** `RETR_EXTERNAL` spans any enclosed
void, and `edgePoints` is a flat ring, so a floor with a courtyard has no
representation regardless of how well it is detected. Ty's floor gains 10.8%
area when its holes are filled; laramie's, 0.72%.

**Segmentation quality bounds everything.** Laramie's Ceiling label fits a plane
at 0.712 median residual while *floor*-labelled points at the same height fit at
0.023 — the Ceiling class kept the worse points. No fitting strategy recovers
from that.

**The survey basis has one genuinely independent check per building, and we have
two.** Laramie and corteva compare against registered matrices; ty's "registered"
value came from this same estimator, so it validates reproducibility only. A
third building with a surveyed basis would be the most useful thing to add.

**`BOUNDARY_CONNECT_C` over-grows for two different reasons, and mishandles
both.** Measured on laramie's ceilings, reproducing the pipeline's own clusters:

*Upper ceiling* — 7 components, the largest holding 99.77% of occupied cells and
99.76% of points; the other six total 0.24% of points and one of them is a
single cell. Point coverage is **complete at 4 × cell**, but the `components ==
1` rule grows to 15 × cell to absorb those specks, inflating area 1.01 → **1.29**
for zero additional evidence. A coverage target plus an area-inflation cap would
stop at 4×.

*Lower ceiling* — 1,356 components; the largest holds 38.6% of points and the
second **35.1%**, 280 cells away. Component count suggests this is two surfaces
being welded together. **It is not** — see the validation below. Growing to 10 ×
cell is the right answer here, and stopping early would badly under-cover it.

Component count cannot tell those two situations apart. Evidence can.

### Validating `fill_gap`

"Reached 1 region" is a stopping condition, not a validation. Closing *asserts*
ceiling in cells where nothing was observed, so checking it means finding
evidence the closing did not itself use. Two sources exist:

**Held-out evidence.** The CSV is a ~0.1% downsample of the e57. For every cell
the closing invented, ask the raw scan whether ceiling is actually there at that
elevation. The closing never saw that data, so agreement is a real test.

**Floor agreement.** A storey's ceiling should span roughly its floor. The floor
is fitted from different points by the same code, so its boundary is an
independent expectation of the extent. A ceiling much larger than the floor
beneath it is geometrically suspect.

Measured on laramie, the two agree with each other on both ceilings:

| | radius | comps | area | vs floor | cells added | e57 backs |
|---|---|---|---|---|---|---|
| **lower** z 12.72 | 2 × | 219 | 5,327 | 0.53× | 21,760 | 87.0% |
| (floor 10,035) | **10 ×** | **1** | **9,914** | **0.99×** | 48,334 | **87.3%** |
| | 20 × | 1 | 10,620 | 1.06× | 52,608 | 81.1% |
| **upper** z 24.55 | **2 ×** | 5 | **9,543** | **1.02×** | 169 | **56.8%** |
| (floor 9,312) | 4 × | 2 | 9,743 | 1.05× | 469 | 35.8% |
| | **15 ×** | **1** | **12,412** | **1.33×** | 4,654 | **8.9%** |

The lower ceiling's growth is **justified**: it lands within 1% of the floor
beneath it and 87% of the invented cells contain real scan points. Its two large
components are the two well-sampled ends of one sparsely-scanned ceiling, not
two surfaces.

The upper ceiling's growth is **not**: it manufactures 4,654 cells of which only
8.9% have any support, ending at a roof a third larger than its own floor.

So the discriminator is not component count or area inflation, it is **whether
the marginal cells have evidence**. A stopping rule follows: grow while marginal
e57 support stays high (~50%) *and* area stays within ~1.1 × the paired floor.
That permits the lower ceiling's 10 × and stops the upper at 2–3 ×.

Caveat: the e57 frame conversion is inferred from matching extents, not recorded
— but an 87% vs 8.9% gap is far too large for a modest offset error to flip, and
the floor check does not involve the e57 at all and agrees.

**Coverage of this work.** Floors and ceilings were tested end-to-end on
laramie, corteva and ty. Walls have been run on **laramie only** — every wall
number in this document comes from one building, and several of them are counts
that cannot distinguish a real improvement from fragmentation. Openings were not
re-tested and consume `floor_bboxz`, which these changes move.

---

## Scripts

| script | purpose |
|---|---|
| [`test.py`](test.py) | run floors + ceilings + walls on one project with the settings above (`RUN_WALLS` toggles the slow stage) |
| [`check_survey_basis.py`](check_survey_basis.py) | registered vs CSV vs e57 basis, all projects |
| [`../survey_basis.py`](../survey_basis.py) | `survey_basis_from_df` / `survey_basis_from_e57` |
| [`../dbscan_experiment/`](../dbscan_experiment/) | the studies behind every number quoted here |

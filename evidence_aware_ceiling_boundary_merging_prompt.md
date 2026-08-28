# Implementation Request: Evidence-Aware Ceiling Boundary Merging

## Objective

Please replace the current global `BOUNDARY_CONNECT_C` growth loop in `floor_ceiling_util.py` with an **evidence-aware component-merging algorithm**.

The intended behavior is:

1. Enclosed holes inside one connected ceiling component should not trigger iterative growth.
2. Detached external components should be evaluated individually.
3. A detached component should be merged only when the additional polygon area created by the merge is reasonably supported by the original occupancy evidence.
4. Tiny distant specks must not force the entire ceiling boundary to expand.
5. Significant but genuinely disconnected ceiling regions must not be connected by an artificial bridge merely to satisfy `contours == 1`.

---

## 1. Enclosed Holes Need No Iterative Merge

Consider this occupancy mask:

```text
###########
##.......##
##.......##
###########
```

This is already one connected component with one exterior boundary.

Because the final output only requires the outer `edgePoints`, the correct path is:

```text
connected component
-> findContours(RETR_EXTERNAL)
-> one exterior contour
-> simplify
-> output outer edgePoints
```

`RETR_EXTERNAL` already ignores the enclosed void when extracting the exterior contour. Therefore, do not enlarge `fill_gap` merely to fill enclosed holes.

A small fixed closing operation may still be retained to repair local raster dropouts, but it should not grow iteratively just to remove interior holes.

---

## 2. Detached External Components Must Be Evaluated Individually

Consider:

```text
##########       #       #
##########

       #               #
```

The large component is the main ceiling. The isolated components may be:

- segmentation noise;
- scanner artifacts;
- small detached patches of real ceiling;
- a second legitimate ceiling region.

Remove the current strategy:

```text
while number_of_contours > 1:
    increase closing radius
```

Instead, evaluate each detached component as an individual merge candidate.

The central question should be:

> Does this candidate contribute enough original occupancy evidence to justify the additional polygon area required to connect it to the main ceiling?

---

## 3. Use Occupied-Cell Area as the Main Evidence Metric

Raw point count is affected by scanner distance, scan angle, scan overlap, downsampling, and segmentation density. Therefore, use the original binary occupancy grid as the primary evidence source.

Define:

```text
occupied support area
=
number of original occupied cells * cell^2
```

Because all ratios use the same `cell^2`, the implementation can use occupied-cell counts directly.

Raw point count may still be logged as a secondary diagnostic, but it should not be the main merge criterion.

All support metrics must be measured from the occupancy grid **before candidate-specific morphology**. Cells created by closing are claimed geometry, not original evidence.

---

## 4. Maintain Three Distinct Mask Types

### 4.1 Original occupancy mask

Let `O` be the original rasterized plane-inlier occupancy mask before morphology.

This is the evidence mask.

### 4.2 Filled exterior masks

For each connected component, extract its exterior contour with `RETR_EXTERNAL` and fill that contour.

Let:

- `C` = filled exterior mask of the currently accepted main region;
- `Q` = filled exterior mask of one candidate component.

Using filled exterior masks is important because enclosed holes are intentionally ignored by the current output representation.

### 4.3 Trial merged exterior mask

Let `M` be the filled exterior mask after applying the minimum candidate-specific bridge required to connect `C` and `Q`.

Evaluate only:

```text
current accepted region + one candidate component
```

Do not close the entire occupancy mask during candidate evaluation. Otherwise, one trial may annex multiple unrelated components and the merge cost cannot be attributed to a specific candidate.

---

## 5. Measure the New Unsupported Area Created Between Components

Define cell-count areas:

```text
A_C     = area of C
A_Q     = area of Q
A_union = area of C union Q
A_M     = area of M
```

The extra area created solely to connect the components is:

```text
A_bridge = max(0, A_M - A_union)
```

This is the most important geometric cost.

Because `C` and `Q` are already filled exterior masks, enclosed holes inside either component are already included in `A_union`. Therefore, `A_bridge` measures only the new void introduced between separate components.

---

## 6. Measure Original Evidence for the Candidate

Define candidate support from the original occupancy mask:

```text
S_Q = area of O intersect Q
```

This is the number of original occupied cells contained in the candidate's exterior mask.

Also calculate:

### Candidate occupancy density

```text
D_Q = S_Q / A_Q
```

This indicates whether the candidate itself is strongly supported or is mostly empty contour area.

### Candidate support fraction

```text
F_Q = S_Q / occupied_cell_count(O)
```

This indicates how much of the total ceiling evidence belongs to the candidate.

Component size should not be the only decision rule, but it is an important diagnostic signal.

---

## 7. Define Merge Efficiency

Relative to the current accepted region, the total newly claimed output area is:

```text
A_new = A_M - A_C
```

This includes both:

- the candidate component;
- the bridge area required to reach it.

Define merge efficiency as:

```text
E_merge = S_Q / max(A_M - A_C, 1)
```

Interpretation:

- high value: much of the newly added output area has original occupancy evidence;
- low value: most of the newly added output area is unsupported bridge or void.

Example of an unreasonable merge:

```text
candidate support cells = 8
newly claimed area      = 2000 cells
E_merge                 = 8 / 2000 = 0.004
```

Only 0.4% of the newly claimed area is supported. Reject this merge.

Example of a plausible merge:

```text
candidate support cells = 350
newly claimed area      = 650 cells
E_merge                 = 350 / 650 = 0.538
```

Approximately 54% of the newly claimed area is supported. This is much more likely to be a reasonable merge.

---

## 8. Define Area Inflation

Define bridge-area inflation as:

```text
I_area = A_bridge / max(A_union, 1)
```

Equivalent form:

```text
I_area = A_M / A_union - 1
```

A large value means the merge creates a substantial output region that belongs to neither original component.

---

## 9. Support-Retention Ratio Is Optional but Redundant

It is acceptable to log:

```text
rho_before = (S_C + S_Q) / A_union
rho_after  = (S_C + S_Q) / A_M
R_support  = rho_after / rho_before
```

However, because the evidence numerator is unchanged:

```text
R_support = A_union / A_M
```

Therefore, support retention and area inflation contain essentially the same information. It is fine to log support retention because it is intuitive, but do not treat it as independent evidence.

`E_merge` is the more useful independent evidence metric.

---

## 10. Find the Minimum Candidate-Specific Connecting Radius

For each external candidate component:

1. Calculate the minimum boundary-to-boundary distance from the candidate to the current accepted region.
2. Convert it to world units:

```text
gap_world = gap_cells * cell
```

3. Use this as a lower bound for the required closing radius.
4. Test actual integer radii with the elliptical morphology kernel.

A theoretical initial estimate is:

```text
r0 = ceil(gap_cells / 2)
```

because both components expand during dilation.

The actual connection must still be verified by running closing and checking connectedness. Kernel discretization and erosion can shift the true minimum by one or more cells.

Test:

```text
r0, r0 + 1, r0 + 2, ...
```

until the two components connect or the maximum physical merge gap is exceeded.

Do not use the current sparse `*1.5` sequence:

```text
2, 3, 4, 7, 10, 15, 23
```

The selected radius should be the smallest actual integer radius that produces a connection.

Every trial must start from the same unmodified trial input. Do not cumulatively close the result of the previous radius.

---

## 11. Use Sufficient Padding for Every Trial

The trial raster or local ROI must have padding of at least:

```text
tested_radius + 1 cells
```

around both components.

Do not reuse padding calculated only from the initial `2 * cell` gap.

Otherwise, large kernels can be clipped by the array boundary, causing:

- asymmetric morphology;
- false area saturation;
- incorrect bridge-area measurements;
- incorrect minimum-radius detection.

A local ROI is acceptable as long as it includes sufficient margin for the tested radius.

---

## 12. Merge Acceptance Rule

Make the thresholds configurable.

A candidate merge may be accepted only when all relevant conditions pass:

```python
accept_merge = (
    gap_world <= max_merge_gap_world
    and merge_efficiency >= min_merge_efficiency
    and area_inflation <= max_area_inflation
    and candidate_density >= min_candidate_density
)
```

Candidate support fraction and raw point count may be used as additional diagnostics or optional noise filters.

Do not accept or reject a component using size alone.

A very small but adjacent and densely supported component may be legitimate. A larger but distant component may be a separate ceiling region and should not be bridged.

Suggested configuration parameters:

```text
BOUNDARY_CONNECT_MODE_C
BOUNDARY_MERGE_MAX_GAP_C
BOUNDARY_MERGE_MIN_EFFICIENCY_C
BOUNDARY_MERGE_MAX_AREA_INFLATION_C
BOUNDARY_COMPONENT_MIN_DENSITY_C
BOUNDARY_COMPONENT_MIN_SUPPORT_FRAC_C
```

Do not choose final universal defaults until the metric distributions have been measured on Laramie, Corteva, and TY.

A useful transition configuration is:

```text
BOUNDARY_CONNECT_MODE_C = "legacy"
BOUNDARY_CONNECT_MODE_C = "evidence"
BOUNDARY_CONNECT_MODE_C = "off"
```

This allows direct A/B testing.

---

## 13. Avoid Candidate-Order Dependence

Do not permanently rely on arbitrary contour order.

Use this iterative process:

1. Evaluate every remaining candidate against the current accepted region.
2. Calculate its minimum bridge and merge metrics.
3. Select the best acceptable candidate, for example by:
   - highest merge efficiency;
   - lowest bridge-area cost per supported cell;
   - smallest physically valid gap.
4. Merge that candidate.
5. Recompute all remaining candidates against the updated accepted region.
6. Stop when no remaining candidate passes the acceptance criteria.

A useful cost metric is:

```text
C_bridge = A_bridge / max(S_Q, 1)
```

Lower values are better.

---

## 14. Components Contained Inside the Main Exterior Polygon

A detached occupancy component may lie inside an enclosed hole of the main component:

```text
###########
##...#...##
##.......##
###########
```

The isolated `#` is disconnected in the occupancy mask, but it is already inside the main exterior polygon.

Because the output intentionally ignores inner holes, classify this component as:

```text
contained
```

It does not need to be bridged or merged.

Only candidates outside the current exterior mask require merge evaluation.

---

## 15. Significant but Distant Components

Consider:

```text
##########                ##########
##########                ##########
```

Both components may have strong ceiling evidence, but the space between them may be real exterior space.

Do not force them together.

If the output schema supports multiple ceiling polygons:

```text
return both as separate ceiling polygons
```

If the schema supports only one polygon per level:

```text
keep the main component
log the significant disconnected component
report its support fraction and area
do not invent a large bridge
```

The warning should distinguish this case from tiny noise:

```text
significant disconnected ceiling component rejected:
support_fraction=...
gap_world=...
bridge_area=...
```

Conservatively omitting a disconnected region is safer than asserting a large unsupported ceiling between two real regions.

---

## 16. Proposed Processing Flow

```text
1. Rasterize plane inliers into the original occupancy mask O.

2. Optionally apply one fixed, small base closing operation using the
   existing BOUNDARY_FILL_GAP_C. Do not grow it globally.

3. Run connected-component analysis.

4. Select the main component using original occupied-cell support,
   not exterior contour area.

5. Extract and fill the main component's RETR_EXTERNAL contour.

6. Ignore enclosed holes. They are already spanned by the exterior contour.

7. Classify secondary components as:
      a. contained inside the current exterior polygon;
      b. external merge candidates;
      c. insignificant noise;
      d. significant disconnected regions.

8. For each external merge candidate:
      a. isolate the current accepted region and this candidate only;
      b. find the minimum actual connecting radius;
      c. simulate the merge with adequate padding;
      d. calculate bridge area, merge efficiency, candidate density,
         physical gap, and area inflation;
      e. accept or reject the candidate.

9. Stop when no additional candidate passes the criteria.

10. Extract the final RETR_EXTERNAL contour.

11. Simplify using the existing absolute approxPolyDP tolerance.

12. Map simplified edge points back to world coordinates.
```

---

## 17. Suggested Pseudocode

```python
original = rasterize(plane_inlier_xy)

# Keep only a fixed local repair close.
base_mask = morphological_close(
    original,
    radius=base_fill_gap_cells,
)

components = connected_components(base_mask)

main = max(
    components,
    key=lambda c: occupied_cell_count(original, c.mask),
)

accepted_output = fill_external_contour(main.mask)
remaining = [c for c in components if c.id != main.id]

while remaining:
    acceptable_trials = []

    for candidate in remaining:
        candidate_output = fill_external_contour(candidate.mask)

        # Already covered by the current exterior output.
        if is_fully_contained(candidate_output, accepted_output):
            mark_contained(candidate)
            continue

        gap_cells = minimum_mask_distance(
            accepted_output,
            candidate_output,
        )
        gap_world = gap_cells * cell

        if gap_world > max_merge_gap_world:
            mark_rejected(candidate, reason="gap")
            continue

        trial_input = accepted_output | candidate_output

        radius, merged_output = find_minimum_connecting_radius(
            trial_input,
            start_radius=ceil(gap_cells / 2),
            max_radius=max_merge_radius_cells,
            padding_mode="radius_plus_one",
            restart_from_original_each_trial=True,
        )

        if merged_output is None:
            mark_rejected(candidate, reason="cannot_connect")
            continue

        current_area = count_cells(accepted_output)
        candidate_area = count_cells(candidate_output)
        union_area = count_cells(accepted_output | candidate_output)
        merged_area = count_cells(merged_output)

        bridge_area = max(0, merged_area - union_area)
        area_inflation = bridge_area / max(union_area, 1)

        candidate_support = count_cells(original & candidate_output)
        candidate_density = candidate_support / max(candidate_area, 1)

        newly_claimed_area = merged_area - current_area
        merge_efficiency = (
            candidate_support / max(newly_claimed_area, 1)
        )

        trial = {
            "candidate": candidate,
            "merged_output": merged_output,
            "radius": radius,
            "gap_world": gap_world,
            "bridge_area": bridge_area,
            "area_inflation": area_inflation,
            "candidate_support": candidate_support,
            "candidate_density": candidate_density,
            "merge_efficiency": merge_efficiency,
        }

        if passes_merge_rules(trial):
            acceptable_trials.append(trial)
        else:
            mark_rejected(candidate, metrics=trial)

    if not acceptable_trials:
        break

    best = select_best_trial(
        acceptable_trials,
        primary="merge_efficiency",
        secondary="bridge_area_per_support",
    )

    accepted_output = best["merged_output"]
    remaining.remove(best["candidate"])

final_contour = largest_external_contour(accepted_output)
final_contour = simplify_absolute(
    final_contour,
    tolerance=simplify_distance,
)

edge_points = pixel_to_world(final_contour)
```

---

## 18. Required Diagnostics

For every candidate, log:

```text
component ID
original occupied-cell count
raw point count
candidate exterior area
candidate occupancy density
candidate support fraction
minimum component distance
minimum distance in world units
minimum tested connecting radius
union area before merge
merged area
bridge area
area inflation
merge efficiency
accepted or rejected
rejection reason
```

Example:

```text
[pp] ceiling boundary candidate 6:
support_cells=8
support_fraction=0.0024
gap=5.32
radius=10
candidate_area=10
bridge_area=1990
area_inflation=0.198
merge_efficiency=0.004
decision=REJECT
reason=unsupported bridge area
```

Also save a diagnostic image such as:

```text
ceiling_boundary_components_N.png
```

The image should distinguish:

- main component;
- accepted components;
- rejected noise components;
- significant disconnected components;
- morphology-created bridge area.

Save the original occupancy mask and final accepted mask separately so morphology-created cells remain distinguishable from original evidence.

---

## 19. Synthetic Tests

### Test 1: Enclosed hole

```text
###########
##.......##
##.......##
###########
```

Expected:

- one exterior contour;
- no iterative growth;
- no merge candidate;
- output spans the enclosed hole.

### Test 2: Tiny distant specks

```text
##########       #       #
##########

       #               #
```

Expected:

- main component retained;
- distant specks rejected;
- no large bridge;
- final area remains close to the main exterior area.

### Test 3: Nearby legitimate fragment

```text
########   ######
########   ######
```

Expected:

- candidate passes support tests;
- minimum connecting radius is found;
- small bridge area;
- candidate accepted.

### Test 4: Large distant second region

```text
##########                ##########
##########                ##########
```

Expected:

- no artificial bridge;
- return two polygons if supported;
- otherwise retain the main polygon and issue a significant-disconnected warning.

### Test 5: Candidate inside an enclosed hole

Expected:

- candidate classified as contained;
- no bridge attempted.

### Test 6: Padding regression

Place a candidate near the raster boundary.

Expected:

- result remains identical when the canvas is enlarged;
- no false area saturation from clipped morphology.

### Test 7: Radius-search regression

Construct a gap that requires exactly 6 or 8 cells.

Expected:

- algorithm finds that exact minimum radius;
- it does not jump from 4 to 7 or from 7 to 10.

---

## 20. Real-Project Regression

Run both legacy and evidence-aware modes on Laramie, Corteva, and TY.

For every ceiling, compare:

```text
number of initial components
number of accepted components
number of rejected components
occupied-cell containment
point containment
output polygon area
area relative to the main component
bridge area
number of edge points
runtime
```

For the Laramie upper ceiling, specifically verify that:

- the six specks containing approximately 0.24% of the points do not cause thousands of unsupported hull cells to be added;
- the previous approximately 1.33x area inflation is reduced;
- real ceiling coverage remains intact.

Do not choose final thresholds using Laramie alone. First inspect whether accepted and rejected candidate metrics form a consistent separation across all available ceiling cases.

---

## Key Design Rule

> Accept a component merge only when the original evidence contributed by that component is sufficient to justify the new output area created by including it.

Enclosed holes require no merge because the exterior contour already spans them. Detached noise should remain detached and be discarded. A real nearby ceiling fragment may be merged when it adds substantial evidence with only a small unsupported bridge.

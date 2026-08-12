# DBSCAN auto-tuning experiments

Sandbox for deciding whether `EPS` / `MIN_SAMPLES` in `cluster_floor_ceiling`
can be derived from the data instead of hand-set. Nothing here is imported by
`post_process_src`, so any scheme can be tried without touching production.

```
python kdistance.py                 # where is the density knee, and is it portable?
python sweep.py                     # does clustering there actually produce good slabs?
python sweep.py --variants current physical --sample 25000
```

Both read `src/laram_inference_prediction.csv` (Laramie, 2 storeys) and
`src/00-10231-20_CortevaYorkTest_Output.csv` (Corteva, 1 storey), rotate them by
the estimated SURVEY_BASIS exactly as the stages do, and filter to `pred_label ==
Floor`. Output plots go to `out/`.

## What `eps` actually means today

`cluster_floor_ceiling` z-scores every column before clustering:

```python
feats = StandardScaler().fit_transform(np.hstack([pts, col]))   # xyz + rgb
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
```

So `eps=0.5` is **0.5 standard deviations in 6-D**, not 0.5 feet — and each
axis is divided by its own std, so one `eps` is a different physical distance
per axis and per building:

| eps=0.5 spans | Laramie | Corteva |
|---|---|---|
| x | 10.11 ft | 29.08 ft |
| z | 3.33 ft | 0.17 ft |
| anisotropy x:z | 3x | 175x |

A feet↔metres conversion changes nothing, because a uniform scaling cancels in
the z-score. Units are not why `eps` needs tuning.

## Feature variants under test

| name | features | `eps` unit |
|---|---|---|
| `current` | z-scored xyz + rgb | std devs (what ships today) |
| `xyz_zscore` | z-scored xyz | std devs |
| `isotropic` | xyz on one shared divisor | multiples of RMS radius |
| `physical` | raw xyz | feet |

## Findings

**1. The k-distance elbow answers the wrong question.** It finds the
noise/density threshold — how far apart neighbouring points are — which is
similar in both clouds (~1.2–2 ft) because both were scanned at similar
resolution. That makes `physical` look highly portable (knees 2.05 vs 1.82 ft,
1.13x). But separating *storeys* is a different question, and the sweep shows
those knee values cluster nothing useful: at eps=2 ft Laramie is 8.9% noise with
storeys still merged, and at eps≤1 ft everything is noise on both datasets.
Use `kdistance.py` for a lower bound on `eps`, not for the answer.

**2. Judge against the storey count, not a fixed number.** `lab.z_profile`
finds slabs from the z-histogram so the sweep can say *merged* vs *fragmented*
rather than just reporting cluster counts. It needs `min_storey_sep` (default
8 ft): without it, Laramie's sloped upper deck spanning z 13.9–17.6 reads as
three storeys and every clustering looks broken. With it, Laramie is 2 levels
(z 0.65 and 15.54, gap 14.89 ft) and Corteva is 1 (z 1.36).

**3. The current scheme is more defensible than it first looks.** The per-axis
anisotropy that seems wrong in the table above is doing adaptive work: Corteva's
tiny z-std makes `eps=0.5` a tight 0.17 ft in z, while Laramie's larger z-std
makes it 3.33 ft — which is what each building needs. At the shipped `EPS=0.5`
both datasets come out correct (Laramie 2 clusters, Corteva 1), and the plateau
runs 0.5–1.0 before Laramie's storeys merge at 1.5.

**4. `physical` also works, over eps 4–8 ft, and is interpretable.** Same
correct answers on both datasets. Its appeal is that the parameter means
something you can reason about and bound — "points within 4 ft are the same
slab" — whereas 0.5 std is not a quantity anyone can sanity-check. Its
weakness is that it must be re-derived if a dataset ever arrives in metres.

## Where this points for auto-tuning

`eps` is bounded on both sides by measurable quantities:

- **lower bound** — in-slab point spacing, from the k-distance curve
- **upper bound** — the smallest inter-storey gap, from `z_profile`

On Laramie those are ~2 ft and ~14.9 ft, a wide window; the shipped default sits
comfortably inside it. So the useful auto-tune is probably not "search for the
best eps" but "verify eps falls inside the measured window, and warn when it
doesn't" — much cheaper, and it catches the real failure (a building whose
storey gap is smaller than the configured eps).

## Not yet covered

- Only `Floor` was swept; `Ceiling` and `Wall` use the same `EPS` default.
- `MIN_SAMPLES` was held at 10 throughout — only `eps` was varied.
- Two buildings is a thin basis for a default. A single-storey and a two-storey
  case do not exercise the hard case: storeys closer together than the eps window.
- Cluster quality is scored by z-thickness and count, not against labelled
  ground truth, which does not exist for these datasets.

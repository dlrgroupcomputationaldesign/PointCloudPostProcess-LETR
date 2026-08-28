"""Cross-project test of the package as configured, one row per surface.

    python para_test/test_package.py                    # every project
    python para_test/test_package.py laramie conroe     # a subset

Runs floors and ceilings through the PACKAGE DEFAULTS -- nothing is set here
except the scan path and the connect mode -- and prints, per surface, the
metrics that say whether the boundary is right:

  vs floor   ceiling area / the area of the floor beneath it. Independent of the
             scan entirely, so when it and the F1 disagree that disagreement is
             the signal.
  F1/P/R     only when connect_mode is "auto": the boundary scored against
             raw-scan points on the same plane. Precision = of what was claimed,
             how much is real surface; recall = of the real surface, how much
             was captured.

A LOW absolute F1 across every candidate means the plane or the segmentation is
wrong, not that the mode choice mattered -- one ceiling scored 0.697/0.718/0.701
with recall never above 0.61 on a 70% inlier fit.
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "post_process_src"))

from post_process import post_process  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("projects", nargs="*", help="default: all in test.py's PROJECTS")
ap.add_argument("--mode", default="auto", help="connect mode for both stages")
ap.add_argument("--out", default="para_test_output",
                help="snapshots go to <out>/<project>/ -- the standard per-project "
                     "dirs, so this OVERWRITES the boundary/mask/edgepoint images "
                     "from earlier runs (survey_basis.json caches are untouched)")
args = ap.parse_args()

PROJECTS = next(
    ast.literal_eval(n.value)
    for n in ast.parse((REPO / "para_test" / "test.py").read_text()).body
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "PROJECTS"
)
names = args.projects or list(PROJECTS)

log = logging.getLogger("Infer")
log.setLevel(logging.INFO)


class Capture(logging.Handler):
    """Keep the [auto] score lines so they can be attached to their surface."""
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def ring_area(pts):
    p = [(q["x"], q["y"]) for q in pts]
    if len(p) < 3:
        return 0.0
    return 0.5 * abs(sum(p[i][0] * p[(i + 1) % len(p)][1]
                         - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p))))


def mean_z(pts):
    return sum(q["z"] for q in pts) / len(pts) if pts else float("nan")


SCORE = re.compile(r"\[auto\] (\w+)\s+F1 ([\d.]+) \(precision ([\d.]+), recall ([\d.]+)\)")
PICK = re.compile(r"\[auto\] -> (\w+)")
FELL_BACK = "needs raw-scan evidence"

rows = []
for name in names:
    spec = PROJECTS.get(name)
    if not spec:
        print(f"\n### {name}: not in PROJECTS, skipped")
        continue
    csv = REPO / spec["csv"]
    if not csv.exists():
        print(f"\n### {name}: {spec['csv']} missing, skipped")
        continue
    e57 = REPO / spec.get("e57", "")
    print(f"\n{'#' * 72}\n### {name}\n{'#' * 72}")

    df = pd.read_csv(csv, low_memory=False)
    params = {
        "LOCAL_OUTPUT_DIR": str(REPO / args.out / name),
        "BOUNDARY_CONNECT_MODE_F": args.mode,
        "BOUNDARY_CONNECT_MODE_C": args.mode,
    }
    if e57.exists():
        params["BOUNDARY_E57_PATH"] = str(e57)
    else:
        print(f"  no scan at {spec.get('e57')} -- auto will fall back")

    cap = Capture()
    log.addHandler(cap)
    try:
        fout, bboxz, _ = post_process.run_floors(df, params)
        cout, _ = post_process.run_ceilings(df, params)
    finally:
        log.removeHandler(cap)

    # Walk the log in order: each surface's scores precede its "-> mode" line.
    picks, scores, pending = [], [], []
    for line in cap.lines:
        m = SCORE.search(line)
        if m:
            pending.append((m.group(1), float(m.group(2)), float(m.group(3)),
                            float(m.group(4))))
            continue
        p = PICK.search(line)
        if p:
            picks.append(p.group(1))
            scores.append(pending)
            pending = []
        elif FELL_BACK in line:
            picks.append(line.rsplit("using ", 1)[-1].strip() + " (fallback)")
            scores.append([])
            pending = []

    floors = [(ring_area(f["edgePoints"]), mean_z(f["edgePoints"]))
              for f in fout["floors"]]
    surfaces = ([("floor", f) for f in fout["floors"]]
                + [("ceiling", c) for c in cout["ceilings"]])

    print(f"\n  {'surface':16} {'pts':>5} {'area':>10} {'vs floor':>9} "
          f"{'chosen':>18} {'F1':>6} {'P':>6} {'R':>6}")
    for i, (kind, s) in enumerate(surfaces):
        area, z = ring_area(s["edgePoints"]), mean_z(s["edgePoints"])
        below = [a for a, fz in floors if fz < z]
        ratio = area / below[-1] if kind == "ceiling" and below else float("nan")
        pick = picks[i] if i < len(picks) else "-"
        won = [x for x in (scores[i] if i < len(scores) else [])
               if x[0] == pick.split()[0]]
        f1, p, r = won[0][1:] if won else (float("nan"),) * 3
        print(f"  {kind + ' ' + str(i):16} {len(s['edgePoints']):5d} {area:10,.0f} "
              f"{ratio:9.2f} {pick:>18} {f1:6.3f} {p:6.3f} {r:6.3f}")
        rows.append((name, f"{kind} {i}", pick, f1, p, r, ratio))

ok = [r for r in rows if r[3] == r[3]]
if ok:
    print(f"\n{'=' * 72}\nsummary: {len(rows)} surfaces, {len(ok)} scored")
    from collections import Counter
    print("  modes chosen:", dict(Counter(r[2] for r in rows)))
    print(f"  mean F1 {sum(r[3] for r in ok) / len(ok):.3f}")
    poor = [r for r in ok if r[3] < 0.75]
    if poor:
        print(f"  {len(poor)} surface(s) below F1 0.75 -- suspect the plane fit, "
              f"not the connect mode:")
        for r in poor:
            print(f"      {r[0]} {r[1]}: F1 {r[3]:.3f} (P {r[4]:.3f}, R {r[5]:.3f})")

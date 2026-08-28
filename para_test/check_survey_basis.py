"""Cross-check SURVEY_BASIS: registered vs estimated from the CSV vs from the e57.

Runs both estimators on every project in test.py's PROJECTS and reports how far
each lands from the registered matrix. Two independent checks are applied:

  yaw agreement    All three should agree mod 90 deg. A 90 deg step only swaps
                   which building axis becomes X and aligns just as well, so the
                   comparison folds that out; a real disagreement shows up as a
                   difference that is NOT a multiple of 90.

  footprint        A correct basis can only SHRINK the axis-aligned bounding
                   area of the wall points. This is the check that catches a
                   transposed matrix, which is a valid rotation applied in the
                   wrong direction and therefore looks plausible on inspection --
                   it makes the footprint LARGER than doing nothing.

    python para_test/check_survey_basis.py
    python para_test/check_survey_basis.py --projects laramie ty
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from survey_basis import survey_basis_from_df, survey_basis_from_e57  # noqa: E402

WALL_LABEL = 3


def yaw_of(matrix):
    """Rotation about Z, in degrees, from SURVEY_BASIS (aligned = xyz @ B)."""
    m = np.asarray(matrix, dtype=float)
    return float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))


def yaw_delta(a, b):
    """Smallest difference between two yaws, folding out the 90 deg ambiguity."""
    return float(abs((yaw_of(a) - yaw_of(b) + 45.0) % 90.0 - 45.0))


def footprint(points_xyz, matrix):
    extent = (points_xyz @ np.asarray(matrix, dtype=float)).ptp(axis=0)
    return float(extent[0] * extent[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=60,
                    help="e57 streaming stride; lower is slower and more precise")
    args = ap.parse_args()

    # Import PROJECTS from test.py without running its pipeline: read the dict
    # literal out of the source rather than executing the module.
    import ast

    tree = ast.parse((REPO / "para_test" / "test.py").read_text())
    projects = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", None) == "PROJECTS"
    )

    names = args.projects or list(projects)
    rows = []

    for name in names:
        spec = projects[name]
        print(f"\n{'=' * 78}\n=== {name} ===")
        df = pd.read_csv(REPO / spec["csv"], low_memory=False)
        walls = df[df["pred_label"] == WALL_LABEL][["x", "y", "z"]].to_numpy(float)
        registered = spec.get("basis")

        print(f"  {len(df):,} points, {len(walls):,} wall-labelled")

        print("\n  -- from CSV wall points --")
        b_csv = survey_basis_from_df(df)

        print(f"\n  -- from e57 ({spec['e57']}) --")
        try:
            b_e57 = survey_basis_from_e57(str(REPO / spec["e57"]), stride=args.stride)
        except Exception as exc:
            print(f"     FAILED: {exc}")
            b_e57 = None

        base = footprint(walls, np.eye(3))
        print(f"\n  {'basis':12} {'yaw':>9} {'vs reg':>9} {'footprint':>12} {'vs identity':>12}")
        for label, mat in (("registered", registered), ("csv", b_csv), ("e57", b_e57)):
            if mat is None:
                continue
            area = footprint(walls, mat)
            delta = f"{yaw_delta(mat, registered):9.3f}" if registered is not None else "        -"
            print(f"  {label:12} {yaw_of(mat):9.3f} {delta} {area:12,.0f} "
                  f"{100 * (1 - area / base):11.1f}%")
        print(f"  {'identity':12} {0.0:9.3f} {'':>9} {base:12,.0f} {0.0:11.1f}%")

        if registered is not None and b_e57 is not None:
            rows.append((name, yaw_delta(b_csv, registered), yaw_delta(b_e57, registered)))

    if rows:
        print(f"\n{'=' * 78}\nyaw error vs the registered basis (deg, mod 90)")
        print(f"  {'project':12} {'csv':>9} {'e57':>9}   better")
        for name, dc, de in rows:
            print(f"  {name:12} {dc:9.3f} {de:9.3f}   {'e57' if de < dc else 'csv'}")


if __name__ == "__main__":
    main()

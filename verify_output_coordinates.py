"""Verify post-process output JSON in blob storage is in the ORIGINAL point-cloud frame.

It downloads the output JSON artifacts for a job, gathers every coordinate
(points, floor/ceiling edgePoints, wall/opening bboxes), and reports their
min/max. If you pass a reference cloud (the original .e57/.las/.npy) or explicit
bounds, it checks the coordinates fall inside that extent -- i.e. they were
converted to original coordinates rather than left in the post-process frame
(scaled feet, shifted to ~0).

Auth: pass --account-name/--container/--folder and either --account-key or set
the AZURE_STORAGE_KEY env var (don't hard-code the key).

Examples:
    python verify_output_coordinates.py --account-name pointcloudapistorage \
        --container jobs --folder postporcess_20260623_ab12cd \
        --reference "opening_detection_exp/e57/00-10231-20_CortevaYorkTest.e57"

    # if you don't have the cloud handy, give its bounds (metres):
    python verify_output_coordinates.py ... --bounds 24.7 24.3 -0.24 89.1 75.1 7.2
"""

import argparse
import json
import os

import numpy as np

DEFAULT_FILES = [
    "elements.json",
    "floors.json",
    "ceilings.json",
    "walls.json",
    "openings.json",
]


def container_client(account_name, account_key, container):
    from azure.storage.blob import BlobServiceClient

    service = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=account_key,
    )
    return service.get_container_client(container)


def download_json(client, blob_path):
    blob = client.get_blob_client(blob_path)
    if not blob.exists():
        return None
    return json.loads(blob.download_blob().readall())


def collect_geometry(output):
    """Gather every coordinate-bearing field from an output dict.

    Returns (pts3d Nx3, xy2d Mx2, zvals K):
      - pts3d : points[].location, floor/ceiling edgePoints, wall & opening bbox
      - xy2d  : wall footprint (x, y only)
      - zvals : wall zRange, levels zMode, opening bottomZ/topZ
    """
    pts3d, xy2d, zvals = [], [], []

    def add3(p):
        if isinstance(p, dict) and "x" in p and "y" in p and "z" in p:
            pts3d.append((p["x"], p["y"], p["z"]))

    for point in output.get("points", []):
        add3(point.get("location", {}))
    for key in ("floors", "ceilings"):
        for element in output.get(key, []):
            for edge in element.get("edgePoints", []):
                add3(edge)
    for wall in output.get("walls", []):
        for corner in wall.get("bbox", []):
            add3(corner)
        for fp in wall.get("footprint", []):
            if isinstance(fp, dict) and "x" in fp and "y" in fp:
                xy2d.append((fp["x"], fp["y"]))
        z_range = wall.get("zRange")
        if isinstance(z_range, dict):
            zvals += [z_range.get("min"), z_range.get("max")]
    for level in output.get("levels", []):
        if "zMode" in level:
            zvals.append(level["zMode"])
    for key in ("doors", "windows", "openings"):
        for element in output.get(key, []):
            for corner in element.get("bbox", []):
                add3(corner)
            for field in ("bottomZ", "topZ"):
                if field in element:
                    zvals.append(element[field])

    return (
        np.asarray(pts3d, dtype=float) if pts3d else np.empty((0, 3)),
        np.asarray(xy2d, dtype=float) if xy2d else np.empty((0, 2)),
        np.asarray([z for z in zvals if z is not None], dtype=float),
    )


def reference_bounds(path):
    """Min/max xyz of a reference cloud (.e57/.las/.laz/.npy)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".e57":
        import pye57

        e57 = pye57.E57(str(path))
        mins = np.array([np.inf, np.inf, np.inf])
        maxs = np.array([-np.inf, -np.inf, -np.inf])
        try:
            for i in range(e57.scan_count):
                h = e57.get_header(i)
                mins = np.minimum(mins, [h.xMinimum, h.yMinimum, h.zMinimum])
                maxs = np.maximum(maxs, [h.xMaximum, h.yMaximum, h.zMaximum])
        finally:
            e57.close()
        return mins, maxs
    if ext in (".las", ".laz"):
        import laspy

        with laspy.open(str(path)) as f:
            h = f.header
            return np.array(h.mins, dtype=float), np.array(h.maxs, dtype=float)
    if ext == ".npy":
        arr = np.load(str(path), mmap_mode="r")
        xyz = np.asarray(arr[:, :3], dtype=float)
        return xyz.min(axis=0), xyz.max(axis=0)
    raise ValueError(f"unsupported reference extension: {ext}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--account-name", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--folder", required=True, help="job_id / blob folder")
    ap.add_argument("--account-key", default=os.environ.get("AZURE_STORAGE_KEY"))
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--reference", help="original cloud (.e57/.las/.npy) for bounds")
    ap.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="original-cloud bounds (metres) if no reference file",
    )
    ap.add_argument("--tolerance", type=float, default=2.0, help="metres of slack")
    args = ap.parse_args()

    if not args.account_key:
        ap.error("provide --account-key or set AZURE_STORAGE_KEY")

    ref_min = ref_max = None
    if args.reference:
        ref_min, ref_max = reference_bounds(args.reference)
        print(f"reference bounds (m): min={np.round(ref_min,3)} max={np.round(ref_max,3)}")
    elif args.bounds:
        ref_min = np.array(args.bounds[:3]); ref_max = np.array(args.bounds[3:])
        print(f"reference bounds (m): min={ref_min} max={ref_max}")
    else:
        print("no reference given -- reporting ranges only (no pass/fail)")

    client = container_client(args.account_name, args.account_key, args.container)
    base = args.folder.strip("/")

    overall_ok = True
    for name in args.files:
        data = download_json(client, f"{base}/{name}")
        if data is None:
            print(f"\n{name}: (not found, skipped)")
            continue
        pts3d, xy2d, zvals = collect_geometry(data)
        total = len(pts3d) + len(xy2d) + len(zvals)
        if total == 0:
            print(f"\n{name}: no coordinates found")
            continue

        print(f"\n{name}: {len(pts3d)} xyz, {len(xy2d)} footprint xy, {len(zvals)} z-vals")
        if len(pts3d):
            print(f"  xyz   min={np.round(pts3d.min(0),3)}  max={np.round(pts3d.max(0),3)}")
        if len(xy2d):
            print(f"  xy    min={np.round(xy2d.min(0),3)}  max={np.round(xy2d.max(0),3)}")
        if len(zvals):
            print(f"  z     min={round(float(zvals.min()),3)}  max={round(float(zvals.max()),3)}")

        if ref_min is not None:
            tol = args.tolerance
            ok3 = (not len(pts3d)) or bool(
                np.all(pts3d >= ref_min - tol) and np.all(pts3d <= ref_max + tol)
            )
            okxy = (not len(xy2d)) or bool(
                np.all(xy2d >= ref_min[:2] - tol) and np.all(xy2d <= ref_max[:2] + tol)
            )
            okz = (not len(zvals)) or bool(
                np.all(zvals >= ref_min[2] - tol) and np.all(zvals <= ref_max[2] + tol)
            )
            inside = ok3 and okxy and okz
            verdict = "ORIGINAL frame (within reference bounds)" if inside else \
                "NOT original frame (outside reference -- still post-process?)"
            failed = [n for n, ok in (("xyz", ok3), ("footprint", okxy), ("z", okz)) if not ok]
            print(f"  -> {verdict}" + (f"  [out of bounds: {', '.join(failed)}]" if failed else ""))
            overall_ok = overall_ok and inside

    if ref_min is not None:
        print("\nRESULT:", "ALL FILES IN ORIGINAL COORDINATES" if overall_ok
              else "SOME FILES NOT IN ORIGINAL COORDINATES")
    else:
        print("\nTip: post-process frame is shifted to ~0 with feet-scale magnitudes; "
              "original frame matches the cloud's metre extent. Pass --reference to auto-check.")


if __name__ == "__main__":
    main()

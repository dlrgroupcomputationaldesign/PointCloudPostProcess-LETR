"""Plot every wall, grouped by floor (levelIndex), from a wall_output.pickle.
One HTML file is written per level. Each wall gets a distinct color and a bbox
outline. Output: plotly HTML in pickle-native coords (feet).

Output filenames are derived from the pickle path stem, e.g.
    WyomingStateFair_Laramie.pickle -> WyomingStateFair_Laramie_level0.html
                                       WyomingStateFair_Laramie_level1.html
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent

name = "00-10231-20_CortevaYorkTest"

PICKLE_PATH = f"{REPO_ROOT} / wall_log_img / wall_output_pickle / {name}.pickle"
POINT_SIZE = 1.5


def _bbox_edge_traces(bbox_pts, color):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs.extend([bbox_pts[a, 0], bbox_pts[b, 0], None])
        ys.extend([bbox_pts[a, 1], bbox_pts[b, 1], None])
        zs.extend([bbox_pts[a, 2], bbox_pts[b, 2], None])
    return go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo="skip",
    )


def main() -> None:
    pickle_path = Path(PICKLE_PATH)
    with open(pickle_path, "rb") as f:
        wall_output = pickle.load(f)

    walls = wall_output["walls"]
    levels = sorted({int(w["levelIndex"]) for w in walls})
    print(f"{pickle_path.name}: {len(walls)} walls across levels {levels}")

    # Bucket points by wall id once, reused across levels.
    points_by_wall: dict[str, list[list[float]]] = {}
    for p in wall_output["points"]:
        wid = str(p["id"])
        points_by_wall.setdefault(wid, []).append(
            [p["location"]["x"], p["location"]["y"], p["location"]["z"]]
        )

    palette = px.colors.qualitative.Light24

    for level in levels:
        same_floor = sorted(
            (w for w in walls if int(w["levelIndex"]) == level),
            key=lambda w: int(w["id"]),
        )
        print(
            f"  level {level}: {len(same_floor)} walls "
            f"{[int(w['id']) for w in same_floor]}"
        )

        fig = go.Figure()
        for i, wall in enumerate(same_floor):
            wid = str(wall["id"])
            raw = points_by_wall.get(wid, [])
            pts = np.array(raw) if raw else np.empty((0, 3))
            color = palette[i % len(palette)]

            if len(pts):
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        marker=dict(size=POINT_SIZE, color=color, opacity=0.7),
                        name=f"wall_{wid} ({len(pts)} pts)",
                    )
                )

            bbox = wall.get("bbox", [])
            if len(bbox) == 8:
                bbox_xyz = np.array([[p["x"], p["y"], p["z"]] for p in bbox])
                fig.add_trace(_bbox_edge_traces(bbox_xyz, color))

        fig.update_layout(
            title=f"{pickle_path.stem} — all walls on levelIndex={level} "
                  f"(pickle feet coords)",
            scene=dict(
                xaxis_title="X (ft)",
                yaxis_title="Y (ft)",
                zaxis_title="Z (ft)",
                aspectmode="data",
            ),
        )

        output_html = f"{pickle_path.stem}_level{level}.html"
        fig.write_html(output_html)
        print(f"  wrote {output_html}")


if __name__ == "__main__":
    main()

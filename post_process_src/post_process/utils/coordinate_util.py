"""Convert post-process output geometry back to the original point-cloud frame.

Post-processing runs on a preprocessed cloud (e.g. metres -> feet, then shifted
to the origin), so all output coordinates are in that frame:

    post = original * scale - offset

This module inverts that, so output JSON can be expressed in the *original*
cloud's coordinates:

    original = (post + offset) / scale          # points / coordinates
    original_length = length / scale            # widths / heights (differences)

It walks the combined ``final_output`` dict as well as the individual stage
dicts (floors / ceilings / walls / openings); only the keys present are touched.

Note: this inverts the preprocessing scale + shift only. It assumes the output
shares the original cloud's orientation (true when SURVEY_BASIS is the identity);
a non-identity survey rotation would also need to be inverted here.
"""

import copy


def _converters(scale, offset):
    scale = float(scale)
    ox, oy, oz = (float(v) for v in offset)

    def to_point(p):
        # {x, y[, z]} -- footprints have no z
        p["x"] = (p["x"] + ox) / scale
        p["y"] = (p["y"] + oy) / scale
        if "z" in p:
            p["z"] = (p["z"] + oz) / scale

    def to_z(value):
        return (value + oz) / scale

    def to_length(value):
        return value / scale

    return to_point, to_z, to_length


def to_original_coordinates(output, scale, offset):
    """Return a deep copy of a post-process output dict in original coordinates.

    Args:
        output: a stage output dict or the combined final-output dict.
        scale: POINT_CLOUD_TO_POST_PROCESSING_SCALE used to build the walls
            (feet-per-meter, or 1.0 if the cloud was not scaled).
        offset: the xyz_offset (3 values) subtracted during preprocessing.
    """
    data = copy.deepcopy(output)
    to_point, to_z, to_length = _converters(scale, offset)

    # Labelled points: [{location: {x,y,z}, ...}]
    for point in data.get("points", []):
        if isinstance(point.get("location"), dict):
            to_point(point["location"])

    # Floors / ceilings: [{edgePoints: [{x,y,z}]}]
    for key in ("floors", "ceilings"):
        for element in data.get(key, []):
            for edge_point in element.get("edgePoints", []):
                to_point(edge_point)

    # Walls: bbox [{x,y,z}], footprint [{x,y}], zRange {min,max}
    for wall in data.get("walls", []):
        for corner in wall.get("bbox", []):
            to_point(corner)
        for footprint_point in wall.get("footprint", []):
            to_point(footprint_point)
        z_range = wall.get("zRange")
        if isinstance(z_range, dict):
            z_range["min"] = to_z(z_range["min"])
            z_range["max"] = to_z(z_range["max"])

    # Levels: [{zMode}]
    for level in data.get("levels", []):
        if "zMode" in level:
            level["zMode"] = to_z(level["zMode"])

    # Openings: bbox [{x,y,z}], bottomZ, topZ (coords), width, height (lengths)
    for key in ("doors", "windows", "openings"):
        for element in data.get(key, []):
            for corner in element.get("bbox", []):
                to_point(corner)
            if "bottomZ" in element:
                element["bottomZ"] = to_z(element["bottomZ"])
            if "topZ" in element:
                element["topZ"] = to_z(element["topZ"])
            if "width" in element:
                element["width"] = to_length(element["width"])
            if "height" in element:
                element["height"] = to_length(element["height"])

    return data

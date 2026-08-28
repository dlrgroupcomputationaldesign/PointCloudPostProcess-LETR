import math

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional
import torch.nn.functional as F
from ..runtime import logger
from .misc import nested_tensor_from_tensor_list
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from .blob_util import (
    plot_clusters_with_aabbs_html_bytes,
    write_html_output,
    write_image_output,
)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return image

class ToTensor(object):
    def __call__(self, img):
        return functional.to_tensor(img)

def resize(image, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = functional.resize(image, size)

    return rescaled_image

class Resize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img):
        size = self.sizes
        return resize(img, size, self.max_size)

# Function to classify lines
def classify_lines(lines, vertical_threshold, horizontal_threshold):
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1 = line.coords[0]
        x2, y2 = line.coords[1]
        
        # Avoid division by zero
        if abs(x2 - x1) < 1e-6:  
            vertical_lines.append(line)
            continue
        
        slope = abs((y2 - y1) / (x2 - x1))  # Compute absolute slope

        if slope > vertical_threshold:
            vertical_lines.append(line)
        elif slope < horizontal_threshold:
            horizontal_lines.append(line)

    return vertical_lines, horizontal_lines

def merge_finalize_polygon(merged_hor, merged_vert, image, resize_ratio):
    # Process each polygon in merged_hor
    corrected_polygons_h = []
    for hploy in merged_hor.geoms:

        # Clamp exterior coordinates
        exterior = clamp_coordinates(hploy.exterior.coords, image.shape[0], image.shape[1], resize_ratio)
        
        # Clamp interior coordinates (holes)
        interiors = [clamp_coordinates(ring.coords, image.shape[0], image.shape[1], resize_ratio) for ring in hploy.interiors]
        
        # Create a new Polygon with corrected coordinates
        corrected_polygons_h.append(Polygon(exterior, interiors))

    # Create a MultiPolygon with corrected polygons
    corrected_merged_hor = MultiPolygon(corrected_polygons_h)

    # Process each polygon in merged_hor
    corrected_polygons_v = []
    for vploy in merged_vert.geoms:
        # Clamp exterior coordinates
        exterior = clamp_coordinates(vploy.exterior.coords, image.shape[0], image.shape[1], resize_ratio)
        
        # Clamp interior coordinates (holes)
        interiors = [clamp_coordinates(ring.coords, image.shape[0], image.shape[1], resize_ratio) for ring in vploy.interiors]
        
        # Create a new Polygon with corrected coordinates
        corrected_polygons_v.append(Polygon(exterior, interiors))

    # Create a MultiPolygon with corrected polygons
    corrected_merged_vert = MultiPolygon(corrected_polygons_v)

    polyhv_arr = []
    for hploy in corrected_merged_hor.geoms:
        polyhv_arr.append(list(hploy.exterior.coords))
    for vploy in corrected_merged_vert.geoms:
        polyhv_arr.append(list(vploy.exterior.coords))

    return polyhv_arr

# Function to clamp negative coordinates to 0
def clamp_coordinates(coords, h, w, resize_ratio):
    return [(max(0, min(h, y*resize_ratio)), max(0, min(w, x*resize_ratio))) for y, x in coords]

def pixel_to_xy(px, py, x_edges, y_edges, img_width, img_height, bins=None):
    """Image pixel -> world XY, as a continuous affine map.

    This used to route through the PROJECTED_BINS grid: work out which bin the
    pixel fell in, then return that BIN'S CENTRE. That quantized every output
    vertex onto a grid far coarser than the image it came from. On one building
    the grid step was 0.55 units against a real wall thickness of 0.92, so every
    wall width collapsed onto an exact multiple of the step -- an 11-inch wall
    was emitted as 1.10 units, 20% too thick, and no intermediate width could be
    expressed at all.

    Nothing required that. project_points_to_floor renders the histogram with
    ``extent=[x_edges[0], x_edges[-1], ...]``, so the image already spans the
    data extent linearly and the pixel grid is finer than the bin grid. Mapping
    straight through keeps the detector's sub-bin precision instead of throwing
    it away, and separates the two concerns: PROJECTED_BINS sets the DETECTION
    resolution, this sets the OUTPUT resolution.

    ``+ 0.5`` because a pixel is a cell, not a sample point -- its centre sits
    half a pixel in. ``bins`` is accepted and ignored, so existing positional
    callers keep working.
    """
    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    x_coord = x_min + (px + 0.5) / img_width * (x_max - x_min)
    y_coord = y_min + (py + 0.5) / img_height * (y_max - y_min)

    return x_coord, y_coord

def find_z(points, filtered_points): ##
    # Initialize new column with NaN (or another default value)
    new_column = np.full(points.shape[0], np.nan)  

    for i, (x, y) in enumerate(points):
        match_idx = np.where((filtered_points[:, 0] == x) & (filtered_points[:, 1] == y))  # Find matching row index
        if match_idx[0].size > 0:
            new_column[i] = filtered_points[match_idx[0][0], 2]  # Assign third column value from A

    # Concatenate the new column 
    points_xyz = np.hstack((points, new_column.reshape(-1, 1)))

    return points_xyz

def find_zrgb(points, lookup):
    out = np.zeros((points.shape[0], 4)) * np.nan  # z,r,g,b columns

    for i, (x, y) in enumerate(points):
        out[i] = lookup.get((float(x), float(y)), [np.nan, np.nan, np.nan, np.nan])

    return np.hstack([points, out])

def load_line_segmentation_model(checkpoint):
    from ..models import build_model

    # load model
    args = checkpoint['args']
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def buffer_from_wall_thickness(x_edges, resize_width, wall_thickness):
    """BUFFER_THR in resized-image pixels, from a physical wall thickness.

    Detected lines are buffered into wall polygons in the RESIZED image's pixel
    space, so the buffer radius has to be half the wall's thickness expressed in
    those pixels -- and that scale changes with both the building's extent and
    RESIZE_WIDTH. A fixed value therefore means a different physical width on
    every project: 2 px on one of ours is 1.65 ft of wall against an actual
    0.92 ft (11 in), so each wall polygon swallowed ~80% more width than the
    wall occupies, pulling in points from the rooms either side.

    ``wall_thickness`` is in the cloud's own units.
    """
    extent = float(x_edges[-1] - x_edges[0])
    px_per_unit = float(resize_width) / max(extent, 1e-9)
    return max(0.5, 0.5 * float(wall_thickness) * px_per_unit)


def slope_thresholds(angle_tol_deg):
    """(VERT_THR, HORI_THR) from ONE angular tolerance, in degrees.

    The two are reciprocals of each other and always were: the shipped 10 and
    0.1 both encode tan(5.71 deg). Keeping them as separate numbers let them
    drift apart into a pair that means nothing geometrically.
    """
    t = math.tan(math.radians(float(angle_tol_deg)))
    t = max(t, 1e-9)
    return 1.0 / t, t


def img_process_model_input(image, RESIZE_WIDTH, INT_THR, morph_kernel=3):
    aspect_ratio = image.shape[1] / image.shape[0]  # width/height
    resize_ratio = image.shape[1] / RESIZE_WIDTH
    new_height = int(RESIZE_WIDTH / aspect_ratio)

    # NOTE ON RESIZE_WIDTH: this is a working resolution, NOT the model's input
    # size -- Resize([test_size]) below sets that. Downsampling here and
    # upsampling there discards detail for nothing: INTER_AREA averages, so a
    # thin wall line drops below INT_THR and vanishes before the model sees it.
    # Measured on one slab, 400 -> 1200 took line detections from 38 to 52.
    resized_image = cv2.resize(image, (RESIZE_WIDTH, new_height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale for image thresholding
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # "auto" uses Otsu, which picks the split from the image's own histogram.
    # A fixed INT_THR is a value on a colormapped, re-rendered, resized image --
    # it has no relation to point density and shifts if any of those change.
    if isinstance(INT_THR, str) and str(INT_THR).lower() == "auto":
        used_thr, binary = cv2.threshold(
            gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        logger.info("wall image: Otsu threshold %.0f", used_thr)
    else:
        _, binary = cv2.threshold(gray_image, float(INT_THR), 255, cv2.THRESH_BINARY)

    # Closing repairs single-pixel dropouts along a wall so the line stays
    # continuous. The shipped kernel was (1, 1), which is a no-op -- dilating
    # and eroding by a 1x1 structuring element returns the input unchanged.
    k = int(morph_kernel)
    if k > 1:
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.erode(cv2.dilate(binary, kernel, iterations=1), kernel, iterations=1)

    color_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    h, w = color_image.shape[0], color_image.shape[1]
    orig_size = torch.as_tensor([int(h), int(w)])

    # normalize image
    test_size = 1100
    normalize = Compose([
            ToTensor(),
            Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
            Resize([test_size]),
        ])
    img = normalize(color_image)
    inputs = nested_tensor_from_tensor_list([img])
    # plt.axis('off')

    return inputs, orig_size, resize_ratio

def line_segmentation_inf(model, inputs, orig_size, image, resize_ratio, SCORE_THR, VERT_THR, HORI_THR, BUFFER_THR):
    # no_grad is not optional here. This is inference, but without it torch
    # keeps every intermediate activation for a backward pass that never comes,
    # which on a transformer roughly doubles peak memory -- and the attention
    # tensor is already the largest allocation in the pipeline. The model is
    # handed a ~2418x1100 image (Resize([test_size]) scales the SHORT side to
    # 1100 regardless of RESIZE_WIDTH), so that allocation runs to gigabytes and
    # this is the difference between running and an OOM.
    with torch.no_grad():
        outputs = model(inputs)[0]
    out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    img_h, img_w = orig_size.unbind(0)
    scale_fct = torch.unsqueeze(torch.stack([img_w, img_h, img_w, img_h], dim=0), dim=0)
    lines = out_line * scale_fct[:, None, :]
    lines = lines.view(1000, 2, 2)
    lines = lines.flip([-1])# this is yxyx format
    scores = scores.detach().numpy()
    keep = scores >= SCORE_THR    # threshold
    keep = keep.squeeze()
    lines = lines[keep]

    # reshape(-1, 4), NOT reshape(lines.shape[0], -1): when no segment clears
    # SCORE_THR the tensor has 0 elements and -1 cannot be inferred, which
    # raises and kills the whole run. Zero segments is a legitimate outcome --
    # merge_finalize_polygon and extract_bbox_minmax both handle an empty list
    # and the level simply yields no walls -- so it must not be a crash.
    # Naming 4 explicitly makes the shape unambiguous at any length.
    lines = lines.reshape(-1, 4)

    # Silently returning no walls is worse than saying so: this is also the
    # number to look at when tuning, since it says whether SCORE_THR or the
    # image thresholding is what starved the detector.
    logger.info("line segmentation: %d of %d segments above SCORE_THR=%.2f",
                len(lines), int(np.size(scores)), SCORE_THR)
    if not len(lines):
        logger.info("  no segments survived -- this level will produce no walls. "
                    "Lower SCORE_THR, or check the projected image (INT_THR / "
                    "PROJECTED_BINS) is not blank.")
        return []

    # Convert tensor to a list of LineStrings
    lst_lines = [
        LineString([(x1, y1), (x2, y2)])
        for x1, y1, x2, y2 in lines.detach().numpy()
    ]
    
    # Classify lines
    vertical_lines, horizontal_lines = classify_lines(lst_lines, VERT_THR, HORI_THR)

    bufferedV_lines = [line.buffer(BUFFER_THR) for line in vertical_lines]
    merged_vert = unary_union(bufferedV_lines)

    bufferedH_lines = [line.buffer(BUFFER_THR) for line in horizontal_lines]
    merged_hor = unary_union(bufferedH_lines)

    polyhv_arr = merge_finalize_polygon(merged_hor, merged_vert, image, resize_ratio)
    
    return polyhv_arr

def extract_bbox_minmax(
        ori_find_z,
        blobs=None,
        snapshot_idx=None,
        local_dir=None
    ):
    import open3d as o3d

    # Generate distinct colors using a colormap
    cmap = plt.get_cmap("jet", len(ori_find_z))
    colors = [cmap(i)[:3] for i in range(len(ori_find_z))]  # Extract RGB values

    clusters = []   # list of dicts: {"points": (Ni,3), "color": (3,), "min": (3,), "max": (3,)}
    bbox_minmax = []

    for i, points in enumerate(ori_find_z):
        # points: (Ni, 3)
        pts = np.asarray(points)

        # Open3D only used here to compute AABB (optional; you can also do np.min/np.max)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = np.asarray(bbox.min_bound)
        max_bound = np.asarray(bbox.max_bound)

        color = np.asarray(colors[i])  # should be (3,) in [0,1] like Open3D

        clusters.append({"points": pts, "color": color, "min": min_bound, "max": max_bound})
        bbox_minmax.append([min_bound, max_bound])

    # Visualize everything
    # o3d.visualization.draw_geometries(geometries)
    # Snapshots go to blob when configured, else to local_dir, else nowhere --
    # the same fallback the floor and ceiling stages use, so a run with
    # logging_blob_location=None still produces its diagnostics while tuning.
    if (blobs is not None or local_dir) and clusters:
        html_bytes = plot_clusters_with_aabbs_html_bytes(clusters, point_size=2)
        write_html_output(html_bytes, f"wall_bbox_{snapshot_idx}.html",
                          blobs, local_dir)

    return bbox_minmax


def wall_overlay_image(image, polyhv_arr):
    """Detected wall footprints drawn over the projected density image.

    This is the diagnostic that wall *counts* cannot give: a count rises both
    when a real wall is recovered and when one wall breaks into three pieces.
    Each polygon is filled in a distinct colour and outlined, so adjacent
    fragments of what should be a single wall are visible as colour changes
    along an unbroken line.

    ``polyhv_arr`` holds (row, col) pixel pairs in the ORIGINAL projected
    image's frame -- merge_finalize_polygon has already undone resize_ratio --
    so no rescaling is needed here. Returns an RGB array.
    """
    base = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_BGR2RGB).copy()
    if not polyhv_arr:
        return canvas

    cmap = plt.get_cmap("hsv", max(len(polyhv_arr), 1))
    fill = canvas.copy()
    for i, poly in enumerate(polyhv_arr):
        pts = np.array([[int(round(x)), int(round(y))] for y, x in poly], np.int32)
        if len(pts) < 3:
            continue
        colour = tuple(int(255 * c) for c in cmap(i)[:3])
        cv2.fillPoly(fill, [pts], colour)
        cv2.polylines(canvas, [pts], True, colour, 1, cv2.LINE_AA)

    # Fills are blended rather than opaque so the underlying point density
    # stays readable -- the question is whether a polygon sits on actual points.
    return cv2.addWeighted(fill, 0.45, canvas, 0.55, 0.0)

def convert_to_edge_points(bboxes):
    """
    Converts a list of bounding boxes (min/max coordinates) into their 8 corner points.

    Parameters:[]
        bboxes (list of tuples): Each tuple contains (min_coords, max_coords),
                                 where min_coords and max_coords are numpy arrays.

    Returns:
        list of lists: Each inner list contains 8 corner points of a bounding box.
    """
    edge_points_list = []
    
    for min_coords, max_coords in bboxes:
        # Generate the 8 corner points
        edge_points = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # (minx, miny, minz)
            [max_coords[0], min_coords[1], min_coords[2]],  # (maxx, miny, minz)
            [max_coords[0], max_coords[1], min_coords[2]],  # (maxx, maxy, minz)
            [min_coords[0], max_coords[1], min_coords[2]],  # (minx, maxy, minz)
            [min_coords[0], min_coords[1], max_coords[2]],  # (minx, miny, maxz)
            [max_coords[0], min_coords[1], max_coords[2]],  # (maxx, miny, maxz)
            [max_coords[0], max_coords[1], max_coords[2]],  # (maxx, maxy, maxz)
            [min_coords[0], max_coords[1], max_coords[2]]   # (minx, maxy, maxz)
        ])
        
        edge_points_list.append(edge_points)
    
    return edge_points_list







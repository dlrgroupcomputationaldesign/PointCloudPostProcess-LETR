import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as functional
import torch.nn.functional as F
from models import build_model
from util.misc import nested_tensor_from_tensor_list
import geopandas as gpd
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
import open3d as o3d
import pickle


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

def pixel_to_xy(px, py, x_edges, y_edges, img_width, img_height, bins):
    bin_x = int(px / img_width * bins)
    bin_y = int(py / img_height * bins)

    # Bin center coordinates
    x_min, x_max = x_edges[0], x_edges[-1]
    y_min, y_max = y_edges[0], y_edges[-1]

    bin_width_x = (x_max - x_min) / bins
    bin_width_y = (y_max - y_min) / bins

    x_coord = x_min + bin_x * bin_width_x + bin_width_x / 2
    y_coord = y_min + bin_y * bin_width_y + bin_width_y / 2

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

def find_zrgb(points, filtered_points):
    # Initialize new columns (4 values: z, r, g, b) with NaN
    new_columns = np.full((points.shape[0], 4), np.nan)

    for i, (x, y) in enumerate(points):
        match_idx = np.where((filtered_points[:, 0] == x) & (filtered_points[:, 1] == y))
        if match_idx[0].size > 0:
            new_columns[i] = filtered_points[match_idx[0][0], 2:6]  # Extract z, r, g, b from filtered_points

    # Concatenate original points with new columns
    points_zrgb = np.hstack((points, new_columns))

    return points_zrgb

def load_line_segmentation_model(checkpoint):
    # load model
    args = checkpoint['args']
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def img_process_model_input(image, RESIZE_WIDTH, INT_THR):  
    aspect_ratio = image.shape[1] / image.shape[0]  # width/height
    resize_ratio = image.shape[1] / RESIZE_WIDTH
    new_height = int(RESIZE_WIDTH / aspect_ratio)

    # Resize the image while keeping the aspect ratio
    resized_image = cv2.resize(image, (RESIZE_WIDTH, new_height), interpolation=cv2.INTER_AREA)

    # Convert to grayscale for image thresholding
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Apply threshold: if pixel intensity > INT_THR, set to 255; otherwise, set to 0
    _, binary = cv2.threshold(gray_image, INT_THR, 255, cv2.THRESH_BINARY)

    kernel = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    eroded_image = cv2.erode(dilation, kernel, iterations=1)

    color_image = cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)

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
    # plt.imshow(color_image)

    return inputs, orig_size, resize_ratio

def line_segmentation_inf(model, inputs, orig_size, image, resize_ratio, SCORE_THR, VERT_THR, HORI_THR, BUFFER_THR):
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
    lines = lines.reshape(lines.shape[0], -1)

    # Convert tensor to a list of LineStrings
    lst_lines = gpd.GeoSeries([LineString([(x1, y1), (x2, y2)]) for x1, y1, x2, y2 in lines.detach().numpy()])
    
    # Classify lines
    vertical_lines, horizontal_lines = classify_lines(lst_lines, VERT_THR, HORI_THR)

    bufferedV_lines = [line.buffer(BUFFER_THR) for line in vertical_lines]
    merged_vert = unary_union(bufferedV_lines)

    bufferedH_lines = [line.buffer(BUFFER_THR) for line in horizontal_lines]
    merged_hor = unary_union(bufferedH_lines)

    polyhv_arr = merge_finalize_polygon(merged_hor, merged_vert, image, resize_ratio)
    
    return polyhv_arr

def extract_bbox_minmax(ori_find_z):
    # Generate distinct colors using a colormap
    cmap = plt.get_cmap("jet", len(ori_find_z))
    colors = [cmap(i)[:3] for i in range(len(ori_find_z))]  # Extract RGB values

    # List to hold Open3D geometries
    geometries = []
    bbox_minmax = []
    # Process each point set
    for i, points in enumerate(ori_find_z):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(colors[i], (points.shape[0], 1)))  # Assign color
        geometries.append(pcd)

        # Compute Axis-Aligned Bounding Box (AABB)
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # Green color for bounding box
        min_bound = bbox.min_bound  # Minimum (x, y, z)
        max_bound = bbox.max_bound 
        bbox_minmax.append([min_bound, max_bound])
        geometries.append(bbox)

    # Visualize everything
    o3d.visualization.draw_geometries(geometries)
    return bbox_minmax

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







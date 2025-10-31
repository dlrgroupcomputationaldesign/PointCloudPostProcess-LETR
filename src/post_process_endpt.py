import pandas as pd
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import alphashape
from shapely.geometry import Point, LineString, Polygon
import torch
from wall_seg_util import *
from io import BytesIO
from PIL import Image
from scipy.stats import mode
import json

# DBSCAN
EPS = 0.5
MIN_SAMPLES = 10

# Floor RANSAC
DIS_THR_F = 1 
RANSAC_N_F = 10 
NUM_ITER_F = 1000 
ALPHA_F = 1

# Ceiling RANSAC
DIS_THR_C = 4
RANSAC_N_C = 10 
NUM_ITER_C = 1000 
ALPHA_C = 1

PROJECTED_BINS = 300
RESIZE_WIDTH = 400
INT_THR = 35
SCORE_THR = 0.55
VERT_THR = 10
HORI_THR = 0.1
BUFFER_THR = 2

def cluster_floor_ceiling(df):
    # Extract xyzrgb columns (x, y, z, r, g, b)
    points = df[['x', 'y', 'z']].values
    colors = df[['r', 'g', 'b']].values
    colors_nor = df[['r', 'g', 'b']].values / 255.0  # Normalize RGB values to [0, 1]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_nor)  # Assign RGB colors
    o3d.visualization.draw_geometries([pcd])
    # Optionally: Normalize the features for DBSCAN (x, y, z, r, g, b)
    features = np.hstack([points, colors_nor])  # Use x, y, z, r, g, b as features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)  # Adjust parameters as needed
    cluster_num = dbscan.fit_predict(features_scaled)

    # Count the number of clusters (excluding noise points, labeled as -1)
    unique_num, counts = np.unique(cluster_num, return_counts=True)
    
    # Exclude noise (-1) & number of points < 10000 from cluster count
    cluster_dict = {label: np.hstack([points[cluster_num == label], colors[cluster_num == label]])for idx, label in enumerate(unique_num) if label != -1 and counts[idx] > 10000}
            
    return cluster_dict
    
def fit_ceiling_floor(df, distance_threshold, ransac_n, num_iterations, alpha_value):
    # Extract XYZ and RGB columns
    xyz = df[['x', 'y', 'z']].values  # Point coordinates
    rgb = df[['r', 'g', 'b']].values / 255.0  # Normalize RGB values (0-1)
    points = df[['x', 'y']].values
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)  # Assign RGB colors

    # Apply RANSAC plane fitting
    plane_model, inliers = pcd.segment_plane(distance_threshold,  # Adjust threshold as needed
                                            ransac_n,
                                            num_iterations)

    # Extract plane parameters (ax + by + cz + d = 0)
    a, b, c, d = plane_model
    # print(f"Plane equation: {a}x + {b}y + {c}z + {d} = 0")

    # Extract inlier and outlier points
    inlier_cloud = pcd.select_by_index(inliers)  # Points belonging to the plane
    outlier_cloud = pcd.select_by_index(inliers, invert=True)  # Points not in the plane

    # Color the plane points (for visualization)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Red for plane
    outlier_cloud.paint_uniform_color([0, 0, 1.0])  # Blue for non-plane

    # Visualize the result
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name="Plane Fitting")

    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
    bbox = pcd.get_oriented_bounding_box()
    bbox.color = (0, 1, 0)  # Green box
    bbox_zmin = bbox.get_min_bound()[2]  # Compute the center
    bbox_zmax = bbox.get_max_bound()[2]
    o3d.visualization.draw_geometries([inlier_cloud, bbox])

    floor_points = np.asarray(pcd.points)[inliers]
    floor_points_2d = floor_points[:, :2]

    # Choose an appropriate alpha value. This is crucial and might require experimentation.
    # A smaller alpha will result in a tighter, more detailed (potentially fragmented) shape.
    # A larger alpha will approach the convex hull.
    
    # Create an alpha shape
    alpha_shape = alphashape.alphashape(floor_points_2d, alpha_value)
    if alpha_shape.geom_type == 'Polygon':
        boundary_coords = alpha_shape.exterior.coords
    elif alpha_shape.geom_type == 'MultiPolygon':
        boundary_coords = max(alpha_shape.geoms, key=lambda p: p.area).exterior.coords
    else:
        print("Alpha shape is not a Polygon or MultiPolygon.")
        boundary_coords = []

    # Plot if boundary was found
    if boundary_coords:
        boundary_array = np.array(boundary_coords)
        plt.figure()
        plt.scatter(points[:, 0], points[:, 1], s=10, label='Floor Points')
        plt.plot(boundary_array[:, 0], boundary_array[:, 1], 'r-', linewidth=2, label='Boundary')
        plt.scatter(boundary_array[:, 0], boundary_array[:, 1], s=1, color='red', label='Corner Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Alpha Shape Boundary (alpha={alpha_value})')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()
    else:
        print("No valid boundary found.")

    
    line = LineString(boundary_coords)
    simplified = line.simplify(tolerance=0.8)  # tweak tolerance

    corner_coords = np.array(simplified.coords)

    extruded_coords = []
    for x, y in corner_coords:
        extruded_coords.append([x, y, bbox_zmin])
        extruded_coords.append([x, y, bbox_zmax])

    # Plot
    plt.scatter(points[:, 0], points[:, 1], s=1)
    plt.scatter(corner_coords[:, 0], corner_coords[:, 1], color="red", label="Corner Points")
    plt.title('Alpha Shape (Concave Hull)')
    plt.show()

    return bbox_zmin, bbox_zmax, extruded_coords

def points_between_level(bbox1, bbox2, xyzrgb):
    z_min_floor = bbox1[1]
    z_max_floor = bbox2[0]
    # Filter points between the floors
    filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor) & (xyzrgb[:, 2] < z_max_floor)]
    return filtered_points

def project_points_to_floor(filtered_points, bins):
    # Project to XY plane (ignore Z)
    xy_projected = filtered_points[:, :2]  # Keep only x, y
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(xy_projected[:, 0], xy_projected[:, 1], bins=bins)

    # Plot histogram
    plt.figure(figsize=(10, 8))
    plt.imshow(hist.T, origin='lower', cmap='hot', extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    plt.axis('off')

    # Save plot to a BytesIO object
    img_bytes = BytesIO()
    # Save figure without extra white space
    img_path = 'bridger.png'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    # img_bytes.seek(0) 
    img = Image.open(img_bytes)
    img_array = np.array(img)
    bgr_arr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

    return xy_projected, x_edges, y_edges, bgr_arr

def point_axis_align(df, survey_basis):
    xyz = df[['x', 'y', 'z']].values
    rotated_xyz = xyz @ survey_basis.T

    # Convert back to DataFrame
    df_rotated = pd.DataFrame(rotated_xyz, columns=['x', 'y', 'z'])
    df_concate = pd.concat([df_rotated, df.iloc[:, 3:]], axis=1)

    return df_concate

def sorted_merged_floor_ceiling_plane(bbox_arr):
    sorted_data = sorted(bbox_arr, key=lambda x: x[1]) #sort by zmin

    # Create the result array
    plane_arr = []
    i = 0
    while i < len(sorted_data):
        current = sorted_data[i]

        if current[0] == 'ceiling' and i + 1 < len(sorted_data):
            next_item = sorted_data[i + 1]
            if next_item[0] == 'floor':
                # Merge ceiling and next floor
                z_min = min(current[1], next_item[1])
                z_max = max(current[2], next_item[2])
                plane_arr.append([z_min, z_max])
                i += 2  # Skip both
                continue

        # Otherwise, just add the current one
        plane_arr.append([current[1], current[2]])
        i += 1
    return plane_arr


def rotate_pt_back(mean, R, rotated_points):
    # Step 1: Shift rotated points back to the origin
    rotated_centered_points = rotated_points - mean

    # Step 2: Apply inverse rotation (transpose of R)
    original_points = (R.T @ rotated_centered_points.T).T

    # Step 3: Shift back to original mean
    original_points += mean
    return original_points

# ensure using GPU 

print("Torch:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Using GPU 0 ->", torch.cuda.get_device_name(0))
    torch.zeros(1, device="cuda")  # hard assert we can allocate on GPU
else:
    raise SystemExit("CUDA not available — running on CPU.")

# Read the label file
with open("src\labels_clean2.txt", "r") as file:
    labels = [line.strip() for line in file.readlines()]  # Read and clean lines

# Create dictionary mapping labels to numeric values
label_dict = {label: idx for idx, label in enumerate(labels)}


# Load the CSV file
# file_name = '00-10231-20_CortevaYorkTest_Output'
# survey_basis = np.array([
#     [1.0, 0.0, 0.0],  # X
#     [0.0, 1.0, 0.0],  # Y
#     [0.0, 0.0, 1.0]   # Z
#     ]).T  # 3x3 rotation matrix

file_name = 'WyomingStateFair_Laramie_Output'
survey_basis = np.array([
    [-0.23829087279918065, 0.9711938323221605, 0.0],   # X
    [-0.9711938323221605, -0.23829087279918065, 0.0],  # Y
    [0.0, 0.0, 0.99999999999999978]                    # Z
    ]).T  # 3x3 rotation matrix

# file_name = 'WyomingStateFair_Bridger_Output'
# survey_basis = np.array([
#     [-0.23636311131930887,
#       0.97166479796659089,
#       0.0],  # X
#     [-0.97166479796659089,
#       -0.23636311131930887,
#       0.0],  # Y
#     [0.0,
#       0.0,
#       0.99999999999999989]   # Z
#     ]).T  # 3x3 rotation matrix

# file_name = 'WyomingStateFair_Fetterman_Output'
# survey_basis = np.array([
#     [0.19714381736663131,
#       0.9803745790635906,
#       0.0],
#     [-0.9803745790635906,
#       0.19714381736663131,
#       0.0],
#     [0.0,
#       0.0,
#       0.99999999999999989]
#     ]).T

# file_name = 'AdultEd_Labled_Output'

file_path = f"src/{file_name}.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# prepare for JSON
points_lst, level_lst, wall_lst = [], [], []

### floor
print('Processing Floor......')
segment_floor_df = df[df['pred_label']==label_dict['Floor']].reset_index(drop=True)
# _, mean, R = point_axis_align_prev(segment_floor_df) 
align_axis_floor_df = point_axis_align(segment_floor_df, survey_basis)  ##
cluster_dict_floor = cluster_floor_ceiling(align_axis_floor_df) ##
# cluster_dict_floor = cluster_floor_ceiling(segment_df)
floor_ceiling_bboxz = [] ##
corner = []
level_id = 1

for num in cluster_dict_floor:
    df_points_colors = pd.DataFrame(cluster_dict_floor[num], columns=['x', 'y', 'z', 'r', 'g', 'b'])

    bbox_zmin, bbox_zmax, corner_xyz = fit_ceiling_floor(df_points_colors, DIS_THR_F, RANSAC_N_F, NUM_ITER_F, ALPHA_F)
    floor_ceiling_bboxz.append(['floor', bbox_zmin, bbox_zmax])
    
    rotated_corner = corner_xyz @ survey_basis 
    corner.append(rotated_corner)

    cluster_xyz_arr = cluster_dict_floor[num][:, :3]
    cluster_rgb_arr = cluster_dict_floor[num][:, 3:]

    # Apply the rotation function to each xyz row
    rotated_xyz = cluster_xyz_arr @ survey_basis 
    # Concatenate rotated xyz with original rgb
    rotated_with_rgb = np.hstack((rotated_xyz, cluster_rgb_arr))

    z_values = [pt[2] for pt in rotated_with_rgb] # Extract the third column (z)
    mode_result = mode(z_values, keepdims=True)
    mode_z = float(mode_result.mode[0])

    # Add points
    for pt in rotated_with_rgb:
        points_lst.append({
            "category": "level",
            "id": str(level_id),
            "location": {
                "x": float(pt[0]),
                "y": float(pt[1]),
                "z": float(pt[2])
            },
            "color": {
                "r": int(pt[3]),
                "g": int(pt[4]),
                "b": int(pt[5])
            }
        })
    # Add level metadata
    level_lst.append({
        "id": str(level_id),
        "end_points": [[float(x), float(y), float(z)] for x, y, z in rotated_corner],
        "z_mode": mode_z
    })

    level_id += 1

### ceiling
print('Processing Ceiling......')
segment_ceiling_df = df[df['pred_label']==label_dict['Ceiling']].reset_index(drop=True)
align_axis_ceiling_df = point_axis_align(segment_ceiling_df, survey_basis) 
cluster_dict_ceiling = cluster_floor_ceiling(align_axis_ceiling_df) ##

for num in cluster_dict_ceiling:
    # Convert NumPy array to DataFrame
    df_points_colors = pd.DataFrame(cluster_dict_ceiling[num], columns=['x', 'y', 'z', 'r', 'g', 'b'])
    bbox_zmin, bbox_zmax, corner_xyz = fit_ceiling_floor(df_points_colors, DIS_THR_C, RANSAC_N_C, NUM_ITER_C, ALPHA_C)
    rotated_corner = corner_xyz @ survey_basis 
    corner.append(rotated_corner)

    cluster_xyz_arr = cluster_dict_ceiling[num][:, :3]
    cluster_rgb_arr = cluster_dict_ceiling[num][:, 3:]

    # Apply the rotation function to each xyz row
    rotated_xyz = cluster_xyz_arr @ survey_basis 
    # Concatenate rotated xyz with original rgb
    rotated_with_rgb = np.hstack((rotated_xyz, cluster_rgb_arr))
    z_values = [pt[2] for pt in rotated_with_rgb] # Extract the third column (z)
    mode_result = mode(z_values, keepdims=True)
    mode_z = float(mode_result.mode[0])

    # Add points
    for pt in rotated_with_rgb:
        points_lst.append({
            "category": "level",
            "id": str(level_id),
            "location": {
                "x": float(pt[0]),
                "y": float(pt[1]),
                "z": float(pt[2])
            },
            "color": {
                "r": int(pt[3]),
                "g": int(pt[4]),
                "b": int(pt[5])
            }
        })
    # Add level metadata
    level_lst.append({
        "id": str(level_id),
        "end_points": [[float(x), float(y), float(z)] for x, y, z in rotated_corner],
        "z_mode": mode_z
    })

    level_id += 1

print('floor_ceiling_bboxz', floor_ceiling_bboxz)
plane_arr = sorted_merged_floor_ceiling_plane(floor_ceiling_bboxz)

### wall
print('Processing Wall......')

num_level = len(cluster_dict_floor)
align_axis_df = point_axis_align(df, survey_basis)
xyzrgb = align_axis_df[['x', 'y', 'z', 'r', 'g', 'b']].values  
# obtain line segmentation model checkpoints
checkpoint = torch.load('checkpoints\checkpoint0024.pth', map_location='cpu')
model = load_line_segmentation_model(checkpoint)
wall_bbox_edge = []
print('no.level', num_level)
wall_id = 1
for level in range(num_level):
    if level == num_level - 1:    
        z_min_floor = plane_arr[-1][1]
        filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor)]
    else:
        filtered_points = points_between_level(plane_arr[level], plane_arr[level+1], xyzrgb)

    xy_projected, x_edges, y_edges, projected_img_arr = project_points_to_floor(filtered_points, PROJECTED_BINS)
    inputs, orig_size, resize_ratio = img_process_model_input(projected_img_arr, RESIZE_WIDTH, INT_THR)
    polyhv_arr = line_segmentation_inf(model, inputs, orig_size, projected_img_arr, resize_ratio, SCORE_THR, VERT_THR, HORI_THR, BUFFER_THR)
    img_width, img_height = projected_img_arr.shape[1], projected_img_arr.shape[0]
    ori_poly = [[pixel_to_xy(x, img_height-y, x_edges, y_edges, img_width, img_height, PROJECTED_BINS) for y, x in poly] for poly in polyhv_arr]
    polygons = [Polygon(row) for row in ori_poly]

    # Check which points from xy_projected are inside each polygon
    pts_in_poly = []
    for poly in polygons:
        inside_points = [[point[0],point[1]] for point in xy_projected if poly.contains(Point(point))]
        if inside_points:
            pts_in_poly.append(inside_points)
        
    points_zrgb = [find_zrgb(np.array(poly), filtered_points) for poly in pts_in_poly]

    points_xyz = [arr[:, :3] for arr in points_zrgb]
    points_rgb = [arr[:, 3:] for arr in points_zrgb]
    bbox_minmax = extract_bbox_minmax(points_xyz)
    edge_points = convert_to_edge_points(bbox_minmax)
    # rotated_edge_prev = [[rotate_pt_back(mean, R, pt) for pt in pts] for pts in edge_points]
    rotated_edge = [pts @ survey_basis for pts in edge_points]

    wall_bbox_edge.append(rotated_edge)

    rotated_wall_xyz = [pts @ survey_basis for pts in points_xyz]
    rotated_with_rgb = [np.hstack((xyz, rgb)) for xyz, rgb in zip(rotated_wall_xyz, points_rgb)]


    # Add points
    for i, bbox in enumerate(rotated_with_rgb):
        for pt in bbox:
            points_lst.append({
                "category": "wall",
                "id": str(wall_id),
                "location": {
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "z": float(pt[2])
                    },
                "color": {
                    "r": int(pt[3]),
                    "g": int(pt[4]),
                    "b": int(pt[5])
                }
            })

        wall_lst.append({
            "id": str(wall_id),
            "bbox": [[float(x), float(y), float(z)] for x, y, z in rotated_edge[i]],
        })

        wall_id += 1

# Final JSON structure
output_dict = {
    "points": points_lst,
    "levels": level_lst,
    "walls": wall_lst
}

# Save to a file
with open(f"{file_name}_test_output_rgb.json", "w") as f:
    json.dump(output_dict, f, indent=2)

with open(f"{file_name}_test_wall_bbox_edge.pkl", "wb") as f:
    pickle.dump(wall_bbox_edge, f)










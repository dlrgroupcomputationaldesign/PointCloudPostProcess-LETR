import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import torch
from .utils.wall_seg_util import *
from .utils.floor_ceiling_util import *
from scipy.stats import mode
import json
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Read the label file
with open("labels_clean2.pkl", 'rb') as file:
    labels = pickle.load(file)

# Create dictionary mapping labels to numeric values
label_dict = {label: idx for idx, label in enumerate(labels)}

def run_floors(csv_path, floors_path, parameters, output_json=False):
    print("Running floor post-processing...")
    df = pd.read_csv(csv_path)
    floor_lst, floor_level, point_lst = [], [], []
    segment_floor_df = df[df['pred_label']==label_dict['Floor']].reset_index(drop=True)

    align_axis_floor_df = point_axis_align(segment_floor_df, np.array(parameters['SURVEY_BASIS']).T)  
    cluster_dict_floor = cluster_floor_ceiling(align_axis_floor_df, parameters['EPS'], parameters['MIN_SAMPLES']) 

    floor_bboxz = [] 
    level_id = 1
    floor_id = 1

    for num in cluster_dict_floor:
        df_points_colors = pd.DataFrame(cluster_dict_floor[num], columns=['x', 'y', 'z', 'r', 'g', 'b'])

        bbox_zmin, bbox_zmax, corner_xyz = fit_ceiling_floor(df_points_colors, parameters['DIS_THR_F'], parameters['RANSAC_N_F'], parameters['NUM_ITER_F'], parameters['ALPHA_F'])
        floor_bboxz.append([bbox_zmin, bbox_zmax])
        
        rotated_corner = corner_xyz @ np.array(parameters['SURVEY_BASIS']).T

        cluster_xyz_arr = cluster_dict_floor[num][:, :3]
        cluster_rgb_arr = cluster_dict_floor[num][:, 3:]

        # Apply the rotation function to each xyz row
        rotated_xyz = cluster_xyz_arr @ np.array(parameters['SURVEY_BASIS']).T
        # Concatenate rotated xyz with original rgb
        rotated_with_rgb = np.hstack((rotated_xyz, cluster_rgb_arr))

        z_values = [pt[2] for pt in rotated_with_rgb] # Extract the third column (z)
        mode_result = mode(z_values, keepdims=True)
        mode_z = float(mode_result.mode[0])

            # Add points
        for pt in rotated_with_rgb:
            point_lst.append({
                "category": "floor",
                "id": str(floor_id),
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
        # level_lst.append({
        #     "id": str(level_id),
        #     "zMode": mode_z
        # })
        # level_id += 1

        floor_level.append(mode_z)

        floor_lst.append({
            "id": str(floor_id),
            "edgePoints": [
                {"x": float(x), "y": float(y), "z": float(z)}   
                for x, y, z in rotated_corner]
        })
        floor_id += 1

    floor_output_dict = {
    "points": point_lst,
    # "levels": level_lst,
    "floors": floor_lst
    }

    # Save to a file
    if output_json:
        with open(f"{floors_path}", "w") as f:
            json.dump(floor_output_dict, f, indent=2)

    return floor_output_dict, floor_bboxz, floor_level
        
def run_ceilings(csv_path, ceilings_path, parameters, output_json=False):
    print("Running ceiling post-processing...")
    df = pd.read_csv(csv_path)
    ceiling_lst, ceiling_level, point_lst = [], [], []   
    segment_ceiling_df = df[df['pred_label']==label_dict['Ceiling']].reset_index(drop=True)
    align_axis_ceiling_df = point_axis_align(segment_ceiling_df, np.array(parameters['SURVEY_BASIS']).T) 
    cluster_dict_ceiling = cluster_floor_ceiling(align_axis_ceiling_df, parameters['EPS'], parameters['MIN_SAMPLES']) 
    ceiling_id = 1
    level_id = 1

    for num in cluster_dict_ceiling:
        # Convert NumPy array to DataFrame
        df_points_colors = pd.DataFrame(cluster_dict_ceiling[num], columns=['x', 'y', 'z', 'r', 'g', 'b'])
        bbox_zmin, bbox_zmax, corner_xyz = fit_ceiling_floor(df_points_colors, parameters['DIS_THR_C'], parameters['RANSAC_N_C'], parameters['NUM_ITER_C'], parameters['ALPHA_C'])
        rotated_corner = corner_xyz @ np.array(parameters['SURVEY_BASIS']).T
        # corner.append(rotated_corner)

        cluster_xyz_arr = cluster_dict_ceiling[num][:, :3]
        cluster_rgb_arr = cluster_dict_ceiling[num][:, 3:]

        # Apply the rotation function to each xyz row
        rotated_xyz = cluster_xyz_arr @ np.array(parameters['SURVEY_BASIS']).T
        # Concatenate rotated xyz with original rgb
        rotated_with_rgb = np.hstack((rotated_xyz, cluster_rgb_arr))
        z_values = [pt[2] for pt in rotated_with_rgb] # Extract the third column (z)
        mode_result = mode(z_values, keepdims=True)
        mode_z = float(mode_result.mode[0])

        # Add points
        for pt in rotated_with_rgb:
            point_lst.append({
                "category": "ceiling",
                "id": str(ceiling_id),
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

        # level_lst.append({
        #     "id": str(level_id),
        #     "zMode": mode_z
        # })
        # level_id += 1

        ceiling_level.append(mode_z)

        ceiling_lst.append({
            "id": str(ceiling_id),
            "edgePoints": [
                {"x": float(x), "y": float(y), "z": float(z)}  
                for x, y, z in rotated_corner]
        })
        ceiling_id += 1

    ceiling_output_dict = {
    "points": point_lst,
    # "levels": level_lst,
    "ceilings": ceiling_lst
    }

    # Save to a file
    if output_json:
        with open(f"{ceilings_path}", "w") as f:
            json.dump(ceiling_output_dict, f, indent=2)

    return ceiling_output_dict, ceiling_level

def run_walls(csv_path, walls_path, parameters, floor_bboxz, line_seg_model, output_json=False):
    print("Running wall post-processing...")
    df = pd.read_csv(csv_path)
    wall_lst, point_lst = [], []
    num_level = len(floor_bboxz)  
    plane_arr = sorted_merged_floor_ceiling_plane(floor_bboxz)
    segment_wall_df = df[df['pred_label']==label_dict['Wall']].reset_index(drop=True)
    # align_axis_wall_df = point_axis_align(segment_wall_df, np.array(parameters['SURVEY_BASIS']).T) 
    # xyzrgb = align_axis_wall_df[['x', 'y', 'z', 'r', 'g', 'b']].values   
    
    align_axis_df = point_axis_align(df, np.array(parameters['SURVEY_BASIS']).T) 
    xyzrgb = align_axis_df[['x', 'y', 'z', 'r', 'g', 'b']].values   

    checkpoint = torch.load(line_seg_model, map_location=device)
    model = load_line_segmentation_model(checkpoint)
    # wall_bbox_edge = []
    wall_id = 1
    for level in range(num_level):
        if level == num_level - 1:    
            z_min_floor = plane_arr[-1][1]
            filtered_points = xyzrgb[(xyzrgb[:, 2] > z_min_floor)]
        else:
            filtered_points = points_between_level(plane_arr[level], plane_arr[level+1], xyzrgb)

        xy_projected, x_edges, y_edges, projected_img_arr = project_points_to_floor(filtered_points, parameters['PROJECTED_BINS'])
        inputs, orig_size, resize_ratio = img_process_model_input(projected_img_arr, parameters['RESIZE_WIDTH'], parameters['INT_THR'])
        polyhv_arr = line_segmentation_inf(model, inputs, orig_size, projected_img_arr, resize_ratio, parameters['SCORE_THR'], parameters['VERT_THR'], parameters['HORI_THR'], parameters['BUFFER_THR'])
        img_width, img_height = projected_img_arr.shape[1], projected_img_arr.shape[0]
        ori_poly = [[pixel_to_xy(x, img_height-y, x_edges, y_edges, img_width, img_height, parameters['PROJECTED_BINS']) for y, x in poly] for poly in polyhv_arr]
        polygons = [Polygon(row) for row in ori_poly]

        # Check which points from xy_projected are inside each polygon
        pts_in_poly = []
        for poly in polygons:
            inside_points = [[point[0],point[1]] for point in xy_projected if poly.contains(Point(point))]
            if inside_points:
                pts_in_poly.append(inside_points)
        
        # filtered_points: [N, 6] → columns: x, y, z, r, g, b
        lookup = {
            (float(x), float(y)): filtered_points[i, 2:6]
            for i, (x, y) in enumerate(filtered_points[:, :2])
        }
        points_zrgb = [find_zrgb(np.array(poly), lookup) for poly in pts_in_poly]

        points_xyz = [arr[:, :3] for arr in points_zrgb]
        points_rgb = [arr[:, 3:] for arr in points_zrgb]
        bbox_minmax = extract_bbox_minmax(points_xyz)
        edge_points = convert_to_edge_points(bbox_minmax)

        rotated_edge = [pts @ np.array(parameters['SURVEY_BASIS']).T for pts in edge_points]

        # wall_bbox_edge.append(rotated_edge)

        rotated_wall_xyz = [pts @ np.array(parameters['SURVEY_BASIS']).T for pts in points_xyz]
        rotated_with_rgb = [np.hstack((xyz, rgb)) for xyz, rgb in zip(rotated_wall_xyz, points_rgb)]

        # Add points
        for i, bbox in enumerate(rotated_with_rgb):
            for pt in bbox:
                point_lst.append({
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
            "bbox": [
                {"x": float(x), "y": float(y), "z": float(z)} 
                for x, y, z in rotated_edge[i]]
        })

        wall_id += 1
        
    wall_output_dict = {
    "points": point_lst,
    "walls": wall_lst
    }

    # Save to a file
    if output_json:
        with open(f"{walls_path}", "w") as f:
            json.dump(wall_output_dict, f, indent=2)

    return wall_output_dict

def final_output(floor_output, ceiling_output, wall_output, floor_level, ceiling_level, all_output_path, output_json=False):
    level_lst = []
    level_id = 1
    level = floor_level + ceiling_level
    for mode_z in level:
        level_lst.append({
            "id": str(level_id),
            "zMode": mode_z
        })
        level_id += 1

    final_output_dict = {
        "points": floor_output["points"] + ceiling_output["points"] + wall_output["points"],
        "levels": level_lst,
        "floors": floor_output["floors"],
        "ceilings": ceiling_output["ceilings"],
        "walls": wall_output["walls"]
    }

    if output_json:
        with open(all_output_path, "w") as f:
            json.dump(final_output_dict, f, indent=2)

    return final_output_dict